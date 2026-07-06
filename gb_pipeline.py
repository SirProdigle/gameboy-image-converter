"""GB Studio hardware-constrained conversion pipeline.

Pure functions over numpy arrays / PIL images — no Gradio imports here.
See docs/superpowers/specs/2026-07-06-gb-pipeline-redesign-design.md for the
authoritative behavior spec and
docs/superpowers/plans/2026-07-06-gb-pipeline-redesign-plan.md for the task
breakdown these functions implement.
"""

import heapq
from dataclasses import dataclass

import numpy as np
from PIL import Image
from skimage.color import lab2rgb, rgb2lab


@dataclass
class GBImage:
    """Mirrors GB/GBC VRAM layout for a converted image."""

    patterns: np.ndarray       # (n, 8, 8) uint8, values 0-3 -- the tileset
    tilemap: np.ndarray        # (th, tw) int32 -> pattern index per cell
    attrs_palette: np.ndarray  # (th, tw) uint8 -- palette id per cell
    attrs_hflip: np.ndarray    # (th, tw) bool
    attrs_vflip: np.ndarray    # (th, tw) bool
    palettes: np.ndarray       # (p, 4, 3) uint8, RGB555-snapped, luminance-sorted


@dataclass
class Preset:
    name: str
    tile_budget: int | None
    n_palettes: int
    allow_flips: bool
    mono: bool
    fixed_size: tuple | None


PRESETS = {
    "color_only": Preset("color_only", 384, 8, True, False, None),
    "mono": Preset("mono", 192, 1, False, True, None),
    "logo_color": Preset("logo_color", None, 8, False, False, (160, 144)),
    "logo_mono": Preset("logo_mono", None, 1, False, True, (160, 144)),
}

# DMG reference ramp, lightest first (from gb_palette.png).
DMG_RAMP = np.array(
    [[224, 248, 208], [136, 192, 112], [52, 104, 86], [8, 24, 32]],
    dtype=np.uint8,
)


def snap_rgb555(colors: np.ndarray) -> np.ndarray:
    """Snap 8-bit-per-channel colors to their RGB555 quantization.

    c5 = floor(c8 * 32 / 256), clamped to [0, 31]; expanded back to 8-bit via
    round(c5 * 255 / 31). Operates on any shape ending in (..., 3) (or any
    shape at all -- it is a plain per-element transform), uint8 in and out.
    """
    colors = np.asarray(colors, dtype=np.float64)
    c5 = np.floor(colors * 32.0 / 256.0)
    c5 = np.clip(c5, 0, 31)
    c8 = np.round(c5 * 255.0 / 31.0)
    return c8.astype(np.uint8)


def luminance_sort(palette: np.ndarray) -> np.ndarray:
    """Sort a (4, 3) RGB palette lightest-first by 2126*R + 7152*G + 722*B."""
    palette = np.asarray(palette)
    luminance = (
        2126 * palette[:, 0].astype(np.int64)
        + 7152 * palette[:, 1].astype(np.int64)
        + 722 * palette[:, 2].astype(np.int64)
    )
    order = np.argsort(-luminance, kind="stable")
    return palette[order]


def pattern_to_2bpp(pattern: np.ndarray) -> bytes:
    """Encode an (8, 8) index pattern (values 0-3) as 16 bytes of GB 2bpp data.

    Per row: byte1 = low bits of each pixel's index, byte2 = high bits;
    within a byte, bit 7 is the leftmost pixel.
    """
    pattern = np.asarray(pattern)
    assert pattern.shape == (8, 8)
    out = bytearray(16)
    for row in range(8):
        low = 0
        high = 0
        for col in range(8):
            value = int(pattern[row, col]) & 0b11
            bit_pos = 7 - col
            low |= (value & 1) << bit_pos
            high |= ((value >> 1) & 1) << bit_pos
        out[row * 2] = low
        out[row * 2 + 1] = high
    return bytes(out)


def pattern_variants(pattern: np.ndarray) -> dict:
    """Return the four flip variants of a pattern: {"", "h", "v", "hv"}."""
    pattern = np.asarray(pattern)
    return {
        "": pattern.copy(),
        "h": np.fliplr(pattern).copy(),
        "v": np.flipud(pattern).copy(),
        "hv": np.flipud(np.fliplr(pattern)).copy(),
    }


# ---------------------------------------------------------------------------
# Stage 3 -- palette packing
# ---------------------------------------------------------------------------
#
# Color distances are Euclidean in CIELAB throughout; every fitted palette
# color is snapped to RGB555 and every returned palette is luminance-sorted
# (lightest first). See the design spec, section "Stage 3 -- palette packing".


def _rgb_to_lab(colors: np.ndarray) -> np.ndarray:
    """Convert (..., 3) uint8-ish RGB to (N, 3) CIELAB (row-flattened)."""
    arr = np.asarray(colors, dtype=np.float64).reshape(-1, 3) / 255.0
    lab = rgb2lab(arr[:, None, :])  # (N, 1, 3)
    return lab.reshape(-1, 3)


def _lab_to_rgb_u8(lab: np.ndarray) -> np.ndarray:
    """Convert (N, 3) CIELAB back to (N, 3) uint8 RGB (gamut-clipped)."""
    lab = np.asarray(lab, dtype=np.float64).reshape(-1, 3)
    rgb = lab2rgb(lab[:, None, :]).reshape(-1, 3)
    return np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)


def _nearest_indices(src_lab: np.ndarray, ref_lab: np.ndarray) -> np.ndarray:
    """Index of the nearest ref row (Euclidean Lab) for each src row."""
    diff = src_lab[:, None, :] - ref_lab[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    return np.argmin(d2, axis=1)


def _pad4(palette: np.ndarray) -> np.ndarray:
    """Pad/truncate a color list to exactly 4 rows (repeat the last color)."""
    palette = np.asarray(palette, dtype=np.uint8).reshape(-1, 3)
    if len(palette) >= 4:
        return palette[:4]
    pad = np.repeat(palette[-1:], 4 - len(palette), axis=0)
    return np.concatenate([palette, pad], axis=0)


def _weighted_kmeans_lab(
    points: np.ndarray, weights: np.ndarray, k: int, seed: int = 42, n_iter: int = 25
) -> np.ndarray:
    """Deterministic weighted k-means in Lab space -> (k, 3) Lab centers.

    Uses a seeded k-means++ initialization followed by Lloyd iterations with
    pixel-count weights. If there are <= k points, each point is its own
    center.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    n = len(points)
    if n <= k:
        return points.copy()

    rng = np.random.RandomState(seed)
    total_w = weights.sum()
    probs = weights / total_w if total_w > 0 else np.full(n, 1.0 / n)
    first = int(rng.choice(n, p=probs))
    centers = [points[first]]
    d2 = np.sum((points - points[first]) ** 2, axis=1)
    for _ in range(1, k):
        p = d2 * weights
        s = p.sum()
        j = int(rng.choice(n, p=p / s)) if s > 0 else int(rng.choice(n))
        centers.append(points[j])
        d2 = np.minimum(d2, np.sum((points - points[j]) ** 2, axis=1))
    centers = np.array(centers, dtype=np.float64)

    for _ in range(n_iter):
        dists = np.sum(
            (points[:, None, :] - centers[None, :, :]) ** 2, axis=2
        )  # (n, k)
        labels = np.argmin(dists, axis=1)
        new_centers = centers.copy()
        for c in range(k):
            mask = labels == c
            if mask.any():
                w = weights[mask]
                new_centers[c] = (points[mask] * w[:, None]).sum(0) / w.sum()
        if np.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers
    return centers


def quantize_working_set(
    image: Image.Image, max_colors: int, custom_palette: np.ndarray | None = None
) -> Image.Image:
    """Reduce an image to a bounded working set of RGB555-snapped colors.

    Uses PIL's quantizer (libimagequant when available, MEDIANCUT fallback)
    with dithering disabled, then snaps the result to RGB555. When
    ``custom_palette`` is given, every pixel is mapped to the nearest color in
    that (snapped) set instead. Returns an RGB-mode image whose pixels contain
    only working-set colors.
    """
    image = image.convert("RGB")

    if custom_palette is not None:
        cust = snap_rgb555(np.asarray(custom_palette, dtype=np.uint8).reshape(-1, 3))
        cust = np.unique(cust, axis=0)
        arr = np.asarray(image, dtype=np.uint8)
        h, w = arr.shape[:2]
        flat = arr.reshape(-1, 3)
        uq, inv = np.unique(flat, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        nearest = _nearest_indices(_rgb_to_lab(uq), _rgb_to_lab(cust))
        mapped = cust[nearest][inv].reshape(h, w, 3)
        return Image.fromarray(mapped.astype(np.uint8), "RGB")

    colors = max(int(max_colors), 1)
    try:
        quant = image.quantize(
            colors=colors,
            method=Image.Quantize.LIBIMAGEQUANT,
            dither=Image.Dither.NONE,
        )
    except (ValueError, OSError):
        quant = image.quantize(
            colors=colors,
            method=Image.Quantize.MEDIANCUT,
            dither=Image.Dither.NONE,
        )
    rgb = np.asarray(quant.convert("RGB"), dtype=np.uint8)
    snapped = snap_rgb555(rgb)
    return Image.fromarray(snapped, "RGB")


def pack_palettes_mono(image: np.ndarray, ramp: np.ndarray) -> tuple:
    """Mono packing: one fixed 4-shade ramp, every cell assigned to it.

    Returns (palettes (1, 4, 3) uint8 snapped+luminance-sorted, assignment
    (H//8, W//8) uint8 of all zeros).
    """
    arr = np.asarray(image, dtype=np.uint8)
    h, w = arr.shape[:2]
    palette = luminance_sort(snap_rgb555(_pad4(ramp)))
    palettes = palette[None, :, :].astype(np.uint8)
    assignment = np.zeros((h // 8, w // 8), dtype=np.uint8)
    return palettes, assignment


def pack_palettes(
    image: np.ndarray, n_palettes: int, custom_palette: np.ndarray | None = None
) -> tuple:
    """Pack an image's tiles into <= ``n_palettes`` 4-color palettes.

    ``image`` is (H, W, 3) uint8 and already working-set-limited; H and W are
    multiples of 8. Implements the two-phase algorithm from the design spec:
    agglomerative seeding (merge cheapest palette pairs until within budget)
    followed by Lloyd refinement (reassign tiles then refit palettes).

    Returns (palettes (p <= n_palettes, 4, 3) uint8 snapped+luminance-sorted,
    assignment (H//8, W//8) uint8 palette id per tile cell).
    """
    arr = np.asarray(image, dtype=np.uint8)
    h, w = arr.shape[:2]
    th, tw = h // 8, w // 8
    n_tiles = th * tw

    # Working set: unique colors of the (already limited) image.
    flat = arr.reshape(-1, 3)
    ws_colors, inverse = np.unique(flat, axis=0, return_inverse=True)
    inv_img = inverse.reshape(h, w)
    ws_lab = _rgb_to_lab(ws_colors)

    # Optional custom-palette restriction set.
    if custom_palette is not None:
        cust = np.unique(
            snap_rgb555(np.asarray(custom_palette, dtype=np.uint8).reshape(-1, 3)),
            axis=0,
        )
        cust_lab = _rgb_to_lab(cust)
    else:
        cust = None
        cust_lab = None

    # Per-tile working-set color indices + pixel counts.
    tile_colors = [None] * n_tiles
    tile_counts = [None] * n_tiles
    for tr in range(th):
        for tc in range(tw):
            block = inv_img[tr * 8 : tr * 8 + 8, tc * 8 : tc * 8 + 8].reshape(-1)
            idx, cnt = np.unique(block, return_counts=True)
            t = tr * tw + tc
            tile_colors[t] = idx
            tile_counts[t] = cnt.astype(np.float64)

    def build_palette(idxs, cnts):
        """(4,3) uint8 palette fitted to working-set colors idxs (+counts).

        <=4 colors are represented exactly; >4 colors are fit by weighted
        4-means. Colors are snapped to RGB555, projected onto the custom set
        if one is given, and luminance-sorted. Used to produce the palettes
        actually returned -- called O(groups + merges) times, not in the
        agglomerative inner loop.
        """
        idxs = np.asarray(idxs)
        if len(idxs) <= 4:
            palette = ws_colors[idxs]
        else:
            centers = _weighted_kmeans_lab(ws_lab[idxs], cnts, 4)
            palette = _lab_to_rgb_u8(centers)
        palette = snap_rgb555(palette)
        if cust is not None:
            proj = _nearest_indices(_rgb_to_lab(palette), cust_lab)
            palette = cust[proj]
        palette = luminance_sort(snap_rgb555(_pad4(palette)))
        return palette.astype(np.uint8)

    def fit_centers_lab(idxs, cnts):
        """Lab centers for working-set colors idxs (each own center if <=4)."""
        idxs = np.asarray(idxs)
        if len(idxs) <= 4:
            return ws_lab[idxs]
        return _weighted_kmeans_lab(ws_lab[idxs], cnts, 4)

    def lab_error(idxs, cnts, centers_lab):
        """Weighted sum of each color's nearest Euclidean-Lab distance."""
        idxs = np.asarray(idxs)
        if len(idxs) == 0:
            return 0.0
        lab = ws_lab[idxs]
        diff = lab[:, None, :] - centers_lab[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2)).min(axis=1)
        return float((dist * np.asarray(cnts)).sum())

    # --- Phase 1: seed groups by ideal palette, dedupe, agglomerate ---------
    groups = {}
    key_to_id = {}
    next_id = 0
    for t in range(n_tiles):
        pal_t = build_palette(tile_colors[t], tile_counts[t])
        key = frozenset(map(tuple, pal_t.tolist()))
        if key in key_to_id:
            g = groups[key_to_id[key]]
            g["members"].append(t)
            for i, c in zip(tile_colors[t], tile_counts[t]):
                g["colors"][int(i)] = g["colors"].get(int(i), 0.0) + float(c)
        else:
            gid = next_id
            next_id += 1
            key_to_id[key] = gid
            colors = {
                int(i): float(c) for i, c in zip(tile_colors[t], tile_counts[t])
            }
            groups[gid] = {"colors": colors, "members": [t]}

    def finalize(gid):
        g = groups[gid]
        idxs = np.array(sorted(g["colors"].keys()), dtype=np.int64)
        cnts = np.array([g["colors"][int(i)] for i in idxs], dtype=np.float64)
        g["palette"] = build_palette(idxs, cnts)
        g["error"] = lab_error(idxs, cnts, fit_centers_lab(idxs, cnts))

    for gid in list(groups.keys()):
        finalize(gid)

    def merge_cost(a, b):
        ga, gb = groups[a], groups[b]
        comb = dict(ga["colors"])
        for i, c in gb["colors"].items():
            comb[i] = comb.get(i, 0.0) + c
        if len(comb) <= 4:
            return 0.0
        idxs = np.array(sorted(comb.keys()), dtype=np.int64)
        cnts = np.array([comb[int(i)] for i in idxs], dtype=np.float64)
        err = lab_error(idxs, cnts, fit_centers_lab(idxs, cnts))
        return max(err - (ga["error"] + gb["error"]), 0.0)

    alive = set(groups.keys())
    heap = []
    alive_list = sorted(alive)
    for i in range(len(alive_list)):
        for j in range(i + 1, len(alive_list)):
            a, b = alive_list[i], alive_list[j]
            heapq.heappush(heap, (merge_cost(a, b), a, b))

    while len(alive) > n_palettes:
        pair = None
        while heap:
            c, a, b = heapq.heappop(heap)
            if a in alive and b in alive:
                pair = (a, b)
                break
        if pair is None:
            break
        a, b = pair
        comb = dict(groups[a]["colors"])
        for i, cc in groups[b]["colors"].items():
            comb[i] = comb.get(i, 0.0) + cc
        members = groups[a]["members"] + groups[b]["members"]
        gid = next_id
        next_id += 1
        groups[gid] = {"colors": comb, "members": members}
        finalize(gid)
        alive.discard(a)
        alive.discard(b)
        alive.add(gid)
        for other in alive:
            if other != gid:
                heapq.heappush(heap, (merge_cost(gid, other), min(gid, other), max(gid, other)))

    # --- Phase 2: Lloyd refinement -----------------------------------------
    alive_ids = sorted(alive)
    palettes = [groups[g]["palette"] for g in alive_ids]
    p = len(palettes)
    assign = np.zeros(n_tiles, dtype=np.int64)
    for pi, g in enumerate(alive_ids):
        for t in groups[g]["members"]:
            assign[t] = pi

    def palette_distance_table(pals):
        # D[c, pi] = min over palette pi's 4 entries of Euclidean Lab dist.
        pal_lab = np.stack([_rgb_to_lab(pp) for pp in pals])  # (p, 4, 3)
        diff = ws_lab[:, None, None, :] - pal_lab[None, :, :, :]  # (nc, p, 4, 3)
        dist = np.sqrt(np.sum(diff * diff, axis=3))  # (nc, p, 4)
        return dist.min(axis=2)  # (nc, p)

    def assign_tiles(dtable):
        out = np.empty(n_tiles, dtype=np.int64)
        for t in range(n_tiles):
            idx = tile_colors[t]
            cnt = tile_counts[t]
            err = (dtable[idx, :] * cnt[:, None]).sum(axis=0)  # (p,)
            out[t] = int(np.argmin(err))
        return out

    def refit(assignment, pals):
        new_pals = []
        for pi in range(len(pals)):
            members = np.where(assignment == pi)[0]
            if len(members) == 0:
                new_pals.append(pals[pi])
                continue
            comb = {}
            for t in members:
                for i, c in zip(tile_colors[t], tile_counts[t]):
                    comb[int(i)] = comb.get(int(i), 0.0) + float(c)
            idxs = np.array(sorted(comb.keys()), dtype=np.int64)
            cnts = np.array([comb[int(i)] for i in idxs], dtype=np.float64)
            new_pals.append(build_palette(idxs, cnts))
        return new_pals

    for _ in range(5):
        dtable = palette_distance_table(palettes)
        new_assign = assign_tiles(dtable)
        changed = not np.array_equal(new_assign, assign)
        assign = new_assign
        palettes = refit(assign, palettes)
        if not changed:
            break

    palettes_arr = np.stack(palettes).astype(np.uint8)  # (p, 4, 3)
    assignment = assign.reshape(th, tw).astype(np.uint8)
    return palettes_arr, assignment


# ---------------------------------------------------------------------------
# Stage 4 -- indexing + Bayer dither
# ---------------------------------------------------------------------------

BAYER4 = np.array(
    [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]
) / 16.0


def index_tiles(
    image: np.ndarray,
    palettes: np.ndarray,
    assignment: np.ndarray,
    dither: str = "none",
) -> GBImage:
    """Render each 8x8 tile against its assigned palette -> index patterns.

    ``image`` is (H, W, 3) uint8 (already working-set/palette-limited);
    ``palettes`` is (p, 4, 3) uint8 luminance-sorted (index 0 == lightest);
    ``assignment`` is (H//8, W//8) uint8 palette id per cell.

    For each pixel, finds the nearest palette entry in CIELAB -> index. With
    ``dither="bayer"``, instead picks between the two nearest palette entries:
    letting d1/d2 be the distances to the nearest/second-nearest entry,
    t = d1/(d1+d2) (0 when the pixel is an exact palette color); the pixel
    takes the second-nearest entry's index iff t exceeds the 4x4 ordered
    Bayer threshold anchored to the pixel's *absolute* image coordinates.
    No error diffusion -- dithering can never pick a color outside the
    tile's assigned 4-color palette, and is position-stable (dedup-friendly).

    Returns a GBImage with one pattern per tile cell (raster order), an
    identity tilemap, no flips set, and ``palettes`` passed through unchanged.
    Dedup (fewer patterns) is a later stage.
    """
    if dither not in ("none", "bayer"):
        raise ValueError(f"unknown dither mode: {dither!r}")

    arr = np.asarray(image, dtype=np.uint8)
    h, w = arr.shape[:2]
    th, tw = h // 8, w // 8
    n_tiles = th * tw

    palettes = np.asarray(palettes, dtype=np.uint8)
    assignment = np.asarray(assignment)
    pal_lab = np.stack([_rgb_to_lab(pp) for pp in palettes])  # (p, 4, 3)

    patterns = np.zeros((n_tiles, 8, 8), dtype=np.uint8)
    tilemap = np.arange(n_tiles, dtype=np.int32).reshape(th, tw)
    attrs_hflip = np.zeros((th, tw), dtype=bool)
    attrs_vflip = np.zeros((th, tw), dtype=bool)

    row_off = np.arange(8)[:, None]
    col_off = np.arange(8)[None, :]

    for tr in range(th):
        for tc in range(tw):
            t = tr * tw + tc
            block = arr[tr * 8 : tr * 8 + 8, tc * 8 : tc * 8 + 8]  # (8,8,3)
            pid = int(assignment[tr, tc])
            plab = pal_lab[pid]  # (4,3)
            blab = _rgb_to_lab(block).reshape(8, 8, 3)
            diff = blab[:, :, None, :] - plab[None, None, :, :]  # (8,8,4,3)
            dist = np.sqrt(np.sum(diff * diff, axis=3))  # (8,8,4)
            order = np.argsort(dist, axis=2, kind="stable")  # nearest-first
            idx1 = order[:, :, 0]

            if dither == "bayer":
                idx2 = order[:, :, 1]
                d1 = np.take_along_axis(dist, idx1[:, :, None], axis=2)[:, :, 0]
                d2 = np.take_along_axis(dist, idx2[:, :, None], axis=2)[:, :, 0]
                denom = d1 + d2
                t_val = np.where(denom > 0, d1 / np.where(denom > 0, denom, 1.0), 0.0)
                yy = tr * 8 + row_off
                xx = tc * 8 + col_off
                thresh = BAYER4[yy % 4, xx % 4]
                index = np.where(t_val > thresh, idx2, idx1)
            else:
                index = idx1

            patterns[t] = index.astype(np.uint8)

    return GBImage(
        patterns=patterns,
        tilemap=tilemap,
        attrs_palette=assignment.astype(np.uint8),
        attrs_hflip=attrs_hflip,
        attrs_vflip=attrs_vflip,
        palettes=palettes,
    )
