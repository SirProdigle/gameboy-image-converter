"""GB Studio hardware-constrained conversion pipeline.

Pure functions over numpy arrays / PIL images — no Gradio imports here.
See docs/superpowers/specs/2026-07-06-gb-pipeline-redesign-design.md for the
authoritative behavior spec and
docs/superpowers/plans/2026-07-06-gb-pipeline-redesign-plan.md for the task
breakdown these functions implement.
"""

import heapq
from dataclasses import dataclass, replace

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


# ---------------------------------------------------------------------------
# Stage 5 -- lossless dedup
# ---------------------------------------------------------------------------


def dedup_patterns(gb: GBImage, allow_flips: bool) -> GBImage:
    """Collapse exact-duplicate (and, if ``allow_flips``, flip-equivalent)
    patterns down to a single stored copy, exactly as GB Studio's importer
    (`tileData.ts`) hashes the 16-byte 2bpp pattern.

    Iterates cells in raster order. Each cell's *effective* pattern (the
    pattern it currently references, with any existing hflip/vflip already
    applied) is hashed via ``pattern_to_2bpp``. On a miss the effective
    pattern becomes a new canonical entry; when ``allow_flips`` its H/V/HV
    variants are hashed too, so a later cell whose effective pattern exactly
    matches one of them reuses that canonical pattern with the matching flip
    bits set instead of minting a new one. Palette assignment is untouched --
    this stage only shrinks `patterns` and remaps `tilemap` / flip attrs.
    """
    th, tw = gb.tilemap.shape
    patterns_out: list = []
    # 2bpp bytes -> (pattern index, hflip, vflip) needed to reproduce them.
    variant_lookup: dict = {}
    new_tilemap = np.zeros((th, tw), dtype=np.int32)
    new_hflip = np.zeros((th, tw), dtype=bool)
    new_vflip = np.zeros((th, tw), dtype=bool)

    for tr in range(th):
        for tc in range(tw):
            src_idx = int(gb.tilemap[tr, tc])
            pattern = gb.patterns[src_idx]
            if gb.attrs_hflip[tr, tc]:
                pattern = np.fliplr(pattern)
            if gb.attrs_vflip[tr, tc]:
                pattern = np.flipud(pattern)
            key = pattern_to_2bpp(pattern)

            if key in variant_lookup:
                out_idx, fh, fv = variant_lookup[key]
            else:
                out_idx = len(patterns_out)
                patterns_out.append(np.ascontiguousarray(pattern))
                if allow_flips:
                    for name, variant in pattern_variants(pattern).items():
                        vkey = pattern_to_2bpp(variant)
                        if vkey not in variant_lookup:
                            variant_lookup[vkey] = (out_idx, "h" in name, "v" in name)
                else:
                    variant_lookup[key] = (out_idx, False, False)
                fh, fv = False, False

            new_tilemap[tr, tc] = out_idx
            new_hflip[tr, tc] = fh
            new_vflip[tr, tc] = fv

    if patterns_out:
        patterns_arr = np.stack(patterns_out).astype(np.uint8)
    else:
        patterns_arr = np.zeros((0, 8, 8), dtype=np.uint8)

    return GBImage(
        patterns=patterns_arr,
        tilemap=new_tilemap,
        attrs_palette=gb.attrs_palette.copy(),
        attrs_hflip=new_hflip,
        attrs_vflip=new_vflip,
        palettes=gb.palettes,
    )


# ---------------------------------------------------------------------------
# Stage 6 -- budget merge (lossy)
# ---------------------------------------------------------------------------
#
# Only runs when the lossless-dedup tile count still exceeds the budget. Each
# pattern gets a static "signature" -- itself rendered under its usage-weighted
# average palette, in CIELAB -- and the globally cheapest merges (perceptual
# signature distance x usage of the replaced pattern) are applied cheapest-first
# from a min-heap until the live pattern count lands exactly on the budget.
# Merges only retarget tilemap references (composing flip bits); they can never
# affect palette constraints. See the design spec, section "Stage 6".

# Above this many patterns the full O(n^2) signature-distance matrix becomes
# expensive; we fall back to KMeans-clustering signatures and only considering
# intra-cluster merge candidates (correctness fallback: recluster survivors,
# and if a pass makes no progress, compare all remaining pairs directly).
_MERGE_MATRIX_MAX = 3500

# Spatial-flip variants of a pattern signature. For a stack of signatures with
# shape (..., 8, 8, 3): axis -3 is rows (vertical flip), axis -2 is columns
# (horizontal flip).
_VARIANT_NAMES_FLIP = ("", "h", "v", "hv")
_VARIANT_NAMES_NOFLIP = ("",)


def _flip_signature(sig: np.ndarray, variant: str) -> np.ndarray:
    """Apply an h/v flip variant to an (..., 8, 8, 3) signature array."""
    out = sig
    if "h" in variant:
        out = np.flip(out, axis=-2)  # columns == horizontal
    if "v" in variant:
        out = np.flip(out, axis=-3)  # rows == vertical
    return out


def _pattern_signatures(gb: GBImage) -> np.ndarray:
    """Render every pattern under its usage-weighted average palette -> Lab.

    For each pattern, the weight of palette id ``pid`` is the number of cells
    that reference the pattern with that palette. The averaged palette (a
    (4, 3) RGB blend) renders the pattern's indices to colors, which are then
    converted to CIELAB. Returns (n, 8, 8, 3) float64. Signatures are computed
    once from the *initial* cell assignments and are treated as static across
    all merges (patterns never change; only tilemap references move).
    """
    n = int(gb.patterns.shape[0])
    p = int(gb.palettes.shape[0])
    flat_pat = gb.tilemap.reshape(-1).astype(np.int64)
    flat_pid = gb.attrs_palette.reshape(-1).astype(np.int64)

    # usage_pal[pattern, palette] = cell count.
    usage_pal = np.zeros((n, p), dtype=np.float64)
    np.add.at(usage_pal, (flat_pat, flat_pid), 1.0)

    wsum = usage_pal.sum(axis=1)  # (n,)
    # Weighted average palette per pattern: (n, 4, 3).
    avg_pal = np.tensordot(usage_pal, gb.palettes.astype(np.float64), axes=([1], [0]))
    avg_pal = avg_pal / np.maximum(wsum[:, None, None], 1e-9)

    idx = gb.patterns.reshape(n, 64).astype(np.int64)  # (n, 64) values 0-3
    sig_rgb = np.take_along_axis(
        avg_pal, idx[:, :, None].repeat(3, axis=2), axis=1
    )  # (n, 64, 3)
    sig_rgb = sig_rgb.reshape(n, 8, 8, 3) / 255.0
    sig_lab = rgb2lab(sig_rgb)  # (n, 8, 8, 3)
    return sig_lab.astype(np.float64)


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, pct: float) -> float:
    """Weighted percentile (0-100) of ``values`` with nonnegative ``weights``."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return 0.0
    order = np.argsort(values, kind="stable")
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w)
    total = float(cw[-1])
    if total <= 0:
        return float(v[-1])
    threshold = (pct / 100.0) * total
    idx = int(np.searchsorted(cw, threshold, side="left"))
    idx = min(idx, v.size - 1)
    return float(v[idx])


def _simple_kmeans_labels(points: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    """Deterministic k-means++ cluster labels for (N, D) points (any D)."""
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    if k >= n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.RandomState(seed)
    first = int(rng.randint(n))
    centers = [points[first]]
    d2 = np.sum((points - points[first]) ** 2, axis=1)
    for _ in range(1, k):
        s = d2.sum()
        j = int(rng.choice(n, p=d2 / s)) if s > 0 else int(rng.randint(n))
        centers.append(points[j])
        d2 = np.minimum(d2, np.sum((points - points[j]) ** 2, axis=1))
    centers = np.array(centers, dtype=np.float64)
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(10):
        dists = np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        for c in range(k):
            mask = labels == c
            if mask.any():
                centers[c] = points[mask].mean(axis=0)
    return labels


def merge_to_budget(gb: GBImage, budget: int, allow_flips: bool) -> tuple:
    """Lossily merge patterns until ``len(patterns) <= budget``, cheapest-first.

    Returns ``(result, n_merges)``. When the input already fits the budget the
    image is returned unchanged with ``0`` merges. The p95 of per-pixel Lab
    distance over all merged cells (a heavy-merge quality signal for Stage 7)
    is stashed on the returned GBImage as ``merge_p95_delta_e``.

    Algorithm (design spec, Stage 6): each pattern's signature is itself
    rendered under its usage-weighted average palette in CIELAB and is static
    across merges. ``distance(A, B)`` is the minimum, over B's flip variants
    (all four when ``allow_flips`` else identity only), of the mean squared Lab
    difference, remembering the best orientation. ``cost(A -> B) =
    distance(A, B) * usage(A)``, and pairs are pushed onto a min-heap oriented
    so the replaced pattern A has the smaller usage. Popped entries are lazily
    validated (both patterns alive, replaced usage unchanged) and re-pushed if
    stale. A merge retargets every cell of A to B, composing the best
    orientation's flip bits, and folds A's usage into B. Repeats until the live
    count reaches the budget, then rebuilds a compact GBImage.
    """
    n = int(gb.patterns.shape[0])
    if n <= budget:
        gb.merge_p95_delta_e = 0.0
        return gb, 0

    variant_names = _VARIANT_NAMES_FLIP if allow_flips else _VARIANT_NAMES_NOFLIP

    sig_lab = _pattern_signatures(gb)  # (n, 8, 8, 3)
    sig_flat = sig_lab.reshape(n, 192)
    # Flattened flip variants of each pattern used as a *target*.
    var_flat = {v: _flip_signature(sig_lab, v).reshape(n, 192) for v in variant_names}

    usage = np.bincount(
        gb.tilemap.reshape(-1).astype(np.int64), minlength=n
    ).astype(np.int64)

    # Working copies mutated in place as cells are retargeted.
    work_tilemap = gb.tilemap.copy().astype(np.int64)
    work_hflip = gb.attrs_hflip.copy()
    work_vflip = gb.attrs_vflip.copy()
    alive = np.ones(n, dtype=bool)

    # Optional full distance/orientation matrices (small-n fast path).
    if n <= _MERGE_MATRIX_MAX:
        best = np.full((n, n), np.inf, dtype=np.float64)
        orient_idx = np.zeros((n, n), dtype=np.int8)
        sqA = np.sum(sig_flat * sig_flat, axis=1)  # (n,)
        for vi, v in enumerate(variant_names):
            B = var_flat[v]
            sqB = np.sum(B * B, axis=1)
            cross = sig_flat @ B.T  # (n, n)
            D = (sqA[:, None] + sqB[None, :] - 2.0 * cross) / 192.0
            np.maximum(D, 0.0, out=D)
            upd = D < best
            best[upd] = D[upd]
            orient_idx[upd] = vi
        np.fill_diagonal(best, np.inf)

        def dist_orient(a, b):
            return float(best[a, b]), variant_names[int(orient_idx[a, b])]

    else:
        def dist_orient(a, b):
            best_d = np.inf
            best_v = ""
            sa = sig_flat[a]
            for v in variant_names:
                diff = sa - var_flat[v][b]
                d = float(np.mean(diff * diff))
                if d < best_d:
                    best_d = d
                    best_v = v
            return best_d, best_v

    heap: list = []
    counter = 0

    def push_pair(i, j):
        nonlocal counter
        a, b = (i, j) if usage[i] <= usage[j] else (j, i)  # a == replaced
        d, _ = dist_orient(a, b)
        cost = d * float(usage[a])
        heapq.heappush(heap, (cost, counter, int(a), int(b), int(usage[a])))
        counter += 1

    def seed_pairs(indices):
        m = len(indices)
        for ii in range(m):
            for jj in range(ii + 1, m):
                push_pair(indices[ii], indices[jj])

    merge_records: list = []  # (per_pixel_lab_dist (64,), weight)
    n_merges = 0
    alive_count = n

    def run_heap():
        nonlocal n_merges, alive_count
        while alive_count > budget and heap:
            cost, _, a, b, snap = heapq.heappop(heap)
            if not alive[a] or not alive[b]:
                continue
            if int(usage[a]) != snap:
                # Replaced pattern absorbed others since push -> stale cost.
                push_pair(a, b)
                continue
            # Valid cheapest merge a -> b.
            _, v = dist_orient(a, b)
            vh = "h" in v
            vv = "v" in v
            mask = work_tilemap == a
            cnt = int(mask.sum())
            work_tilemap[mask] = b
            if vh:
                work_hflip[mask] = ~work_hflip[mask]
            if vv:
                work_vflip[mask] = ~work_vflip[mask]

            sa = sig_lab[a]
            sb_v = var_flat[v][b].reshape(8, 8, 3)
            diff = sa - sb_v
            per_pixel = np.sqrt(np.sum(diff * diff, axis=2)).reshape(-1)  # (64,)
            merge_records.append((per_pixel, cnt))

            usage[b] += usage[a]
            usage[a] = 0
            alive[a] = False
            alive_count -= 1
            n_merges += 1

    if n <= _MERGE_MATRIX_MAX:
        seed_pairs(list(range(n)))
        run_heap()
    else:
        # Clustering fast path: only push intra-cluster pairs, reclustering the
        # survivors between passes. A pass that makes no progress falls back to
        # comparing all remaining pairs directly (correctness guarantee).
        while alive_count > budget:
            survivors = [int(i) for i in np.nonzero(alive)[0]]
            if len(survivors) <= budget:
                break
            before = n_merges
            k = max(1, int(round(np.sqrt(len(survivors)))))
            labels = _simple_kmeans_labels(sig_flat[survivors], k)
            heap.clear()
            for c in range(int(labels.max()) + 1):
                members = [survivors[t] for t in np.nonzero(labels == c)[0]]
                seed_pairs(members)
            run_heap()
            if n_merges == before:
                # No intra-cluster merge possible -> compare everything.
                heap.clear()
                seed_pairs(survivors)
                run_heap()
                break

    # --- Rebuild a compact GBImage over the surviving patterns --------------
    alive_idx = np.nonzero(alive)[0]
    remap = -np.ones(n, dtype=np.int64)
    remap[alive_idx] = np.arange(alive_idx.size, dtype=np.int64)
    new_tilemap = remap[work_tilemap].astype(np.int32)
    assert new_tilemap.min() >= 0, "dangling reference to a merged-away pattern"

    result = GBImage(
        patterns=gb.patterns[alive_idx].copy(),
        tilemap=new_tilemap,
        attrs_palette=gb.attrs_palette.copy(),
        attrs_hflip=work_hflip,
        attrs_vflip=work_vflip,
        palettes=gb.palettes,
    )

    if merge_records:
        all_dists = np.concatenate([pp for pp, _ in merge_records])
        all_weights = np.concatenate(
            [np.full(pp.shape[0], w, dtype=np.float64) for pp, w in merge_records]
        )
        p95 = _weighted_percentile(all_dists, all_weights, 95.0)
    else:
        p95 = 0.0
    result.merge_p95_delta_e = float(p95)

    return result, n_merges


# ---------------------------------------------------------------------------
# Stage 7 / Stage 8 -- verification, render, and the public entry point
# ---------------------------------------------------------------------------
#
# `render` turns a GBImage back into an RGB PNG (index -> palette color, with
# flip bits applied); `verify_roundtrip` re-imports that PNG with a
# reimplementation of GB Studio's tile hashing and asserts the hardware
# invariants hold; `convert_for_hardware` is the single public entry point that
# runs the whole pipeline (downscale/crop -> quantize -> pack -> index -> dedup
# -> budget merge -> verify -> render) and returns a ConversionResult.

# p95 per-pixel Lab distance (from heavy budget merging) above which the result
# gets an actionable quality warning.
_HEAVY_MERGE_P95_DELTA_E = 12.0

# Relative luminance weights, matching ``luminance_sort``.
_LUM_WEIGHTS = np.array([2126.0, 7152.0, 722.0], dtype=np.float64)


@dataclass
class ConversionResult:
    image: Image.Image            # final RGB render
    reference: Image.Image        # render before budget merge (natural compare)
    stats: dict                   # tiles_used, tile_budget, palettes_used,
                                  # n_merges, p95_delta_e, preset, dither
    warnings: list                # list[str]
    palette_hex: list             # per palette, 4 hex strings


def _pixel_luminance(rgb: np.ndarray) -> np.ndarray:
    """Relative luminance of an (..., 3) RGB array (float, same lead shape)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    return rgb @ _LUM_WEIGHTS


def render(gb: GBImage) -> Image.Image:
    """Render a GBImage to an RGB PIL image (the importable PNG).

    Each cell places its pattern (flip bits applied) rendered through its
    assigned palette, so the output contains only palette colors.
    """
    th, tw = gb.tilemap.shape
    out = np.zeros((th * 8, tw * 8, 3), dtype=np.uint8)
    palettes = np.asarray(gb.palettes, dtype=np.uint8)
    for tr in range(th):
        for tc in range(tw):
            pattern = gb.patterns[int(gb.tilemap[tr, tc])]
            if gb.attrs_hflip[tr, tc]:
                pattern = np.fliplr(pattern)
            if gb.attrs_vflip[tr, tc]:
                pattern = np.flipud(pattern)
            pal = palettes[int(gb.attrs_palette[tr, tc])]  # (4, 3)
            out[tr * 8 : tr * 8 + 8, tc * 8 : tc * 8 + 8] = pal[pattern]
    return Image.fromarray(out, "RGB")


def _colors_to_indices(block: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Map an (8, 8, 3) tile's pixels to palette indices by exact color match.

    A pixel that equals several palette entries (duplicate colors in a padded
    palette) takes the lowest such index -- mirroring how ``index_tiles``
    resolves ties. Asserts every pixel matches some palette color.
    """
    block = np.asarray(block, dtype=np.uint8)
    palette = np.asarray(palette, dtype=np.uint8)
    idx = np.full(block.shape[:2], -1, dtype=np.int64)
    for i in range(palette.shape[0]):
        match = np.all(block == palette[i], axis=2) & (idx < 0)
        idx[match] = i
    assert idx.min() >= 0, "rendered pixel is not one of its palette colors"
    return idx.astype(np.uint8)


def _canonical_2bpp(pattern: np.ndarray, allow_flips: bool) -> bytes:
    """2bpp bytes for a pattern, minimized over flip variants when allowed."""
    if not allow_flips:
        return pattern_to_2bpp(pattern)
    return min(pattern_to_2bpp(v) for v in pattern_variants(pattern).values())


def verify_roundtrip(png: Image.Image, preset: Preset, expected: GBImage) -> None:
    """Independently re-import ``png`` and assert the hardware invariants.

    Independent round-trip (spec Stage 7.2): crop each 8x8 tile, require <=4
    unique colors, require every color to be RGB555-stable, recover the tile's
    index pattern against its assigned palette and hash it as 2bpp
    (canonicalizing flips when the preset allows them); assert the
    reconstructed unique-tile count equals ``len(expected.patterns)`` and stays
    within the tile budget (skipped for logo presets, whose tiles are stored
    sequentially without dedup). Structural GBImage assertions (spec Stage 7.1):
    the palette count is within ``preset.n_palettes`` and the stored palettes
    are RGB555-only. Because every tile's pixels are matched exactly against
    its assigned palette, a passing round-trip also proves each tile renders
    from one of those <= n_palettes 4-color palettes.
    """
    arr = np.asarray(png.convert("RGB"), dtype=np.uint8)
    th, tw = expected.tilemap.shape
    assert arr.shape[0] == th * 8 and arr.shape[1] == tw * 8, "png size mismatch"

    # Structural assertions on the GBImage itself (Stage 7.1).
    assert expected.palettes.shape[0] <= preset.n_palettes, (
        f"{expected.palettes.shape[0]} palettes exceeds limit {preset.n_palettes}"
    )
    assert np.array_equal(snap_rgb555(expected.palettes), expected.palettes), (
        "stored palette is not RGB555-stable"
    )

    budget = preset.tile_budget
    canon_hashes = set()
    for tr in range(th):
        for tc in range(tw):
            block = arr[tr * 8 : tr * 8 + 8, tc * 8 : tc * 8 + 8]
            uniq = np.unique(block.reshape(-1, 3), axis=0)
            assert uniq.shape[0] <= 4, "tile has more than 4 colors"
            assert np.array_equal(snap_rgb555(uniq), uniq), "color not RGB555-stable"

            pal = expected.palettes[int(expected.attrs_palette[tr, tc])]
            idx_pattern = _colors_to_indices(block, pal)
            canon_hashes.add(_canonical_2bpp(idx_pattern, preset.allow_flips))

    n_expected = int(expected.patterns.shape[0])
    if budget is None:
        # Logo: sequential storage, no dedup -- one stored tile per cell.
        assert n_expected == th * tw, "logo tile count must equal cell count"
    else:
        assert len(canon_hashes) == n_expected, (
            f"reimport found {len(canon_hashes)} tiles, expected {n_expected}"
        )
        assert len(canon_hashes) <= budget, "tile count exceeds budget"


def _index_tiles_mono(
    image: np.ndarray, palettes: np.ndarray, dither: str = "none"
) -> GBImage:
    """Index every pixel to the mono ramp by luminance rank (Stage 4, mono).

    ``palettes`` is the (1, 4, 3) luminance-sorted ramp. Each pixel takes the
    ramp entry nearest in luminance; with ``dither="bayer"`` it picks between
    the two nearest ramp levels using the absolute-coordinate Bayer threshold.
    """
    if dither not in ("none", "bayer"):
        raise ValueError(f"unknown dither mode: {dither!r}")
    arr = np.asarray(image, dtype=np.uint8)
    h, w = arr.shape[:2]
    th, tw = h // 8, w // 8

    ramp = np.asarray(palettes, dtype=np.uint8)[0]  # (4, 3)
    ramp_lum = _pixel_luminance(ramp)               # (4,)
    px_lum = _pixel_luminance(arr)                   # (h, w)

    dist = np.abs(px_lum[:, :, None] - ramp_lum[None, None, :])  # (h, w, 4)
    order = np.argsort(dist, axis=2, kind="stable")
    idx1 = order[:, :, 0]
    if dither == "bayer":
        idx2 = order[:, :, 1]
        d1 = np.take_along_axis(dist, idx1[:, :, None], axis=2)[:, :, 0]
        d2 = np.take_along_axis(dist, idx2[:, :, None], axis=2)[:, :, 0]
        denom = d1 + d2
        t_val = np.where(denom > 0, d1 / np.where(denom > 0, denom, 1.0), 0.0)
        yy = np.arange(h)[:, None]
        xx = np.arange(w)[None, :]
        thresh = BAYER4[yy % 4, xx % 4]
        index = np.where(t_val > thresh, idx2, idx1)
    else:
        index = idx1

    index = index.astype(np.uint8)
    n_tiles = th * tw
    patterns = np.zeros((n_tiles, 8, 8), dtype=np.uint8)
    for tr in range(th):
        for tc in range(tw):
            patterns[tr * tw + tc] = index[tr * 8 : tr * 8 + 8, tc * 8 : tc * 8 + 8]

    return GBImage(
        patterns=patterns,
        tilemap=np.arange(n_tiles, dtype=np.int32).reshape(th, tw),
        attrs_palette=np.zeros((th, tw), dtype=np.uint8),
        attrs_hflip=np.zeros((th, tw), dtype=bool),
        attrs_vflip=np.zeros((th, tw), dtype=bool),
        palettes=ramp[None, :, :].astype(np.uint8),
    )


def _palette_hex(palettes: np.ndarray) -> list:
    """Per-palette list of 4 ``#RRGGBB`` hex strings."""
    palettes = np.asarray(palettes, dtype=np.uint8)
    return [
        ["#{:02X}{:02X}{:02X}".format(int(c[0]), int(c[1]), int(c[2])) for c in pal]
        for pal in palettes
    ]


def convert_for_hardware(
    image: Image.Image,
    preset: str,
    *,
    tile_budget: int | None = None,
    reserve_ui_palette: bool = True,
    dither: str = "none",
    custom_palette: np.ndarray | None = None,
    mono_ramp: np.ndarray | None = None,
) -> ConversionResult:
    """Convert ``image`` to a GB Studio-importable render for a named preset.

    ``preset`` is a key of ``PRESETS``. Runs the full pipeline: RGB-convert,
    crop to multiples of 8 (logo presets are resized to their fixed size),
    quantize to a bounded working set, pack palettes, index tiles (mono uses
    the ramp path), losslessly dedup (skipped for logo), lossily merge to the
    tile budget (skipped for logo / no budget), then verify and render.
    ``reserve_ui_palette`` reserves palette 8 for the dialogue/UI palette by
    reducing an 8-palette color preset to 7. Returns a ConversionResult with
    the final and pre-merge reference images, verified stats, warnings, and
    per-palette hex swatches.
    """
    if preset not in PRESETS:
        raise ValueError(f"unknown preset: {preset!r}")
    ps = PRESETS[preset]
    warnings: list = []

    img = image.convert("RGB")
    is_logo = ps.fixed_size is not None

    # Crop down to multiples of 8 (notice if it changed anything).
    ow, oh = img.size
    nw, nh = (ow // 8) * 8, (oh // 8) * 8
    if nw < 1 or nh < 1:
        raise ValueError("image is smaller than one 8x8 tile")
    if (nw, nh) != (ow, oh):
        img = img.crop((0, 0, nw, nh))
        warnings.append(
            f"Input cropped from {ow}x{oh} to {nw}x{nh} "
            "(dimensions must be multiples of 8)."
        )

    # Logo presets are locked to a fixed screen size.
    if is_logo:
        fw, fh = ps.fixed_size
        if img.size != (fw, fh):
            img = img.resize((fw, fh), Image.LANCZOS)
            warnings.append(f"Input resized to {fw}x{fh} for the {ps.name} preset.")

    # Effective palette budget: reserve palette 8 for color presets.
    n_palettes = ps.n_palettes
    if reserve_ui_palette and not ps.mono and n_palettes >= 8:
        n_palettes -= 1

    quantized = quantize_working_set(img, 4 * n_palettes, custom_palette)
    arr = np.asarray(quantized, dtype=np.uint8)

    if ps.mono:
        ramp = DMG_RAMP if mono_ramp is None else np.asarray(mono_ramp, dtype=np.uint8)
        palettes, _ = pack_palettes_mono(arr, ramp)
        gb = _index_tiles_mono(arr, palettes, dither)
    else:
        palettes, assignment = pack_palettes(arr, n_palettes, custom_palette)
        gb = index_tiles(arr, palettes, assignment, dither)

    if not is_logo:
        gb = dedup_patterns(gb, ps.allow_flips)

    reference_gb = gb
    reference_img = render(reference_gb)

    budget = tile_budget if tile_budget is not None else ps.tile_budget
    n_merges = 0
    p95 = 0.0
    if not is_logo and budget is not None:
        gb, n_merges = merge_to_budget(gb, budget, ps.allow_flips)
        p95 = float(getattr(gb, "merge_p95_delta_e", 0.0))

    final_img = render(gb)

    # Independent round-trip verification against the effective palette limit.
    verify_roundtrip(final_img, replace(ps, n_palettes=n_palettes), gb)

    if n_merges > 0 and p95 > _HEAVY_MERGE_P95_DELTA_E:
        warnings.append(
            f"Heavy tile merging (p95 dE ~ {p95:.1f}): raise the tile budget, "
            "simplify the image, or disable dithering to reduce quality loss."
        )

    stats = {
        "tiles_used": int(gb.patterns.shape[0]),
        "tile_budget": budget,
        "palettes_used": int(gb.palettes.shape[0]),
        "n_merges": int(n_merges),
        "p95_delta_e": p95,
        "preset": ps.name,
        "dither": dither,
    }

    return ConversionResult(
        image=final_img,
        reference=reference_img,
        stats=stats,
        warnings=warnings,
        palette_hex=_palette_hex(gb.palettes),
    )
