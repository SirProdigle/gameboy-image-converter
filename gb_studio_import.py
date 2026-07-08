"""Faithful Python port of GB Studio v4.3.2's background importer.

Purpose: report the tile / palette counts GB Studio *actually* computes when a
converted PNG is imported as a Color-Only background, and detect the cases where
GB Studio's import would silently corrupt colors (it re-extracts >8 palettes and
wraps the overflow ``% 8``) or overflow the tile budget. The pipeline's own
``verify_roundtrip`` re-derives tiles with a *different* algorithm (per-tile
luminance reindex) than GB Studio's color path, so its counts can diverge; this
module mirrors GB Studio's real logic instead.

Ported verbatim in behavior from GB Studio's TypeScript source (v4.3.2):
  src/shared/lib/tiles/indexedImage.ts  -- indexedImageTo2bppTileData
  src/shared/lib/tiles/tileData.ts      -- hashTileData / toTileLookup
  src/shared/lib/tiles/autoColor.ts     -- autoPalette / compressPalettes
  src/shared/lib/tiles/autoFlip.ts      -- autoFlipTiles
"""

from dataclasses import dataclass

import numpy as np
from skimage.color import rgb2lab

BRIGHTNESS_ANCHOR = (100.0, 66.0, 33.0, 0.0)


def _green_index(g: int) -> int:
    """tileDataIndexFn: GB Studio's fixed green-channel bucket (0=lightest)."""
    if g < 65:
        return 3
    if g < 130:
        return 2
    if g < 205:
        return 1
    return 0


def _hex(r: int, g: int, b: int) -> str:
    """GB Studio rgb2hex for colorCorrection='none': raw 8-bit -> 'rrggbb'."""
    return f"{r & 0xFF:02x}{g & 0xFF:02x}{b & 0xFF:02x}"


def _lab_lightness(hexstr: str) -> float:
    """Perceptual lightness (CIELAB L*) of an 'rrggbb' hex, as GB Studio's
    chroma(hex).lab()[0] uses for palette sorting."""
    r = int(hexstr[0:2], 16) / 255.0
    g = int(hexstr[2:4], 16) / 255.0
    b = int(hexstr[4:6], 16) / 255.0
    return float(rgb2lab(np.array([[[r, g, b]]], dtype=np.float64))[0, 0, 0])


def _sort_hex_palette(colors: list) -> list:
    """sortHexPalette: brightest first (stable on ties -> keeps scan order)."""
    return sorted(colors, key=lambda h: -_lab_lightness(h))


def extract_tile_palette(tile_rgb: np.ndarray) -> list:
    """extractTilePalette (colorCorrection='none'): the first <=4 distinct
    colors found scanning the 8x8 tile row-major, sorted brightest-first."""
    tile_rgb = np.asarray(tile_rgb, dtype=np.uint8)
    seen = set()
    colors = []
    for y in range(8):
        for x in range(8):
            r, g, b = (int(v) for v in tile_rgb[y, x])
            key = (r, g, b)
            if key in seen:
                continue
            seen.add(key)
            colors.append(_hex(r, g, b))
            if len(colors) == 4:
                return _sort_hex_palette(colors)
    return _sort_hex_palette(colors)


def _unique_keep_order(seq):
    """JS Array.from(new Set(seq)): dedup preserving first-seen order."""
    out = []
    seen = set()
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _greedy_compress_groups(palettes: list):
    """compressPalettes' merge phase: greedily merge any two palettes whose color
    union is <=4 (repeatedly, first compatible pair wins).

    Returns ``(out_palettes, origins)`` where ``out_palettes`` is the list of
    merged variable-length hex palettes (NOT yet filled to 4) in their *un-wrapped*
    merged order, and ``origins[new_index]`` lists the original palette indices
    that collapsed into it. ``len(out_palettes) > 8`` is exactly GB Studio's
    silent-corruption condition, and ``new_index >= 8`` marks a palette GB drops
    (wraps ``% 8``) rather than keeps.
    """
    out = [list(p) for p in palettes]
    origins = [[i] for i in range(len(palettes))]

    merged = True
    while merged:
        merged = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                unique = _unique_keep_order(out[i] + out[j])
                if len(unique) <= 4:
                    out[i] = unique
                    origins[i] = origins[i] + origins[j]
                    del out[j]
                    del origins[j]
                    merged = True
                    break
            if merged:
                break
    return out, origins


def _greedy_compress(palettes: list):
    """compressPalettes: greedy-merge (``_greedy_compress_groups``) then build the
    original->new mapping table with the overflow wrapped ``% 8``.

    Returns ``(out_palettes, mapping_table)`` where ``mapping_table[original_index]
    = new_index % 8``.
    """
    out, origins = _greedy_compress_groups(palettes)
    mapping = [i % 8 for i in range(len(palettes))]
    for new_index, group in enumerate(origins):
        for origin in group:
            mapping[origin] = new_index % 8
    return out, mapping


def _rgb_of(hexstr: str):
    return (int(hexstr[0:2], 16), int(hexstr[2:4], 16), int(hexstr[4:6], 16))


def fill_variable_palette(palette: list) -> list:
    """fillVariablePalette: place a variable-length (<=4) hex palette into a
    strict 4-slot palette, preferring each color's green-bucket slot and
    resolving contention by nearest CIELAB-lightness to the slot's anchor."""
    colors = []
    for hexstr in palette:
        r, g, b = _rgb_of(hexstr)
        colors.append({
            "hex": hexstr,
            "index": _green_index(g),
            "lightness": _lab_lightness(hexstr),
        })
    colors.sort(key=lambda c: c["index"])  # stable

    result = [None, None, None, None]
    claimed = [False, False, False, False]
    overflow = []

    i = 0
    while i < len(colors):
        slot_index = colors[i]["index"]
        start = i
        while i < len(colors) and colors[i]["index"] == slot_index:
            i += 1
        group = colors[start:i]
        anchor = BRIGHTNESS_ANCHOR[slot_index]
        group.sort(key=lambda c: abs(c["lightness"] - anchor))  # stable
        winner = group[0]
        if not claimed[slot_index] and result[slot_index] is None:
            result[slot_index] = winner["hex"]
            claimed[slot_index] = True
        else:
            overflow.append(winner)
        for j in range(1, len(group)):
            overflow.append(group[j])

    overflow.sort(key=lambda c: abs(c["lightness"] - BRIGHTNESS_ANCHOR[c["index"]]))

    def find_forward(start):
        for k in range(start + 1, 4):
            if result[k] is None:
                return k
        return None

    def find_backward(start):
        for k in range(start - 1, -1, -1):
            if result[k] is None:
                return k
        return None

    for col in overflow:
        slot = find_forward(col["index"])
        if slot is None:
            slot = find_backward(col["index"])
        if slot is None:
            slot = find_backward(4)
        if slot is not None:
            result[slot] = col["hex"]
            claimed[slot] = True

    for k in range(4):
        if result[k] is None:
            result[k] = "000000"
    return result


def compress_palettes(all_palettes: list):
    """compressPalettes: greedy-merge, sort each brightest-first, then fill each
    into a strict 4-slot palette. Returns ``(filled_palettes, mapping_table)``."""
    out, mapping = _greedy_compress(all_palettes)
    out = [_sort_hex_palette(p) for p in out]
    filled = [fill_variable_palette(p) for p in out]
    return filled, mapping


def _closest_index(hexstr: str, palette: list) -> int:
    """findClosestHexColor + indexOf: nearest palette entry by Manhattan RGB,
    returning its slot index (first on ties)."""
    r, g, b = _rgb_of(hexstr)
    best_i = 0
    best = None
    best_hex = palette[0]
    for pc in palette:
        pr, pg, pb = _rgb_of(pc)
        d = abs(r - pr) + abs(g - pg) + abs(b - pb)
        if best is None or d < best:
            best = d
            best_hex = pc
    return palette.index(best_hex)


def _extract_tile_palettes(arr: np.ndarray):
    """autoPalette's first half: extract each tile's <=4-color palette and dedup
    identical palettes into a global list. Returns ``(all_palettes, tile_map,
    (h, w, xt, yt))`` where ``tile_map[ti]`` indexes into ``all_palettes``
    (BEFORE compression)."""
    h, w = arr.shape[:2]
    xt, yt = w // 8, h // 8

    all_palettes = []
    palette_cache = {}
    tile_palette_cache = {}
    tile_map = [0] * (xt * yt)
    for ty in range(yt):
        for tx in range(xt):
            ti = ty * xt + tx
            block = arr[ty * 8:ty * 8 + 8, tx * 8:tx * 8 + 8]
            tkey = block.tobytes()
            palette = tile_palette_cache.get(tkey)
            if palette is None:
                palette = extract_tile_palette(block)
                tile_palette_cache[tkey] = palette
            key = "".join(palette)
            if key in palette_cache:
                tile_map[ti] = palette_cache[key]
            else:
                tile_map[ti] = len(all_palettes)
                palette_cache[key] = tile_map[ti]
                all_palettes.append(palette)
    return all_palettes, tile_map, (h, w, xt, yt)


def autopalette(arr: np.ndarray):
    """autoPalette (colorCorrection='none', no UI palette): extract per-tile
    palettes, compress, then index every pixel to its tile-palette slot.

    Returns ``(indexed, palettes, tile_palette_map)`` where ``indexed`` is an
    (H, W) uint8 slot-index image, ``palettes`` the list of filled 4-hex
    palettes GB Studio derived (``len`` may exceed 8), and ``tile_palette_map``
    the per-tile palette index (already wrapped ``% 8``)."""
    arr = np.asarray(arr, dtype=np.uint8)
    all_palettes, tile_map, (h, w, xt, yt) = _extract_tile_palettes(arr)

    palettes, mapping = compress_palettes(all_palettes)

    indexed = np.zeros((h, w), dtype=np.uint8)
    for ty in range(yt):
        for tx in range(xt):
            ti = ty * xt + tx
            tile_map[ti] = mapping[tile_map[ti]]
            pal = palettes[tile_map[ti]]
            idx_cache = {}
            for y in range(8):
                for x in range(8):
                    r, g, b = (int(v) for v in arr[ty * 8 + y, tx * 8 + x])
                    ck = (r, g, b)
                    slot = idx_cache.get(ck)
                    if slot is None:
                        slot = _closest_index(_hex(r, g, b), pal)
                        idx_cache[ck] = slot
                    indexed[ty * 8 + y, tx * 8 + x] = slot
    return indexed, palettes, tile_map


def _tile_reconstruction_error(tile_rgb: np.ndarray, palette: list) -> int:
    """Total per-pixel Manhattan-RGB error of reconstructing an 8x8 tile under
    GB Studio's own nearest-color rule (``_closest_index``) against a filled
    4-slot ``palette`` -- i.e. how badly that palette would recolor the tile."""
    total = 0
    cache = {}
    for y in range(8):
        for x in range(8):
            r, g, b = (int(v) for v in tile_rgb[y, x])
            ck = (r, g, b)
            err = cache.get(ck)
            if err is None:
                pr, pg, pb = _rgb_of(palette[_closest_index(_hex(r, g, b), palette)])
                err = abs(r - pr) + abs(g - pg) + abs(b - pb)
                cache[ck] = err
            total += err
    return total


def recolor_overflow(arr: np.ndarray):
    """Pre-empt GB Studio's silent ``% 8`` palette-overflow corruption by
    recoloring the offending tiles ourselves, with a best-fit palette choice.

    Runs GB's autoPalette machinery and inspects the *un-wrapped* greedy-merge
    result. Tiles whose merged palette index is < 8 land on a palette GB keeps,
    so they import faithfully and are copied through untouched. Tiles whose
    merged index is >= 8 are the ones GB would wrap ``% 8`` (arbitrary, lossy):
    each is rewritten as GB's own reconstruction (nearest-color, Manhattan RGB)
    against the kept palette (index < 8) that minimizes the tile's total
    reconstruction error -- the best fit, not GB's blind wrap.

    Pure: does not mutate ``arr``. Returns ``(new_arr, n_recolored)``."""
    arr = np.asarray(arr, dtype=np.uint8)
    all_palettes, tile_map, (h, w, xt, yt) = _extract_tile_palettes(arr)
    out, origins = _greedy_compress_groups(all_palettes)

    # Un-wrapped merged index per original palette: GB's greedy result BEFORE the
    # corrupting ``% 8`` wrap, so we can tell keepers (< 8) from overflow (>= 8).
    unwrapped = [0] * len(all_palettes)
    for new_index, group in enumerate(origins):
        for origin in group:
            unwrapped[origin] = new_index
    filled = [fill_variable_palette(_sort_hex_palette(p)) for p in out]
    kept = filled[:8]  # the 8 palettes GB actually keeps

    new_arr = arr.copy()
    n_recolored = 0
    for ty in range(yt):
        for tx in range(xt):
            ti = ty * xt + tx
            if unwrapped[tile_map[ti]] < 8:
                continue  # maps to a kept master -> GB imports it faithfully
            block = arr[ty * 8:ty * 8 + 8, tx * 8:tx * 8 + 8]
            best_pal = kept[0]
            best_err = None
            for pal in kept:
                err = _tile_reconstruction_error(block, pal)
                if best_err is None or err < best_err:
                    best_err = err
                    best_pal = pal
            for y in range(8):
                for x in range(8):
                    r, g, b = (int(v) for v in block[y, x])
                    slot = _closest_index(_hex(r, g, b), best_pal)
                    new_arr[ty * 8 + y, tx * 8 + x] = _rgb_of(best_pal[slot])
            n_recolored += 1
    return new_arr, n_recolored


def _flip_tile(tile, fx, fy):
    if fx:
        tile = tile[:, ::-1]
    if fy:
        tile = tile[::-1, :]
    return tile


def _count_tiles(indexed: np.ndarray, autoflip: bool) -> int:
    """Count unique stored background tiles as GB Studio does: exact 2bpp-byte
    dedup, optionally collapsing H/V/both flips (autoFlipTiles / color default)."""
    h, w = indexed.shape
    xt, yt = w // 8, h // 8
    lookup = set()
    for ty in range(yt):
        for tx in range(xt):
            tile = indexed[ty * 8:ty * 8 + 8, tx * 8:tx * 8 + 8]
            orig = tile_to_2bpp(tile)
            if not autoflip:
                lookup.add(orig)
                continue
            variants = [
                orig,
                tile_to_2bpp(_flip_tile(tile, True, False)),
                tile_to_2bpp(_flip_tile(tile, False, True)),
                tile_to_2bpp(_flip_tile(tile, True, True)),
            ]
            if not any(v in lookup for v in variants):
                lookup.add(orig)
    return len(lookup)


def _snap5(v: int) -> int:
    return int(round(round((v / 255.0) * 31) / 31 * 255))


@dataclass
class GBStudioStats:
    tiles_noflip: int
    tiles_autoflip: int
    palettes_extracted: int
    corrupted_tiles: int
    corrupted_pixels: int


def gbstudio_color_stats(arr: np.ndarray) -> GBStudioStats:
    """Run GB Studio v4.3.2's Color-Only background import on an RGB image and
    report the tile/palette counts it actually computes plus how many tiles it
    would color-corrupt (from extracting >8 palettes and wrapping ``% 8``)."""
    arr = np.asarray(arr, dtype=np.uint8)
    indexed, palettes, tile_map = autopalette(arr)
    h, w = arr.shape[:2]
    xt, yt = w // 8, h // 8

    corrupt_tiles = 0
    corrupt_pixels = 0
    for ty in range(yt):
        for tx in range(xt):
            ti = ty * xt + tx
            pal = palettes[tile_map[ti]]
            bad = False
            for y in range(8):
                for x in range(8):
                    r, g, b = (int(v) for v in arr[ty * 8 + y, tx * 8 + x])
                    gb = pal[int(indexed[ty * 8 + y, tx * 8 + x])]
                    gr, gg, gbb = _rgb_of(gb)
                    if (_snap5(r), _snap5(g), _snap5(b)) != (_snap5(gr), _snap5(gg), _snap5(gbb)):
                        corrupt_pixels += 1
                        bad = True
            if bad:
                corrupt_tiles += 1

    return GBStudioStats(
        tiles_noflip=_count_tiles(indexed, autoflip=False),
        tiles_autoflip=_count_tiles(indexed, autoflip=True),
        palettes_extracted=len(palettes),
        corrupted_tiles=corrupt_tiles,
        corrupted_pixels=corrupt_pixels,
    )


def gbstudio_report(stats: GBStudioStats, budget):
    """Turn GBStudioStats into user-facing warnings + honest stat fields.

    Warns when GB Studio's Color-Only import would (a) extract >8 palettes --
    it silently wraps the overflow ``% 8`` and recolors those tiles -- or
    (b) store more unique tiles than ``budget`` after its own flip dedup.
    """
    warnings = []
    if stats.palettes_extracted > 8:
        warnings.append(
            f"GB Studio will re-extract {stats.palettes_extracted} color "
            f"palettes on import (max 8) and recolor {stats.corrupted_tiles} "
            "tiles with the wrong palette -- reduce the number of distinct "
            "colors/palettes in this scene."
        )
    if budget is not None and stats.tiles_autoflip > budget:
        warnings.append(
            f"GB Studio counts {stats.tiles_autoflip} unique tiles on import, "
            f"over the {budget} budget -- simplify the image or reduce detail."
        )
    extra = {
        "gbstudio_tiles": stats.tiles_autoflip,
        "gbstudio_palettes_extracted": stats.palettes_extracted,
        "gbstudio_corrupted_tiles": stats.corrupted_tiles,
    }
    return warnings, extra


def _mono_tile_indices(tile_rgb: np.ndarray) -> np.ndarray:
    """Per-pixel GB Studio mono green-threshold index (0-3) for an 8x8 RGB
    tile, via the same fixed ``_green_index`` buckets the color path uses."""
    tile_rgb = np.asarray(tile_rgb, dtype=np.uint8)
    idx = np.zeros((8, 8), dtype=np.uint8)
    for y in range(8):
        for x in range(8):
            idx[y, x] = _green_index(int(tile_rgb[y, x, 1]))
    return idx


@dataclass
class GBStudioMonoStats:
    tiles: int


def gbstudio_mono_stats(arr: np.ndarray) -> GBStudioMonoStats:
    """GB Studio's Mono/DMG background import tile counting: per-tile
    green-threshold indices (``_green_index``), packed to 2bpp, deduped by
    exact bytes. Unlike the color path's ``autoFlipTiles``, GB Studio's mono
    importer does NOT collapse horizontal/vertical flips."""
    arr = np.asarray(arr, dtype=np.uint8)
    h, w = arr.shape[:2]
    xt, yt = w // 8, h // 8
    lookup = set()
    for ty in range(yt):
        for tx in range(xt):
            block = arr[ty * 8:ty * 8 + 8, tx * 8:tx * 8 + 8]
            lookup.add(tile_to_2bpp(_mono_tile_indices(block)))
    return GBStudioMonoStats(tiles=len(lookup))


def mono_ramp_green_buckets(ramp) -> list:
    """Green-channel bucket (``_green_index``) for each entry of a 4-shade
    mono ramp, so callers can detect a custom ramp whose shades collide under
    GB Studio's fixed thresholds (non-distinct buckets -> shades silently
    merge on import)."""
    ramp = np.asarray(ramp, dtype=np.uint8)
    return [_green_index(int(ramp[i, 1])) for i in range(len(ramp))]


def tile_to_2bpp(tile: np.ndarray) -> bytes:
    """Pack an 8x8 array of 0-3 indices to GB 2bpp (16 bytes).

    Mirrors indexedImageTo2bppTileData: each row -> two bytes
    [low-bit plane, high-bit plane], with the leftmost pixel as the MSB.
    """
    tile = np.asarray(tile, dtype=np.uint8)
    out = bytearray(16)
    i = 0
    for y in range(8):
        low = 0
        high = 0
        for x in range(8):
            idx = int(tile[y, x])
            low = (low << 1) | (idx & 1)
            high = (high << 1) | ((idx >> 1) & 1)
        out[i] = low
        out[i + 1] = high
        i += 2
    return bytes(out)
