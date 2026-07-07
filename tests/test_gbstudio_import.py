"""Tests for gb_studio_import.py -- a faithful Python port of GB Studio v4.3.2's
background importer (autoPalette / autoFlip / tile counting), used to report the
tile/palette counts GB Studio *actually* computes and to warn when its import
would corrupt colors (>8 extracted palettes) or exceed the tile budget.

Ground truth is GB Studio's own TypeScript source (v4.3.2):
  src/shared/lib/tiles/{indexedImage,tileData,autoColor,autoFlip}.ts
"""

import os

import numpy as np
from PIL import Image

import gb_studio_import as gsi

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _load(name):
    return np.asarray(Image.open(os.path.join(FIXTURES, name)).convert("RGB"), np.uint8)


def test_2bpp_all_index_one_is_low_plane_set():
    """indexedImageTo2bppTileData: index 1 sets only the low bit-plane.

    GB Studio packs each row as two bytes [low-plane, high-plane], MSB = the
    leftmost pixel. A tile of all-index-1 -> every row low-plane=0xFF, high=0x00.
    """
    tile = np.ones((8, 8), dtype=np.uint8)  # all index 1
    data = gsi.tile_to_2bpp(tile)
    assert list(data) == [0xFF, 0x00] * 8


def test_2bpp_leftmost_pixel_is_msb():
    """A single index-1 pixel at x=0 sets bit7 of the low-plane byte."""
    tile = np.zeros((8, 8), dtype=np.uint8)
    tile[0, 0] = 1
    data = gsi.tile_to_2bpp(tile)
    assert data[0] == 0x80 and data[1] == 0x00


def test_extract_tile_palette_first_four_distinct_sorted_bright_first():
    """extractTilePalette: keep the first <=4 distinct colors in scan order,
    drop the rest, then sort by perceptual lightness (brightest first)."""
    A, B, C, D, E = (
        (10, 10, 10), (200, 200, 200), (100, 100, 100),
        (150, 150, 150), (50, 50, 50),
    )
    tile = np.zeros((8, 8, 3), dtype=np.uint8)
    tile[0, 0:8] = [A, B, C, D, E, A, A, A]  # E is the 5th distinct -> dropped
    tile[1:] = A
    assert gsi.extract_tile_palette(tile) == ["c8c8c8", "969696", "646464", "0a0a0a"]


def test_greedy_compress_merges_when_union_fits_four():
    """compressPalettes greedily merges any two palettes whose color union is
    <=4, repeatedly. Three overlapping palettes collapse to one."""
    palettes = [
        ["ff0000", "00ff00"],
        ["0000ff", "ffffff"],
        ["ff0000", "00ff00", "0000ff"],
    ]
    out, mapping = gsi._greedy_compress(palettes)
    assert len(out) == 1
    assert mapping == [0, 0, 0]


def test_greedy_compress_keeps_disjoint_palettes_separate():
    """Two palettes whose union exceeds 4 colors cannot merge."""
    palettes = [
        ["010101", "020202", "030303", "040404"],
        ["050505", "060606", "070707", "080808"],
    ]
    out, mapping = gsi._greedy_compress(palettes)
    assert len(out) == 2
    assert mapping == [0, 1]


def test_greedy_compress_wraps_overflow_mod_eight():
    """>8 resulting palettes wrap the mapping table % 8 (the silent-corruption
    behavior): the 9th and 10th disjoint palettes map back onto slots 0 and 1."""
    palettes = [
        [f"{i:02x}0000", f"{i:02x}0001", f"{i:02x}0002", f"{i:02x}0003"]
        for i in range(1, 11)  # 10 mutually disjoint 4-color palettes
    ]
    out, mapping = gsi._greedy_compress(palettes)
    assert len(out) == 10          # greedy cannot merge any (all disjoint)
    assert mapping[8] == 0 and mapping[9] == 1  # wrapped % 8


def test_color_stats_matches_gbstudio_on_overflow_fixture():
    """End-to-end vs GB Studio's real importer (node oracle ground truth):
    12 disjoint 4-color tile palettes -> GB extracts 12 (>8) -> wraps %8, so
    the 4 overflow tiles render with the wrong palette (256 px corrupted).
    All tiles share one arrangement, so the tileset dedups to a single tile."""
    arr = _load("overflow_palettes.png")
    s = gsi.gbstudio_color_stats(arr)
    assert s.palettes_extracted == 12
    assert s.tiles_noflip == 1
    assert s.tiles_autoflip == 1
    assert s.corrupted_tiles == 4
    assert s.corrupted_pixels == 256


def test_gbstudio_report_warns_on_palette_overflow():
    """>8 extracted palettes => GB Studio silently recolors overflow tiles; the
    report must warn and surface the honest counts."""
    stats = gsi.GBStudioStats(
        tiles_noflip=250, tiles_autoflip=247, palettes_extracted=10,
        corrupted_tiles=53, corrupted_pixels=3392,
    )
    warnings, extra = gsi.gbstudio_report(stats, budget=384)
    assert any("palette" in w.lower() for w in warnings)
    assert extra["gbstudio_palettes_extracted"] == 10
    assert extra["gbstudio_tiles"] == 247
    assert extra["gbstudio_corrupted_tiles"] == 53


def test_gbstudio_report_clean_when_within_limits():
    stats = gsi.GBStudioStats(300, 290, 7, 0, 0)
    warnings, extra = gsi.gbstudio_report(stats, budget=384)
    assert warnings == []


def test_gbstudio_report_warns_over_budget():
    stats = gsi.GBStudioStats(400, 390, 8, 0, 0)
    warnings, extra = gsi.gbstudio_report(stats, budget=384)
    assert any("tile" in w.lower() for w in warnings)


def test_convert_for_hardware_color_includes_gbstudio_stats(photo_like_image):
    """Color conversions surface GB Studio's real import counts in stats, and
    they match a direct run of the importer on the rendered PNG."""
    from gb_pipeline import convert_for_hardware
    res = convert_for_hardware(photo_like_image, "color_only")
    assert "gbstudio_tiles" in res.stats
    assert "gbstudio_palettes_extracted" in res.stats
    assert "gbstudio_corrupted_tiles" in res.stats
    direct = gsi.gbstudio_color_stats(np.asarray(res.image.convert("RGB"), np.uint8))
    assert res.stats["gbstudio_tiles"] == direct.tiles_autoflip
    assert res.stats["gbstudio_palettes_extracted"] == direct.palettes_extracted


def test_convert_for_hardware_mono_skips_gbstudio_color_stats(photo_like_image):
    """Mono uses the DMG path, not GB Studio's color auto-palette, so the
    color-only stats are not attached."""
    from gb_pipeline import convert_for_hardware
    res = convert_for_hardware(photo_like_image, "mono")
    assert "gbstudio_palettes_extracted" not in res.stats
