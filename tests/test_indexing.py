"""Tests for gb_pipeline.py Task 1.3 indexing + Bayer dither (spec Stage 4)."""

import numpy as np

import gb_pipeline
from gb_pipeline import BAYER4, index_tiles, snap_rgb555


# A simple 4-color palette, distinct colors, luminance-descending order
# assumed (not required by index_tiles itself, but kept realistic).
PALETTE = snap_rgb555(
    np.array(
        [[240, 240, 240], [170, 170, 170], [90, 90, 90], [10, 10, 10]],
        dtype=np.uint8,
    )
)[None, :, :]  # (1, 4, 3)


def _solid_tile(color, tiles_w=1, tiles_h=1):
    arr = np.zeros((tiles_h * 8, tiles_w * 8, 3), dtype=np.uint8)
    arr[:, :] = color
    return arr


def test_flat_tile_all_one_index():
    color = PALETTE[0, 2]  # exact palette color
    image = _solid_tile(color)
    assignment = np.zeros((1, 1), dtype=np.uint8)
    gb = index_tiles(image, PALETTE, assignment, dither="none")
    assert gb.patterns.shape == (1, 8, 8)
    assert np.all(gb.patterns[0] == gb.patterns[0][0, 0])
    assert gb.patterns[0][0, 0] == 2


def test_exact_palette_colors_map_regardless_of_dither():
    # Each 2x2 pixel block uses a distinct exact palette color.
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    expected = np.zeros((8, 8), dtype=np.uint8)
    quadrant_colors = [0, 1, 2, 3]
    for qi, color_idx in enumerate(quadrant_colors):
        r0 = (qi // 2) * 4
        c0 = (qi % 2) * 4
        image[r0 : r0 + 4, c0 : c0 + 4] = PALETTE[0, color_idx]
        expected[r0 : r0 + 4, c0 : c0 + 4] = color_idx
    assignment = np.zeros((1, 1), dtype=np.uint8)

    for mode in ("none", "bayer"):
        gb = index_tiles(image, PALETTE, assignment, dither=mode)
        assert np.array_equal(gb.patterns[0], expected), mode


def test_bayer_dither_mixes_indices_at_midtone():
    # A flat tile filled with the color exactly halfway between palette
    # entries 0 and 1. Without dither every pixel goes to a single nearest
    # index; with Bayer dithering the ordered matrix splits it between both.
    c0 = PALETTE[0, 0].astype(np.int32)
    c1 = PALETTE[0, 1].astype(np.int32)
    mid = ((c0 + c1) // 2).astype(np.uint8)
    image = _solid_tile(mid)
    assignment = np.zeros((1, 1), dtype=np.uint8)

    gb_none = index_tiles(image, PALETTE, assignment, dither="none")
    assert len(np.unique(gb_none.patterns[0])) == 1

    gb_bayer = index_tiles(image, PALETTE, assignment, dither="bayer")
    uniques = np.unique(gb_bayer.patterns[0])
    assert len(uniques) == 2
    assert set(uniques.tolist()) == {0, 1}


def test_absolute_coordinate_anchoring_identical_tiles_match():
    # Two tiles (side by side, and stacked) with identical content -- since
    # tile size (8) is a multiple of the Bayer matrix size (4), the (y%4,
    # x%4) phase used for dithering is always the same at any 8-aligned tile
    # origin, so identical tile content must dither identically everywhere.
    c0 = PALETTE[0, 0].astype(np.int32)
    c1 = PALETTE[0, 1].astype(np.int32)
    mid = ((c0 + c1) // 2).astype(np.uint8)

    tile = np.zeros((8, 8, 3), dtype=np.uint8)
    tile[:, :] = mid
    # A small horizontal gradient within the tile so it isn't perfectly flat.
    for col in range(8):
        t = col / 7.0
        tile[:, col] = np.round(c0 * (1 - t) + c1 * t).astype(np.uint8)

    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[0:8, 0:8] = tile
    image[0:8, 8:16] = tile
    image[8:16, 0:8] = tile
    image[8:16, 8:16] = tile
    assignment = np.zeros((2, 2), dtype=np.uint8)

    gb = index_tiles(image, PALETTE, assignment, dither="bayer")
    assert np.array_equal(gb.patterns[0], gb.patterns[1])
    assert np.array_equal(gb.patterns[0], gb.patterns[2])
    assert np.array_equal(gb.patterns[0], gb.patterns[3])


def test_output_values_all_in_0_to_3():
    rng = np.random.RandomState(42)
    image = rng.randint(0, 256, size=(16, 24, 3)).astype(np.uint8)
    assignment = np.zeros((2, 3), dtype=np.uint8)
    for mode in ("none", "bayer"):
        gb = index_tiles(image, PALETTE, assignment, dither=mode)
        assert gb.patterns.min() >= 0
        assert gb.patterns.max() <= 3
        assert gb.patterns.dtype == np.uint8


def test_index_tiles_structure():
    image = _solid_tile(PALETTE[0, 0], tiles_w=2, tiles_h=3)
    assignment = np.zeros((3, 2), dtype=np.uint8)
    gb = index_tiles(image, PALETTE, assignment, dither="none")
    assert gb.patterns.shape == (6, 8, 8)
    assert np.array_equal(gb.tilemap, np.arange(6).reshape(3, 2))
    assert not gb.attrs_hflip.any()
    assert not gb.attrs_vflip.any()
    assert np.array_equal(gb.attrs_palette, assignment)
    assert np.array_equal(gb.palettes, PALETTE)


def test_bayer_matrix_shape_and_values():
    assert BAYER4.shape == (4, 4)
    assert np.isclose(BAYER4.max(), 15 / 16.0)
    assert np.isclose(BAYER4.min(), 0.0)


# --- mono luminance path (Task 3) -----------------------------------------


def test_mono_bayer_dithers_continuous_gradient():
    w, h = 160, 8
    grad = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    arr = np.stack([grad] * 3, axis=2)
    ramp = gb_pipeline.DMG_RAMP
    palettes = np.asarray(gb_pipeline.luminance_sort(gb_pipeline.snap_rgb555(ramp)))[None]
    gb = gb_pipeline._index_tiles_mono(arr, palettes, dither="bayer")
    idx = np.concatenate([gb.patterns[t] for t in range(gb.patterns.shape[0])], axis=1)
    # Monotonic mean index (lightest-first ramp: index 0 is lightest, gradient
    # goes dark->light left->right, so mean index must be non-increasing along x,
    # allowing dither noise).
    col_means = idx.mean(axis=0)
    smooth = np.convolve(col_means, np.ones(8) / 8, mode="valid")
    assert np.all(np.diff(smooth) <= 0.15)
    # Transition zones actually mix two adjacent levels.
    mid = idx[:, w // 3 : 2 * w // 3]
    assert len(np.unique(mid)) >= 2


def test_mono_no_dither_gives_clean_bands():
    w, h = 160, 8
    grad = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    arr = np.stack([grad] * 3, axis=2)
    palettes = np.asarray(gb_pipeline.DMG_RAMP)[None]
    gb = gb_pipeline._index_tiles_mono(arr, palettes, dither="none")
    idx = np.concatenate([gb.patterns[t] for t in range(gb.patterns.shape[0])], axis=1)
    # Every column is a single level; 4 bands total.
    assert all(len(np.unique(idx[:, c])) == 1 for c in range(w))
    assert len(np.unique(idx)) == 4


def test_mono_constant_image_no_crash():
    arr = np.full((16, 16, 3), 137, dtype=np.uint8)
    palettes = np.asarray(gb_pipeline.DMG_RAMP)[None]
    gb = gb_pipeline._index_tiles_mono(arr, palettes, dither="bayer")
    assert len(np.unique(gb.patterns)) == 1
