"""Tests for gb_pipeline.py Task 1.3 indexing + Bayer dither (spec Stage 4)."""

import numpy as np
import pytest

import gb_pipeline
from gb_pipeline import BAYER4, index_tiles, snap_rgb555
from gb_pipeline import GBImage, _rgb_to_lab


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


def test_projection_dither_no_speckle_beyond_endpoints():
    # Palette: mid-gray and dark-gray. Pixels are pure white -- beyond the
    # light end of the segment. Old d1/(d1+d2) ratio dithered these; the
    # projection must clamp t to 0 -> no dithering at all.
    pal = np.array([[[128, 128, 128], [128, 128, 128],
                     [64, 64, 64], [0, 0, 0]]], dtype=np.uint8)
    pal = gb_pipeline.snap_rgb555(pal)
    arr = np.full((8, 8, 3), 255, dtype=np.uint8)
    assignment = np.zeros((1, 1), dtype=np.uint8)
    gb = gb_pipeline.index_tiles(arr, pal, assignment, dither="bayer")
    assert len(np.unique(gb.patterns)) == 1  # all pixels -> single nearest entry


def test_projection_dither_midpoint_mixes():
    # Entries 0/1 are a duplicated far-away filler color (never the nearest
    # two -- with a fully-duplicated palette the two *nearest* entries are
    # structurally guaranteed to be a duplicate pair of the *same* color,
    # which is exactly the ||c2-c1||^2==0 guard case and correctly never
    # dithers; that would not exercise the on-segment midpoint path this
    # test targets). Entries 2/3 are the two genuinely distinct colors we
    # place the pixel exactly between.
    pal = np.array([[[255, 255, 255], [255, 255, 255],
                     [200, 200, 200], [40, 40, 40]]], dtype=np.uint8)
    pal = gb_pipeline.snap_rgb555(pal)
    c1 = gb_pipeline._rgb_to_lab(pal[0, 2:3])[0]
    c2 = gb_pipeline._rgb_to_lab(pal[0, 3:4])[0]
    mid_lab = ((c1 + c2) / 2)[None, :]
    mid_rgb = gb_pipeline._lab_to_rgb_u8(mid_lab)[0]
    arr = np.tile(mid_rgb, (8, 8, 1)).astype(np.uint8)
    assignment = np.zeros((1, 1), dtype=np.uint8)
    gb = gb_pipeline.index_tiles(arr, pal, assignment, dither="bayer")
    vals, counts = np.unique(gb.patterns, return_counts=True)
    assert len(vals) == 2
    assert set(vals.tolist()) == {2, 3}
    assert 0.25 <= counts[0] / counts.sum() <= 0.75


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


def _index_tiles_reference(image, palettes, assignment, dither="none"):
    """Frozen copy of the post-Task-5 per-tile index_tiles body.

    Task 7 vectorizes index_tiles; this plain reference (projection-dither
    spec) pins bit-identical behavior through the rewrite.
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
            block = arr[tr * 8 : tr * 8 + 8, tc * 8 : tc * 8 + 8]
            pid = int(assignment[tr, tc])
            plab = pal_lab[pid]
            blab = _rgb_to_lab(block).reshape(8, 8, 3)
            diff = blab[:, :, None, :] - plab[None, None, :, :]
            dist = np.sqrt(np.sum(diff * diff, axis=3))
            order = np.argsort(dist, axis=2, kind="stable")
            idx1 = order[:, :, 0]

            if dither == "bayer":
                idx2 = order[:, :, 1]
                c1 = plab[idx1]
                c2 = plab[idx2]
                seg = c2 - c1
                seg_len2 = np.sum(seg * seg, axis=2)
                proj = np.sum((blab - c1) * seg, axis=2)
                t_val = np.where(
                    seg_len2 > 0, proj / np.where(seg_len2 > 0, seg_len2, 1.0), 0.0
                )
                t_val = np.clip(t_val, 0.0, 1.0)
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


@pytest.mark.parametrize("dither", ["none", "bayer"])
def test_index_tiles_matches_reference(dither):
    rng = np.random.RandomState(7)
    for _ in range(3):
        h, w = 24, 32
        arr = rng.randint(0, 256, (h, w, 3)).astype(np.uint8)
        pals = gb_pipeline.snap_rgb555(rng.randint(0, 256, (3, 4, 3)).astype(np.uint8))
        pals = np.stack([gb_pipeline.luminance_sort(p) for p in pals])
        assignment = rng.randint(0, 3, (h // 8, w // 8)).astype(np.uint8)
        got = gb_pipeline.index_tiles(arr, pals, assignment, dither=dither)
        want = _index_tiles_reference(arr, pals, assignment, dither=dither)
        assert np.array_equal(got.patterns, want.patterns)
