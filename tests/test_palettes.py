"""Tests for gb_pipeline.py Task 1.2 palette packing (spec Stage 3)."""

import numpy as np
from PIL import Image
from skimage.color import rgb2lab

from gb_pipeline import (
    DMG_RAMP,
    luminance_sort,
    pack_palettes,
    pack_palettes_mono,
    quantize_working_set,
    snap_rgb555,
)


def _lab(colors):
    arr = np.asarray(colors, dtype=np.float64).reshape(-1, 3) / 255.0
    return rgb2lab(arr[:, None, :]).reshape(-1, 3)


def _per_tile_max_delta_e(image, palettes, assignment):
    """Max Euclidean-Lab distance of any pixel to its assigned palette."""
    arr = np.asarray(image, dtype=np.uint8)
    th, tw = assignment.shape
    worst = 0.0
    for tr in range(th):
        for tc in range(tw):
            block = arr[tr * 8 : tr * 8 + 8, tc * 8 : tc * 8 + 8].reshape(-1, 3)
            pal_lab = _lab(palettes[assignment[tr, tc]])
            blk_lab = _lab(block)
            diff = blk_lab[:, None, :] - pal_lab[None, :, :]
            dist = np.sqrt(np.sum(diff * diff, axis=2)).min(axis=1)
            worst = max(worst, float(dist.max()))
    return worst


def _assert_invariants(palettes, n_palettes):
    assert palettes.ndim == 3
    assert palettes.shape[0] <= n_palettes
    assert palettes.shape[1:] == (4, 3)
    assert palettes.dtype == np.uint8
    # All colors are RGB555-stable.
    assert np.array_equal(snap_rgb555(palettes), palettes)
    # Every palette is luminance-sorted (lightest first).
    for pal in palettes:
        assert np.array_equal(luminance_sort(pal), pal)


def _flat_array():
    arr = np.full((144, 160, 3), (96, 128, 160), dtype=np.uint8)
    return snap_rgb555(arr)


def _quadrant_array():
    """160x144, four 8-aligned quadrants each using its own 4 colors."""
    width, height = 160, 144
    quad_colors = snap_rgb555(
        np.array(
            [
                [[200, 30, 30], [150, 60, 20], [220, 90, 40], [180, 20, 10]],
                [[30, 200, 40], [60, 150, 20], [90, 220, 30], [20, 180, 50]],
                [[30, 40, 200], [60, 20, 150], [30, 90, 220], [50, 20, 180]],
                [[220, 220, 220], [160, 160, 160], [100, 100, 100], [40, 40, 40]],
            ],
            dtype=np.uint8,
        )
    ).reshape(4, 4, 3)
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:height, 0:width]
    region = (yy >= height // 2).astype(int) * 2 + (xx >= width // 2).astype(int)
    sel = (xx + yy) % 4
    for q in range(4):
        for c in range(4):
            arr[(region == q) & (sel == c)] = quad_colors[q, c]
    return arr


def _gradient_working_set():
    ramp = np.linspace(0, 255, num=160, endpoint=True, dtype=np.uint8)
    row = np.stack([ramp, ramp, ramp], axis=-1)
    grad = np.tile(row, (144, 1, 1)).astype(np.uint8)
    quantized = quantize_working_set(Image.fromarray(grad, "RGB"), 28)
    return np.asarray(quantized, dtype=np.uint8)


# --- quantize_working_set --------------------------------------------------


def test_quantize_working_set_bounds_and_snaps():
    ramp = np.linspace(0, 255, num=160, endpoint=True, dtype=np.uint8)
    row = np.stack([ramp, ramp, ramp], axis=-1)
    grad = np.tile(row, (144, 1, 1)).astype(np.uint8)
    out = quantize_working_set(Image.fromarray(grad, "RGB"), 12)
    arr = np.asarray(out, dtype=np.uint8)
    colors = np.unique(arr.reshape(-1, 3), axis=0)
    assert out.mode == "RGB"
    assert len(colors) <= 12
    # Every working-set color is RGB555-stable.
    assert np.array_equal(snap_rgb555(colors), colors)


def test_quantize_working_set_custom_palette_restricts_colors():
    custom = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    src = np.random.RandomState(1).randint(0, 256, size=(16, 16, 3)).astype(np.uint8)
    out = quantize_working_set(Image.fromarray(src, "RGB"), 8, custom_palette=custom)
    colors = np.unique(np.asarray(out).reshape(-1, 3), axis=0)
    allowed = {tuple(c) for c in snap_rgb555(custom).tolist()}
    for c in colors:
        assert tuple(int(v) for v in c) in allowed


# --- pack_palettes ---------------------------------------------------------


def test_pack_palettes_flat_image_single_palette():
    palettes, assignment = pack_palettes(_flat_array(), 7)
    assert palettes.shape[0] == 1
    assert np.array_equal(np.unique(assignment), np.array([0]))
    _assert_invariants(palettes, 7)


def test_pack_palettes_quadrants_zero_reconstruction_error():
    arr = _quadrant_array()
    assert len(np.unique(arr.reshape(-1, 3), axis=0)) == 16
    palettes, assignment = pack_palettes(arr, 7)
    assert palettes.shape[0] <= 4
    assert _per_tile_max_delta_e(arr, palettes, assignment) == 0.0
    _assert_invariants(palettes, 7)


def test_pack_palettes_gradient_small_error_per_tile():
    ws = _gradient_working_set()
    palettes, assignment = pack_palettes(ws, 7)
    assert _per_tile_max_delta_e(ws, palettes, assignment) < 8.0
    _assert_invariants(palettes, 7)


def test_pack_palettes_invariants_shape_snap_sorted():
    ws = _gradient_working_set()
    palettes, assignment = pack_palettes(ws, 7)
    _assert_invariants(palettes, 7)
    assert assignment.shape == (144 // 8, 160 // 8)
    assert assignment.dtype == np.uint8
    assert assignment.max() < palettes.shape[0]


def test_pack_palettes_deterministic():
    ws = _gradient_working_set()
    p1, a1 = pack_palettes(ws, 7)
    p2, a2 = pack_palettes(ws, 7)
    assert np.array_equal(p1, p2)
    assert np.array_equal(a1, a2)


def test_pack_palettes_respects_palette_budget():
    # 8 distinct 4-color tile palettes (a bounded 32-color working set) must
    # be merged down to a budget of 3 -- exercises agglomerative merging.
    rng = np.random.RandomState(7)
    base = snap_rgb555(rng.randint(0, 256, size=(8, 4, 3)).astype(np.uint8))
    # Ensure 8 genuinely distinct palettes (32 distinct colors).
    assert len(np.unique(base.reshape(-1, 3), axis=0)) == 32
    width, height = 160, 144
    th, tw = height // 8, width // 8
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for tr in range(th):
        for tc in range(tw):
            pal = base[(tr * tw + tc) % 8]
            block = pal[rng.randint(0, 4, size=(8, 8))]
            arr[tr * 8 : tr * 8 + 8, tc * 8 : tc * 8 + 8] = block

    # Under a generous budget, all 8 distinct palettes survive.
    palettes_wide, _ = pack_palettes(arr, 8)
    assert palettes_wide.shape[0] == 8

    # Under a tight budget they are merged down to exactly the budget.
    palettes, assignment = pack_palettes(arr, 3)
    assert palettes.shape[0] == 3
    _assert_invariants(palettes, 3)
    assert assignment.max() < palettes.shape[0]


def test_pack_palettes_custom_palette_restriction():
    custom = snap_rgb555(
        np.array(
            [[240, 240, 240], [170, 120, 60], [60, 90, 170], [20, 20, 20]],
            dtype=np.uint8,
        )
    )
    src = np.random.RandomState(3).randint(0, 256, size=(64, 64, 3)).astype(np.uint8)
    ws = np.asarray(
        quantize_working_set(Image.fromarray(src, "RGB"), 16, custom_palette=custom)
    )
    palettes, _ = pack_palettes(ws, 7, custom_palette=custom)
    allowed = {tuple(c) for c in custom.tolist()}
    for pal in palettes:
        for color in pal:
            assert tuple(int(v) for v in color) in allowed


# --- pack_palettes_mono ----------------------------------------------------


def test_pack_palettes_mono_single_ramp_palette():
    arr = _flat_array()
    palettes, assignment = pack_palettes_mono(arr, DMG_RAMP)
    assert palettes.shape == (1, 4, 3)
    assert np.array_equal(np.unique(assignment), np.array([0]))
    assert assignment.shape == (144 // 8, 160 // 8)
    expected = luminance_sort(snap_rgb555(DMG_RAMP))
    assert np.array_equal(palettes[0], expected)


def test_pack_palettes_mono_custom_ramp():
    ramp = np.array(
        [[250, 250, 250], [180, 180, 180], [90, 90, 90], [10, 10, 10]],
        dtype=np.uint8,
    )
    arr = _flat_array()
    palettes, _ = pack_palettes_mono(arr, ramp)
    assert np.array_equal(palettes[0], luminance_sort(snap_rgb555(ramp)))
