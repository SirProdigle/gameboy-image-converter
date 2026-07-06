"""Unit tests for gb_pipeline.py Task 1.1 primitives."""

import numpy as np

from gb_pipeline import (
    DMG_RAMP,
    luminance_sort,
    pattern_to_2bpp,
    pattern_variants,
    snap_rgb555,
)


def test_snap_rgb555_idempotent_all_byte_values():
    values = np.arange(256, dtype=np.uint8)
    colors = np.stack([values, values[::-1], values], axis=-1)
    once = snap_rgb555(colors)
    twice = snap_rgb555(once)
    assert np.array_equal(once, twice)


def test_snap_rgb555_white_stays_white():
    # floor(255*32/256) = 31 -> round(31*255/31) = 255
    result = snap_rgb555(np.array([255, 255, 255]))
    assert np.array_equal(result, np.array([255, 255, 255]))


def test_snap_rgb555_hand_computed():
    # 8  -> c5 = floor(8*32/256)   = 1  -> round(1*255/31)  = 8
    # 16 -> c5 = floor(16*32/256)  = 2  -> round(2*255/31)  = 16
    # 250-> c5 = floor(250*32/256) = 31 -> round(31*255/31) = 255
    result = snap_rgb555(np.array([8, 16, 250]))
    assert np.array_equal(result, np.array([8, 16, 255]))


def test_snap_rgb555_dtype_and_shape_preserved():
    colors = np.zeros((4, 3), dtype=np.uint8) + 100
    result = snap_rgb555(colors)
    assert result.dtype == np.uint8
    assert result.shape == colors.shape


def test_luminance_sort_orders_white_before_black():
    palette = np.array(
        [[0, 0, 0], [255, 255, 255], [128, 128, 128], [64, 64, 64]],
        dtype=np.uint8,
    )
    sorted_palette = luminance_sort(palette)
    assert np.array_equal(sorted_palette[0], [255, 255, 255])
    assert np.array_equal(sorted_palette[-1], [0, 0, 0])


def test_luminance_sort_dmg_ramp_is_already_lightest_first():
    sorted_ramp = luminance_sort(DMG_RAMP)
    assert np.array_equal(sorted_ramp, DMG_RAMP)


def test_luminance_sort_shape():
    palette = np.random.RandomState(0).randint(0, 256, size=(4, 3)).astype(np.uint8)
    result = luminance_sort(palette)
    assert result.shape == (4, 3)


def test_pattern_to_2bpp_checkerboard(checker_tile):
    encoded = pattern_to_2bpp(checker_tile)
    assert isinstance(encoded, bytes)
    assert len(encoded) == 16
    # checker_tile rows alternate [0,1,0,1,0,1,0,1] / [1,0,1,0,1,0,1,0];
    # values are 0/1 only so the high-bit byte is always 0.
    expected = bytes([0x55, 0x00, 0xAA, 0x00] * 4)
    assert encoded == expected


def test_pattern_to_2bpp_length_always_16():
    pattern = np.random.RandomState(1).randint(0, 4, size=(8, 8)).astype(np.uint8)
    assert len(pattern_to_2bpp(pattern)) == 16


def _asymmetric_pattern():
    r = np.arange(8).reshape(8, 1)
    c = np.arange(8).reshape(1, 8)
    return ((r + 2 * c + (r * c) % 3) % 4).astype(np.uint8)


def test_pattern_variants_asymmetric_are_four_distinct_arrays():
    pattern = _asymmetric_pattern()
    variants = pattern_variants(pattern)
    assert set(variants.keys()) == {"", "h", "v", "hv"}
    keys = list(variants.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert not np.array_equal(variants[keys[i]], variants[keys[j]]), (
                keys[i],
                keys[j],
            )


def test_pattern_variants_symmetric_pattern_collides():
    # Each row is a horizontal palindrome (c0==c7, c1==c6, c2==c5, c3==c4)
    # but rows differ from each other and aren't vertically symmetric.
    # So fliplr is a no-op ("" collides with "h", and "v" collides with
    # "hv"), while flipud still changes the pattern ("" != "v").
    rows = []
    for i in range(8):
        a, b, c, d = i % 4, (i + 1) % 4, (i + 2) % 4, (i + 3) % 4
        rows.append([a, b, c, d, d, c, b, a])
    pattern = np.array(rows, dtype=np.uint8)

    variants = pattern_variants(pattern)
    assert np.array_equal(variants[""], variants["h"])
    assert np.array_equal(variants["v"], variants["hv"])
    assert not np.array_equal(variants[""], variants["v"])


def test_pattern_variants_correctness_vs_numpy():
    pattern = _asymmetric_pattern()
    variants = pattern_variants(pattern)
    assert np.array_equal(variants[""], pattern)
    assert np.array_equal(variants["h"], np.fliplr(pattern))
    assert np.array_equal(variants["v"], np.flipud(pattern))
    assert np.array_equal(variants["hv"], np.flipud(np.fliplr(pattern)))
