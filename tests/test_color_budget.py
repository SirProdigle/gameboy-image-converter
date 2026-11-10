"""Colour-budget control: a requested total colour count drives the palette
count, is hard-capped at the hardware ceiling (with a warning), and is
enforced exactly even when it is not a whole number of 4-colour palettes."""

import numpy as np
import pytest
from PIL import Image

import gb_pipeline
import main


def _rainbow_image(w=80, h=72) -> Image.Image:
    """Smooth multi-hue image: plenty of distinct colours to cap."""
    yy, xx = np.mgrid[0:h, 0:w]
    r = (xx * 255 // (w - 1)).astype(np.uint8)
    g = (yy * 255 // (h - 1)).astype(np.uint8)
    b = (((xx // 8 + yy // 8) % 4) * 85).astype(np.uint8)
    return Image.fromarray(np.stack([r, g, b], axis=-1), "RGB")


def _unique_colors(img: Image.Image) -> int:
    arr = np.asarray(img.convert("RGB")).reshape(-1, 3)
    return len(np.unique(arr, axis=0))


# --- cap_palette_colors ----------------------------------------------------

def _twelve_color_palettes():
    cols = np.array(
        [[i * 20, 255 - i * 15, (i * 70) % 256] for i in range(12)], dtype=np.uint8
    )
    cols = gb_pipeline.snap_rgb555(cols)
    return np.stack([gb_pipeline.luminance_sort(cols[i * 4:(i + 1) * 4]) for i in range(3)])


def test_cap_palette_colors_noop_when_within_budget():
    pals = _twelve_color_palettes()
    img = np.asarray(_rainbow_image())
    out = gb_pipeline.cap_palette_colors(pals, img, 12)
    assert np.array_equal(out, pals)


@pytest.mark.parametrize("cap", [1, 2, 5, 9, 11])
def test_cap_palette_colors_reduces_to_cap(cap):
    pals = _twelve_color_palettes()
    img = np.asarray(_rainbow_image())
    out = gb_pipeline.cap_palette_colors(pals, img, cap)
    assert out.shape == pals.shape
    assert gb_pipeline.count_palette_colors(out) <= cap
    for p in out:  # still luminance-sorted (lightest first)
        assert np.array_equal(p, gb_pipeline.luminance_sort(p))


def test_cap_palette_colors_without_new_colors_stays_in_original_set():
    pals = _twelve_color_palettes()
    img = np.asarray(_rainbow_image())
    out = gb_pipeline.cap_palette_colors(pals, img, 7, allow_new_colors=False)
    original = {tuple(c) for c in pals.reshape(-1, 3)}
    assert {tuple(c) for c in out.reshape(-1, 3)} <= original
    assert gb_pipeline.count_palette_colors(out) <= 7


# --- convert_for_hardware --------------------------------------------------

@pytest.mark.parametrize("n", [4, 9, 12, 20])
def test_convert_respects_color_budget(n):
    res = gb_pipeline.convert_for_hardware(
        _rainbow_image(), "color_only", max_colors=n
    )
    assert res.stats["colors_target"] == n
    assert res.stats["colors_used"] <= n
    assert _unique_colors(res.image) <= n
    assert res.stats["palettes_used"] <= -(-n // 4)
    assert res.stats["gbstudio_corrupted_tiles"] == 0
    assert not any("capped" in w for w in res.warnings)


def test_convert_default_uses_full_hardware_budget():
    res = gb_pipeline.convert_for_hardware(_rainbow_image(), "color_only")
    assert res.stats["colors_target"] == 28  # palette 8 reserved by default
    assert res.stats["palettes_used"] <= 7


def test_convert_default_without_reserve_is_32():
    res = gb_pipeline.convert_for_hardware(
        _rainbow_image(), "color_only", reserve_ui_palette=False
    )
    assert res.stats["colors_target"] == 32


def test_convert_warns_when_reserved_palette_caps_request():
    res = gb_pipeline.convert_for_hardware(
        _rainbow_image(), "color_only", max_colors=32
    )
    assert res.stats["colors_target"] == 28
    assert any("palette 8 is reserved" in w for w in res.warnings)


def test_convert_warns_when_over_hardware_limit():
    res = gb_pipeline.convert_for_hardware(
        _rainbow_image(), "color_only", max_colors=99, reserve_ui_palette=False
    )
    assert res.stats["colors_target"] == 32
    assert any("capped at 32" in w for w in res.warnings)


def test_convert_mono_ignores_color_budget():
    res = gb_pipeline.convert_for_hardware(_rainbow_image(), "mono", max_colors=2)
    assert res.stats["colors_target"] is None
    assert not res.warnings or not any("colors" in w for w in res.warnings)


def test_convert_logo_color_respects_budget():
    res = gb_pipeline.convert_for_hardware(
        _rainbow_image(160, 144), "logo_color", max_colors=8
    )
    assert res.stats["colors_used"] <= 8
    assert res.stats["palettes_used"] <= 2


def test_convert_custom_palette_with_budget_stays_in_palette():
    custom = np.array(
        [[0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255],
         [255, 255, 0]], dtype=np.uint8,
    )
    res = gb_pipeline.convert_for_hardware(
        _rainbow_image(), "color_only", custom_palette=custom, max_colors=5
    )
    allowed = {tuple(c) for c in gb_pipeline.snap_rgb555(custom)}
    got = {tuple(c) for c in np.asarray(res.image).reshape(-1, 3)}
    assert res.stats["colors_used"] <= 5
    assert got <= allowed


# --- UI glue ---------------------------------------------------------------

def test_process_image_passes_color_budget_and_notice_shows_it():
    out, _t, _ref, notice, _h = main.process_image(
        _rainbow_image(), main.MODE_COLOR, 80, 72, False,
        True, 4, "Median cut", "None", False, None,
        False, False, 128,
        False, 0, 1, 1, 1.5, 0.5, False, False, False, False,
        True, "None", 384, "Color", "Auto", 12,
    )
    assert _unique_colors(out) <= 12
    assert "/12 colors" in notice


def test_artistic_warns_over_gb_studio_limit():
    out = main._process_artistic(
        _rainbow_image(), True, 48, "Median cut", "None", False, None,
        False, False, 128, False, 0, 1, 1, 1.5, 0.5, False, False, False, False,
    )
    assert "32-color" in out[3]
    ok = main._process_artistic(
        _rainbow_image(), True, 16, "Median cut", "None", False, None,
        False, False, 128, False, 0, 1, 1, 1.5, 0.5, False, False, False, False,
    )
    assert ok[3] == "No Warnings"


def test_reserve_toggle_moves_slider_ceiling():
    u = main.on_reserve_change_color_ceiling(False, 28)
    assert u["maximum"] == 32 and u["value"] == 32      # at old max -> follows
    u = main.on_reserve_change_color_ceiling(True, 32)
    assert u["maximum"] == 28 and u["value"] == 28
    u = main.on_reserve_change_color_ceiling(False, 12)
    assert u["maximum"] == 32 and u["value"] == 12      # user's choice kept
    u = main.on_reserve_change_color_ceiling(True, 30)
    assert u["maximum"] == 28 and u["value"] == 28
