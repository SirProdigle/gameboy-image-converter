"""Tests for gb_pipeline.py Task 1.6 entry point / verify / render (Stages 7-8).

Exercises ``convert_for_hardware`` end to end across the color, mono, and logo
presets: property-style checks on seeded random images (every result passes its
own independent ``verify_roundtrip`` and stays within hardware limits), the
logo preset's fixed 160x144 no-merge behavior, the mono preset's ramp-only
output, and golden tile/palette counts pinned for the deterministic
``photo_like_image`` fixture. Also covers ``render`` round-tripping a GBImage
(including flip bits) and ``verify_roundtrip`` catching an over-limit tile.
"""

import numpy as np
import pytest
from PIL import Image

import gb_pipeline
from gb_pipeline import (
    DMG_RAMP,
    GBImage,
    PRESETS,
    ConversionResult,
    convert_for_hardware,
    render,
    snap_rgb555,
    verify_roundtrip,
    pattern_to_2bpp,
)


def _seeded_image(seed, width=80, height=72):
    """A deterministic blocky RGB image with a handful of flat regions.

    Built from a coarse grid of random colors upsampled to full resolution so
    tiles repeat (giving dedup / merge something to chew on) while regions stay
    photo-like rather than pure noise.
    """
    rng = np.random.RandomState(seed)
    gh, gw = height // 8, width // 8
    grid = rng.randint(0, 256, size=(gh, gw, 3)).astype(np.uint8)
    arr = np.repeat(np.repeat(grid, 8, axis=0), 8, axis=1)
    # Add a mild gradient + noise so tiles are not perfectly flat.
    yy, xx = np.mgrid[0:height, 0:width]
    shade = ((xx + yy) / (width + height) * 60).astype(np.int16)
    noise = rng.normal(0, 8, size=arr.shape)
    arr = np.clip(arr.astype(np.int16)[..., :] + shade[..., None] + noise, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), "RGB")


# ---------------------------------------------------------------------------
# render


def test_render_roundtrips_indices_and_flips():
    # Two patterns; cell (0,1) references pattern 0 with an hflip.
    p0 = np.zeros((8, 8), dtype=np.uint8)
    p0[:, 0] = 3  # a left stripe -> asymmetric so the flip is observable
    p1 = np.full((8, 8), 1, dtype=np.uint8)
    palettes = snap_rgb555(
        np.array([[240, 240, 240], [160, 160, 160], [80, 80, 80], [0, 0, 0]],
                 dtype=np.uint8)
    )[None, :, :]
    gb = GBImage(
        patterns=np.stack([p0, p1]),
        tilemap=np.array([[0, 0]], dtype=np.int32),
        attrs_palette=np.zeros((1, 2), dtype=np.uint8),
        attrs_hflip=np.array([[False, True]], dtype=bool),
        attrs_vflip=np.zeros((1, 2), dtype=bool),
        palettes=palettes,
    )
    img = render(gb)
    arr = np.asarray(img)
    assert arr.shape == (8, 16, 3)
    # Cell 0: left column is palette[3]; cell 1 (hflipped): right column is [3].
    assert np.array_equal(arr[:, 0], np.tile(palettes[0, 3], (8, 1)))
    assert np.array_equal(arr[:, 15], np.tile(palettes[0, 3], (8, 1)))
    # And cell 1's left column is now the background (palette[0]).
    assert np.array_equal(arr[:, 8], np.tile(palettes[0, 0], (8, 1)))


# ---------------------------------------------------------------------------
# verify_roundtrip


def test_verify_roundtrip_passes_inside_convert():
    # convert_for_hardware calls verify_roundtrip on its final render; if that
    # assertion tripped, convert would raise. A clean return proves it passed.
    result = convert_for_hardware(_seeded_image(1), "color_only")
    assert isinstance(result, ConversionResult)


def test_verify_roundtrip_detects_over_budget():
    # Build a GBImage whose rendered PNG has more unique tiles than the budget.
    rng = np.random.RandomState(11)
    n = 12
    patterns = []
    seen = set()
    while len(patterns) < n:
        p = rng.choice(np.array([0, 3], dtype=np.uint8), size=(8, 8))
        k = pattern_to_2bpp(p)
        if k in seen:
            continue
        seen.add(k)
        patterns.append(p)
    patterns = np.stack(patterns)
    palettes = snap_rgb555(
        np.array([[255, 255, 255], [170, 170, 170], [85, 85, 85], [0, 0, 0]],
                 dtype=np.uint8)
    )[None, :, :]
    gb = GBImage(
        patterns=patterns,
        tilemap=np.arange(n, dtype=np.int32).reshape(1, n),
        attrs_palette=np.zeros((1, n), dtype=np.uint8),
        attrs_hflip=np.zeros((1, n), dtype=bool),
        attrs_vflip=np.zeros((1, n), dtype=bool),
        palettes=palettes,
    )
    png = render(gb)
    tight = gb_pipeline.Preset("t", tile_budget=4, n_palettes=1,
                               allow_flips=False, mono=False, fixed_size=None)
    with pytest.raises(AssertionError):
        verify_roundtrip(png, tight, gb)


# ---------------------------------------------------------------------------
# property-style: seeded images x presets


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("preset", ["color_only", "mono", "logo_color"])
def test_convert_property_invariants(seed, preset):
    image = _seeded_image(seed)
    result = convert_for_hardware(image, preset)

    assert isinstance(result, ConversionResult)
    ps = PRESETS[preset]

    # Stats reflect the hardware limits.
    if ps.tile_budget is not None:
        assert result.stats["tiles_used"] <= ps.tile_budget
    limit = ps.n_palettes - 1 if (not ps.mono and ps.n_palettes >= 8) else ps.n_palettes
    assert result.stats["palettes_used"] <= limit

    # verify_roundtrip already ran inside convert; the rendered image must have
    # <= 4 colors per tile and only RGB555-stable colors.
    arr = np.asarray(result.image)
    assert snap_rgb555(arr).tolist() == arr.tolist()

    # palette_hex shape matches palettes.
    assert len(result.palette_hex) == result.stats["palettes_used"]
    for pal in result.palette_hex:
        assert len(pal) == 4
        for h in pal:
            assert h.startswith("#") and len(h) == 7


def test_convert_is_deterministic():
    a = convert_for_hardware(_seeded_image(3), "color_only")
    b = convert_for_hardware(_seeded_image(3), "color_only")
    assert a.stats == b.stats
    assert np.array_equal(np.asarray(a.image), np.asarray(b.image))


# ---------------------------------------------------------------------------
# logo preset


def test_logo_output_is_fixed_size_no_merge():
    # Non-160x144 input is resized; output is exactly 160x144 with no merges.
    image = _seeded_image(7, width=80, height=72)
    result = convert_for_hardware(image, "logo_color")
    assert result.image.size == (160, 144)
    assert result.reference.size == (160, 144)
    assert result.stats["n_merges"] == 0
    # 160x144 -> 20x18 = 360 tile cells, stored sequentially.
    assert result.stats["tiles_used"] <= 360
    assert result.stats["tile_budget"] is None
    assert any("resized" in w for w in result.warnings)


def test_logo_already_correct_size_not_resized():
    image = _seeded_image(8, width=160, height=144)
    result = convert_for_hardware(image, "logo_color")
    assert result.image.size == (160, 144)
    assert result.stats["tiles_used"] == 360  # no dedup for logo
    assert not any("resized" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# mono preset


def test_mono_uses_only_ramp_colors():
    result = convert_for_hardware(_seeded_image(4), "mono")
    ramp_colors = {tuple(c) for c in gb_pipeline.luminance_sort(snap_rgb555(DMG_RAMP)).tolist()}
    out = np.asarray(result.image).reshape(-1, 3)
    used = {tuple(c) for c in np.unique(out, axis=0).tolist()}
    assert used <= ramp_colors
    assert result.stats["palettes_used"] == 1


def test_mono_custom_ramp():
    ramp = np.array([[255, 255, 255], [180, 180, 180], [90, 90, 90], [0, 0, 0]],
                    dtype=np.uint8)
    result = convert_for_hardware(_seeded_image(4), "mono", mono_ramp=ramp)
    ramp_colors = {tuple(c) for c in gb_pipeline.luminance_sort(snap_rgb555(ramp)).tolist()}
    out = np.asarray(result.image).reshape(-1, 3)
    used = {tuple(c) for c in np.unique(out, axis=0).tolist()}
    assert used <= ramp_colors


# ---------------------------------------------------------------------------
# golden counts (pinned on first run for the deterministic fixture)


def test_golden_counts_photo_like(photo_like_image):
    color = convert_for_hardware(photo_like_image, "color_only")
    mono = convert_for_hardware(photo_like_image, "mono")
    logo = convert_for_hardware(photo_like_image, "logo_color")

    # Pinned observed counts (regenerate deliberately if the pipeline changes).
    assert color.stats["tiles_used"] == GOLDEN["color_tiles"]
    assert color.stats["palettes_used"] == GOLDEN["color_palettes"]
    assert mono.stats["tiles_used"] == GOLDEN["mono_tiles"]
    assert mono.stats["palettes_used"] == 1
    assert logo.stats["tiles_used"] == 360


# Golden values observed on the first green run; see test above. Regenerate
# deliberately (and review the diff) only if the pipeline math changes.
GOLDEN = {
    "color_tiles": 357,     # 360 cells (20x18) deduped, under the 384 budget
    "color_palettes": 7,    # reserve_ui_palette caps color at 7
    "mono_tiles": 143,
}
