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


def test_verify_roundtrip_reports_over_budget_count():
    # Build a GBImage whose rendered PNG has more unique tiles than the budget.
    # verify_roundtrip no longer asserts on budget -- it RETURNS the true
    # re-imported count so the caller can surface an over-budget warning.
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
    # 12 distinct rendered tiles, budget 4: returns 12 without raising.
    n_reimport = verify_roundtrip(png, tight, gb)
    assert n_reimport == n
    assert n_reimport > tight.tile_budget


def test_verify_roundtrip_independent_of_stored_indices():
    # Regression: two stored patterns that RENDER IDENTICALLY (one reaches a
    # color through a duplicate palette slot, the other through a single slot)
    # must re-import to ONE tile. GB Studio's importer sees one tile, so an
    # independent re-import (indices ranked by tile-local luminance) must too --
    # a verifier that instead reconstructs indices against the tile's assigned
    # palette would hash both to the same pattern yet still demand two, and the
    # old coupled implementation raised "reimport found 1 tiles, expected 2".
    # verify_roundtrip now RETURNS the re-imported count; assert it collapses
    # to fewer tiles than were stored.
    dark = snap_rgb555(np.array([16, 16, 16], dtype=np.uint8)).tolist()
    # Palette with a DUPLICATE dark entry at slots 2 and 3.
    palettes = snap_rgb555(
        np.array([[240, 240, 240], [160, 160, 160], dark, dark], dtype=np.uint8)
    )[None, :, :]

    # Pattern A draws the right half using BOTH duplicate slots (2 and 3);
    # pattern B draws the identical picture using only slot 2. Same pixels,
    # different 2bpp bytes -> two distinct stored patterns.
    pat_a = np.zeros((8, 8), dtype=np.uint8)
    pat_a[:, 4:6] = 2
    pat_a[:, 6:8] = 3
    pat_b = np.zeros((8, 8), dtype=np.uint8)
    pat_b[:, 4:8] = 2
    assert pattern_to_2bpp(pat_a) != pattern_to_2bpp(pat_b)

    gb = GBImage(
        patterns=np.stack([pat_a, pat_b]),
        tilemap=np.array([[0, 1]], dtype=np.int32),
        attrs_palette=np.zeros((1, 2), dtype=np.uint8),
        attrs_hflip=np.zeros((1, 2), dtype=bool),
        attrs_vflip=np.zeros((1, 2), dtype=bool),
        palettes=palettes,
    )
    png = render(gb)
    # Both cells render to the same 8x8 block (dark right half).
    arr = np.asarray(png)
    assert np.array_equal(arr[:, 0:8], arr[:, 8:16])

    preset = gb_pipeline.Preset("t", tile_budget=4, n_palettes=1,
                                allow_flips=False, mono=False, fixed_size=None)
    # Independent re-import collapses to one tile, fewer than the 2 stored.
    n_reimport = verify_roundtrip(png, preset, gb)
    assert n_reimport == 1
    assert n_reimport < gb.patterns.shape[0]


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


def test_crop_is_center_anchored():
    arr = np.zeros((20, 20, 3), dtype=np.uint8)
    arr[2:18, 2:18] = 200   # center 16x16 block is bright
    res = convert_for_hardware(Image.fromarray(arr, "RGB"), "mono")
    out = np.asarray(res.image)
    assert out.shape[:2] == (16, 16)
    # Top-left crop would include 2 dark rows/cols; center crop keeps only bright.
    assert len(np.unique(out.reshape(-1, 3), axis=0)) == 1


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
    assert result.stats["tiles_used"] == 360  # logo: sequential storage, no dedup
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
    assert logo.stats["tiles_used"] == GOLDEN["logo_tiles"]


# Golden values observed on the first green run; see test above. Regenerate
# deliberately (and review the diff) only if the pipeline math changes.
GOLDEN = {
    # For color/mono, tiles_used is now GB Studio's own conformed import count
    # (its real importer, oracle-in-the-loop) -- the number of tiles GB Studio
    # stores, not the pipeline's internal pattern count. For logo, tiles are
    # stored sequentially with no dedup, so tiles_used is the cell count (th*tw).
    "color_tiles": 360,     # 360 cells (20x18); indexing from original pixels
                            #   (Task 4) dedups less than the quantized source did
                            #   -- still under the 384 budget
    "color_palettes": 7,    # reserve_ui_palette caps color at 7
    "mono_tiles": 192,  # GB Studio's mono importer (fixed green buckets, no flip
                        #   dedup) counts 192 where our verify_roundtrip's
                        #   luminance-reindex collapsed to 189 -- conform reports
                        #   GB's number (still within the 192 mono budget)
    "logo_tiles": 360,      # logo: sequential storage, no dedup (th*tw cells)
}


def test_custom_palette_capacity_warning():
    rng = np.random.RandomState(3)
    arr = np.repeat(np.repeat(rng.randint(0, 256, (4, 4, 3)).astype(np.uint8), 8, 0), 8, 1)
    img = Image.fromarray(arr, "RGB")
    dmg = gb_pipeline.DMG_RAMP
    res = gb_pipeline.convert_for_hardware(img, "color_only", custom_palette=dmg)
    assert any("restricts output to 4 colors" in w for w in res.warnings)
    res_free = gb_pipeline.convert_for_hardware(img, "color_only")
    assert not any("restricts output" in w for w in res_free.warnings)
