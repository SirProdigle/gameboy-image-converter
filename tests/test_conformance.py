"""Integration tests for the GB Studio conformance loop (``_conform_to_gbstudio``).

Design: docs/superpowers/specs/2026-07-08-gbstudio-conformance-loop-design.md.

``convert_for_hardware`` now conforms its rendered output to GB Studio v4.3.2's
*actual* importer (``gb_studio_import``) so the shipped PNG is pixel-exact what
GB reconstructs on import: the tile count lands within budget, GB extracts <=8
palettes, and there is zero silent recoloring left for GB to do. Any recoloring
needed to fit the 8-palette import is applied by *us* (best-fit) and reported as
``gbstudio_recolored_tiles``; ``gbstudio_corrupted_tiles`` is 0 by construction.
"""

import os

import numpy as np
import pytest
from PIL import Image, ImageFilter

import gb_pipeline as gp
import gb_studio_import as gsi
from tests.conftest import make_photo_like_image

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _coarse_grid(seed, w, h):
    """Coarse per-tile color grid + gradient + noise (mirrors the e2e image).

    Small 2-3 color per-tile subsets drawn from different masters make GB
    Studio's greedy ``compressPalettes`` spawn >8 hybrid palettes on our
    otherwise GB-legal render -- exactly defect #1 the conform loop repairs.
    (``overflow_palettes.png`` cannot exercise this: the packer collapses its 12
    disjoint palettes into <=7 masters, so its render imports cleanly -- see
    ``test_overflow_fixture_imports_gbstudio_clean``.)
    """
    rng = np.random.RandomState(seed)
    gh, gw = h // 8, w // 8
    grid = rng.randint(0, 256, size=(gh, gw, 3)).astype(np.uint8)
    arr = np.repeat(np.repeat(grid, 8, axis=0), 8, axis=1)
    yy, xx = np.mgrid[0:h, 0:w]
    shade = ((xx + yy) / (w + h) * 60).astype(np.int16)
    noise = rng.normal(0, 8, size=arr.shape)
    arr = np.clip(arr.astype(np.int16) + shade[..., None] + noise, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), "RGB")


# -- (a) palette overflow: conform recolors to a GB-clean fixed point ----------


def test_color_overflow_recolors_to_gbstudio_clean():
    """A render GB Studio would re-extract as >8 palettes is conformed: we
    recolor the overflow tiles ourselves, so the shipped PNG imports with <=8
    palettes and zero corruption -- and re-importing it is a fixed point."""
    res = gp.convert_for_hardware(_coarse_grid(1, 160, 144), "color_only")

    assert res.stats["gbstudio_palettes_extracted"] <= 8
    assert res.stats["gbstudio_corrupted_tiles"] == 0
    # recolored > 0 proves the overflow defect was present and we repaired it.
    assert res.stats["gbstudio_recolored_tiles"] > 0

    # Fixed point: a direct re-run of GB Studio's importer on the shipped PNG
    # confirms corruption 0 / <=8 palettes and agrees with the reported stats.
    direct = gsi.gbstudio_color_stats(np.asarray(res.image.convert("RGB"), np.uint8))
    assert direct.corrupted_tiles == 0
    assert direct.palettes_extracted <= 8
    assert direct.palettes_extracted == res.stats["gbstudio_palettes_extracted"]
    assert direct.tiles_autoflip == res.stats["gbstudio_tiles"]
    assert res.stats["tiles_used"] == direct.tiles_autoflip


def test_overflow_fixture_imports_gbstudio_clean():
    """The overflow_palettes.png fixture through the full pipeline imports
    GB-clean (its 12 disjoint palettes collapse to <=7 masters, so no recolor is
    even needed) -- the conformance guarantee holds regardless of fixture."""
    arr = np.asarray(
        Image.open(os.path.join(FIXTURES, "overflow_palettes.png")).convert("RGB"),
        np.uint8,
    )
    res = gp.convert_for_hardware(Image.fromarray(arr, "RGB"), "color_only")

    assert res.stats["gbstudio_corrupted_tiles"] == 0
    assert res.stats["gbstudio_palettes_extracted"] <= 8
    direct = gsi.gbstudio_color_stats(np.asarray(res.image.convert("RGB"), np.uint8))
    assert direct.corrupted_tiles == 0
    assert direct.palettes_extracted <= 8


# -- (b) tile-count divergence: the tile loop tightens until GB fits -----------


def test_tile_loop_conforms_320x288_to_budget():
    """A high-detail 320x288 image whose un-conformed render GB Studio counts at
    >384 tiles is tightened by the tile loop until GB's own autoflip count lands
    within the 384 budget; the reported ``tiles_used`` is GB's number."""
    img = Image.fromarray(
        np.random.RandomState(11).randint(0, 256, (288, 320, 3)).astype(np.uint8),
        "RGB",
    ).filter(ImageFilter.GaussianBlur(2.0))

    res = gp.convert_for_hardware(img, "color_only")

    direct = gsi.gbstudio_color_stats(np.asarray(res.image.convert("RGB"), np.uint8))
    assert direct.tiles_autoflip <= 384
    assert res.stats["tiles_used"] <= 384
    assert res.stats["tiles_used"] == direct.tiles_autoflip
    assert res.stats["gbstudio_corrupted_tiles"] == 0


# -- (c) mono custom-ramp green-bucket collision warning -----------------------


def test_mono_colliding_custom_ramp_warns():
    """A custom mono ramp whose two mid shades share a GB Studio green bucket
    warns that those shades will merge on import."""
    ramp = np.array(
        [[255, 255, 255], [0, 150, 0], [0, 190, 0], [0, 0, 0]], dtype=np.uint8
    )
    res = gp.convert_for_hardware(_coarse_grid(2, 160, 144), "mono", mono_ramp=ramp)
    assert any(
        "merge" in w.lower() and "gb studio import" in w.lower()
        for w in res.warnings
    )


def test_mono_distinct_ramp_does_not_warn():
    """The default DMG ramp lands in 4 distinct buckets, so no merge warning."""
    res = gp.convert_for_hardware(_coarse_grid(2, 160, 144), "mono")
    assert not any("shades will merge" in w.lower() for w in res.warnings)


# -- (d) clean image: conform is a no-op that costs nothing extra --------------


def test_clean_image_no_recolor_identical(photo_like_image):
    """A GB-legal image (<=8 palettes, within budget, no merges) recolors 0
    tiles and its shipped PNG is byte-identical to the pre-conform render."""
    res = gp.convert_for_hardware(photo_like_image, "color_only")
    assert res.stats["gbstudio_recolored_tiles"] == 0
    assert res.stats["gbstudio_corrupted_tiles"] == 0
    assert np.array_equal(
        np.asarray(res.image.convert("RGB")),
        np.asarray(res.reference.convert("RGB")),
    )


# -- (e) sweep: final oracle verdict holds across size x dither x preset ------

SWEEP_SIZES = [(160, 144), (320, 288)]
SWEEP_DITHERS = ["none", "bayer"]
SWEEP_PRESETS = ["color_only", "mono"]


@pytest.mark.slow
@pytest.mark.parametrize("preset", SWEEP_PRESETS)
@pytest.mark.parametrize("dither", SWEEP_DITHERS)
@pytest.mark.parametrize("size", SWEEP_SIZES, ids=["160x144", "320x288"])
def test_sweep_final_oracle_verdict(size, dither, preset):
    """Exhaustive combinatorial sweep of the conform loop's guarantee: for
    every (size, dither, preset) combo, the SHIPPED render (``res.image``) is
    a GB Studio-clean fixed point -- tiles within budget, and for color, <=8
    extracted palettes with zero corruption. Each combo gets its own
    deterministic seed (photo-like generator) so a failure reproduces."""
    width, height = size
    seed = hash((width, height, dither, preset)) & 0xFFFF
    img = make_photo_like_image(width=width, height=height, seed=seed)

    res = gp.convert_for_hardware(img, preset, dither=dither)
    budget = gp.PRESETS[preset].tile_budget
    out_arr = np.asarray(res.image.convert("RGB"), np.uint8)

    if preset == "mono":
        mono_stats = gsi.gbstudio_mono_stats(out_arr)
        assert mono_stats.tiles <= budget
    else:
        color_stats = gsi.gbstudio_color_stats(out_arr)
        assert color_stats.tiles_autoflip <= budget
        assert color_stats.palettes_extracted <= 8
        assert color_stats.corrupted_tiles == 0
