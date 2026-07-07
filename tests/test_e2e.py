"""Task 3.1: end-to-end + perf tests.

Exercises `convert_for_hardware` across every preset (color_only, mono,
logo_color, logo_mono) at two realistic screen sizes -- 160x144 (a single GB
Studio scene) and 320x288 (2x2 scenes, exercising the pipeline at a size
above a single screen) -- with deterministic seeded images. Asserts the same
hardware invariants the property tests in test_convert.py check (tile budget,
palette count, RGB555-only colors, verified roundtrip already having run
inside convert_for_hardware) plus a generous, CI-safe wall-clock bound per
the plan ("complete within seconds").
"""

import time

import numpy as np
import pytest
from PIL import Image

import gb_pipeline
from gb_pipeline import PRESETS, ConversionResult, convert_for_hardware, snap_rgb555

# Generous but real perf bound: plan calls for "within seconds" on a CI box;
# 10s per conversion leaves ample headroom without masking a real regression
# (a correct run on a dev machine completes in well under a second).
PERF_BOUND_SECONDS = 10.0


def _seeded_image(seed: int, width: int, height: int) -> Image.Image:
    """Deterministic blocky RGB image sized to (width, height).

    Mirrors the generator in test_convert.py: a coarse per-tile color grid
    (so tiles repeat, giving dedup/merge real work to do) plus a mild
    gradient and noise so tiles are not perfectly flat.
    """
    rng = np.random.RandomState(seed)
    gh, gw = height // 8, width // 8
    grid = rng.randint(0, 256, size=(gh, gw, 3)).astype(np.uint8)
    arr = np.repeat(np.repeat(grid, 8, axis=0), 8, axis=1)
    yy, xx = np.mgrid[0:height, 0:width]
    shade = ((xx + yy) / (width + height) * 60).astype(np.int16)
    noise = rng.normal(0, 8, size=arr.shape)
    arr = np.clip(arr.astype(np.int16) + shade[..., None] + noise, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), "RGB")


SIZES = [(160, 144), (320, 288)]
PRESET_NAMES = ["color_only", "mono", "logo_color", "logo_mono"]

# Collected across the parametrized run for the human-readable summary
# printed by test_print_summary (module-level so it survives across tests
# within one pytest process; order of collection follows collection order).
_STATS_LOG = []


@pytest.mark.parametrize("preset", PRESET_NAMES)
@pytest.mark.parametrize("size", SIZES, ids=["160x144", "320x288"])
def test_e2e_preset_at_size_within_perf_bound(size, preset):
    width, height = size
    image = _seeded_image(seed=hash((width, height, preset)) & 0xFFFF, width=width, height=height)

    start = time.monotonic()
    result = convert_for_hardware(image, preset)
    elapsed = time.monotonic() - start

    assert isinstance(result, ConversionResult)
    assert elapsed < PERF_BOUND_SECONDS, (
        f"{preset} at {width}x{height} took {elapsed:.2f}s, exceeding the "
        f"{PERF_BOUND_SECONDS}s perf smoke bound"
    )

    ps = PRESETS[preset]

    # Tile budget honored (logo presets have none).
    if ps.tile_budget is not None:
        assert result.stats["tiles_used"] <= ps.tile_budget
    else:
        assert result.stats["tile_budget"] is None

    # Palette count honored (reserve_ui_palette default caps 8-palette color
    # presets to 7; mono/logo_mono always have exactly 1).
    limit = ps.n_palettes - 1 if (not ps.mono and ps.n_palettes >= 8) else ps.n_palettes
    assert result.stats["palettes_used"] <= limit

    # Logo presets are pinned to the fixed GB screen size regardless of input.
    if ps.fixed_size is not None:
        assert result.image.size == ps.fixed_size
        assert result.reference.size == ps.fixed_size
        assert result.stats["n_merges"] == 0

    # verify_roundtrip already ran inside convert_for_hardware; independently
    # re-check the output only contains RGB555-stable colors here too.
    arr = np.asarray(result.image)
    assert snap_rgb555(arr).tolist() == arr.tolist()

    # palette_hex is consistent with the verified palette count.
    assert len(result.palette_hex) == result.stats["palettes_used"]

    _STATS_LOG.append(
        {
            "size": f"{width}x{height}",
            "preset": preset,
            "elapsed_s": round(elapsed, 3),
            **result.stats,
        }
    )


def test_print_summary():
    # Not a real assertion -- a human-readable recap of every case run above,
    # printed for the final task report (visible with `pytest -s`). Guarded
    # so it still passes (skips cleanly) if test selection excluded the
    # parametrized cases above.
    if not _STATS_LOG:
        pytest.skip("no e2e cases collected (ran with a narrower -k selection)")
    print("\n--- Task 3.1 e2e/perf summary ---")
    for row in _STATS_LOG:
        print(
            f"{row['size']:>9} {row['preset']:<11} "
            f"{row['elapsed_s']:>6.3f}s  "
            f"tiles={row['tiles_used']}/{row['tile_budget']}  "
            f"palettes={row['palettes_used']}  merges={row['n_merges']}"
        )


def _mean_lab_error(a_img, b_img):
    a = gb_pipeline._rgb_to_lab(np.asarray(a_img, dtype=np.uint8))
    b = gb_pipeline._rgb_to_lab(np.asarray(b_img, dtype=np.uint8))
    return float(np.sqrt(((a - b) ** 2).sum(axis=1)).mean())


def test_indexing_from_original_beats_quantized_source():
    rng = np.random.RandomState(0)
    yy, xx = np.mgrid[0:64, 0:64]
    arr = np.stack([xx * 4, yy * 4, ((xx + yy) * 2)], axis=2)
    arr = np.clip(arr + rng.normal(0, 6, arr.shape), 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, "RGB")

    quant = gb_pipeline.quantize_working_set(img, 28)
    qarr = np.asarray(quant, dtype=np.uint8)
    palettes, assignment = gb_pipeline.pack_palettes(qarr, 7)

    # dither="none" isolates the indexing-source fidelity property: mapping the
    # original pixels to the palette loses less than mapping the already-
    # quantized pixels. (Bayer dithering's off-axis speckle -- fixed in Task 5's
    # projection dither -- would otherwise dominate this per-pixel Lab metric.)
    gb_orig = gb_pipeline.index_tiles(arr, palettes, assignment, dither="none")
    gb_quant = gb_pipeline.index_tiles(qarr, palettes, assignment, dither="none")
    err_orig = _mean_lab_error(np.asarray(gb_pipeline.render(gb_orig)), arr)
    err_quant = _mean_lab_error(np.asarray(gb_pipeline.render(gb_quant)), arr)
    assert err_orig < err_quant


def test_convert_for_hardware_color_indexes_from_original():
    # The public entry point must produce the err_orig result, not err_quant.
    rng = np.random.RandomState(1)
    yy, xx = np.mgrid[0:64, 0:64]
    arr = np.clip(np.stack([xx * 4, yy * 4, (xx + yy) * 2], axis=2)
                  + rng.normal(0, 6, (64, 64, 3)), 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, "RGB")
    res = gb_pipeline.convert_for_hardware(img, "color_only", dither="none")
    quant = gb_pipeline.quantize_working_set(img, 28)
    qarr = np.asarray(quant, dtype=np.uint8)
    palettes, assignment = gb_pipeline.pack_palettes(qarr, 7)
    gb_quant = gb_pipeline.index_tiles(qarr, palettes, assignment, dither="none")
    err_pipeline = _mean_lab_error(np.asarray(res.image), arr)
    err_quant = _mean_lab_error(np.asarray(gb_pipeline.render(gb_quant)), arr)
    assert err_pipeline < err_quant
