"""Tests for Task 2.1 (main.py mode-first UI wiring).

`main.py` is a Gradio app; we do not drive it through an actual browser/
`gradio_client` session here. Instead -- per the plan's documented fallback --
we verify the module imports cleanly, the Blocks interface builds without
raising, and `process_image` (the function every mode's Convert button calls)
routes correctly and completes without exception when called directly for
each of the four modes: Artistic, GB Studio: Color, GB Studio: Mono, and
GB Studio: Logo (both Color and Mono sub-toggles).
"""

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import main

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GB_PALETTE_PATH = os.path.join(REPO_ROOT, "gb_palette.png")


def _small_test_image(width=80, height=72, seed=1):
    rng = np.random.RandomState(seed)
    gh, gw = height // 8, width // 8
    grid = rng.randint(0, 256, size=(gh, gw, 3)).astype(np.uint8)
    arr = np.repeat(np.repeat(grid, 8, axis=0), 8, axis=1)
    return Image.fromarray(arr, "RGB")


# Default keyword-shared arguments for `process_image` that only matter for
# one branch or the other; every call below supplies all of them since the
# signature takes them positionally regardless of mode.
_ARTISTIC_DEFAULTS = dict(
    color_limit=True,
    num_colors=4,
    # "libimagequant" (the UI default) requires an optional Pillow build-time
    # dependency that may not be present in every environment; "Median cut"
    # exercises the same code path without that environment dependency.
    quant_method="Median cut",
    artistic_dither_method="None",
    use_custom_palette=True,
    grayscale=False,
    black_and_white=False,
    bw_threshold=128,
    enable_gothic_filter=False,
    brightness_threshold=0,
    dot_size=1,
    spacing=1,
    contrast_boost=1.5,
    noise_factor=0.5,
    edge_enhance=False,
    apply_blur=False,
    irregular_shape=False,
    irregular_size=False,
)

_HARDWARE_DEFAULTS = dict(
    reserve_ui_palette=True,
    hw_dither_method="None",
    tile_budget=384,
    logo_subtype="Color",
)


def _call_process_image(mode, image, custom_palette, **overrides):
    kwargs = dict(_ARTISTIC_DEFAULTS)
    kwargs.update(_HARDWARE_DEFAULTS)
    kwargs.update(overrides)
    return main.process_image(
        image, mode, 80, 72, False,
        kwargs["color_limit"], kwargs["num_colors"], kwargs["quant_method"],
        kwargs["artistic_dither_method"], kwargs["use_custom_palette"], custom_palette,
        kwargs["grayscale"], kwargs["black_and_white"], kwargs["bw_threshold"],
        kwargs["enable_gothic_filter"], kwargs["brightness_threshold"], kwargs["dot_size"],
        kwargs["spacing"], kwargs["contrast_boost"], kwargs["noise_factor"],
        kwargs["edge_enhance"], kwargs["apply_blur"], kwargs["irregular_shape"],
        kwargs["irregular_size"],
        kwargs["reserve_ui_palette"], kwargs["hw_dither_method"], kwargs["tile_budget"],
        kwargs["logo_subtype"],
    )


def test_import_main_succeeds():
    # Regression guard: `import main` must not raise (module-level gr.State
    # races removed, no syntax errors introduced by the UI rework).
    import importlib
    importlib.reload(main)


def test_create_gradio_interface_builds():
    demo = main.create_gradio_interface()
    assert demo is not None


@pytest.mark.parametrize("mode", [main.MODE_ARTISTIC, main.MODE_COLOR, main.MODE_MONO])
def test_process_image_each_simple_mode(mode):
    image = _small_test_image()
    palette = Image.open(GB_PALETTE_PATH).convert("RGB")
    out_image, palette_text, reference_image, notice, palette_html = _call_process_image(mode, image, palette)

    assert isinstance(out_image, Image.Image)
    assert out_image.mode == "RGB"
    assert isinstance(reference_image, Image.Image)
    assert isinstance(palette_text, str) and palette_text
    assert isinstance(notice, str) and notice
    assert isinstance(palette_html, str) and palette_html
    assert "background:#" in palette_html.lower()
    assert "palette 1" in palette_html.lower()


@pytest.mark.parametrize("logo_subtype", ["Color", "Mono"])
def test_process_image_logo_mode(logo_subtype):
    image = _small_test_image()
    palette = Image.open(GB_PALETTE_PATH).convert("RGB")
    out_image, palette_text, reference_image, notice, palette_html = _call_process_image(
        main.MODE_LOGO, image, palette, logo_subtype=logo_subtype,
    )

    assert isinstance(out_image, Image.Image)
    assert out_image.size == (160, 144)
    assert "no tile limit" in notice
    assert isinstance(palette_html, str) and palette_html
    assert "background:#" in palette_html.lower()


def test_process_image_hardware_routes_to_convert_for_hardware(monkeypatch):
    calls = []
    real_convert = main.gb_pipeline.convert_for_hardware

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real_convert(*args, **kwargs)

    monkeypatch.setattr(main.gb_pipeline, "convert_for_hardware", spy)
    image = _small_test_image()
    palette = Image.open(GB_PALETTE_PATH).convert("RGB")
    _call_process_image(main.MODE_COLOR, image, palette)

    # Color mode with an active custom palette dual-runs: the restricted left
    # pane plus the natural (custom_palette=None) right pane.
    assert len(calls) == 2
    _, kwargs = calls[0]
    assert kwargs["tile_budget"] == 384
    assert kwargs["reserve_ui_palette"] is True
    assert kwargs["dither"] == "none"
    assert kwargs["custom_palette"] is not None
    _, natural_kwargs = calls[1]
    assert natural_kwargs["custom_palette"] is None
    assert natural_kwargs["tile_budget"] == 384


def test_process_image_artistic_does_not_route_to_hardware(monkeypatch):
    calls = []
    monkeypatch.setattr(
        main.gb_pipeline, "convert_for_hardware",
        lambda *a, **k: calls.append((a, k)),
    )
    image = _small_test_image()
    palette = Image.open(GB_PALETTE_PATH).convert("RGB")
    _call_process_image(main.MODE_ARTISTIC, image, palette)
    assert calls == []


def test_mono_ramp_defaults_to_gb_palette():
    # gb_palette.png IS the DMG reference ramp -- confirms the "reuse the
    # custom-palette image input as the mono ramp" wiring actually threads
    # the image through rather than silently ignoring it.
    palette = Image.open(GB_PALETTE_PATH).convert("RGB")
    ramp = main._mono_ramp_array(palette)
    assert ramp is not None
    assert ramp.shape[1] == 3


def test_on_mode_or_logo_change_visibility():
    artistic, hardware, reserve, tiles, logo, effects = main.on_mode_or_logo_change(
        main.MODE_ARTISTIC, "Color"
    )
    assert artistic["visible"] is True
    assert hardware["visible"] is False
    assert effects["visible"] is True

    artistic, hardware, reserve, tiles, logo, effects = main.on_mode_or_logo_change(
        main.MODE_COLOR, "Color"
    )
    assert artistic["visible"] is False
    assert hardware["visible"] is True
    assert reserve["visible"] is True
    assert tiles["visible"] is True
    assert tiles["value"] == 384

    artistic, hardware, reserve, tiles, logo, effects = main.on_mode_or_logo_change(
        main.MODE_MONO, "Color"
    )
    assert reserve["visible"] is False
    # Mono is always 192 tiles (DMG) -- no budget choice, so the picker hides.
    assert tiles["visible"] is False
    assert tiles["value"] == 192

    artistic, hardware, reserve, tiles, logo, effects = main.on_mode_or_logo_change(
        main.MODE_LOGO, "Mono"
    )
    assert logo["visible"] is True
    assert tiles["visible"] is False
    assert reserve["visible"] is False


def test_on_mode_change_lock_logo_size():
    keep_aspect, width, height = main.on_mode_change_lock_logo_size(main.MODE_LOGO)
    assert (keep_aspect, width, height) == (False, 160, 144)

    result = main.on_mode_change_lock_logo_size(main.MODE_ARTISTIC)
    assert len(result) == 3


def test_adjust_for_aspect_ratio_uses_explicit_state_not_globals():
    # original_width/original_height are no longer module globals.
    assert not hasattr(main, "original_width")
    assert not hasattr(main, "original_height")
    assert not hasattr(main, "quantize_for_GBC")
    assert not hasattr(main, "use_tile_variance")

    width, height = main.adjust_for_aspect_ratio(True, 100, 50, 200, 100)
    assert (width, height) == (100, 50)
    width, height = main.adjust_for_aspect_ratio(False, 100, 50, 200, 100)
    assert (width, height) == (100, 50)


def test_format_palette_html_empty():
    assert main._format_palette_html([]) == "<div>No palette</div>"


def test_format_palette_html_renders_swatches_and_hex():
    html = main._format_palette_html([["#AABBCC", "#001122"], ["#FFFFFF"]])
    assert "Palette 1" in html
    assert "Palette 2" in html
    assert "#AABBCC" in html
    assert "#001122" in html
    assert "#FFFFFF" in html
    assert html.count("background:#") == 3


def test_capture_original_dimensions_handles_none():
    assert main.capture_original_dimensions(None) == (None, 0, 0)
    img = _small_test_image()
    out_img, width, height = main.capture_original_dimensions(img)
    assert out_img is img
    assert (width, height) == img.size


def test_no_committed_secrets_or_shell_deletes():
    src = Path(main.__file__).read_text()
    assert "discord.com/api/webhooks" not in src
    assert "boobiess" not in src
    assert "os.system" not in src


def test_color_mode_custom_palette_natural_pane_differs():
    image = _small_test_image()
    palette = Image.open(GB_PALETTE_PATH)
    out_image, _text, reference_image, notice, _html = _call_process_image(
        main.MODE_COLOR, image, palette, use_custom_palette=True)
    out_colors = np.unique(np.asarray(out_image.convert("RGB")).reshape(-1, 3), axis=0)
    ref_colors = np.unique(np.asarray(reference_image.convert("RGB")).reshape(-1, 3), axis=0)
    assert len(out_colors) <= 4          # left: clamped to the 4-color custom set
    assert len(ref_colors) > 4           # right: the converter's own choices
    assert "restricts output" in notice


def test_color_mode_without_custom_palette_single_run(monkeypatch):
    calls = {"n": 0}
    real = main.gb_pipeline.convert_for_hardware
    def counting(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)
    monkeypatch.setattr(main.gb_pipeline, "convert_for_hardware", counting)
    _call_process_image(main.MODE_COLOR, _small_test_image(),
                        Image.open(GB_PALETTE_PATH), use_custom_palette=False)
    assert calls["n"] == 1


def test_mode_switch_to_color_unticks_custom_palette():
    upd = main.on_mode_change_custom_palette(main.MODE_COLOR)
    assert upd["value"] is False
    for mode in (main.MODE_ARTISTIC, main.MODE_MONO, main.MODE_LOGO):
        upd = main.on_mode_change_custom_palette(mode)
        assert "value" not in upd


def test_resize_filter_resolution():
    assert main._resolve_resample("Auto", main.MODE_COLOR) == Image.LANCZOS
    assert main._resolve_resample("Auto", main.MODE_ARTISTIC) == Image.NEAREST
    assert main._resolve_resample("Nearest (pixel art)", main.MODE_COLOR) == Image.NEAREST
    assert main._resolve_resample("Lanczos (smooth)", main.MODE_ARTISTIC) == Image.LANCZOS


def test_hardware_notice_shows_gbstudio_import_counts():
    """Color notices surface GB Studio's real import figures (tiles/palettes)
    and flag tiles it would recolor, so the count shown is the honest one."""
    import types
    import main
    result = types.SimpleNamespace(
        stats={
            "tiles_used": 360, "tile_budget": 384, "palettes_used": 7,
            "n_merges": 0, "gbstudio_tiles": 247,
            "gbstudio_palettes_extracted": 10, "gbstudio_corrupted_tiles": 39,
        },
        warnings=[],
    )
    notice = main._format_hardware_notice(result)
    assert "GB Studio" in notice
    assert "247" in notice and "10" in notice
    assert "39" in notice  # recolored tiles surfaced


def test_hardware_notice_shows_adapted_for_import_when_recolored():
    """When we've conformed the output ourselves (gbstudio_recolored_tiles > 0),
    the notice says tiles were adapted for import instead of warning about
    GB Studio recoloring -- corruption is 0 by construction in this case."""
    import types
    import main
    result = types.SimpleNamespace(
        stats={
            "tiles_used": 360, "tile_budget": 384, "palettes_used": 7,
            "n_merges": 0, "gbstudio_tiles": 360,
            "gbstudio_palettes_extracted": 8, "gbstudio_corrupted_tiles": 0,
            "gbstudio_recolored_tiles": 5,
        },
        warnings=[],
    )
    notice = main._format_hardware_notice(result)
    assert "5 tiles adapted for import" in notice
    assert "⚠️" not in notice


def test_hardware_notice_unchanged_when_no_tiles_recolored():
    """gbstudio_recolored_tiles == 0 (or absent) leaves the GB Studio line
    exactly as it was before this stat existed."""
    import types
    import main
    result = types.SimpleNamespace(
        stats={
            "tiles_used": 360, "tile_budget": 384, "palettes_used": 7,
            "n_merges": 0, "gbstudio_tiles": 360,
            "gbstudio_palettes_extracted": 8, "gbstudio_corrupted_tiles": 0,
            "gbstudio_recolored_tiles": 0,
        },
        warnings=[],
    )
    notice = main._format_hardware_notice(result)
    assert "adapted for import" not in notice
    assert "⚠️" not in notice


def test_hardware_notice_safety_net_still_warns_on_corruption():
    """If the fallback ladder is somehow exhausted and corrupted tiles remain,
    the old honest ⚠️ warning still fires instead of crashing or hiding it."""
    import types
    import main
    result = types.SimpleNamespace(
        stats={
            "tiles_used": 360, "tile_budget": 384, "palettes_used": 7,
            "n_merges": 0, "gbstudio_tiles": 360,
            "gbstudio_palettes_extracted": 10, "gbstudio_corrupted_tiles": 12,
            "gbstudio_recolored_tiles": 0,
        },
        warnings=[],
    )
    notice = main._format_hardware_notice(result)
    assert "⚠️ 12 tiles recolored" in notice
    assert "adapted for import" not in notice
