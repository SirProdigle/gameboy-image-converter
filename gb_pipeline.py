"""GB Studio hardware-constrained conversion pipeline.

Pure functions over numpy arrays / PIL images — no Gradio imports here.
See docs/superpowers/specs/2026-07-06-gb-pipeline-redesign-design.md for the
authoritative behavior spec and
docs/superpowers/plans/2026-07-06-gb-pipeline-redesign-plan.md for the task
breakdown these functions implement.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class GBImage:
    """Mirrors GB/GBC VRAM layout for a converted image."""

    patterns: np.ndarray       # (n, 8, 8) uint8, values 0-3 -- the tileset
    tilemap: np.ndarray        # (th, tw) int32 -> pattern index per cell
    attrs_palette: np.ndarray  # (th, tw) uint8 -- palette id per cell
    attrs_hflip: np.ndarray    # (th, tw) bool
    attrs_vflip: np.ndarray    # (th, tw) bool
    palettes: np.ndarray       # (p, 4, 3) uint8, RGB555-snapped, luminance-sorted


@dataclass
class Preset:
    name: str
    tile_budget: int | None
    n_palettes: int
    allow_flips: bool
    mono: bool
    fixed_size: tuple | None


PRESETS = {
    "color_only": Preset("color_only", 384, 8, True, False, None),
    "mono": Preset("mono", 192, 1, False, True, None),
    "logo_color": Preset("logo_color", None, 8, False, False, (160, 144)),
    "logo_mono": Preset("logo_mono", None, 1, False, True, (160, 144)),
}

# DMG reference ramp, lightest first (from gb_palette.png).
DMG_RAMP = np.array(
    [[224, 248, 208], [136, 192, 112], [52, 104, 86], [8, 24, 32]],
    dtype=np.uint8,
)


def snap_rgb555(colors: np.ndarray) -> np.ndarray:
    """Snap 8-bit-per-channel colors to their RGB555 quantization.

    c5 = floor(c8 * 32 / 256), clamped to [0, 31]; expanded back to 8-bit via
    round(c5 * 255 / 31). Operates on any shape ending in (..., 3) (or any
    shape at all -- it is a plain per-element transform), uint8 in and out.
    """
    colors = np.asarray(colors, dtype=np.float64)
    c5 = np.floor(colors * 32.0 / 256.0)
    c5 = np.clip(c5, 0, 31)
    c8 = np.round(c5 * 255.0 / 31.0)
    return c8.astype(np.uint8)


def luminance_sort(palette: np.ndarray) -> np.ndarray:
    """Sort a (4, 3) RGB palette lightest-first by 2126*R + 7152*G + 722*B."""
    palette = np.asarray(palette)
    luminance = (
        2126 * palette[:, 0].astype(np.int64)
        + 7152 * palette[:, 1].astype(np.int64)
        + 722 * palette[:, 2].astype(np.int64)
    )
    order = np.argsort(-luminance, kind="stable")
    return palette[order]


def pattern_to_2bpp(pattern: np.ndarray) -> bytes:
    """Encode an (8, 8) index pattern (values 0-3) as 16 bytes of GB 2bpp data.

    Per row: byte1 = low bits of each pixel's index, byte2 = high bits;
    within a byte, bit 7 is the leftmost pixel.
    """
    pattern = np.asarray(pattern)
    assert pattern.shape == (8, 8)
    out = bytearray(16)
    for row in range(8):
        low = 0
        high = 0
        for col in range(8):
            value = int(pattern[row, col]) & 0b11
            bit_pos = 7 - col
            low |= (value & 1) << bit_pos
            high |= ((value >> 1) & 1) << bit_pos
        out[row * 2] = low
        out[row * 2 + 1] = high
    return bytes(out)


def pattern_variants(pattern: np.ndarray) -> dict:
    """Return the four flip variants of a pattern: {"", "h", "v", "hv"}."""
    pattern = np.asarray(pattern)
    return {
        "": pattern.copy(),
        "h": np.fliplr(pattern).copy(),
        "v": np.flipud(pattern).copy(),
        "hv": np.flipud(np.fliplr(pattern)).copy(),
    }
