"""Shared pytest fixtures for the gb_pipeline test suite.

Fixtures here are intentionally plain data (numpy arrays / PIL Images) with
no dependency on gb_pipeline.py, so they can be authored ahead of the
pipeline code itself (Task 0.1 is pure scaffolding).
"""

import os

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GB_PALETTE_PATH = os.path.join(REPO_ROOT, "gb_palette.png")


@pytest.fixture
def checker_tile() -> np.ndarray:
    """8x8 two-color checkerboard pattern, values alternating 0/1.

    Suitable as a minimal `pattern` input (0-3 valued (8,8) uint8 array)
    for exercising pattern_to_2bpp / pattern_variants once implemented.
    """
    tile = np.zeros((8, 8), dtype=np.uint8)
    tile[1::2, 0::2] = 1
    tile[0::2, 1::2] = 1
    return tile


@pytest.fixture
def gradient_image() -> Image.Image:
    """160x144 RGB image with a horizontal gradient (varies by x only)."""
    width, height = 160, 144
    ramp = np.linspace(0, 255, num=width, endpoint=True, dtype=np.uint8)
    row = np.stack([ramp, ramp, ramp], axis=-1)  # (width, 3)
    arr = np.tile(row, (height, 1, 1)).astype(np.uint8)  # (height, width, 3)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def flat_image() -> Image.Image:
    """160x144 solid-color RGB image."""
    width, height = 160, 144
    arr = np.full((height, width, 3), fill_value=(96, 128, 160), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def photo_like_image() -> Image.Image:
    """160x144 deterministic multi-region gradient + noise image.

    Built from a fixed numpy seed so results are reproducible across runs
    (used for golden-count style tests in later tasks).
    """
    width, height = 160, 144
    rng = np.random.RandomState(42)

    yy, xx = np.mgrid[0:height, 0:width]

    # Four quadrant regions, each with its own gradient direction, so the
    # image has distinct "photo-like" areas rather than one smooth ramp.
    region = (yy >= height // 2).astype(np.uint8) * 2 + (xx >= width // 2).astype(np.uint8)

    r = np.zeros((height, width), dtype=np.float32)
    g = np.zeros((height, width), dtype=np.float32)
    b = np.zeros((height, width), dtype=np.float32)

    # Region 0: horizontal red ramp.
    m0 = region == 0
    r[m0] = (xx[m0] / max(width - 1, 1)) * 255
    g[m0] = 40
    b[m0] = 60

    # Region 1: vertical green ramp.
    m1 = region == 1
    r[m1] = 30
    g[m1] = (yy[m1] / max(height - 1, 1)) * 255
    b[m1] = 50

    # Region 2: diagonal blue ramp.
    m2 = region == 2
    diag = (xx[m2].astype(np.float32) + yy[m2].astype(np.float32))
    diag_max = float((width - 1) + (height - 1))
    r[m2] = 20
    g[m2] = 20
    b[m2] = (diag / max(diag_max, 1)) * 255

    # Region 3: radial-ish gray ramp from the region's own center.
    m3 = region == 3
    cx, cy = 3 * width / 4, 3 * height / 4
    dist = np.sqrt((xx[m3].astype(np.float32) - cx) ** 2 + (yy[m3].astype(np.float32) - cy) ** 2)
    dist_max = float(np.sqrt((width / 2) ** 2 + (height / 2) ** 2))
    shade = 255 - (dist / max(dist_max, 1)) * 255
    r[m3] = shade
    g[m3] = shade
    b[m3] = shade

    arr = np.stack([r, g, b], axis=-1)

    noise = rng.normal(loc=0.0, scale=12.0, size=arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def sample_palette_image() -> Image.Image:
    """The repo's `gb_palette.png` (DMG reference ramp), loaded as RGB."""
    return Image.open(GB_PALETTE_PATH).convert("RGB")
