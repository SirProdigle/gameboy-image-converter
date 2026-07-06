"""Task 0.1 smoke test: verify fixtures load with expected shapes/types.

There is no pipeline code yet (Phase 1), so this file only exercises the
conftest fixtures themselves.
"""

import numpy as np
from PIL import Image


def test_checker_tile_shape_and_values(checker_tile):
    assert checker_tile.shape == (8, 8)
    assert checker_tile.dtype == np.uint8
    assert set(np.unique(checker_tile)) == {0, 1}
    # Checkerboard: every 2x2 block alternates.
    assert checker_tile[0, 0] != checker_tile[0, 1]
    assert checker_tile[0, 0] != checker_tile[1, 0]
    assert checker_tile[0, 0] == checker_tile[1, 1]


def test_gradient_image_loads(gradient_image):
    assert isinstance(gradient_image, Image.Image)
    assert gradient_image.size == (160, 144)
    assert gradient_image.mode == "RGB"
    arr = np.array(gradient_image)
    # Horizontal gradient: left edge darker than right edge, constant per column.
    assert arr[:, 0, 0].max() < arr[:, -1, 0].min()
    assert np.all(arr[0] == arr[-1])


def test_flat_image_loads(flat_image):
    assert isinstance(flat_image, Image.Image)
    assert flat_image.size == (160, 144)
    assert flat_image.mode == "RGB"
    arr = np.array(flat_image)
    assert np.all(arr == arr[0, 0])


def test_photo_like_image_loads(photo_like_image):
    assert isinstance(photo_like_image, Image.Image)
    assert photo_like_image.size == (160, 144)
    assert photo_like_image.mode == "RGB"
    arr = np.array(photo_like_image)
    # Not flat, and reasonably varied (multi-region gradient + noise).
    assert arr.std() > 10


def test_photo_like_image_deterministic(photo_like_image):
    """The fixture factory is seeded, so independent calls must match."""
    from tests.conftest import photo_like_image as photo_like_image_fixture

    other = photo_like_image_fixture.__wrapped__()
    assert np.array_equal(np.array(photo_like_image), np.array(other))


def test_sample_palette_image_loads(sample_palette_image):
    assert isinstance(sample_palette_image, Image.Image)
    assert sample_palette_image.mode == "RGB"
    arr = np.array(sample_palette_image)
    assert arr.size > 0
