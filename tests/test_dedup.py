"""Tests for gb_pipeline.py Task 1.4 lossless dedup (spec Stage 5)."""

import numpy as np

from gb_pipeline import GBImage, dedup_patterns, pattern_to_2bpp, pattern_variants


def _asymmetric_pattern(seed=0):
    """An 8x8 index pattern (values 0-3) with no flip symmetry."""
    r = np.arange(8).reshape(8, 1)
    c = np.arange(8).reshape(1, 8)
    base = ((r + 2 * c + (r * c) % 3 + seed) % 4).astype(np.uint8)
    return base


def _dummy_palettes(n=1):
    return np.tile(
        np.array([[240, 240, 240], [170, 170, 170], [90, 90, 90], [10, 10, 10]], dtype=np.uint8),
        (n, 1, 1),
    )


def _pre_dedup_gb(patterns, tilemap, attrs_palette=None, attrs_hflip=None, attrs_vflip=None, palettes=None):
    """Build a GBImage as index_tiles would emit it (one pattern per cell,
    identity references) or with explicit flip attrs already set.
    """
    th, tw = tilemap.shape
    if attrs_palette is None:
        attrs_palette = np.zeros((th, tw), dtype=np.uint8)
    if attrs_hflip is None:
        attrs_hflip = np.zeros((th, tw), dtype=bool)
    if attrs_vflip is None:
        attrs_vflip = np.zeros((th, tw), dtype=bool)
    if palettes is None:
        palettes = _dummy_palettes(int(attrs_palette.max()) + 1)
    return GBImage(
        patterns=patterns,
        tilemap=tilemap,
        attrs_palette=attrs_palette,
        attrs_hflip=attrs_hflip,
        attrs_vflip=attrs_vflip,
        palettes=palettes,
    )


def test_one_repeated_tile_collapses_to_one_pattern():
    tile = _asymmetric_pattern()
    # 2x3 grid of tiles, every cell identical content, one pattern per cell
    # (as index_tiles would emit before dedup).
    th, tw = 2, 3
    patterns = np.stack([tile.copy() for _ in range(th * tw)])
    tilemap = np.arange(th * tw, dtype=np.int32).reshape(th, tw)
    gb = _pre_dedup_gb(patterns, tilemap)

    result = dedup_patterns(gb, allow_flips=False)

    assert result.patterns.shape == (1, 8, 8)
    assert np.array_equal(result.tilemap, np.zeros((th, tw), dtype=np.int32))
    assert not result.attrs_hflip.any()
    assert not result.attrs_vflip.any()
    assert np.array_equal(result.attrs_palette, gb.attrs_palette)
    assert np.array_equal(result.palettes, gb.palettes)


def test_mirror_pair_with_flips_allowed_collapses_and_sets_hflip():
    tile = _asymmetric_pattern()
    mirror = np.fliplr(tile)
    patterns = np.stack([tile, mirror])
    tilemap = np.array([[0, 1]], dtype=np.int32)
    gb = _pre_dedup_gb(patterns, tilemap)

    result = dedup_patterns(gb, allow_flips=True)

    assert result.patterns.shape == (1, 8, 8)
    # The first cell is the canonical (no-flip) orientation.
    assert result.tilemap[0, 0] == result.tilemap[0, 1]
    assert not result.attrs_hflip[0, 0]
    assert not result.attrs_vflip[0, 0]
    assert result.attrs_hflip[0, 1]
    assert not result.attrs_vflip[0, 1]
    # Reconstructing cell 1 from the canonical pattern + its flip bits must
    # reproduce the original mirrored content exactly.
    canonical = result.patterns[result.tilemap[0, 1]]
    assert np.array_equal(np.fliplr(canonical), mirror)


def test_mirror_pair_without_flips_stays_two_patterns():
    tile = _asymmetric_pattern()
    mirror = np.fliplr(tile)
    patterns = np.stack([tile, mirror])
    tilemap = np.array([[0, 1]], dtype=np.int32)
    gb = _pre_dedup_gb(patterns, tilemap)

    result = dedup_patterns(gb, allow_flips=False)

    assert result.patterns.shape == (2, 8, 8)
    assert result.tilemap[0, 0] != result.tilemap[0, 1]
    assert not result.attrs_hflip.any()
    assert not result.attrs_vflip.any()


def test_count_matches_independent_set_of_bytes_computation():
    # A grid mixing several distinct random tiles with deliberate exact
    # duplicates (no mirrors involved), so with allow_flips=False the result
    # must match a plain "unique raw bytes" count computed independently.
    rng = np.random.RandomState(7)
    distinct = [rng.randint(0, 4, size=(8, 8)).astype(np.uint8) for _ in range(5)]
    th, tw = 3, 4
    layout = [0, 1, 2, 0, 1, 3, 4, 4, 2, 0, 3, 1]  # reuses distinct[i] many times
    patterns = np.stack([distinct[i].copy() for i in layout])
    tilemap = np.arange(th * tw, dtype=np.int32).reshape(th, tw)
    gb = _pre_dedup_gb(patterns, tilemap)

    result = dedup_patterns(gb, allow_flips=False)

    expected_unique = len({pattern_to_2bpp(distinct[i]) for i in layout})
    assert result.patterns.shape[0] == expected_unique

    # Every cell's reconstructed effective pattern must hash to the same
    # bytes as its original source pattern (bytes-identity check).
    for tr in range(th):
        for tc in range(tw):
            orig_bytes = pattern_to_2bpp(patterns[tr * tw + tc])
            idx = result.tilemap[tr, tc]
            recon = result.patterns[idx]
            if result.attrs_hflip[tr, tc]:
                recon = np.fliplr(recon)
            if result.attrs_vflip[tr, tc]:
                recon = np.flipud(recon)
            assert pattern_to_2bpp(recon) == orig_bytes


def test_dedup_is_idempotent():
    tile_a = _asymmetric_pattern(seed=0)
    tile_b = _asymmetric_pattern(seed=1)
    mirror_a = np.fliplr(tile_a)
    vflip_b = np.flipud(tile_b)

    patterns = np.stack([tile_a, tile_b, mirror_a, vflip_b, tile_a.copy()])
    tilemap = np.arange(5, dtype=np.int32).reshape(1, 5)
    gb = _pre_dedup_gb(patterns, tilemap)

    once = dedup_patterns(gb, allow_flips=True)
    twice = dedup_patterns(once, allow_flips=True)

    assert once.patterns.shape == twice.patterns.shape
    assert np.array_equal(once.patterns, twice.patterns)
    assert np.array_equal(once.tilemap, twice.tilemap)
    assert np.array_equal(once.attrs_hflip, twice.attrs_hflip)
    assert np.array_equal(once.attrs_vflip, twice.attrs_vflip)
    assert np.array_equal(once.attrs_palette, twice.attrs_palette)
    assert np.array_equal(once.palettes, twice.palettes)


def test_dedup_preserves_palette_assignment():
    tile = _asymmetric_pattern()
    patterns = np.stack([tile.copy(), tile.copy()])
    tilemap = np.array([[0, 1]], dtype=np.int32)
    attrs_palette = np.array([[0, 1]], dtype=np.uint8)
    palettes = _dummy_palettes(2)
    gb = _pre_dedup_gb(patterns, tilemap, attrs_palette=attrs_palette, palettes=palettes)

    result = dedup_patterns(gb, allow_flips=False)

    assert result.patterns.shape == (1, 8, 8)
    assert np.array_equal(result.attrs_palette, attrs_palette)
    assert np.array_equal(result.palettes, palettes)


def test_pattern_with_self_colliding_variants_still_dedups_correctly():
    # A pattern whose "" and "h" variants collide (horizontally palindromic
    # rows), per test_primitives.py's test_pattern_variants_symmetric_pattern
    # _collides. All three of {itself, its fliplr, its flipud} are then
    # variants of the *same* underlying canonical tile, so allow_flips=True
    # must collapse all three cells to one stored pattern without raising --
    # exercising the "if vkey not in variant_lookup" collision guard.
    rows = []
    for i in range(8):
        a, b, c, d = i % 4, (i + 1) % 4, (i + 2) % 4, (i + 3) % 4
        rows.append([a, b, c, d, d, c, b, a])
    tile = np.array(rows, dtype=np.uint8)
    variants = pattern_variants(tile)
    assert np.array_equal(variants[""], variants["h"])  # sanity, matches spec

    patterns = np.stack([tile, np.flipud(tile), np.fliplr(tile)])
    tilemap = np.array([[0, 1, 2]], dtype=np.int32)
    gb = _pre_dedup_gb(patterns, tilemap)

    result = dedup_patterns(gb, allow_flips=True)

    assert result.patterns.shape == (1, 8, 8)
    for tc in range(3):
        idx = result.tilemap[0, tc]
        recon = result.patterns[idx]
        if result.attrs_hflip[0, tc]:
            recon = np.fliplr(recon)
        if result.attrs_vflip[0, tc]:
            recon = np.flipud(recon)
        assert pattern_to_2bpp(recon) == pattern_to_2bpp(patterns[tc])
