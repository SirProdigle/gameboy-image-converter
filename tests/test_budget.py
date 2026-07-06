"""Tests for gb_pipeline.py Task 1.5 budget merge (spec Stage 6).

merge_to_budget lossily collapses patterns until the live tile count lands
exactly on the budget, cheapest-first, where cost = signature distance x usage
of the replaced pattern. These tests exercise: the already-under-budget no-op,
landing exactly on budget with only-live references, cheap near-duplicate
merges preferred over dissimilar ones, usage-weighting (high-usage patterns
survive), flip-orientation composition, determinism, and the clustering fast
path.
"""

import numpy as np
import pytest

import gb_pipeline
from gb_pipeline import GBImage, merge_to_budget, pattern_to_2bpp

# A grayscale ramp palette: index 0 white ... index 3 black. A one-pixel change
# between adjacent ramp levels is a tiny signature perturbation, while flipping
# a pixel between white (0) and black (3) is a large one -- this lets tests
# build guaranteed "near-duplicate" vs "very different" patterns.
RAMP = np.array(
    [[255, 255, 255], [170, 170, 170], [85, 85, 85], [0, 0, 0]], dtype=np.uint8
)


def _build_gb(patterns, usage=None, palettes=None, attrs_palette=None,
              attrs_hflip=None, attrs_vflip=None):
    """Build a single-row GBImage from a list of (8,8) patterns.

    ``usage[i]`` (default 1) is how many cells reference pattern i; the tilemap
    is a 1xN row listing each pattern index that many times in order.
    """
    patterns = np.stack([np.asarray(p, dtype=np.uint8) for p in patterns])
    n = patterns.shape[0]
    if usage is None:
        usage = [1] * n
    flat = []
    for i, u in enumerate(usage):
        flat.extend([i] * u)
    total = len(flat)
    tilemap = np.array(flat, dtype=np.int32).reshape(1, total)
    if palettes is None:
        palettes = RAMP[None, :, :].copy()
    if attrs_palette is None:
        attrs_palette = np.zeros((1, total), dtype=np.uint8)
    if attrs_hflip is None:
        attrs_hflip = np.zeros((1, total), dtype=bool)
    if attrs_vflip is None:
        attrs_vflip = np.zeros((1, total), dtype=bool)
    return GBImage(
        patterns=patterns,
        tilemap=tilemap,
        attrs_palette=attrs_palette,
        attrs_hflip=attrs_hflip,
        attrs_vflip=attrs_vflip,
        palettes=palettes,
    )


def _hi_lo_tile(rng):
    """Random 8x8 tile using only the extreme ramp indices 0 and 3."""
    return rng.choice(np.array([0, 3], dtype=np.uint8), size=(8, 8))


def _distinct_hi_lo_tiles(n, rng):
    seen = set()
    out = []
    while len(out) < n:
        t = _hi_lo_tile(rng)
        key = pattern_to_2bpp(t)
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def _near_twin(base):
    """A one-pixel, one-ramp-step perturbation of ``base`` (tiny signature
    distance) that is still a distinct 2bpp pattern."""
    twin = np.asarray(base, dtype=np.uint8).copy()
    twin[0, 0] = 2 if twin[0, 0] == 3 else 1
    return twin


def _asymmetric_tile():
    r = np.arange(8).reshape(8, 1)
    c = np.arange(8).reshape(1, 8)
    return ((r + 2 * c + (r * c) % 3) % 4).astype(np.uint8)


def _byteset(gb):
    return {pattern_to_2bpp(gb.patterns[i]) for i in range(gb.patterns.shape[0])}


def _references_only_live(gb):
    n = gb.patterns.shape[0]
    return int(gb.tilemap.min()) >= 0 and int(gb.tilemap.max()) < n


# ---------------------------------------------------------------------------


def test_already_under_budget_is_unchanged_zero_merges():
    rng = np.random.RandomState(1)
    tiles = _distinct_hi_lo_tiles(5, rng)
    gb = _build_gb(tiles)

    result, n_merges = merge_to_budget(gb, budget=10, allow_flips=False)

    assert n_merges == 0
    assert result.patterns.shape[0] == 5
    assert np.array_equal(result.patterns, gb.patterns)
    assert np.array_equal(result.tilemap, gb.tilemap)
    assert result.merge_p95_delta_e == 0.0


def test_exactly_on_budget_when_all_distinct():
    rng = np.random.RandomState(2)
    tiles = _distinct_hi_lo_tiles(100, rng)
    gb = _build_gb(tiles)  # each used once, 100 cells

    result, n_merges = merge_to_budget(gb, budget=40, allow_flips=False)

    assert result.patterns.shape[0] == 40
    assert n_merges == 60
    assert _references_only_live(result)
    # Every surviving pattern is referenced (survivors keep their own cells),
    # and the tilemap still covers all original cells.
    assert result.tilemap.shape == gb.tilemap.shape
    assert len(np.unique(result.tilemap)) == 40


def test_near_duplicates_merge_before_dissimilar():
    rng = np.random.RandomState(3)
    # 5 outliers + 3 group bases (all very different), plus one near-twin each.
    distinct = _distinct_hi_lo_tiles(8, rng)
    outliers = distinct[:5]
    bases = distinct[5:]
    twins = [_near_twin(b) for b in bases]
    tiles = outliers + bases + twins  # 11 patterns
    gb = _build_gb(tiles)

    result, n_merges = merge_to_budget(gb, budget=8, allow_flips=False)

    # Exactly the three cheap near-twin merges happen; the outliers survive.
    assert n_merges == 3
    assert result.patterns.shape[0] == 8
    assert _references_only_live(result)
    survivors = _byteset(result)
    for o in outliers:
        assert pattern_to_2bpp(o) in survivors


def test_high_usage_pattern_survives_single_use_twin():
    # A (high usage) and B (single use) are a near-duplicate pair; the merge is
    # oriented to replace the lower-usage pattern, so A survives.
    rng = np.random.RandomState(4)
    distinct = _distinct_hi_lo_tiles(6, rng)
    a = distinct[0]
    b = _near_twin(a)
    outliers = distinct[1:]  # 5 far-apart, single use
    tiles = [a, b] + outliers  # indices: 0=A, 1=B, 2..6 outliers
    usage = [5, 1] + [1] * len(outliers)
    gb = _build_gb(tiles, usage=usage)

    result, n_merges = merge_to_budget(gb, budget=6, allow_flips=False)

    assert n_merges == 1
    survivors = _byteset(result)
    assert pattern_to_2bpp(a) in survivors        # high-usage kept
    assert pattern_to_2bpp(b) not in survivors    # single-use absorbed
    assert _references_only_live(result)
    # The cells that referenced B now reference A's surviving pattern.
    a_new = [i for i in range(result.patterns.shape[0])
             if pattern_to_2bpp(result.patterns[i]) == pattern_to_2bpp(a)][0]
    assert int((result.tilemap == a_new).sum()) == 6  # 5 original A + 1 from B


def test_flip_orientation_composed_when_flips_allowed():
    tile = _asymmetric_tile()
    mirror = np.fliplr(tile)
    gb = _build_gb([tile, mirror])  # cell0 -> tile, cell1 -> mirror

    result, n_merges = merge_to_budget(gb, budget=1, allow_flips=True)

    assert n_merges == 1
    assert result.patterns.shape[0] == 1
    # Reconstruct each original cell from the survivor + its composed flip bits.
    for col, original in enumerate([tile, mirror]):
        recon = result.patterns[result.tilemap[0, col]]
        if result.attrs_hflip[0, col]:
            recon = np.fliplr(recon)
        if result.attrs_vflip[0, col]:
            recon = np.flipud(recon)
        assert pattern_to_2bpp(recon) == pattern_to_2bpp(original)


def test_p95_delta_e_is_stashed_and_nonnegative():
    rng = np.random.RandomState(5)
    tiles = _distinct_hi_lo_tiles(20, rng)
    gb = _build_gb(tiles)

    result, _ = merge_to_budget(gb, budget=8, allow_flips=False)

    assert hasattr(result, "merge_p95_delta_e")
    assert result.merge_p95_delta_e >= 0.0


def test_determinism():
    rng = np.random.RandomState(6)
    tiles = _distinct_hi_lo_tiles(60, rng)
    gb1 = _build_gb(tiles)
    gb2 = _build_gb(tiles)

    r1, m1 = merge_to_budget(gb1, budget=25, allow_flips=True)
    r2, m2 = merge_to_budget(gb2, budget=25, allow_flips=True)

    assert m1 == m2
    assert np.array_equal(r1.patterns, r2.patterns)
    assert np.array_equal(r1.tilemap, r2.tilemap)
    assert np.array_equal(r1.attrs_hflip, r2.attrs_hflip)
    assert np.array_equal(r1.attrs_vflip, r2.attrs_vflip)
    assert r1.merge_p95_delta_e == r2.merge_p95_delta_e


def test_clustering_fast_path_lands_on_budget(monkeypatch):
    # Force the clustering path by lowering the matrix threshold; it must still
    # land exactly on budget with only-live references.
    monkeypatch.setattr(gb_pipeline, "_MERGE_MATRIX_MAX", 5)
    rng = np.random.RandomState(7)
    tiles = _distinct_hi_lo_tiles(20, rng)
    gb1 = _build_gb(tiles)
    gb2 = _build_gb(tiles)

    result, n_merges = merge_to_budget(gb1, budget=8, allow_flips=False)

    assert result.patterns.shape[0] == 8
    assert n_merges == 12
    assert _references_only_live(result)
    # Deterministic across runs even on the clustering path.
    result2, n2 = merge_to_budget(gb2, budget=8, allow_flips=False)
    assert n2 == n_merges
    assert np.array_equal(result.tilemap, result2.tilemap)
    assert np.array_equal(result.patterns, result2.patterns)
