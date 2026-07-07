# GB Studio Conformance Loop — Design

**Date:** 2026-07-08 · **Branch:** `gb-pipeline-redesign` · **Status:** Approved

## Problem

GB Studio's importer discards our conversion decisions and re-derives palettes,
tile indices, and dedup from the rendered PNG with its own algorithms
(v4.3.2 `autoPalette`/`compressPalettes`/`autoFlipTiles`/green-threshold mono).
Two defects follow, verified against GB Studio's real code (see
`gb_studio_import.py`, validated 18/18 vs the node oracle):

1. **Palette-overflow recoloring (color).** Our output is GB-legal (every tile
   drawn from one of ≤8 four-color masters), but GB's greedy `compressPalettes`
   merges *any* two per-tile palettes whose color union ≤4. Small (2–3 color)
   per-tile subsets from *different* masters merge into **hybrids** that are a
   subset of no master, can never be absorbed (union >4), and survive as 9th+
   palettes. GB then wraps the palette mapping `% 8` with no warning →
   silently recolored tiles. Root cause confirmed empirically: photo1 no-dither
   has 5 full masters but GB extracts 10 palettes (4 hybrids); the dithered
   variant (whose tiles nearly all contain 4 master colors) converges to 8.
2. **Tile-count divergence (color, multi-screen).** Our packer counts tiles
   with its own dedup; GB's autoflip count on its re-quantized tiles ran
   389–390 at 320×288 where we reported ≤380 → GB rejects the import
   ("too many tiles"). Priority per Liam: **tile count matters most**;
   recoloring is acceptable (though not ideal). Scope: **any image size**.

## Decision

**Approach A — oracle-in-the-loop conformance.** Keep the existing packer for
candidate generation; use the validated Python port (`gb_studio_import`) as an
in-loop oracle and conform the output until GB's own measurement passes. The
shipped PNG is then pixel-exact what GB Studio reconstructs on import
(corruption = 0 by construction), and any recoloring is applied by *us* with a
best-fit palette choice instead of GB's arbitrary `% 8` wrap.

Rejected: (B) rewriting the packer to natively count like GB — large refactor
of freshly perf-tuned merge machinery, high regression risk; (C) post-process
tile patching on final pixels — same counts as A but visibly worse merges.

## Components

### `gb_studio_import.py` (additions)

- `gbstudio_mono_stats(arr) -> GBStudioMonoStats` — GB's mono import counting:
  green-threshold indices (`g<65→3, <130→2, <205→1, else 0`) → exact 2bpp
  dedup, **no flips**. Fields: `tiles`, and per-ramp-entry green buckets so the
  caller can detect non-distinct ramps. No LAB, cheap.
- `recolor_overflow(arr) -> (new_arr, n_recolored)` — run `autopalette`; for
  every tile whose merged palette index ≥8, rewrite the tile as GB's own
  reconstruction (`buildIndexedTile` nearest-color, Manhattan RGB) against the
  **best-fit** kept palette (index <8 minimizing total per-pixel color error
  for that tile), not the `% 8` one. Tiles mapping <8 are untouched.

### `gb_pipeline.py` — `_conform_to_gbstudio(...)`

Called from `convert_for_hardware` where the report-only honest check sits
today; the check becomes the loop's oracle and final assertion.

```
target = budget                      # color 384 / mono 192 (or user tile_budget)
# — Tile loop (max 5 iterations) —
repeat:
    pack + render candidate at target          # existing pipeline stages
    s = oracle(candidate)                      # GB's real tile count
    if s.tiles <= budget: break
    target -= (s.tiles - budget)               # tighten by the overshoot
# — Palette conform loop, color only (max 4 iterations) —
while s.palettes_extracted > 8:
    candidate, n = recolor_overflow(candidate) # best-fit, our choice
    re-render, re-measure
# — Fallback ladder (provably terminates) —
if still >8 palettes: repack with n_palettes-1 masters and restart
    # at n=1 every per-tile palette is a subset of one master; any two such
    # subsets union within the master (≤4) so GB's greedy always converges
# — Final assertion —
oracle reports tiles <= budget AND palettes <= 8 AND corrupted == 0
```

Ordering is safe: recoloring is deterministic per pixel-content, so identical
tiles recolor identically and distinct tiles can only collapse — the palette
loop can only *shrink* the tile count and cannot undo the tile loop.
Convergence: observed tile overshoot ≤10 → ~2 iterations; palette conform makes
overflow tiles exact subsets of kept masters → typically 1–2 passes.

Presets: `color_only` gets both loops; `mono` gets the tile loop with the mono
oracle plus a warning when a custom `mono_ramp` is not green-bucket-distinct
under GB's fixed thresholds (today that silently merges shades on import);
logo presets are tile-exempt in GB (`!r` guard) — `logo_color` gets only the
palette conform, `logo_mono` only the ramp-distinctness warning.

### Reporting (`convert_for_hardware` stats + `main._format_hardware_notice`)

- `gbstudio_corrupted_tiles` is 0 by construction → the "GB Studio will
  recolor N tiles" warning disappears.
- New stat `gbstudio_recolored_tiles`: tiles *we* recolored to fit the
  8-palette import. Notice line gains `· N tiles adapted for import` when >0.
- `tiles_used` shown to the UI = GB's count (identical by construction).
- Safety net: if the fallback ladder is somehow exhausted (not expected to be
  reachable), keep today's honest warning instead of crashing.

## Performance

Clean images pay exactly today's cost (one oracle run, ~140ms @160×144,
~464ms @320×288). Overshooting images pay +1–3 pack/oracle rounds — worst
≈ +2s at 320×288 on a ~3.4s conversion. No micro-optimization now; a
changed-tiles-only oracle is a later option if it hurts.

## Testing (TDD, red first)

- **Unit:** `gbstudio_mono_stats` expected counts on hand-built arrays +
  existing fixtures; `recolor_overflow` on `tests/fixtures/overflow_palettes.png`
  → ≤8 palettes at fixed point, corruption 0, and best-fit total error
  strictly below the `% 8` wrap error.
- **Integration:** `overflow_palettes.png` and a 320×288 photo-like image
  through `convert_for_hardware` → final oracle-clean stats
  (tiles ≤ budget, palettes ≤ 8, corrupted 0); mono custom-ramp warning.
- **Sweep (slow-marked):** photo-like generated images × {160×144, 320×288} ×
  {dither none, bayer} × {color_only, mono} → final assertion holds everywhere.
- Existing suite (134 tests) must stay green; `test_gbstudio_import.py`
  oracle-parity tests are the regression anchor for the port itself.
