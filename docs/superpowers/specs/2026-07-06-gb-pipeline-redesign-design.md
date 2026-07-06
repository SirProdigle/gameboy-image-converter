# GB Studio Hardware Pipeline Redesign — Design Spec

**Date:** 2026-07-06
**Status:** Approved design, pending implementation plan

## Goal

Replace the hardware-constrained conversion paths (4-colors-per-tile palette mode and tile reduction) with a new module that produces images guaranteed to satisfy GB Studio's limits — exact tile budget, palette count, and colors-per-tile — by construction rather than by tuning. Redesign the Gradio UI around a mode-first layout. The free-form "Artistic" flow is preserved unchanged.

## Ground truth (verified against GB Studio source, docs, and Pan Docs)

| Constraint | Value |
|---|---|
| Unique BG tiles, Mono scenes | 192 (`MAX_BACKGROUND_TILES = 16*12`) |
| Unique BG tiles, Color Only scenes | 384 (both CGB VRAM banks) |
| BG palettes per scene | 8 × 4 colors; **palette 8 is overwritten by the dialogue/UI palette at runtime** |
| Palette assignment | One palette per 8×8 tile (CGB attribute map bits 0–2) |
| Tile flipping | CGB BG tiles support X/Y flip; GB Studio ≥4.2 auto-dedupes flipped variants in Color Only scenes. DMG has no BG flips |
| Color depth | RGB555; GB Studio snaps imports via `c5 = floor(c8·32/256)` per channel |
| Tile identity | The 16-byte 2bpp index pattern, not RGB — identical patterns under different palettes cost **one** tile |
| Over limit | GB Studio warns but compiles; glitches on hardware |

Key algorithmic context: palette packing is the NP-complete pagination problem (rgbgfx cites arXiv:1605.00558 and uses an "overload-and-remove" heuristic). GB Studio's own importer is a greedy first-4-colors-per-tile + exact-union merge; it has no lossy fallback and wraps overflow modulo 8.

## Why the current implementation over/undershoots

- All `reduce_tiles*` variants use greedy first-fit in raster order with the similarity threshold as the only knob; the final count is emergent, never targeted. High threshold → budget exhausted early → spatial quality cliff. Low threshold → undershoot.
- Overflow fallbacks pick `min(...)` by similarity — the **least** similar tile (main.py:566, 570, 627, 740).
- `reduce_tiles_index_palette_aware` merges only same-palette tiles — backwards vs. hardware, which shares patterns across palettes.
- Palette generation (`analyze_and_construct_palettes`) k-means all pixels globally, chops centers sequentially into groups of 4, then scatters colors round-robin by frequency — optimizing global coverage, not per-tile renderability. It also computes distances between RGB and LAB values (main.py:1271), a unit bug.
- No flip awareness, no RGB555 snapping, no verification of final output; `apply_mapped_colors_to_tile` silently passes through out-of-palette colors.
- Module-level `gr.State` mutation races across concurrent users on the hosted app.

## Architecture

New file **`gb_pipeline.py`** — pure functions over numpy arrays, no Gradio imports. Entry point:

```python
result = convert_for_hardware(image, preset, options)
# preset: COLOR_ONLY (384 tiles, 8 palettes, flips) | MONO (192 tiles, 1×4 shades)
#         | LOGO (fixed 160×144, 360 sequential tiles, no tile limit, palettes still apply)
# options: tile_budget (default from preset), reserve_ui_palette (default True),
#          dither ("none" | "bayer"), custom_palette (optional restriction set),
#          mono_ramp (4 colors, default DMG green)
```

### Data model

```python
@dataclass
class GBImage:
    patterns: np.ndarray   # (n_tiles, 8, 8) uint8 values 0-3 — the tileset
    tilemap: np.ndarray    # (H/8, W/8) int — pattern index per cell
    attrs: np.ndarray      # (H/8, W/8) — palette id + hflip/vflip flags
    palettes: np.ndarray   # (n_palettes, 4, 3) uint8, RGB555-snapped, luminance-sorted
```

This mirrors VRAM. After stage 4 every operation is on `GBImage`; RGB reappears only at render. Colors-per-tile ≤ 4 and palette-independence of patterns are unrepresentable-to-violate rather than checked-for.

### Stage flow

```
RGB image
 1. downscale                      (existing code reused)
 2. RGB555 snap + global quantize  (working set ≤ 4 × palette budget)
 3. palette packing                → palettes + per-cell palette assignment
 4. per-tile indexing + dither     → GBImage
 5. lossless dedup (exact + flips) → fewer patterns
 6. budget merge (lossy, if over)  → len(patterns) ≤ budget, exactly
 7. verify + report                → counts, swatches, quality warnings
 8. render                         → PNG importable directly into GB Studio
```

**LOGO preset:** GB Studio Logo scenes (`is360` in the compiler) require exactly 160×144 and store all 360 tiles sequentially with no deduplication and no tile limit — so stages 5–6 are skipped entirely. Palette constraints are hardware-level and still enforced (stage 3 runs unchanged: ≤7/8 palettes, 4 colors per tile, RGB555). This is the highest-fidelity target for full-screen art: every tile keeps its own pixels, only colors are constrained. Input is auto-resized/cropped to 160×144 with a notice if it differs.

## Stage 3 — palette packing

Two phases; distances in CIELAB throughout; every fitted color snapped to RGB555.

**Phase 1 — agglomerative seeding.** Each tile gets an ideal palette (its unique colors if ≤4, else 4-means of its pixels). Dedupe identical palettes. While count > budget (7 by default, 8 if `reserve_ui_palette=False`): merge the cheapest pair via priority queue. A merge whose union has ≤4 colors is free (this is all GB Studio's importer can do); otherwise fit 4 colors to the union by pixel-count-weighted k-means and cost = added quantization error over member tiles.

**Phase 2 — Lloyd refinement** (~3–5 iterations or until stable): (a) reassign each tile to the palette that renders it with least frequency-weighted ΔE; (b) refit each palette as weighted 4-means over member tiles' pixels, snapped to RGB555. This is the step no existing GB tool performs.

**Mono preset:** collapses to one fixed 4-shade ramp (default DMG green from `gb_palette.png`, or user-supplied); tiles map by luminance.
**Custom palette image:** when provided, palette colors are selected from the custom set instead of free-fitted; packing/assignment machinery identical.

## Stage 4 — indexing + dithering

Each tile is rendered against its assigned palette (nearest color in Lab → index 0–3). Palette entries are luminance-sorted so index 0 means "lightest" in every palette — tiles with the same shading structure under different palettes produce identical patterns, which stage 5 dedups free.

**Dithering:** optional ordered Bayer, applied per pixel against a threshold matrix anchored to absolute image coordinates, choosing between the two nearest shades of the tile's own palette. Properties:

- Cannot violate 4-colors-per-tile (only picks from the assigned palette).
- Position-stable: identical content → identical patterns → dedup preserved.
- Trade-off: gradients become fine checkerboards that vary tile-to-tile, consuming more of the tile budget; the merger absorbs this (budget still lands exactly) and the verified stats make the cost visible. Flip-dedup is slightly less effective under dither (mirrored Bayer ≠ Bayer of mirror); the lossy merger catches those pairs.
- Floyd–Steinberg is excluded in hardware modes: error diffusion crosses tile/palette boundaries and is position-unstable (manufactures unique tiles).

## Stage 5 — lossless dedup

Hash each 16-byte 2bpp pattern exactly as GB Studio's `tileData.ts` does. For Color Only, also hash H/V/HV flips and set attribute flip bits on match. After this stage our unique count equals GB Studio's. Many images fit here and stage 6 never runs.

## Stage 6 — budget merge

Only if count > budget. Each pattern's signature = the tile rendered under its usage-weighted palette as an 8×8 Lab vector. Cheapest-first from a priority queue:

- cost = perceptual distance between signatures (flip variants included) × usage count of the replaced pattern;
- a merge remaps tilemap references (+ flip bits) only — it cannot affect palette constraints;
- repeat until `len(patterns) ≤ budget`, then stop — images already under budget after stage 5 are never merged at all. No threshold; when merging is needed, error concentrates in the globally cheapest merges instead of the bottom of the image.

Pairwise distances are one vectorized numpy matrix at screen sizes; above ~3–4k unique patterns, pre-cluster signatures and compare within clusters (fast path for oversized images — slower but functional, per scope decision).

## Stage 7 — verification

1. Assert on `GBImage`: patterns ≤ budget, palettes ≤ 7/8, RGB555-only.
2. Independent round-trip: re-import the rendered PNG with a reimplementation of GB Studio's hashing and assert counts match. The UI notice reports these verified numbers.
3. Quality warnings: when merging was heavy, report p95 ΔE of merged pixels with guidance (raise budget / simplify image / disable dither).

## UI redesign

Mode-first layout; a mode radio at top: **Artistic | GB Studio: Color | GB Studio: Mono | GB Studio: Logo**.

- **Input:** tabs (Single Image | Batch Folder), one Convert button acting on the active tab.
- **Size:** W×H, aspect lock, preset buttons "GB Screen (160×144)" (replaces "Use Logo Resolution") and "Original".
- **Color panel, per mode:**
  - Artistic: color count 2–64, quantization method, dither incl. Floyd–Steinberg, custom palette image, grayscale/B&W — today's flow unchanged.
  - GB Studio Color: reserve-palette-8 checkbox (default on), dither None/Bayer, optional custom palette restriction. Color count derived (palettes × 4), no slider.
  - GB Studio Mono: 4-shade ramp picker (default DMG green), dither None/Bayer.
  - GB Studio Logo: a Color/Mono sub-toggle picks which color panel applies (color logo for CGB projects, 4-shade ramp for DMG projects); size locked to 160×144 (the "GB Screen" preset applied automatically); tiles panel hidden — no tile limit applies.
- **Tiles panel (Color/Mono modes only):** tile budget number, prefilled 384/192, editable downward for aesthetic crunch.
- **Effects accordion (collapsed):** Gothic filter, grayscale, B&W.
- **Output panel:** converted image + natural-palette reference; verified stats line ("✅ 337/384 tiles · 7 palettes · 214 merges"); rendered palette swatches with copyable hex (replacing the text dump); PNG/ZIP download.
- Removed controls: similarity-threshold slider, "sort by tile complexity", "limit to 4 colors per tile", "reduce to 192 tiles" — all subsumed by mode + budget.
- Unchanged: queue/heartbeat infra, analytics header, ko-fi/Discord/GitHub links.

## Error handling

- Non-multiple-of-8 dimensions auto-crop with a notice (current behavior).
- No hard mathematical failure cases (any budget ≥1 and palette count ≥1 is achievable); quality degradation is reported, not errored.
- Deterministic output: fixed random seeds on all k-means.
- Input mode conversions (P/RGBA/L → RGB) at the boundary.

## Code cleanup (in blast radius)

- Delete: both `reduce_tiles_index` definitions, `reduce_tiles`, `reduce_tiles_index_palette_aware`, duplicate `tile_similarity_indexed`, `process_tiles`, `analyze_and_construct_palettes`, `create_refined_palettes`, `find_best_matching_palette`, `map_pattern_to_palette`, `apply_mapped_colors_to_tile` (replaced by `gb_pipeline.py`).
- Replace module-level `gr.State` globals (`quantize_for_GBC`, `use_tile_variance`, `original_width/height`) with ordinary event-handler inputs — fixes cross-user races on the hosted app.
- Scattered mid-function imports in `main.py` consolidated where touched.

## Testing

New `tests/` with pytest (first tests in the repo):

- **Unit:** 2bpp hashing matches GB Studio semantics incl. flips; luminance sort; RGB555 snap idempotence; budget merge lands exactly on budget; palette packing respects ≤4 colors and ≤7/8 palettes.
- **Property:** random images × presets → all invariants hold.
- **Golden:** sample images with pinned tile/palette counts and SSIM-vs-baseline regression thresholds.
- **Perf smoke:** 160×144 and 320×288 complete within seconds.

## Out of scope

- Sprite conversion (3+transparent palettes).
- Direct .gbr / tileset binary export — PNG round-trip through GB Studio's importer is the delivery mechanism.
- Performance work for 2040×2040 beyond the clustering fast path.
