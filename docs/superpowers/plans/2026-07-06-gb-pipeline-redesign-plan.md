# Implementation Plan: GB Studio Hardware Pipeline

**Spec:** `docs/superpowers/specs/2026-07-06-gb-pipeline-redesign-design.md` — read it first; it is the authority on behavior. This plan defines the work breakdown, signatures, and acceptance criteria. Do not deviate from signatures without strong reason.

**Environment:** Python venv at `./venv`. Run tests with `./venv/bin/python -m pytest tests/ -x -q`. The Gradio app is `main.py`; the new module is `gb_pipeline.py` (no Gradio imports allowed in it).

**Git:** work on branch `gb-pipeline-redesign`. Commit after each task with a conventional message.

---

## Phase 0 — Setup

### Task 0.1: Test scaffolding
- Install pytest into venv if missing (`./venv/bin/pip install pytest`).
- Create `tests/__init__.py`, `tests/conftest.py` with fixtures: `checker_tile` (8×8 two-color checkerboard), `gradient_image` (160×144 RGB horizontal gradient), `flat_image` (160×144 solid), `photo_like_image` (160×144 deterministic multi-region gradient+noise, seeded numpy), `sample_palette_image` (loads `gb_palette.png`).
- Verify: `./venv/bin/python -m pytest tests/ -q` collects and passes (no tests yet is fine, use a trivial smoke test asserting fixtures load).

## Phase 1 — `gb_pipeline.py` core

All numeric randomness seeded (`random_state=42`). Color distance = Euclidean in CIELAB (`skimage.color.rgb2lab`). All palette colors RGB555-snapped. Palettes always sorted lightest→darkest by luminance `2126*R + 7152*G + 722*B` (descending).

### Task 1.1: Data model + primitives
Create `gb_pipeline.py` with:
```python
@dataclass
class GBImage:
    patterns: np.ndarray   # (n, 8, 8) uint8, values 0-3
    tilemap: np.ndarray    # (th, tw) int32 -> pattern index
    attrs_palette: np.ndarray  # (th, tw) uint8 palette id
    attrs_hflip: np.ndarray    # (th, tw) bool
    attrs_vflip: np.ndarray    # (th, tw) bool
    palettes: np.ndarray   # (p, 4, 3) uint8

@dataclass
class Preset:
    name: str; tile_budget: int | None; n_palettes: int
    allow_flips: bool; mono: bool; fixed_size: tuple | None

PRESETS = {
  "color_only": Preset("color_only", 384, 8, True, False, None),
  "mono":       Preset("mono", 192, 1, False, True, None),
  "logo_color": Preset("logo_color", None, 8, False, False, (160, 144)),
  "logo_mono":  Preset("logo_mono", None, 1, False, True, (160, 144)),
}

DMG_RAMP = np.array([[224,248,208],[136,192,112],[52,104,86],[8,24,32]], dtype=np.uint8)  # from gb_palette.png, lightest first

def snap_rgb555(colors: np.ndarray) -> np.ndarray  # c5=floor(c8*32/256) clamp 31; back via round(c5*255/31). uint8 in/out, any shape (...,3)
def luminance_sort(palette: np.ndarray) -> np.ndarray  # (4,3) -> (4,3) lightest first
def pattern_to_2bpp(pattern: np.ndarray) -> bytes  # 16 bytes, GB format: per row, byte1=low bits, byte2=high bits, MSB=leftmost pixel
def pattern_variants(pattern: np.ndarray) -> dict[str, np.ndarray]  # {"": p, "h": fliplr, "v": flipud, "hv": both}
```
- `reserve_ui_palette` option reduces `n_palettes` 8→7 at the entry point (Task 1.6), not in PRESETS.
- Tests (`tests/test_primitives.py`): RGB555 snap idempotence (snap(snap(x))==snap(x)); snap(255)==255 is False — assert actual formula values e.g. snap([255,255,255])==[255,255,255]? compute: floor(255*32/256)=31 → round(31*255/31)=255 ✓; snap([8,16,250]) matches hand-computed; luminance_sort orders white before black; pattern_to_2bpp of checkerboard matches hand-computed bytes; variants of asymmetric pattern are 4 distinct arrays, variants of symmetric pattern collide appropriately.

### Task 1.2: Palette packing (spec "Stage 3") — HARD, read spec section fully
```python
def quantize_working_set(image: Image.Image, max_colors: int, custom_palette: np.ndarray | None) -> Image.Image
    # PIL quantize (libimagequant, fall back to MEDIANCUT), dither NONE, then snap palette to RGB555; if custom_palette given, quantize to that set instead. Returns RGB-mode image containing only working-set colors.

def pack_palettes(image: np.ndarray, n_palettes: int, custom_palette: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]
    # image: (H, W, 3) uint8 already working-set-limited. H,W multiples of 8.
    # returns (palettes (p<=n_palettes, 4, 3) uint8 snapped+sorted, assignment (H//8, W//8) uint8)
```
Algorithm (per spec):
1. Per-tile ideal palette: unique colors if ≤4 else KMeans(4, random_state=42) on the tile's pixels in Lab, centers→RGB→snap.
2. Dedupe ideal palettes by frozenset of color tuples; track member tiles + per-color pixel counts.
3. Agglomerative: min-heap of pair merge costs. |union|≤4 → cost 0. Else fit 4 colors = weighted KMeans(4) in Lab over union colors weighted by pixel counts; cost = Σ over member-tile pixels of (new nearest-ΔE − old nearest-ΔE), computed on working-set colors × counts (not per-pixel). Lazy heap invalidation (skip popped pairs referencing dead palettes). Merge until count ≤ n_palettes.
4. Lloyd refinement, ≤5 iterations, stop when assignment unchanged: (a) assign each tile to argmin palette of Σ pixel-count-weighted ΔE(color, nearest palette entry); (b) refit each palette = weighted KMeans(4) in Lab over member tiles' colors×counts (keep old palette if a group is empty), snap RGB555, luminance_sort.
5. Mono path: `pack_palettes_mono(image, ramp)` → single palette = luminance-sorted ramp, assignment all zeros. (Callers map pixels by L channel rank later; keep it here for symmetry.)
- `custom_palette` restriction: in steps 1/3/4 candidate colors come only from the custom set (nearest-member projection after each fit).
- Tests (`tests/test_palettes.py`): flat image → 1 palette; 4-quadrant image with 4 colors/quadrant, 16 total colors → ≤4 palettes, zero reconstruction error; gradient 160×144 with n_palettes=7 → every tile's pixels within small ΔE of its assigned palette; invariants: shape ≤(7,4,3), all colors RGB555, all palettes luminance-sorted; determinism: two runs identical.

### Task 1.3: Indexing + Bayer dither (spec "Stage 4")
```python
BAYER4 = np.array([[0,8,2,10],[12,4,14,6],[3,11,1,9],[15,7,13,5]]) / 16.0

def index_tiles(image: np.ndarray, palettes: np.ndarray, assignment: np.ndarray, dither: str = "none") -> GBImage
```
- Per tile: convert pixels + its palette to Lab; nearest entry → index. `dither="bayer"`: per pixel find nearest c1 and second-nearest c2; t = d1/(d1+d2) (0 when exact); pixel gets c2's index iff t > BAYER4[y%4, x%4] where y,x are **absolute image coordinates**. No error diffusion.
- patterns initially = one per tile cell (dedup is next task); tilemap = arange; flips false.
- Tests (`tests/test_indexing.py`): flat tile → all one index; exact-palette-color pixels map to their index regardless of dither; dither on gradient produces both indices in mixed regions; absolute-coordinate anchoring: two identical tiles at different 8-aligned positions with same (y%4,x%4) phase produce identical patterns; output values all in 0..3.

### Task 1.4: Lossless dedup (spec "Stage 5")
```python
def dedup_patterns(gb: GBImage, allow_flips: bool) -> GBImage
```
- Iterate cells in raster order. Key = `pattern_to_2bpp(pattern)`. If allow_flips, on miss also try "h","v","hv" variants; on variant hit set the cell's flip bits. Rebuild compact patterns array + remapped tilemap.
- Tests (`tests/test_dedup.py`): image of one repeated tile → 1 pattern; tile + its mirror with allow_flips=True → 1 pattern + hflip set on the mirrored cell; same with allow_flips=False → 2 patterns; count matches independent set-of-bytes computation; idempotent.

### Task 1.5: Budget merge (spec "Stage 6") — HARD, read spec section fully
```python
def merge_to_budget(gb: GBImage, budget: int, allow_flips: bool) -> tuple[GBImage, int]  # (result, n_merges)
```
- If `len(patterns) <= budget`: return unchanged, 0.
- Signature per pattern: render under its usage-weighted average palette (weight = count of cells using it per palette id), to Lab, flatten (192,). Signatures are static across merges (patterns never change; only tilemap refs move).
- Distance(A,B) = min over B's variants (all 4 if allow_flips else identity) of mean squared Lab difference; remember best orientation.
- Cost(A→B) = distance(A,B) × usage(A). Min-heap over ordered pairs where usage(A) ≤ usage(B) at push time; lazy validation on pop (both alive, usage(A) still current — else recompute/repush).
- Merge: retarget all cells of A to B (compose flip bits with the best orientation), usage(B) += usage(A), mark A dead. Repeat until alive count ≤ budget. Rebuild compact GBImage.
- O(n²) signature distance matrix via vectorized numpy is acceptable to ~3–4k patterns; above that, KMeans-cluster signatures into ~√n groups and only push intra-cluster pairs (correctness fallback: if a cluster exhausts merges, recluster the survivors).
- Track p95 of per-pixel Lab distance over all merged cells → return in stats later (stash on the GBImage or return a third value; pick one and keep it consistent with Task 1.6).
- Tests (`tests/test_budget.py`): already-under-budget → unchanged, 0 merges; 100 distinct random tiles, budget 40 → exactly ≤40 (and ==40 if all distinct), tilemap references only live patterns; near-duplicate pairs merge before dissimilar ones (construct 3 groups of near-identical + distinct outliers, small budget → outliers survive); high-usage patterns survive vs single-use (usage weighting test); determinism.

### Task 1.6: Entry point, verify, render (spec "Stage 7"/"Stage 8")
```python
@dataclass
class ConversionResult:
    image: Image.Image            # final RGB render
    reference: Image.Image        # render before budget merge (natural comparison)
    stats: dict                   # tiles_used, tile_budget, palettes_used, n_merges, p95_delta_e, preset, dither
    warnings: list[str]
    palette_hex: list[list[str]]  # per palette, 4 hex strings

def render(gb: GBImage) -> Image.Image
def verify_roundtrip(png: Image.Image, preset: Preset, expected: GBImage) -> None
    # GB Studio-importer-equivalent: crop 8x8 tiles from png; per tile unique colors (assert <=4);
    # map colors->indices by luminance order; hash 2bpp (with flip variants if preset.allow_flips);
    # assert unique count == len(expected.patterns) and <= budget (skip budget when None);
    # assert distinct tile color-sets <= preset.n_palettes; assert every color RGB555-stable.
def convert_for_hardware(image: Image.Image, preset: str, *, tile_budget: int | None = None,
                         reserve_ui_palette: bool = True, dither: str = "none",
                         custom_palette: np.ndarray | None = None, mono_ramp: np.ndarray | None = None) -> ConversionResult
```
- convert_for_hardware: RGB-convert → crop to multiples of 8 (warning if cropped) → resize to fixed_size for logo presets (warning if it changed) → quantize_working_set(4×n_palettes) → pack → index (mono: ramp path) → dedup (skip for logo) → merge_to_budget (skip for logo/None) → verify_roundtrip → build result. `reserve_ui_palette` maps 8→7 for color presets. Warnings when p95_delta_e high (heavy merging) with actionable text per spec.
- Tests (`tests/test_convert.py`): property-style — seeded random images (5 seeds × {color_only, mono, logo_color}) → verify_roundtrip passes, stats within limits; logo preset output is exactly 160×144 and stats["tiles_used"] ≤ 360 with no merges; mono output uses only the 4 ramp colors; golden counts for `photo_like_image` pinned on first run (write the observed counts into the test with a comment).

## Phase 2 — `main.py` UI rework (spec "UI redesign")

### Task 2.1: Mode-first controls + wiring
- Add mode radio (Artistic | GB Studio: Color | GB Studio: Mono | GB Studio: Logo). Group existing artistic controls under Artistic visibility; build hardware panels: reserve-palette-8 checkbox (Color/Logo-color), dither dropdown (None/Bayer), tile budget number (Color: 384, Mono: 192; hidden for Logo), Logo Color/Mono sub-toggle, mono ramp via existing custom-palette image input.
- `process_image` routes hardware modes to `convert_for_hardware`; artistic path unchanged. Remove `quantize_for_GBC`/`use_tile_variance` module `gr.State`s — pass values as event inputs (`original_width/height` likewise).
- Remove UI: similarity-threshold slider, tile-variance checkbox, limit-4-colors checkbox, reduce-tiles checkbox. Rename "Use Logo Resolution" → "GB Screen (160×144)".
- Keep: batch folder flow (route through same function), queue/heartbeat wiring, analytics header, ko-fi/discord/github, gothic/grayscale/BW accordion (artistic only).
- Verify: `./venv/bin/python -c "import main"` succeeds; manual smoke: launch app, convert `gb_palette.png`-based test image in each mode without exceptions (drive with `demo.launch(prevent_thread_lock=True)` + `gradio_client` in a script if feasible, else document manual check performed via the process_image function called directly with each mode).

### Task 2.2: Output panel
- Notice line shows verified stats: "✅ 337/384 tiles · 7 palettes · 214 merges" (or "no tile limit" for Logo; warnings appended).
- Palette report: HTML swatches (colored div per color, hex text under) replacing the text dump; keep copyable hex via the existing textbox as fallback below.
- Reference image output = `ConversionResult.reference` in hardware modes.
- Verify: direct calls to the handler return populated stats/swatch HTML for each mode.

### Task 2.3: Dead code removal
- Delete: both `reduce_tiles_index`, `reduce_tiles`, `reduce_tiles_index_palette_aware`, both `tile_similarity_indexed`, `normalize_tile`, `map_pattern_to_palette`, `process_tiles`, `analyze_and_construct_palettes`, `create_refined_palettes`, `find_best_matching_palette`, `apply_palette`, `apply_mapped_colors_to_tile`, `replace_tile_palettes`, `convert_to_rgb_with_four_colors`, `generate_palette`, `get_color_distribution`, solid-tile helpers, and now-unused imports/mid-function duplicate imports. Keep gothic filter + artistic helpers.
- Verify: `import main` clean; grep confirms no references to deleted names; full pytest green.

## Phase 3 — Final verification

### Task 3.1: End-to-end + perf
- `tests/test_e2e.py`: convert 160×144 and 320×288 seeded images through all four presets; assert invariants + wall time <10s each (mark perf test with generous bound, CI-safe).
- Run full suite; run the app import smoke; produce a short summary of stats per preset for the final report.

---

**Definition of done per task:** implementation + its tests written, full `pytest` green, committed on `gb-pipeline-redesign`. Never mark done with failing tests.
