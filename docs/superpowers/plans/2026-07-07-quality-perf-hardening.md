# GB Pipeline Quality, Performance & Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. (This plan is being executed by a Workflow of Opus/Sonnet agents, one task per agent, strictly in task order; each agent implements exactly one task, runs the full test suite, and commits.)

**Goal:** Implement all findings from the 2026-07-07 pipeline review: two large image-quality fixes (dither against original pixels; a dedicated mono luminance path), the pack_palettes/index_tiles performance work, security/robustness cleanup in main.py, input-geometry polish, and restore the intended two-output semantics (left = hardware/custom-palette result, right = converter's natural color choices) including fixing the "custom palette collapses GBC mode to 4 colors" regression.

**Architecture:** All pipeline changes stay inside `gb_pipeline.py` (pure functions, no Gradio imports); UI/wiring changes stay in `main.py`. Every task is one commit, verified by the full pytest suite (`venv/bin/python -m pytest tests/ -q`, currently 101 passing). Quality changes land before performance changes so the perf work optimizes final semantics. A benchmark script is added mid-plan and used for perf acceptance.

**Tech Stack:** Python 3.12, numpy, scikit-image (rgb2lab/lab2rgb), PIL/Pillow, Gradio, pytest.

**User decisions (already made):**
- "Plan all of these and set up a workflow of opus/sonnet to execute" — execution via Workflow subagents; Fable does planning and final review only.
- Left output image = the GBC/custom-palette conversion; right output = the converter's own (natural/free) color choices. Current behavior (right = pre-budget-merge render) is wrong.
- "Custom palette ticked on, even on GBC mode, seems to only produce those 4 colours" — confirmed root cause: `gb_palette.png` (4 DMG colors) is the default palette image and `use_custom_palette` defaults to True, so Color mode's working set is restricted to 4 colors. Must be smoothed out per Task 2.
- All 2026-07-07 review findings are in scope, including the optional Stage-6 signature refresh.

**Baseline numbers (must not regress, perf targets reference these):** photo-like content, `convert_for_hardware`: 160×144 color_only ≈ 2.05s; 320×288 color_only ≈ 4.97s; 320×288 mono ≈ 0.39s. Test suite: 101 passed in ~51s.

**Global rules for every task:**
- Run the FULL suite before starting (must be green) and after finishing: `venv/bin/python -m pytest tests/ -q`.
- Never weaken an existing test's intent. If a test asserts behavior this plan deliberately changes, update the test to assert the *new* specified behavior and say so in the commit message.
- One commit per task, message format given in each task. Work on branch `gb-pipeline-redesign`.
- Repo root: `/home/liam/projects/gameboy-image-converter`. Python: `venv/bin/python`.

---

### Task 1: main.py security & robustness cleanup

**Goal:** Remove the committed Discord webhook secret and stale password comment, replace `os.system("rm -rf ...")` with `shutil.rmtree`, make `active_tasks` thread-safe, and fix the stale cleanup comment.

**Files:**
- Modify: `main.py:49-52` (webhook), `main.py:997` (stale comment), `main.py:732` and `main.py:968-979` (rm -rf), `main.py:63-101` (active_tasks), `main.py:983` (comment)
- Test: `tests/test_main_ui.py` (add one test)

**Acceptance Criteria:**
- [ ] `grep -n "discord.com/api/webhooks" main.py` returns nothing; `HEARTBEAT_WEBHOOK_URL` comes only from the environment (empty default), and the existing `if HEARTBEAT_WEBHOOK_URL:` guard still disables the monitor when unset.
- [ ] `grep -n "boobiess" main.py` returns nothing.
- [ ] `grep -n 'os.system' main.py` returns nothing; both cleanup sites use `shutil.rmtree(path, ignore_errors=True)`.
- [ ] `active_tasks` increment/decrement guarded by a `threading.Lock`.
- [ ] Full suite passes.

**Verify:** `venv/bin/python -m pytest tests/ -q` → all pass; the three greps above are empty.

**Steps:**

- [ ] **Step 1: Write the failing test** in `tests/test_main_ui.py`:

```python
def test_no_committed_secrets_or_shell_deletes():
    src = Path(main.__file__).read_text()
    assert "discord.com/api/webhooks" not in src
    assert "boobiess" not in src
    assert "os.system" not in src
```

(Match the file's existing import style; it already imports `main`. Add `from pathlib import Path` if missing.) Run: `venv/bin/python -m pytest tests/test_main_ui.py -q` → the new test FAILS.

- [ ] **Step 2: Fix the webhook.** Replace `main.py:49-52` with:

```python
HEARTBEAT_WEBHOOK_URL = os.environ.get("HEARTBEAT_WEBHOOK_URL", "")
```

- [ ] **Step 3: Remove the stale auth comment** at `main.py:997` (`# use http basic auth with password of boobiess`) — delete the line entirely.

- [ ] **Step 4: Replace shell deletes.** Add `import shutil` to the imports. At `main.py:732` replace `os.system("rm -rf " + folder_name)` with `shutil.rmtree(folder_name, ignore_errors=True)`. In `clear_temporary_files` replace the `os.system("rm -rf " + folder)` call with `shutil.rmtree(folder, ignore_errors=True)` (keep the surrounding try/except). Fix the comment above it: it says "more than 1 hour ago" but the code uses 600s — change the comment to "more than 10 minutes ago".

- [ ] **Step 5: Thread-safe active_tasks.** Next to `active_tasks = 0` add `active_tasks_lock = threading.Lock()`. In `task_log`, wrap both mutations:

```python
    with active_tasks_lock:
        active_tasks += 1
        active_snapshot = active_tasks
```

and in the `finally` block:

```python
        with active_tasks_lock:
            active_tasks = max(active_tasks - 1, 0)
            active_snapshot = active_tasks
```

Use `active_snapshot` in the two `json.dumps` log payloads instead of reading the global again.

- [ ] **Step 6: Run full suite** → all pass. Commit:

```bash
git add main.py tests/test_main_ui.py
git commit -m "fix: remove committed webhook secret, shell deletes, and active_tasks race"
```

**Post-merge note for the user (not automatable):** the old webhook is in git history — rotate/delete it in Discord.

---

### Task 2: Restore two-output semantics + fix the custom-palette 4-color collapse

**Goal:** In GB Studio hardware modes, the left pane ("Output Image") shows the custom-palette-restricted conversion and the right pane ("Output Image (Natural Palette)") shows the converter's OWN color choices — a full conversion run without the custom palette. Also: warn when a custom palette restricts a color preset below its capacity, and stop the default 4-color `gb_palette.png` from silently collapsing Color mode (untick "Use Custom Color Palette" when the user switches into Color mode).

**Background (investigated 2026-07-07, pre-redesign commit `a8b5dc0`):** the old UI's right pane was always a genuinely *natural* render (`limit_colors` with no palette image — the image's own adaptive palette). The redesign repurposed it as the pre-budget-merge reference, and `quantize_working_set`'s custom-palette branch is a whole-image nearest-color clamp, so the default ticked checkbox + 4-color `gb_palette.png` collapses Color mode (capacity 28-32 colors) to 4 colors in BOTH panes. The clamp semantics themselves are correct for a deliberate restriction set and are kept; what's restored is the natural comparison pane, plus guardrails around the trap default.

**Files:**
- Modify: `main.py` (`process_image` hardware path ~651-682; new mode-change handler + wiring ~898-905)
- Modify: `gb_pipeline.py` (`convert_for_hardware`: capacity warning)
- Test: `tests/test_main_ui.py`, `tests/test_convert.py`

**Acceptance Criteria:**
- [ ] Hardware modes with an active custom palette run the pipeline twice: left = restricted `result.image`, right = unrestricted `natural.image` (same preset/budget/dither/reserve flags, `custom_palette=None`). With no active custom palette, behavior is unchanged (right = `result.reference`, the pre-merge render).
- [ ] Mono modes are untouched (the palette image is a mono *ramp* there, not a restriction set — single run as today).
- [ ] `convert_for_hardware` appends a warning when `custom_palette` is given for a non-mono preset and its snapped unique-color count `k < 4 * n_palettes` (effective, i.e. after UI-palette reservation): `f"Custom palette restricts output to {k} colors (this preset supports up to {4 * n_palettes}). The right pane shows the converter's own choices."`
- [ ] Switching the mode radio TO "GB Studio: Color" unticks `use_custom_palette` (one-way; switching elsewhere leaves it alone; logo-subtype changes never touch it).
- [ ] Full suite passes.

**Verify:** `venv/bin/python -m pytest tests/test_main_ui.py tests/test_convert.py -q` then the full suite.

**Steps:**

- [ ] **Step 1: Failing pipeline-level test** in `tests/test_convert.py`:

```python
def test_custom_palette_capacity_warning():
    rng = np.random.RandomState(3)
    arr = np.repeat(np.repeat(rng.randint(0, 256, (4, 4, 3)).astype(np.uint8), 8, 0), 8, 1)
    img = Image.fromarray(arr, "RGB")
    dmg = gb_pipeline.DMG_RAMP
    res = gb_pipeline.convert_for_hardware(img, "color_only", custom_palette=dmg)
    assert any("restricts output to 4 colors" in w for w in res.warnings)
    res_free = gb_pipeline.convert_for_hardware(img, "color_only")
    assert not any("restricts output" in w for w in res_free.warnings)
```

- [ ] **Step 2: Failing UI-level tests** in `tests/test_main_ui.py` (reuse `_call_process_image` and `_small_test_image`):

```python
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
        assert "value" not in upd or upd.get("__type__") == "update" and "value" not in {k: v for k, v in upd.items() if k != "__type__"}
```

(For the last test: `gr.update()` with no kwargs produces a dict without `value`; assert simply `"value" not in main.on_mode_change_custom_palette(mode)` for the three non-color modes if that holds for the installed Gradio version — check `gr.update()`'s actual shape once and pin the simplest true assertion.)

- [ ] **Step 3: Capacity warning in `convert_for_hardware`.** After the `n_palettes` reservation block, add:

```python
    if custom_palette is not None and not ps.mono:
        k = int(np.unique(
            snap_rgb555(np.asarray(custom_palette, dtype=np.uint8).reshape(-1, 3)),
            axis=0).shape[0])
        if k < 4 * n_palettes:
            warnings.append(
                f"Custom palette restricts output to {k} colors (this preset "
                f"supports up to {4 * n_palettes}). The right pane shows the "
                "converter's own choices."
            )
```

- [ ] **Step 4: Dual-run in `process_image`.** Replace the tail of the hardware branch:

```python
        result = gb_pipeline.convert_for_hardware(
            image, preset,
            tile_budget=budget,
            reserve_ui_palette=bool(reserve_ui_palette),
            dither=dither_key,
            custom_palette=custom_palette_arr,
            mono_ramp=mono_ramp_arr,
        )

        if custom_palette_arr is not None:
            # Right pane: the converter's own color choices, same settings
            # minus the custom-palette restriction.
            natural = gb_pipeline.convert_for_hardware(
                image, preset,
                tile_budget=budget,
                reserve_ui_palette=bool(reserve_ui_palette),
                dither=dither_key,
                custom_palette=None,
                mono_ramp=mono_ramp_arr,
            )
            reference_out = natural.image
        else:
            reference_out = result.reference

        notice = _format_hardware_notice(result)
        palette_text = _format_palette_text(result.palette_hex)
        palette_html = _format_palette_html(result.palette_hex)
        return result.image, palette_text, reference_out, notice, palette_html
```

- [ ] **Step 5: One-way untick on entering Color mode.** Add to `main.py`:

```python
def on_mode_change_custom_palette(mode):
    """Entering GB Studio: Color unticks the custom palette by default: the
    bundled gb_palette.png is a 4-color DMG ramp, which would clamp a 28-32
    color preset to 4 colors. One-way -- other modes leave the box alone."""
    if mode == MODE_COLOR:
        return gr.update(value=False)
    return gr.update()
```

Wire it on `mode_radio.change` ONLY (not `logo_subtype_radio.change`):

```python
                mode_radio.change(fn=on_mode_change_custom_palette, inputs=[mode_radio],
                                  outputs=[use_custom_palette])
```

Place it next to the existing `mode_radio.change` wirings. NOTE: `use_custom_palette` is defined in a `gr.Group` after the mode wiring block — if the component isn't in scope at that point, move this `.change` wiring below the `use_custom_palette` definition (Gradio wiring order within `Blocks` doesn't matter functionally).

- [ ] **Step 6: Full suite; the existing `test_process_image_each_simple_mode[GB Studio: Color]` passes `use_custom_palette=True` and now dual-runs — confirm it still passes (slower but valid). Commit:**

```bash
git add main.py gb_pipeline.py tests/
git commit -m "feat: natural-palette right pane for hardware modes + custom-palette capacity guardrails"
```

---

### Task 3: Dedicated mono luminance path (continuous-tone Bayer)

**Goal:** Mono (and logo_mono) stop quantizing to 4 colors before dithering. Instead: original pixels → relative luminance → percentile contrast stretch → even-spaced 4-level ordered dither (or rounding when dither is off) → ramp indices. This restores real gradient dithering (the "GB camera" look).

**Files:**
- Modify: `gb_pipeline.py:1203-1252` (`_index_tiles_mono`), `gb_pipeline.py:1314-1327` (`convert_for_hardware` mono branch)
- Test: `tests/test_indexing.py`, `tests/test_e2e.py`

**Acceptance Criteria:**
- [ ] `_index_tiles_mono` takes the ORIGINAL (post-crop/resize) RGB array, not a quantized one; `convert_for_hardware` no longer calls `quantize_working_set` for mono presets.
- [ ] With `dither="bayer"`, a smooth horizontal gradient produces mixed adjacent ramp levels in the transition zones (checkerboard-style), and mean output level is monotonic along the gradient axis.
- [ ] With `dither="none"`, the same gradient produces exactly 4 clean bands.
- [ ] A constant-color image does not crash (degenerate percentile range) and maps to a single ramp level.
- [ ] Output pixels are always exactly ramp colors; `verify_roundtrip` still passes in e2e tests.
- [ ] Full suite passes (update any test that asserted the old quantize-first mono behavior, preserving each test's intent).

**Verify:** `venv/bin/python -m pytest tests/test_indexing.py tests/test_e2e.py -q` then the full suite.

**Steps:**

- [ ] **Step 1: Write failing tests** in `tests/test_indexing.py`:

```python
def test_mono_bayer_dithers_continuous_gradient():
    w, h = 160, 8
    grad = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    arr = np.stack([grad] * 3, axis=2)
    ramp = gb_pipeline.DMG_RAMP
    palettes = np.asarray(gb_pipeline.luminance_sort(gb_pipeline.snap_rgb555(ramp)))[None]
    gb = gb_pipeline._index_tiles_mono(arr, palettes, dither="bayer")
    idx = np.concatenate([gb.patterns[t] for t in range(gb.patterns.shape[0])], axis=1)
    # Monotonic mean index (lightest-first ramp: index falls as x brightens? No —
    # gradient goes dark->light left->right, ramp index 0 is lightest, so mean
    # index must be non-increasing along x, allowing dither noise of +-0.6).
    col_means = idx.mean(axis=0)
    smooth = np.convolve(col_means, np.ones(8) / 8, mode="valid")
    assert np.all(np.diff(smooth) <= 0.15)
    # Transition zones actually mix two adjacent levels.
    mid = idx[:, w // 3 : 2 * w // 3]
    assert len(np.unique(mid)) >= 2

def test_mono_no_dither_gives_clean_bands():
    w, h = 160, 8
    grad = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    arr = np.stack([grad] * 3, axis=2)
    palettes = np.asarray(gb_pipeline.DMG_RAMP)[None]
    gb = gb_pipeline._index_tiles_mono(arr, palettes, dither="none")
    idx = np.concatenate([gb.patterns[t] for t in range(gb.patterns.shape[0])], axis=1)
    # Every column is a single level; 4 bands total.
    assert all(len(np.unique(idx[:, c])) == 1 for c in range(w))
    assert len(np.unique(idx)) == 4

def test_mono_constant_image_no_crash():
    arr = np.full((16, 16, 3), 137, dtype=np.uint8)
    palettes = np.asarray(gb_pipeline.DMG_RAMP)[None]
    gb = gb_pipeline._index_tiles_mono(arr, palettes, dither="bayer")
    assert len(np.unique(gb.patterns)) == 1
```

Run them: they FAIL against the current implementation (current code maps by nearest ramp luminance of already-quantized colors; the gradient tests will fail because `_index_tiles_mono` currently receives raw `arr` fine, but produces `t_val` from absolute luminance distances — the no-dither bands test may pass; keep whichever fail as the red state and keep all three as regression tests).

- [ ] **Step 2: Reimplement `_index_tiles_mono`** (replace the luminance-nearest logic; keep signature and GBImage construction):

```python
def _index_tiles_mono(
    image: np.ndarray, palettes: np.ndarray, dither: str = "none"
) -> GBImage:
    """Index pixels to the mono ramp by contrast-stretched luminance (Stage 4, mono).

    ``image`` is the ORIGINAL (post-crop/resize) RGB array -- mono deliberately
    skips the working-set quantizer so ordered dithering sees continuous tone.
    Pixel luminance is percentile-stretched (1st..99th -> 0..1) and quantized to
    the 4 ramp levels evenly: x = lum_n * 3, base = floor(x), and the pixel
    takes ``base + 1`` iff the fractional part exceeds the threshold (0.5 when
    dither="none", else the absolute-coordinate 4x4 Bayer threshold). Ramp
    index 0 is lightest, so the stored index is ``3 - level``.
    """
    if dither not in ("none", "bayer"):
        raise ValueError(f"unknown dither mode: {dither!r}")
    arr = np.asarray(image, dtype=np.uint8)
    h, w = arr.shape[:2]
    th, tw = h // 8, w // 8

    ramp = np.asarray(palettes, dtype=np.uint8)[0]  # (4, 3), lightest first

    lum = _pixel_luminance(arr)  # (h, w) float
    lo, hi = np.percentile(lum, [1.0, 99.0])
    if hi - lo < 1e-9:
        lum_n = np.full_like(lum, 0.5, dtype=np.float64)
    else:
        lum_n = np.clip((lum - lo) / (hi - lo), 0.0, 1.0)

    x = lum_n * 3.0
    base = np.minimum(np.floor(x), 2.0)
    frac = x - base
    if dither == "bayer":
        yy = np.arange(h)[:, None]
        xx = np.arange(w)[None, :]
        thresh = BAYER4[yy % 4, xx % 4]
    else:
        thresh = 0.5
    level = base + (frac > thresh)
    level = np.clip(level, 0, 3)
    index = (3 - level).astype(np.uint8)  # ramp is lightest-first

    n_tiles = th * tw
    patterns = np.zeros((n_tiles, 8, 8), dtype=np.uint8)
    for tr in range(th):
        for tc in range(tw):
            patterns[tr * tw + tc] = index[tr * 8 : tr * 8 + 8, tc * 8 : tc * 8 + 8]

    return GBImage(
        patterns=patterns,
        tilemap=np.arange(n_tiles, dtype=np.int32).reshape(th, tw),
        attrs_palette=np.zeros((th, tw), dtype=np.uint8),
        attrs_hflip=np.zeros((th, tw), dtype=bool),
        attrs_vflip=np.zeros((th, tw), dtype=bool),
        palettes=ramp[None, :, :].astype(np.uint8),
    )
```

Note `frac > thresh` with `dither="none"` and `frac` exactly 0.5 rounds down — fine, deterministic.

- [ ] **Step 3: Skip working-set quantization for mono** in `convert_for_hardware`. Replace the current block:

```python
    quantized = quantize_working_set(img, 4 * n_palettes, custom_palette)
    arr = np.asarray(quantized, dtype=np.uint8)

    if ps.mono:
        ramp = DMG_RAMP if mono_ramp is None else np.asarray(mono_ramp, dtype=np.uint8)
        palettes, _ = pack_palettes_mono(arr, ramp)
        gb = _index_tiles_mono(arr, palettes, dither)
    else:
        palettes, assignment = pack_palettes(arr, n_palettes, custom_palette)
        gb = index_tiles(arr, palettes, assignment, dither)
```

with:

```python
    orig_arr = np.asarray(img, dtype=np.uint8)

    if ps.mono:
        # Mono skips the working-set quantizer entirely: the luminance path
        # dithers continuous tone straight to the 4-level ramp.
        ramp = DMG_RAMP if mono_ramp is None else np.asarray(mono_ramp, dtype=np.uint8)
        palettes, _ = pack_palettes_mono(orig_arr, ramp)
        gb = _index_tiles_mono(orig_arr, palettes, dither)
    else:
        quantized = quantize_working_set(img, 4 * n_palettes, custom_palette)
        arr = np.asarray(quantized, dtype=np.uint8)
        palettes, assignment = pack_palettes(arr, n_palettes, custom_palette)
        gb = index_tiles(arr, palettes, assignment, dither)
```

- [ ] **Step 4: Fix broken existing tests.** Run the full suite. Tests that quantized mono inputs or asserted nearest-luminance behavior must be updated to the new spec above (contrast-stretched even levels). Do not delete tests; rewrite assertions to the new behavior while keeping what they guard (e.g. "output only contains ramp colors", "verify_roundtrip passes").

- [ ] **Step 5: Full suite green → commit:**

```bash
git add gb_pipeline.py tests/
git commit -m "feat: mono converts continuous luminance with contrast stretch, skipping working-set quantize"
```

---

### Task 4: Color/logo dithering & indexing against original pixels

**Goal:** `index_tiles` receives the original (post-crop/resize) pixels instead of the quantized working-set image, so nearest-entry mapping and Bayer dithering see real gradients. Palette packing still runs on the quantized working set.

**Files:**
- Modify: `gb_pipeline.py` (`convert_for_hardware` color branch; `index_tiles` docstring)
- Test: `tests/test_e2e.py`

**Acceptance Criteria:**
- [ ] `convert_for_hardware` passes the original array to `index_tiles`; `pack_palettes` still gets the quantized array.
- [ ] New e2e test proves fidelity: for a photo-like gradient fixture, mean per-pixel Lab error between final render and the ORIGINAL image is strictly lower when indexing from original pixels than from the quantized image (computed by calling `index_tiles` both ways with identical palettes/assignment).
- [ ] `index_tiles` docstring no longer claims the input must be working-set-limited.
- [ ] Full suite passes.

**Verify:** `venv/bin/python -m pytest tests/test_e2e.py -q` then the full suite.

**Steps:**

- [ ] **Step 1: Write the failing e2e test** in `tests/test_e2e.py`:

```python
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

    gb_orig = gb_pipeline.index_tiles(arr, palettes, assignment, dither="bayer")
    gb_quant = gb_pipeline.index_tiles(qarr, palettes, assignment, dither="bayer")
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
    res = gb_pipeline.convert_for_hardware(img, "color_only", dither="bayer")
    quant = gb_pipeline.quantize_working_set(img, 28)
    qarr = np.asarray(quant, dtype=np.uint8)
    palettes, assignment = gb_pipeline.pack_palettes(qarr, 7)
    gb_quant = gb_pipeline.index_tiles(qarr, palettes, assignment, dither="bayer")
    err_pipeline = _mean_lab_error(np.asarray(res.image), arr)
    err_quant = _mean_lab_error(np.asarray(gb_pipeline.render(gb_quant)), arr)
    assert err_pipeline < err_quant
```

(The first test passes already — `index_tiles` is input-agnostic; it documents the property. The second FAILS until the pipeline change lands. Note: `convert_for_hardware` runs dedup + budget merge after indexing, which can only *increase* error; if the strict `<` in the second test is flaky because of that, compare against `err_quant + 0.0` computed after running the same dedup/merge on `gb_quant` via `gb_pipeline.dedup_patterns` + `gb_pipeline.merge_to_budget` with the same preset budget — mirror the pipeline exactly.)

- [ ] **Step 2: Make the pipeline change.** In `convert_for_hardware`'s color branch (as restructured by Task 3), change `index_tiles(arr, palettes, assignment, dither)` to `index_tiles(orig_arr, palettes, assignment, dither)` where `orig_arr = np.asarray(img, dtype=np.uint8)` (already defined by Task 3).

- [ ] **Step 3: Update `index_tiles` docstring** — replace "(already working-set/palette-limited)" with "(any RGB content; typically the original post-crop pixels — palette assignment is computed from the quantized working set, but per-pixel indexing/dithering is most faithful from the source)".

- [ ] **Step 4: Full suite; fix any test asserting the old source semantics (preserve intent). Commit:**

```bash
git add gb_pipeline.py tests/
git commit -m "feat: index and dither color tiles from original pixels, not the quantized working set"
```

---

### Task 5: Dither mix ratio by segment projection

**Goal:** Replace `t = d1/(d1+d2)` in `index_tiles`'s Bayer branch with the pixel's projection onto the Lab segment between its two nearest palette entries, clamped to [0, 1]. Off-axis and out-of-range pixels stop speckling.

**Files:**
- Modify: `gb_pipeline.py:639-648` (Bayer branch of `index_tiles`)
- Test: `tests/test_indexing.py`

**Acceptance Criteria:**
- [ ] `t` is computed as `dot(pix - c1, c2 - c1) / ||c2 - c1||²` in Lab, clamped to [0, 1], with a guard for `||c2 - c1||² == 0` (t = 0).
- [ ] A pixel "beyond" its nearest entry (e.g. lighter than the lightest palette color) never dithers to the second entry.
- [ ] A pixel exactly midway on the segment still dithers ~50%.
- [ ] Full suite passes.

**Verify:** `venv/bin/python -m pytest tests/test_indexing.py -q` then the full suite.

**Steps:**

- [ ] **Step 1: Failing test** in `tests/test_indexing.py`:

```python
def test_projection_dither_no_speckle_beyond_endpoints():
    # Palette: mid-gray and dark-gray. Pixels are pure white -- beyond the
    # light end of the segment. Old d1/(d1+d2) ratio dithered these; the
    # projection must clamp t to 0 -> no dithering at all.
    pal = np.array([[[128, 128, 128], [128, 128, 128],
                     [64, 64, 64], [0, 0, 0]]], dtype=np.uint8)
    pal = gb_pipeline.snap_rgb555(pal)
    arr = np.full((8, 8, 3), 255, dtype=np.uint8)
    assignment = np.zeros((1, 1), dtype=np.uint8)
    gb = gb_pipeline.index_tiles(arr, pal, assignment, dither="bayer")
    assert len(np.unique(gb.patterns)) == 1  # all pixels -> single nearest entry

def test_projection_dither_midpoint_mixes():
    pal = np.array([[[200, 200, 200], [200, 200, 200],
                     [40, 40, 40], [40, 40, 40]]], dtype=np.uint8)
    pal = gb_pipeline.snap_rgb555(pal)
    lab = gb_pipeline._rgb_to_lab(np.unique(pal.reshape(-1, 3), axis=0))
    mid_lab = lab.mean(axis=0, keepdims=True)
    mid_rgb = gb_pipeline._lab_to_rgb_u8(mid_lab)[0]
    arr = np.tile(mid_rgb, (8, 8, 1)).astype(np.uint8)
    assignment = np.zeros((1, 1), dtype=np.uint8)
    gb = gb_pipeline.index_tiles(arr, pal, assignment, dither="bayer")
    vals, counts = np.unique(gb.patterns, return_counts=True)
    assert len(vals) == 2
    assert 0.25 <= counts[0] / counts.sum() <= 0.75
```

Run → first test FAILS on current code (white vs {128,64,0} grays: d1/(d1+d2) > 0 so some Bayer cells flip).

- [ ] **Step 2: Implement.** In `index_tiles`'s `if dither == "bayer":` branch, replace the `d1/d2/t_val` computation with:

```python
                idx2 = order[:, :, 1]
                c1 = np.take_along_axis(
                    plab[None, None, :, :].repeat(8, 0).repeat(8, 1),
                    idx1[:, :, None, None].repeat(3, 3), axis=2)[:, :, 0, :]
                c2 = np.take_along_axis(
                    plab[None, None, :, :].repeat(8, 0).repeat(8, 1),
                    idx2[:, :, None, None].repeat(3, 3), axis=2)[:, :, 0, :]
                seg = c2 - c1                          # (8,8,3)
                seg_len2 = np.sum(seg * seg, axis=2)   # (8,8)
                proj = np.sum((blab - c1) * seg, axis=2)
                t_val = np.where(seg_len2 > 0, proj / np.where(seg_len2 > 0, seg_len2, 1.0), 0.0)
                t_val = np.clip(t_val, 0.0, 1.0)
```

Simpler equivalent gathers are fine (`plab[idx1]` fancy-indexes to (8,8,3) directly: `c1 = plab[idx1]`, `c2 = plab[idx2]` — prefer that form). Keep the existing threshold comparison lines unchanged.

- [ ] **Step 3: Full suite (existing dither-distribution tests may need retuning to the projection semantics — preserve intent). Commit:**

```bash
git add gb_pipeline.py tests/test_indexing.py
git commit -m "feat: Bayer dither mixes by Lab segment projection, killing off-axis speckle"
```

---

### Task 6: pack_palettes performance (proxy heap + k-means micro-opts)

**Goal:** Cut the dominant agglomeration cost: stop running an exact weighted k-means for every candidate pair up front. Seed the merge heap with a cheap union-of-centers proxy cost and compute the exact cost lazily on pop. Also: deduplicate the double k-means in `finalize`, replace `np.allclose` in the Lloyd loop, and replace `rng.choice(p=...)` with cumsum+searchsorted.

**Files:**
- Modify: `gb_pipeline.py` (`_weighted_kmeans_lab`, `pack_palettes` internals: `finalize`, `merge_cost`, `agglomerate`)
- Create: `scripts/bench_pipeline.py`
- Test: `tests/test_palettes.py`

**Acceptance Criteria:**
- [ ] `finalize` runs weighted k-means at most ONCE per group (centers reused for both palette and error).
- [ ] Groups store their fitted Lab centers (`g["centers"]`); heap seeding uses a proxy cost (error of combined colors against the union of both groups' stored centers, minus the two self-errors, floored at 0) with NO k-means call; exact `merge_cost` runs only when a proxy entry is popped, and the entry is re-pushed with its exact cost (marked exact) rather than merged immediately. Only exact-marked entries may trigger a merge.
- [ ] `_weighted_kmeans_lab` convergence check is a direct `np.max(np.abs(new_centers - centers)) < 1e-9` (no `np.allclose`); k-means++ sampling uses `np.searchsorted(np.cumsum(p), rng.random_sample() * p.sum())` instead of `rng.choice(..., p=...)` (still fully deterministic under the seed).
- [ ] `scripts/bench_pipeline.py` exists (code below) and 320×288 color_only lands ≤ 2.5s (baseline 4.97s) on the same machine.
- [ ] Full suite passes. Palette-quality tests may shift numerically (different RNG draws / merge order); update only assertions that check exact colors, never ones that check error bounds or hardware constraints — those must still hold.

**Verify:** `venv/bin/python -m pytest tests/ -q` and `venv/bin/python scripts/bench_pipeline.py` → printed 320×288 color_only time ≤ 2.5s.

**Steps:**

- [ ] **Step 1: Commit the benchmark script** `scripts/bench_pipeline.py`:

```python
"""Reproducible pipeline benchmark. Run: venv/bin/python scripts/bench_pipeline.py"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from PIL import Image
import gb_pipeline

rng = np.random.RandomState(0)

def photo(w, h):
    yy, xx = np.mgrid[0:h, 0:w]
    r = xx / w * 255
    g = yy / h * 255
    b = (np.sin(xx / 9.0) + np.cos(yy / 7.0)) * 60 + 120
    arr = np.stack([r, g, b], axis=2) + rng.normal(0, 14, (h, w, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")

if __name__ == "__main__":
    for name, (w, h) in [("160x144", (160, 144)), ("320x288", (320, 288))]:
        img = photo(w, h)
        for preset, kw in [("color_only", dict(tile_budget=192, dither="bayer")),
                           ("mono", dict(dither="bayer"))]:
            t0 = time.perf_counter()
            res = gb_pipeline.convert_for_hardware(img, preset, **kw)
            dt = time.perf_counter() - t0
            print(f"{name} {preset:11s} {dt:6.2f}s  tiles={res.stats['tiles_used']}"
                  f" merges={res.stats['n_merges']} p95={res.stats['p95_delta_e']:.1f}")
```

Run it once and record the pre-change numbers in the commit message of Step 6.

- [ ] **Step 2: Add a perf-shape regression test** in `tests/test_palettes.py` (counts k-means calls, not wall time — wall time is machine-dependent):

```python
def test_agglomerate_kmeans_call_budget(monkeypatch):
    calls = {"n": 0}
    real = gb_pipeline._weighted_kmeans_lab
    def counting(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)
    monkeypatch.setattr(gb_pipeline, "_weighted_kmeans_lab", counting)
    rng = np.random.RandomState(0)
    arr = rng.randint(0, 255, (96, 96, 3)).astype(np.uint8)
    arr = np.asarray(gb_pipeline.quantize_working_set(Image.fromarray(arr, "RGB"), 28))
    gb_pipeline.pack_palettes(arr, 7)
    # Pre-change this is O(pairs) ~ thousands; the lazy-proxy heap must keep it
    # within a small multiple of the number of merges (144 tiles -> < 800).
    assert calls["n"] < 800
```

Run → FAILS pre-change (record the observed count in a comment).

- [ ] **Step 3: `finalize` single-fit.** Restructure so `build_palette`'s k-means isn't repeated by `fit_centers_lab`:

```python
    def finalize(gid):
        g = groups[gid]
        idxs = np.array(sorted(g["colors"].keys()), dtype=np.int64)
        cnts = np.array([g["colors"][int(i)] for i in idxs], dtype=np.float64)
        centers = fit_centers_lab(idxs, cnts)
        g["centers"] = centers
        g["palette"] = palette_from_centers(idxs, centers)
        g["error"] = lab_error(idxs, cnts, centers)
```

Add `palette_from_centers(idxs, centers)`: identical to `build_palette` except it takes precomputed Lab centers — when `len(idxs) <= 4` use `ws_colors[idxs]` directly, else `_lab_to_rgb_u8(centers)`; then the same snap → custom-projection → luminance-sort tail. Keep `build_palette` for the per-tile Phase-1 seeding loop (which never needs the error), or refactor it to call `palette_from_centers` internally — either way no duplicated k-means anywhere.

- [ ] **Step 4: Lazy proxy heap in `agglomerate`.** Replace heap seeding and the pop loop:

```python
    def proxy_cost(a, b):
        ga, gb = groups[a], groups[b]
        comb = dict(ga["colors"])
        for i, c in gb["colors"].items():
            comb[i] = comb.get(i, 0.0) + c
        if len(comb) <= 4:
            return 0.0
        idxs = np.array(sorted(comb.keys()), dtype=np.int64)
        cnts = np.array([comb[int(i)] for i in idxs], dtype=np.float64)
        union = np.concatenate([ga["centers"], gb["centers"]], axis=0)
        err = lab_error(idxs, cnts, union)
        return max(err - (ga["error"] + gb["error"]), 0.0)
```

Heap entries become `(cost, exact, a, b)` where `exact` is a bool (`False` sorts before `True` for equal costs — fine). Seeding pushes `(proxy_cost(a, b), False, a, b)`. Pop loop:

```python
        while len(alive) > target:
            pair = None
            while heap:
                c, exact, a, b = heapq.heappop(heap)
                if a not in alive or b not in alive:
                    continue
                if not exact:
                    heapq.heappush(heap, (merge_cost(a, b), True, a, b))
                    continue
                pair = (a, b)
                break
            ...
```

`merge_cost` itself is unchanged (still the exact k-means fit). New pairs created after a merge are pushed as proxies too. `comb ≤ 4` pairs are exact by construction (cost 0 either way) — push them with `exact=True` to skip a wasted cycle.

- [ ] **Step 5: k-means micro-opts** in `_weighted_kmeans_lab`:
  - Convergence: replace `if np.allclose(new_centers, centers):` with `if float(np.max(np.abs(new_centers - centers))) < 1e-9:`.
  - Seeding draws: replace both `rng.choice(n, p=...)` calls with:

```python
def _weighted_draw(rng, p):
    cum = np.cumsum(p)
    return int(np.searchsorted(cum, rng.random_sample() * cum[-1], side="right").clip(0, len(p) - 1))
```

(module-level helper; `first = _weighted_draw(rng, probs)`, and in the loop `j = _weighted_draw(rng, p) if s > 0 else int(rng.randint(n))`). This changes which RNG draws happen — outputs shift slightly; that is expected and accepted.

- [ ] **Step 6: Full suite + benchmark.** Fix exact-color assertions if any shifted (never loosen error-bound/constraint tests). Run `scripts/bench_pipeline.py`; require 320×288 color_only ≤ 2.5s. Commit:

```bash
git add gb_pipeline.py scripts/bench_pipeline.py tests/test_palettes.py
git commit -m "perf: lazy proxy merge heap + single-fit finalize + kmeans micro-opts (320x288 color: <before>s -> <after>s)"
```

---

### Task 7: Vectorize index_tiles via unique-color distance tables

**Goal:** Eliminate the per-tile Python loop and per-tile `rgb2lab` calls in `index_tiles`. Compute everything from a `(n_unique_colors, p, 4)` Lab distance table plus vectorized gathers. Output must be bit-identical to the Task-5 semantics.

**Files:**
- Modify: `gb_pipeline.py:582-661` (`index_tiles`)
- Test: `tests/test_indexing.py`

**Acceptance Criteria:**
- [ ] No Python-level per-tile loop for index computation (a final reshape into `(n, 8, 8)` patterns is fine).
- [ ] Bit-identical output to the previous implementation: a property test compares against a straightforward per-tile reference implementation (kept in the test file, written to the Task-5 projection spec) on randomized images, both dither modes.
- [ ] Lab conversion happens once for unique colors (`np.unique(..., return_inverse=True)`) and once for palettes; large-unique-count inputs are chunked (chunk size 8192 unique colors) to bound memory.
- [ ] Full suite passes.

**Verify:** `venv/bin/python -m pytest tests/test_indexing.py -q` then the full suite; `venv/bin/python scripts/bench_pipeline.py` should show 320×288 color_only improved again vs Task 6's number (record both in the commit message).

**Steps:**

- [ ] **Step 1: Freeze the reference.** Copy the CURRENT (post-Task-5) body of `index_tiles` into `tests/test_indexing.py` as `_index_tiles_reference(image, palettes, assignment, dither)` (a plain function calling gb_pipeline helpers). Add the property test:

```python
@pytest.mark.parametrize("dither", ["none", "bayer"])
def test_index_tiles_matches_reference(dither):
    rng = np.random.RandomState(7)
    for _ in range(3):
        h, w = 24, 32
        arr = rng.randint(0, 256, (h, w, 3)).astype(np.uint8)
        pals = gb_pipeline.snap_rgb555(rng.randint(0, 256, (3, 4, 3)).astype(np.uint8))
        pals = np.stack([gb_pipeline.luminance_sort(p) for p in pals])
        assignment = rng.randint(0, 3, (h // 8, w // 8)).astype(np.uint8)
        got = gb_pipeline.index_tiles(arr, pals, assignment, dither=dither)
        want = _index_tiles_reference(arr, pals, assignment, dither=dither)
        assert np.array_equal(got.patterns, want.patterns)
```

This passes trivially before the rewrite (both are the same code) — its job is to pin behavior through Step 2.

- [ ] **Step 2: Rewrite `index_tiles`** (keep signature, docstring updated):

```python
    arr = np.asarray(image, dtype=np.uint8)
    h, w = arr.shape[:2]
    th, tw = h // 8, w // 8
    n_tiles = th * tw
    palettes = np.asarray(palettes, dtype=np.uint8)
    assignment = np.asarray(assignment)
    p = palettes.shape[0]

    uq, inv = np.unique(arr.reshape(-1, 3), axis=0, return_inverse=True)
    pal_lab = np.stack([_rgb_to_lab(pp) for pp in palettes])  # (p, 4, 3)

    n_uq = uq.shape[0]
    idx1_t = np.empty((n_uq, p), dtype=np.int8)
    idx2_t = np.empty((n_uq, p), dtype=np.int8)
    tval_t = np.zeros((n_uq, p), dtype=np.float32)
    for lo in range(0, n_uq, 8192):
        hi_ = min(lo + 8192, n_uq)
        ulab = _rgb_to_lab(uq[lo:hi_])                       # (m, 3)
        diff = ulab[:, None, None, :] - pal_lab[None, :, :, :]  # (m, p, 4, 3)
        dist = np.sqrt(np.sum(diff * diff, axis=3))          # (m, p, 4)
        order = np.argsort(dist, axis=2, kind="stable")
        i1 = order[:, :, 0]; i2 = order[:, :, 1]
        c1 = np.take_along_axis(pal_lab[None].repeat(hi_ - lo, 0), i1[:, :, None, None].repeat(3, 3), axis=2)[:, :, 0, :]
        c2 = np.take_along_axis(pal_lab[None].repeat(hi_ - lo, 0), i2[:, :, None, None].repeat(3, 3), axis=2)[:, :, 0, :]
        seg = c2 - c1
        seg_len2 = np.sum(seg * seg, axis=2)
        proj = np.sum((ulab[:, None, :] - c1) * seg, axis=2)
        t = np.where(seg_len2 > 0, proj / np.where(seg_len2 > 0, seg_len2, 1.0), 0.0)
        idx1_t[lo:hi_] = i1; idx2_t[lo:hi_] = i2
        tval_t[lo:hi_] = np.clip(t, 0.0, 1.0)

    inv2d = inv.reshape(h, w)
    pid_px = np.repeat(np.repeat(assignment.astype(np.int64), 8, axis=0), 8, axis=1)
    i1_px = idx1_t[inv2d, pid_px]
    if dither == "bayer":
        i2_px = idx2_t[inv2d, pid_px]
        t_px = tval_t[inv2d, pid_px]
        yy = np.arange(h)[:, None]; xx = np.arange(w)[None, :]
        thresh = BAYER4[yy % 4, xx % 4]
        index = np.where(t_px > thresh, i2_px, i1_px)
    else:
        index = i1_px
    patterns = (index.astype(np.uint8)
                .reshape(th, 8, tw, 8).transpose(0, 2, 1, 3).reshape(n_tiles, 8, 8))
```

(GBImage construction tail unchanged.) Watch float dtype: the reference computes `t` in float64; storing `tval_t` as float32 can flip pixels sitting exactly on a Bayer threshold. If the property test catches mismatches, use float64 for `tval_t` — bit-identity wins over the small memory saving.

- [ ] **Step 3: Full suite + benchmark; commit:**

```bash
git add gb_pipeline.py tests/test_indexing.py
git commit -m "perf: vectorize index_tiles via unique-color Lab distance tables (bit-identical)"
```

---

### Task 8: Raise the working-set color budget

**Goal:** Give palette packing more colors to choose from: quantize the working set to `4 * n_palettes * WORKING_SET_FACTOR` (factor 4, capped at 128) instead of exactly `4 * n_palettes`. Final palettes are still ≤ n_palettes × 4 colors; this only widens the candidate pool.

**Files:**
- Modify: `gb_pipeline.py` (module constant + `convert_for_hardware` color branch)
- Test: `tests/test_e2e.py`

**Acceptance Criteria:**
- [ ] `WORKING_SET_FACTOR = 4` module constant with a comment explaining the trade-off; color branch calls `quantize_working_set(img, min(4 * n_palettes * WORKING_SET_FACTOR, 128), custom_palette)`.
- [ ] New test: on the photo fixture from Task 4's test, mean Lab error of the final render (vs original) with factor 4 is ≤ the error with factor 1 (compute factor-1 by calling the stages manually with `max_colors = 4 * n_palettes`).
- [ ] `venv/bin/python scripts/bench_pipeline.py`: 320×288 color_only stays ≤ 2× the Task-7 number (record both in the commit message) — the pre-cluster fast path (`GROUP_EXACT_TRIGGER`) must be keeping the group blow-up in check.
- [ ] Full suite passes.

**Verify:** full suite + benchmark, numbers in commit message.

**Steps:**

- [ ] **Step 1: Failing test** (add to `tests/test_e2e.py`, reusing `_mean_lab_error` from Task 4):

```python
def test_wider_working_set_does_not_hurt_fidelity():
    rng = np.random.RandomState(2)
    yy, xx = np.mgrid[0:64, 0:64]
    arr = np.clip(np.stack([xx * 4, yy * 4, (xx + yy) * 2], axis=2)
                  + rng.normal(0, 6, (64, 64, 3)), 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, "RGB")

    def run(max_colors):
        quant = gb_pipeline.quantize_working_set(img, max_colors)
        qarr = np.asarray(quant, dtype=np.uint8)
        palettes, assignment = gb_pipeline.pack_palettes(qarr, 7)
        gb = gb_pipeline.index_tiles(arr, palettes, assignment, dither="none")
        return _mean_lab_error(np.asarray(gb_pipeline.render(gb)), arr)

    assert run(min(28 * gb_pipeline.WORKING_SET_FACTOR, 128)) <= run(28) * 1.02
```

(≤ with 2% slack, not strict <: on easy images both reach the same optimum. The pipeline-level improvement claim is directional; the guard is "wider never hurts".) FAILS only because `WORKING_SET_FACTOR` doesn't exist yet — that's the red state.

- [ ] **Step 2: Implement** the constant and the call-site change:

```python
# Working-set head-room: the global quantizer keeps FACTOR x the final color
# budget so per-palette 4-means can place centers regionally instead of being
# limited to the global quantizer's exact final picks. Cost: more Phase-1
# groups (bounded by the GROUP_EXACT_TRIGGER pre-cluster fast path).
WORKING_SET_FACTOR = 4
```

- [ ] **Step 3: Full suite + benchmark (≤ 2× Task-7 time). If the bench blows past 2×, lower `GROUP_CLUSTER_TARGET` to 40 and re-measure before anything else. Commit:**

```bash
git add gb_pipeline.py tests/test_e2e.py
git commit -m "feat: 4x working-set head-room for palette packing (bench: <numbers>)"
```

---

### Task 9: Stage-6 signature refresh on merge

**Goal:** When pattern A merges into B, blend A's signature into B (usage-weighted, orientation-aligned) so subsequent merge costs measure B's *current* visual content, reducing drift in heavy-merge chains.

**Files:**
- Modify: `gb_pipeline.py:852-1033` (`merge_to_budget`)
- Test: `tests/test_budget.py`

**Acceptance Criteria:**
- [ ] After each merge a→b with orientation v: `sig_lab[b] = (u_b * sig_lab[b] + u_a * _flip_signature(sig_lab[a], v)) / (u_a + u_b)` using pre-merge usages; `sig_flat` and every `var_flat[v'][b]` row refreshed; per-pattern `version` counter bumped for b.
- [ ] Heap entries carry version snapshots of both endpoints and are re-pushed (recomputed) when stale — same lazy-invalidation pattern already used for the usage snapshot.
- [ ] Matrix fast path (n ≤ `_MERGE_MATRIX_MAX`): after a merge, recompute row/column `best[:, b]`/`best[b, :]` and `orient_idx` for b (vectorized over survivors) instead of leaving stale distances.
- [ ] New test: constructed 3-pattern chain where static signatures pick a provably worse second merge than refreshed signatures (see Step 1); refreshed behavior wins.
- [ ] p95 tracking still works; full suite passes.

**Verify:** `venv/bin/python -m pytest tests/test_budget.py -q` then the full suite.

**Steps:**

- [ ] **Step 1: Failing test** in `tests/test_budget.py`:

```python
def _flat_pattern(v):
    return np.full((8, 8), v, dtype=np.uint8)

def test_signature_refresh_prevents_drift():
    # Grayscale palette; three flat patterns at indices 0, 1, 3 (light, mid, dark).
    # usages: light=1, mid=1, dark=100. Budget forces two merges.
    # Merge 1: light->mid (cheapest). With STATIC signatures, merge 2 compares
    # dark against mid's ORIGINAL signature. With refresh, mid's signature has
    # drifted toward light, making the (mid+light)->dark merge measurably
    # different. Assert the refreshed p95 reflects the blended distance:
    # the final single pattern must be the high-usage dark one, and mid's
    # cells' recorded per-pixel distance must be measured against dark, not
    # against mid's stale pre-merge self.
    pal = gb_pipeline.snap_rgb555(np.array(
        [[[220, 220, 220], [150, 150, 150], [90, 90, 90], [30, 30, 30]]],
        dtype=np.uint8))
    th, tw = 1, 102
    tilemap = np.zeros((th, tw), dtype=np.int32)
    tilemap[0, 0] = 0
    tilemap[0, 1] = 1
    tilemap[0, 2:] = 2
    patterns = np.stack([_flat_pattern(0), _flat_pattern(1), _flat_pattern(3)])
    gb = gb_pipeline.GBImage(
        patterns=patterns, tilemap=tilemap,
        attrs_palette=np.zeros((th, tw), dtype=np.uint8),
        attrs_hflip=np.zeros((th, tw), dtype=bool),
        attrs_vflip=np.zeros((th, tw), dtype=bool),
        palettes=pal)
    merged, n = gb_pipeline.merge_to_budget(gb, budget=1, allow_flips=False)
    assert n == 2
    assert merged.patterns.shape[0] == 1
    # Survivor must be the dark pattern (usage 100 dominates both merges).
    assert np.array_equal(merged.patterns[0], _flat_pattern(3))
```

Then extend it: monkeypatch-free introspection is hard, so ALSO add a direct unit test for the new `_refresh_signature` helper (write the helper in Step 2 as a module-level function so it is testable):

```python
def test_refresh_signature_blend():
    sig = np.zeros((2, 8, 8, 3))
    sig[0, :, :, 0] = 10.0   # pattern a
    sig[1, :, :, 0] = 30.0   # pattern b
    out = gb_pipeline._refresh_signature(sig[1], sig[0], u_b=3, u_a=1, orientation="")
    assert np.allclose(out[:, :, 0], 25.0)  # (3*30 + 1*10) / 4
```

Run → FAILS (`_refresh_signature` doesn't exist).

- [ ] **Step 2: Implement.** Module-level helper:

```python
def _refresh_signature(sig_b, sig_a, u_b, u_a, orientation):
    """Usage-weighted blend of a merged-away pattern's signature into its
    survivor, after aligning a into b's frame (flips are involutions, so the
    same variant name maps b->a and a->b)."""
    aligned = _flip_signature(sig_a, orientation)
    return (float(u_b) * sig_b + float(u_a) * aligned) / float(u_b + u_a)
```

In `merge_to_budget`:
  - Add `version = np.zeros(n, dtype=np.int64)`.
  - Heap entries become `(cost, counter, a, b, usage_snap, ver_a, ver_b)`; `push_pair` records `version[a]`, `version[b]`; the pop loop's staleness check becomes `if int(usage[a]) != snap or version[a] != ver_a or version[b] != ver_b: push_pair(a, b); continue`.
  - In the merge body, BEFORE `usage[b] += usage[a]`: compute `new_sig = _refresh_signature(sig_lab[b], sig_lab[a], usage[b], usage[a], v)`, then `sig_lab[b] = new_sig`, `sig_flat[b] = new_sig.reshape(192)`, and for each variant name `var_flat[vn][b] = _flip_signature(new_sig, vn).reshape(192)`; bump `version[b] += 1`.
  - Matrix path: after refreshing b, recompute its row/col:

```python
            if best is not None:
                sb = sig_flat[b]
                for vi, vn in enumerate(variant_names):
                    dbt = np.mean((var_flat[vn] - sb[None, :]) ** 2, axis=1)      # b as source vs all targets' variant vn? 
                # Careful with asymmetry: best[i, j] uses sig_flat[i] vs var_flat[v][j].
                # Recompute BOTH directions involving b:
                dsrc = np.stack([np.mean((sig_flat[b][None, :] - var_flat[vn]) ** 2, axis=1)
                                 for vn in variant_names])   # (V, n): b as A, others as B
                dtgt = np.stack([np.mean((sig_flat - var_flat[vn][b][None, :]) ** 2, axis=1)
                                 for vn in variant_names])   # (V, n): others as A, b as B
                best[b, :] = dsrc.min(axis=0); orient_idx[b, :] = dsrc.argmin(axis=0)
                best[:, b] = dtgt.min(axis=0); orient_idx[:, b] = dtgt.argmin(axis=0)
                best[b, b] = np.inf
                # push refreshed candidate pairs for b against all alive patterns
                for other in np.nonzero(alive)[0]:
                    if int(other) != b:
                        push_pair(int(other), b)
```

  (`best` must be initialized to `None` on the cluster path so the `if best is not None:` guard works; hold `var_flat[vn]` as the (n, 192) arrays already built. Note `var_flat[vn][b]` row must be refreshed BEFORE these recomputes.)
  - Cluster fast path (n > `_MERGE_MATRIX_MAX`): no matrix to fix; version-stale entries re-push automatically. After each `run_heap` pass survivors are re-clustered anyway.
  - `dist_orient` (matrix path) reads the refreshed matrix; non-matrix `dist_orient` reads refreshed `sig_flat`/`var_flat` directly — both stay correct.

- [ ] **Step 3: Full suite** (existing budget tests use small constructed cases; merge order can shift — re-verify each affected test's intent and update expected values with a comment explaining the refresh semantics). Commit:

```bash
git add gb_pipeline.py tests/test_budget.py
git commit -m "feat: refresh merge signatures on absorb so heavy-merge chains stop drifting"
```

---

### Task 10: Input geometry polish — center crop + resample filter option

**Goal:** Crop-to-multiple-of-8 becomes center-anchored, and the UI gains a resize-filter choice (Auto / Nearest / Lanczos) where Auto = Lanczos for GB Studio hardware modes, Nearest for Artistic.

**Files:**
- Modify: `gb_pipeline.py:1294-1304` (crop), `main.py` (`downscale_image`, `process_image`, `process_image_folder`, UI dropdown + `shared_inputs`)
- Test: `tests/test_convert.py`, `tests/test_main_ui.py`

**Acceptance Criteria:**
- [ ] `convert_for_hardware` center-crops: a 20×20 input keeps rows/cols 2..17 (offsets `(ow - nw) // 2`, `(oh - nh) // 2`), warning text unchanged in spirit.
- [ ] `downscale_image(image, w, h, keep_aspect, resample=Image.NEAREST)` accepts a resample argument; `process_image` chooses it from a new `resize_filter` input: `"Auto"` → `Image.LANCZOS` when `mode in HARDWARE_MODES` else `Image.NEAREST`; `"Nearest (pixel art)"` → NEAREST; `"Lanczos (smooth)"` → LANCZOS.
- [ ] New `gr.Dropdown(choices=["Auto", "Nearest (pixel art)", "Lanczos (smooth)"], value="Auto", label="Resize Filter")` placed in the size `gr.Row` next to Keep Aspect Ratio, appended to `shared_inputs` (and threaded through `process_image_folder`'s signature and its inner `process_image` call).
- [ ] Full suite passes; `test_main_ui.py` updated for the new input.

**Verify:** `venv/bin/python -m pytest tests/test_convert.py tests/test_main_ui.py -q` then the full suite.

**Steps:**

- [ ] **Step 1: Failing tests.** In `tests/test_convert.py`:

```python
def test_crop_is_center_anchored():
    arr = np.zeros((20, 20, 3), dtype=np.uint8)
    arr[2:18, 2:18] = 200   # center 16x16 block is bright
    res = gb_pipeline.convert_for_hardware(Image.fromarray(arr, "RGB"), "mono")
    out = np.asarray(res.image)
    assert out.shape[:2] == (16, 16)
    # Top-left crop would include 2 dark rows/cols; center crop keeps only bright.
    assert len(np.unique(out.reshape(-1, 3), axis=0)) == 1
```

In `tests/test_main_ui.py`:

```python
def test_resize_filter_resolution():
    assert main._resolve_resample("Auto", main.MODE_COLOR) == Image.LANCZOS
    assert main._resolve_resample("Auto", main.MODE_ARTISTIC) == Image.NEAREST
    assert main._resolve_resample("Nearest (pixel art)", main.MODE_COLOR) == Image.NEAREST
    assert main._resolve_resample("Lanczos (smooth)", main.MODE_ARTISTIC) == Image.LANCZOS
```

- [ ] **Step 2: Center crop** in `convert_for_hardware`:

```python
    if (nw, nh) != (ow, oh):
        left = (ow - nw) // 2
        top = (oh - nh) // 2
        img = img.crop((left, top, left + nw, top + nh))
        warnings.append(
            f"Input center-cropped from {ow}x{oh} to {nw}x{nh} "
            "(dimensions must be multiples of 8)."
        )
```

- [ ] **Step 3: main.py.** Add:

```python
RESIZE_FILTERS = {"Nearest (pixel art)": Image.NEAREST, "Lanczos (smooth)": Image.LANCZOS}

def _resolve_resample(resize_filter: str, mode: str):
    if resize_filter in RESIZE_FILTERS:
        return RESIZE_FILTERS[resize_filter]
    return Image.LANCZOS if mode in HARDWARE_MODES else Image.NEAREST
```

`downscale_image` gains `resample=Image.NEAREST` and uses it in `image.resize(...)`. `process_image` gains a `resize_filter` parameter (append at the END of the signature to keep positional wiring stable, before `logo_subtype` is fine too — but then update BOTH `.click` wirings and `process_image_folder` consistently; simplest is appending to the end of `shared_inputs` and the end of both function signatures) and calls `downscale_image(image, int(width), int(height), aspect_ratio, _resolve_resample(resize_filter, mode))`. Add the dropdown component and append it to `shared_inputs`.

- [ ] **Step 4: Full suite; commit:**

```bash
git add gb_pipeline.py main.py tests/
git commit -m "feat: center crop + resize filter choice (auto Lanczos for hardware modes)"
```

---

## Execution & review protocol (for the coordinating session)

- Tasks run strictly in order 1 → 10 (shared files; no parallelism).
- Model assignment: Tasks 1, 5, 8, 10 → Sonnet; Tasks 2, 3, 4, 6, 7, 9 → Opus.
- Each task agent: read this plan section, implement TDD-style, run the FULL suite, commit exactly one commit, and report {commit hash, tests passed count, notes, any deviations}.
- A verification agent (Sonnet) checks each task's acceptance criteria after the implementer reports; on failure, one repair round with the verifier's findings, then re-verify. A task that fails twice is SKIPPED (left uncommitted / reverted) and reported — later tasks must still leave the suite green.
- Final review is done by Fable in the main session: full-branch diff review, full suite, benchmark comparison against the baseline numbers above, and a visual sanity pass.
