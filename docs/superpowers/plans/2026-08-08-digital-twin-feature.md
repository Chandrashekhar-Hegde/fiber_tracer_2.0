# Digital Twin Feature Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the digital twin feature per `docs/superpowers/specs/2026-08-08-digital-twin-feature-design.md`, promoting `scripts/digital_twin_poc.py`'s validated functions into `fiber_tracer/twin/fitting.py` and wiring config/pipeline/CLI/TUI, mirroring the shipped DVC/DIC features.

**Architecture:** New `twin/fitting.py` module (fit → regenerate → modulus, unchanged from the PoC). `TwinConfig` has no reference/deformed paths (fits from the same run's own resolved-regime summary). `Pipeline._run_twin` is called after regime dispatch, skips with a warning on non-resolved regimes, writes `twin_summary.json` only (no CSV/HTML — not a per-node grid).

**Tech Stack:** Python, NumPy. No new dependency.

## Global Constraints

- Run `black --check`, `ruff check`, `mypy`, `pytest` before every commit.
- Add `twin=file_config.twin` to `cli.py`'s `Config(...)` reconstruction in the SAME commit as the config field — this omission has been a real bug twice (DVC, then caught proactively for DIC); don't let it regress a third time.
- The cylinder-inversion fix and its fallback-length limitation (comments explaining why) must survive the promotion from script to module verbatim — they document a real, still-open bug (issue #28).

---

### Task 1: `twin/fitting.py`

**Files:**
- Create: `src/fiber_tracer/twin/__init__.py`, `src/fiber_tracer/twin/fitting.py`

- [ ] **Step 1:** Create `twin/__init__.py` with a one-line module docstring.
- [ ] **Step 2:** Create `twin/fitting.py` by moving `scripts/digital_twin_poc.py`'s `_cross_sectional_diameters_um`, `fit_twin_parameters`, `regenerate_twin`, `effective_modulus_halpin_tsai` (and their docstrings/comments) unchanged. Update the module docstring to describe this as the shipped feature's fitting logic (not a spike script), and add a reference to issue #28 for the fallback-length limitation.
- [ ] **Step 3:** `black --check && ruff check && mypy src/fiber_tracer/twin/`, then commit:

```bash
git add src/fiber_tracer/twin/
git commit -m "Add twin/fitting.py: promote the digital twin PoC's fitting logic"
```

---

### Task 2: `TwinConfig`

**Files:**
- Modify: `src/fiber_tracer/config.py`, `tests/test_config.py`

- [ ] **Step 1:** Add `TwinConfig` (enabled, fiber_modulus_gpa=72.0, matrix_modulus_gpa=3.0, aspect_ratio=20.0) next to `DICConfig`. Add `twin: TwinConfig = field(default_factory=TwinConfig)` to `Config`.
- [ ] **Step 2:** In `Config.validate()`, add a block: if `self.twin.enabled`, check all three numeric fields are positive (no path-existence checks — no paths).
- [ ] **Step 3:** In `Config.from_dict()`, add the `"twin"` branch.
- [ ] **Step 4:** In `tests/test_config.py`, add `TwinConfig` to imports and two tests mirroring the DVC/DIC config tests (round-trip + validation), substituted for `twin`/`TwinConfig` (validation test: negative `aspect_ratio` should raise).
- [ ] **Step 5:** `black --check && ruff check && mypy src/fiber_tracer/config.py && pytest tests/test_config.py -v`, then commit.

---

### Task 3: Pipeline wiring

**Files:**
- Modify: `src/fiber_tracer/pipeline.py`

- [ ] **Step 1:** Import `fit_twin_parameters, regenerate_twin, effective_modulus_halpin_tsai` from `fiber_tracer.twin.fitting`.
- [ ] **Step 2:** Add `_run_twin(self, summary: dict, out: Path) -> dict | None`:
  - If `summary.get("regime") != "resolved"`: `logger.warning(...)`, return `None`.
  - Else: call `fit_twin_parameters(summary, volume_shape=self.volume.shape)` (the pipeline already stores `self.volume` after `_run_resolved`), `regenerate_twin`, compute volume fraction the same way the PoC's `main()` did, call `effective_modulus_halpin_tsai` with `self.config.twin.{fiber_modulus_gpa,matrix_modulus_gpa,aspect_ratio}`, build a `twin_summary` dict (fitted params, twin fiber count, volume fraction, effective modulus, config echo), `write_json_report(out / "twin_summary.json", twin_summary)`, return it.
- [ ] **Step 3:** In `run()`, after the existing `if self.config.dic.enabled:` block, add `if self.config.twin.enabled: twin_result = self._run_twin(summary, out); if twin_result is not None: summary["twin"] = twin_result`.
- [ ] **Step 4:** Manual smoke test: run the pipeline with `twin.enabled=True` on a resolved-regime phantom (reuse the PoC's own test setup), confirm `twin_summary.json` exists and contains sane values; also test with a config that resolves to `marginal`/`subvoxel` and confirm the twin section is absent with a logged warning, not a crash.
- [ ] **Step 5:** `black --check && ruff check && mypy src/fiber_tracer/pipeline.py`, then commit.

---

### Task 4: Pipeline-level tests + CLI wiring

**Files:**
- Modify: `tests/test_pipeline.py`, `src/fiber_tracer/cli.py`

- [ ] **Step 1:** Add `twin=file_config.twin,` to `cli.py`'s `Config(...)` reconstruction, in the same commit as the tests below (not split into a separate "fix" commit this time).
- [ ] **Step 2:** Add three tests to `tests/test_pipeline.py`: (a) `twin.enabled=True` on a resolved-regime run produces `twin_summary.json` with sane fitted values; (b) `twin.enabled=True` on a config that resolves to marginal/subvoxel omits the twin section (no crash); (c) CLI-level round-trip test via `main(["--config", ...])`, mirroring the DVC/DIC CLI tests, asserting `twin_summary.json` exists.
- [ ] **Step 3:** `pytest tests/test_pipeline.py -v -k twin`, then `black --check && ruff check && mypy src/fiber_tracer/cli.py`, then commit.

---

### Task 5: TUI wiring

**Files:**
- Modify: `tui/src/types.ts`, `bridge.ts`, `app.tsx`, `components/configure.tsx`, `components/review.tsx`, `bridge.test.ts`, `history.test.ts`

- [ ] **Step 1:** Add `computeTwin: boolean;` to `AnalysisConfig` (no path fields).
- [ ] **Step 2:** Add `twin: { enabled: config.computeTwin }` to `buildJson`.
- [ ] **Step 3:** Add `computeTwin: false,` to `DEFAULT_CONFIG`.
- [ ] **Step 4:** Add `{ key: "computeTwin", label: "Digital twin (fitted microstructure + effective modulus)" }` to `Configure`'s `TOGGLES` — no file-picker rows needed (not in `FILE_PICKER_KEYS`).
- [ ] **Step 5:** Add a DIC-mirroring line to `review.tsx` (no reference/deformed paths to show, just on/off).
- [ ] **Step 6:** Fix `bridge.test.ts`/`history.test.ts` fixtures with `computeTwin: false,`.
- [ ] **Step 7:** `tsc --noEmit && bun test && bun run build`, then commit.

---

### Task 6: Docs

**Files:**
- Modify: `docs/CAPABILITIES.md`, `ROADMAP.md`

- [ ] **Step 1:** Flip the digital-twin status-matrix row to ✅ in `CAPABILITIES.md`; add the twin switches to the switch-flow table; move the "Digital twin" bullet out of the research track into a "Shipped" note, referencing issue #28's known limitation.
- [ ] **Step 2:** Add a "Digital Twin (epic #21)" section to `ROADMAP.md` mirroring the DVC/DIC ones.
- [ ] **Step 3:** Commit.

---

### Task 7: Final verification, PR, epic tracking

- [ ] **Step 1:** Full sweep: `pytest tests/ -q`, `black --check .`, `ruff check src/ tests/ scripts/`, `mypy src/fiber_tracer`, TUI `tsc --noEmit && bun test`.
- [ ] **Step 2:** Manual end-to-end run via the real `fiber-tracer run --config ...` CLI (not just the Python API) with `twin.enabled=true` — confirms `twin_summary.json` is produced through the actual CLI path, the exact check that has caught real bugs twice before.
- [ ] **Step 3:** Push branch, open a PR (mirror PRs #25/#26's structure).
- [ ] **Step 4:** Confirm with the user before posting to epic #21 (checklist + closing comment; note this epic's acceptance criteria was scope sign-off, already given).
- [ ] **Step 5:** Watch CI; squash-merge once green.

## Self-Review

**Spec coverage:** module (Task 1), config (Task 2), pipeline (Task 3), tests+CLI (Task 4), TUI (Task 5), docs (Task 6), verification/PR/tracking (Task 7) — all spec sections covered.

**Placeholder scan:** none; the one open item (issue #28's fallback-length limitation) is explicitly a carried-forward known limitation, not a placeholder.

**Consistency:** `twin_summary.json`, `computeTwin`, `TwinConfig` field names used identically across all tasks.
