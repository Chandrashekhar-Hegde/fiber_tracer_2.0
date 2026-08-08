# Digital Twin Feature — Design Spec

## Context

The digital twin spike (`research/digital-twin-spike` branch, `scripts/digital_twin_poc.py`; its validated functions now ship unchanged in `src/fiber_tracer/twin/fitting.py`) validated the scope defined in `docs/superpowers/specs/2026-08-07-digital-twin-spike-design.md`: fitting phantom-generator parameters from a resolved-regime summary, regenerating a statistically-matched synthetic volume, and computing an effective modulus via Halpin-Tsai. The round-trip fit passed within documented tolerances after two real bugs were found and fixed/worked around (sphere- vs. cylinder-equivalent diameter; `extract_fiber_paths` failing on most fibers in dense unidirectional bundles, now filed as issue #28). This spec turns that PoC into a real feature, mirroring the DVC/DIC features (PRs #25/#26).

## Design

**Module.** `fiber_tracer/twin/fitting.py` — promotes the PoC's three functions unchanged: `fit_twin_parameters(summary, volume_shape)`, `regenerate_twin(fitted_params, shape, voxel_spacing_um)`, `effective_modulus_halpin_tsai(volume_fraction, fiber_modulus_gpa, matrix_modulus_gpa, aspect_ratio)`. The cylinder-inversion fix and its fallback-length limitation (still real until issue #28 lands) move over verbatim, including the explanatory comments — this is correctness-critical context, not boilerplate to trim.

**Config.** `TwinConfig`:

```python
@dataclass
class TwinConfig:
    enabled: bool = False
    fiber_modulus_gpa: float = 72.0
    matrix_modulus_gpa: float = 3.0
    aspect_ratio: float = 20.0
```

No `reference_path`/`deformed_path` — unlike `DVCConfig`/`DICConfig`, the twin fits from the *same* run's own resolved-regime output, not a separate file pair. Added as `Config.twin`, with `from_dict`/`validate` wiring following the established pattern (positivity checks on the three numeric fields; no path-existence checks needed).

**Pipeline.** After the regime dispatch in `run()` (same call site as the existing `if self.config.dvc.enabled` / `if self.config.dic.enabled` blocks): if `self.config.twin.enabled`, call `self._run_twin(summary, out)`. That method checks `summary["regime"] == "resolved"` — if not, logs a warning and returns `None` (the twin section is simply omitted, same disabled-note convention used elsewhere) rather than raising, since `regime="auto"` may not resolve to `resolved` and that's a legitimate outcome, not a user error. If resolved, runs `fit_twin_parameters` → `regenerate_twin` → `effective_modulus_halpin_tsai` (volume fraction computed from the fitted twin's own fiber count/diameter/shape, matching the PoC's `main()`), and writes `twin_summary.json` via the existing `write_json_report` (no CSV/HTML — this produces a small parameter dict, not a per-node grid like DVC/DIC's `dvc_windows`/`dic_windows`, so `write_csv_report`/`write_html_report` don't apply and aren't called).

**CLI.** Add `twin=file_config.twin` to `_run_pipeline`'s `Config(...)` reconstruction in the same commit as the config field — no separate "proactive fix" task this time, since the DVC/DIC pattern is now established practice, not something to rediscover.

**TUI.** `computeTwin: boolean` only, added to `AnalysisConfig`/`buildJson`/`DEFAULT_CONFIG`/`Configure`'s `TOGGLES`/`Review` — no file-picker rows (reuses `dataPath`, not a separate reference/deformed pair).

**Critical files:**
- `src/fiber_tracer/twin/__init__.py`, `twin/fitting.py` (new package).
- `src/fiber_tracer/config.py` — `TwinConfig`, `Config.twin`, `from_dict`/`validate`.
- `src/fiber_tracer/pipeline.py` — `_run_twin`, call site in `run()`.
- `src/fiber_tracer/cli.py` — `twin=file_config.twin`.
- `tui/src/types.ts`, `bridge.ts`, `app.tsx`, `components/configure.tsx`, `components/review.tsx`, `bridge.test.ts`, `history.test.ts`.
- `docs/CAPABILITIES.md`, `ROADMAP.md`.
- `tests/test_config.py` (round-trip), `tests/test_pipeline.py` (toggle on/off including the non-resolved-regime skip path, CLI round-trip).

## Not doing

Same exclusions as the spike spec, carried forward: no CAD/nominal-design reconciliation, no DVC/DIC deformation coupling, no FE mesh export, no continuous orientation-tensor fitting into the phantom generator.

## Testing

Pytest, following the DVC/DIC pattern: config round-trip/validation, a pipeline-level test asserting `twin_summary.json` exists when enabled and resolved, a test asserting the twin section is *absent* (with a logged warning, not a crash) when the regime isn't resolved, and the CLI-level `main()` round-trip test that has caught a real config-reconstruction omission twice before.
