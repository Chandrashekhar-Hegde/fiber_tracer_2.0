# Fiber Tracer Terminal UI Design Spec

**Date:** 2026-06-25  
**Status:** Draft — pending implementation planning  
**Author:** Kimi Code (with user review)  
**Related:** `docs/PERFORMANCE.md`, `docs/CLI_REFERENCE.md`, `ROADMAP.md`

## 1. Overview

Build a keyboard-driven, beautiful terminal user interface (TUI) for Fiber Tracer using [termcn](https://www.termcn.dev/) components on top of [Ink](https://github.com/vadimdemedes/ink) and [OpenTUI](https://github.com/AllanChain/openTUI), bundled with [Bun](https://bun.sh/). The TUI complements the existing CLI by providing a guided, interactive experience for running analyses, inspecting results, managing models, and conducting experiments.

## 2. Goals

- **Guided for newcomers:** A sidebar-embedded wizard makes the analysis pipeline discoverable.
- **Fast for power users:** Keyboard-first navigation with vim-style shortcuts, quick jumps, and searchable lists.
- **Future-proof:** The navigation model supports future AI methods, model registries, training workflows, and experiments without redesign.
- **Beautiful & consistent:** Multiple built-in terminal themes (Dracula default, Catppuccin, etc.) selectable in Settings.
- **Integrated:** Reuses the existing Python analysis engine via a JSON-RPC / CLI wrapper so the TUI is a thin, stateful shell.

## 3. Non-Goals

- The TUI does not replace the existing `fiber-tracer` CLI; it wraps it.
- It does not perform heavy compute inside the terminal process (model inference stays in Python).
- It is not a web application; everything renders in the terminal.

## 4. Architecture

```
┌─────────────────────────────────────────────┐
│  Terminal UI (Bun + React + Ink + termcn)   │
│  - Keyboard input, layout, theme, charts    │
└──────────────┬──────────────────────────────┘
               │ stdin/stdout / JSON-RPC
┌──────────────▼──────────────────────────────┐
│  Python bridge (`fiber_tracer tui-server`)  │
│  - Config validation, pipeline orchestration│
│  - File scanning, progress events, results  │
└──────────────┬──────────────────────────────┘
               │
┌──────────────▼──────────────────────────────┐
│  Existing Fiber Tracer library              │
│  - Segmentation, analysis, reporting, IO    │
└─────────────────────────────────────────────┘
```

The bridge can initially be a simple subprocess wrapper that runs `fiber-tracer` with JSON output. Later it may evolve into a long-running JSON-RPC server for live progress streaming.

## 5. Navigation Model

A **persistent left sidebar** doubles as the wizard step list. This gives newcomers a clear path while letting experts jump directly to any section.

### Sidebar Sections

```
Fiber Tracer
▸ New Analysis
  1. Select Data
  2. Configure
  3. Review
  4. Run & Watch
Dashboard
History
Experiments
Model Registry
Training
Logs
Settings
```

- Selecting **New Analysis** expands the four wizard steps.
- Selecting **Dashboard / History / Experiments / Model Registry / Training / Logs / Settings** switches the main panel.
- A **status bar** at the top shows version, current mode, and global shortcuts (`?` help, `q` quit).
- A **footer** at the bottom shows context-sensitive keyboard shortcuts.

## 6. Screens

### 6.1 Select Data

- **Recent files list:** Last 10 analyzed volumes, searchable with `/`.
- **Browse:** Open a terminal file browser modal (`b`).
- **Path input:** Manual path entry with validation.
- **Preview panel:** Shape, dtype, voxel spacing guess, file size.
- **Drag-and-drop not applicable** in terminal; instead, accept clipboard path paste.

### 6.2 Configure

Editable fields:

| Field | Type | Default |
|-------|------|---------|
| Voxel spacing (Z Y X) | numeric triple | from metadata or 1.0 1.0 1.0 |
| Fiber diameter (µm) | float | 10.0 |
| Regime | select | auto |
| Segmentation method | select | otsu |
| Model | select from registry | fiber_unet_v2_full.pt |
| U-Net batch size | integer | 1 |
| Output directory | path | ./results/<name> |
| Compute morphometry | checkbox | true |
| Compute orientation tensor | checkbox | true |
| Compute TDA descriptors | checkbox | false |

The form should validate inline and show warnings (e.g., large volume size, missing model for `unet`).

### 6.3 Review

- Read-only summary of all configured parameters.
- Warnings / estimated time / disk usage.
- **Edit** button returns to Configure.
- **Run** button starts the pipeline.

### 6.4 Run & Watch

- Live progress bar and current stage name.
- Elapsed time and estimated remaining time.
- Log tail (last N lines).
- **Cancel** (`Ctrl+c`) and **Go to Dashboard** (`d`) actions.
- On completion, auto-offer to open Dashboard.

### 6.5 Dashboard

Multi-panel layout:

- **Summary panel:** N fibers, mean equivalent diameter, volume fraction, regime.
- **Fiber table:** Label, volume, equivalent diameter, orientation vector (if computed).
- **Orientation panel:** A2 tensor, fractional anisotropy.
- **Histogram panel:** Diameter distribution (termcn Bar Chart).
- **Actions:** Open output folder, regenerate report, export CSV/JSON.

### 6.6 History

- List of past runs with status icon, name, timestamp, duration.
- Actions: rerun with same config (`r`), view dashboard (`Enter`), delete record, open output folder.
- State persisted in `~/.config/fiber-tracer/history.jsonl`.

### 6.7 Experiments

- Create a batch of runs with parameter sweeps (method, model, diameter, etc.).
- Table comparing runs with metrics (Dice, IoU, accuracy, time, memory).
- Actions: add run, remove run, rerun all, export comparison CSV.

### 6.8 Model Registry

- List installed models with name, version, source, size, validation metrics.
- Actions: activate, download from GitHub release, import local `.pt`, delete, view model card.
- Plugin backends appear here as additional cards.

### 6.9 Training

Guided wizard:

1. **Prepare:** Select raw data directory, patch size, augmentation options.
2. **Train:** Choose base model, epochs, learning rate, device (cpu/mps/cuda), output name.
3. **Validate:** Run validation on a hold-out set and show metrics.
4. **Export:** Save checkpoint to `models/` and optionally register it.

### 6.10 Logs

- Scrollable, filterable pipeline log viewer.
- Toggle level (DEBUG/INFO/WARNING/ERROR).
- Export current session log.

### 6.11 Settings

- **Theme:** Dracula (default), Catppuccin Mocha, Catppuccin Latte, One Dark, Nord.
- **Default output directory.**
- **Default model checkpoint.**
- **Log level default.**
- **Keybinding style:** vim (default) or arrows-only.
- **Reset to defaults.**

Theme changes apply instantly and persist to `~/.config/fiber-tracer/tui-config.json`.

## 7. Theme System

Use OpenTUI theme definitions. The TUI reads a theme name from config and maps it to an OpenTUI color palette. termcn components inherit the palette automatically.

Initial themes:

1. **Dracula** — default; high contrast, research-friendly.
2. **Catppuccin Mocha** — soft dark.
3. **Catppuccin Latte** — light option for presentations.
4. **One Dark** — familiar IDE feel.
5. **Nord** — cool, low-distraction.

Users can add custom themes by dropping a JSON file into `~/.config/fiber-tracer/themes/`.

## 8. Keyboard Shortcuts

Global shortcuts are always available; context shortcuts appear in the footer.

### Global

| Key | Action |
|-----|--------|
| `q` | Quit (confirm if a run is active) |
| `?` / `F1` | Toggle help overlay |
| `1`–`7` | Jump to sidebar sections |
| `Tab` / `Shift+Tab` | Cycle focus between sidebar and main panel |
| `/` | Search in current list |

### Navigation (vim style)

| Key | Action |
|-----|--------|
| `j` / `↓` | Next item |
| `k` / `↑` | Previous item |
| `h` / `←` | Back / collapse section |
| `l` / `→` / `Enter` | Select / expand |
| `g` / `G` | Top / bottom of list |

### Wizard

| Key | Action |
|-----|--------|
| `n` | Next step |
| `p` | Previous step |
| `r` | Run analysis (from Review) |

### Run & Watch

| Key | Action |
|-----|--------|
| `Ctrl+c` | Cancel run |
| `d` | Open Dashboard |
| `l` | Focus log tail |

## 9. Data Flow

1. TUI starts and loads config/history.
2. User navigates wizard and builds a `Config` object.
3. On **Run**, TUI spawns `fiber-tracer --config /tmp/... --output results/...`.
4. Python pipeline writes progress events to a JSONL file (`results/progress.jsonl`) or streams to stdout.
5. TUI polls the progress file and updates UI.
6. On completion, TUI reads `results/summary.json` and renders Dashboard.
7. History is appended with run metadata.

For live progress, prefer JSONL streaming over polling where possible.

## 10. Future-Proofing

- **Plugin registry:** New segmentation backends register via a `backends.json` manifest; the TUI discovers them without code changes.
- **Model registry:** Versioned model entries with arbitrary metadata fields leave room for v3, v4, and domain-specific models.
- **Experiments table:** Generic metric columns allow comparing any future method.
- **Training wizard:** Step labels and required fields are data-driven, so new training modes can be added.
- **Theme system:** External theme JSON files; no rebuild needed to add themes.

## 11. Tech Stack

| Layer | Technology |
|-------|------------|
| Runtime | Bun 1.3+ |
| Framework | React 18 + Ink 5 |
| Components | termcn / OpenTUI |
| Build | `bun build` or `bun run build` |
| Packaging | Single executable via `bun build --compile` or distributed as `npm` package |
| State | React hooks + local JSON config files |
| Python bridge | Subprocess JSON-RPC / CLI wrapper |

## 12. Project Structure

```
tui/
├── package.json
├── bun.lockb
├── tsconfig.json
├── src/
│   ├── app.tsx                 # Ink root
│   ├── components/
│   │   ├── layout.tsx          # Sidebar + header + footer
│   │   ├── wizard.tsx          # Wizard shell
│   │   ├── select-data.tsx
│   │   ├── configure.tsx
│   │   ├── review.tsx
│   │   ├── run-watch.tsx
│   │   ├── dashboard.tsx
│   │   ├── history.tsx
│   │   ├── experiments.tsx
│   │   ├── model-registry.tsx
│   │   ├── training.tsx
│   │   ├── logs.tsx
│   │   └── settings.tsx
│   ├── theme.ts                # OpenTUI theme loader
│   ├── config.ts               # User config persistence
│   ├── history.ts              # Run history persistence
│   ├── bridge.ts               # Python subprocess wrapper
│   └── keybindings.ts          # Shortcut definitions
└── themes/
    ├── dracula.json
    ├── catppuccin-mocha.json
    ├── catppuccin-latte.json
    ├── one-dark.json
    └── nord.json
```

## 13. Error Handling

- Validate file paths before running.
- Show user-friendly alerts when Python backend is missing or returns errors.
- Surface model download failures with retry action.
- Preserve partial progress on cancellation; allow resume where feasible.

## 14. Testing

- Unit tests for config/history persistence and bridge parsing using Bun's built-in test runner.
- Component render tests with Ink's testing utilities.
- End-to-end smoke test: launch TUI against a synthetic phantom and verify Dashboard renders.

## 15. Open Questions

1. Should the Python bridge be a long-running JSON-RPC server or a per-run subprocess?  
   *Recommendation:* Start with subprocess; migrate to server if live progress streaming needs improvement.
2. Should the TUI be shipped as a separate `fiber-tracer-tui` package or bundled inside the main Python package?  
   *Recommendation:* Keep as a separate `tui/` directory in the repo with its own `package.json`; Python `pyproject.toml` can optionally include it as an extra.
3. Which exact termcn components will we use?  
   *Recommendation:* Select, Table, Bar Chart, Spinner, Alert, Badge, Progress, Input, Box/Text from Ink. Pull components via shadcn CLI as needed.

## 16. Next Steps

1. Finalize this spec with user approval.
2. Create implementation plan (writing-plans skill).
3. Scaffold `tui/` with Bun + Ink + termcn.
4. Implement Layout, Settings, and Theme system first.
5. Implement Select Data → Configure → Review → Run wizard.
6. Integrate Python bridge and live progress.
7. Implement Dashboard, History, Logs.
8. Implement Model Registry, Experiments, Training.
9. Write tests and documentation.
10. Update CI to build/test the TUI.
