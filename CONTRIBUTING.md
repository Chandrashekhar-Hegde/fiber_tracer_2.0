# Contributing to Fiber Tracer

Thank you for considering a contribution. This document explains how to set up a development environment, run tests, open issues, and submit pull requests.

## Development setup

1. Fork and clone the repository:

   ```bash
   git clone https://github.com/Chandrashekhar-Hegde/fiber_tracer_2.0.git
   cd fiber_tracer_2.0
   ```

2. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the package in editable mode with development dependencies:

   ```bash
   pip install -e ".[dev]"
   ```

   To also work on the ML backend, install the `ml` extra:

   ```bash
   pip install -e ".[dev,ml]"
   ```

## Running tests and checks

Before submitting a PR, run the full quality suite:

```bash
ruff check .
black --check .
pytest tests/
mypy src/fiber_tracer
python scripts/check_doc_links.py
```

If `black` reports formatting issues, run `black .` to fix them.

`check_doc_links.py` fails if `src/`, a top-level `docs/` page, or a root
`*.md` cites a doc or script path that does not exist in the repo. See
[Graduating a spike](#graduating-a-spike) for the mistake it exists to catch.

## Branch and commit style

- Create a feature branch from `main`:

  ```bash
  git checkout -b feature/your-feature-name
  ```

- Use clear, descriptive commit messages in the imperative mood:

  ```text
  Add domain-randomization augmentation for U-Net training
  Fix foreground sampling threshold for thin fibers
  Update validation protocol with GF-PA66 metrics
  ```

- Keep commits focused and atomic.

## Graduating a spike

Research spikes are explored on a `research/*-spike` branch. When a spike
graduates into a shipped feature, the feature PR must leave `main`
self-contained:

- **Land the spike's design spec** with the feature PR. Shipped code routinely
  cites it for the reasoning behind a threshold or an architecture choice, and
  that citation has to resolve for anyone reading `main`.
- **Do not land proof-of-concept scripts the feature supersedes.** If the PoC's
  functions moved into the package, or its checks became real tests, the script
  is a stale duplicate. Point readers at the shipped code instead.
- **Never cite a branch from `src/` or a user-facing doc.** Branches get
  deleted; `main` should not depend on one surviving.

This is worth spelling out because it went wrong three times in a row (DVC,
DIC, digital twin): each spike branch was worked directly and never opened as
a PR, so only the follow-up feature PR landed and `main` was left citing spec
files it did not contain. `scripts/check_doc_links.py` now enforces the
reference half of this in CI.

## Pull request process

1. Ensure the checklist in the PR template is complete.
2. Update `CHANGELOG.md` for user-facing changes.
3. Update relevant documentation (`README.md`, `docs/`, model card).
4. If you add a dataset, cite it in `CITATIONS.md` and confirm its open license.
5. Open the PR against `main`; the project owner will review it when time permits.

## Adding datasets

We welcome open-licensed XCT datasets that improve training or benchmarking. Please open a **Dataset request** issue with:

- Dataset name, URL/DOI, and license.
- Fiber and matrix type, resolution, and whether ground truth is available.
- Why it fills a gap in the current corpus.

Do not commit raw data or model checkpoints to git. Use `scripts/download_datasets.py` patterns for downloaders.

## Adding models

Model checkpoints should be:

- Trained using scripts in `scripts/` so the process is reproducible.
- Validated against GF-PA66 or another open ground-truth dataset.
- Uploaded to a GitHub Release, not committed to the repository.
- Documented in `docs/MODEL_CARD.md` with architecture, training data, and limitations.

## Contributing to the AI workflow

The model registry, experiment tracking, training CLI, and TUI screens are the newest parts of the codebase. If you want to extend them, the relevant modules are:

| Module / file | Responsibility |
|---------------|----------------|
| `src/fiber_tracer/models/registry.py` | Add/remove/default local model entries stored in `~/.config/fiber-tracer/models.json`. |
| `src/fiber_tracer/experiments/store.py` | Create/update/list experiment records stored in `~/.config/fiber-tracer/experiments.jsonl`. |
| `src/fiber_tracer/training/trainer.py` | The reusable `UNetTrainer` and JSON progress emission. |
| `src/fiber_tracer/cli.py` | CLI subcommands `model`, `experiment`, and `train`. |
| `tui/src/bridge.ts` | TypeScript bridge that calls the new CLI subcommands and parses JSON output. |
| `tui/src/components/model-registry.tsx`, `experiments.tsx`, `training.tsx` | TUI screens that display live registry/experiment/training data. |

Guidelines:

- Keep the registry and store file-based and cross-platform (POSIX and Windows paths). Use `fiber_tracer.utils.paths.get_config_dir()` for the config directory.
- Maintain backward compatibility when changing the JSON/JSONL schemas; provide a migration path for existing user data.
- Add tests for new registry/store operations in `tests/test_model_registry.py` and `tests/test_experiment_store.py`.
- Add CLI tests in `tests/test_cli_model_experiment_train.py` or `tests/test_cli_train_integration.py`.
- Add TUI bridge tests in `tui/src/bridge.test.ts`.
- Training changes should still run in a reasonable time on CPU for CI; use small synthetic patches for unit tests.

## Code of conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting help

- Open a [Discussion](https://github.com/Chandrashekhar-Hegde/fiber_tracer_2.0/discussions) for questions and ideas.
- Open an [Issue](https://github.com/Chandrashekhar-Hegde/fiber_tracer_2.0/issues) for bugs and concrete feature requests.
- See `ROADMAP.md` for planned work.
