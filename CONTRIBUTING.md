# Contributing to Fiber Tracer

Thank you for considering a contribution. This document explains how to set up a development environment, run tests, open issues, and submit pull requests.

## Development setup

1. Fork and clone the repository:

   ```bash
   git clone https://github.com/llMr-Sweetll/fiber_tracer_2.0.git
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
```

If `black` reports formatting issues, run `black .` to fix them.

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

## Pull request process

1. Ensure the checklist in the PR template is complete.
2. Update `CHANGELOG.md` for user-facing changes.
3. Update relevant documentation (`README.md`, `docs/`, model card).
4. If you add a dataset, cite it in `CITATIONS.md` and confirm its open license.
5. Request review from a maintainer.

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

## Code of conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting help

- Open a [Discussion](https://github.com/llMr-Sweetll/fiber_tracer_2.0/discussions) for questions and ideas.
- Open an [Issue](https://github.com/llMr-Sweetll/fiber_tracer_2.0/issues) for bugs and concrete feature requests.
- See `ROADMAP.md` for planned work.
