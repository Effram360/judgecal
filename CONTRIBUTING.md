# Contributing to judgecal

Thanks for contributing! judgecal is small and statistics-forward; the bar is
correctness, determinism, and tests.

## Dev setup

We use [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/Effram360/judgecal
cd judgecal
uv venv
uv pip install -e ".[dev]"
```

(Plain `python -m venv` + `pip install -e ".[dev]"` works too.)

## Test and lint

```bash
.venv/bin/python -m pytest                  # fast suite (default; excludes slow/network)
.venv/bin/python -m pytest -m slow          # long statistical coverage checks
.venv/bin/python -m ruff check src tests    # lint (line length 100)
.venv/bin/python -m mypy src                # types (pragmatic ignores allowed)
```

All of `pytest` (fast suite) and `ruff check` must pass before a PR is mergeable;
CI runs them on Linux and macOS across Python 3.10/3.12/3.13.

## The validation gate

Any change touching `src/judgecal/stats/`, `src/judgecal/probes/`, or
`src/judgecal/fixtures/` must keep the planted-bias recovery suite green:

```bash
.venv/bin/judgecal validate
```

This is the "we test the tester" contract: probes must recover analytically known
planted biases within their CIs, and stay silent on the null judge. If your change
legitimately moves an analytic truth, update the mock judge's expectation functions
and the validation scenarios *in the same PR*, with an explanation.

## Hard rules

- **No network in tests.** Tests run fully offline on an 8GB laptop. Anything
  touching Hugging Face or a real model must be mocked, or marked
  `@pytest.mark.network` (excluded by default). Never invoke a real LLM in tests.
- **Determinism.** Every stochastic function takes an explicit `seed` or
  `rng: np.random.Generator`. No `random` module, no wall-clock-dependent behavior
  in `src/` (the CLI owns the clock).
- **Runtime deps are frozen** at numpy/scipy/pandas/pydantic/click. `statsmodels`
  is dev/test-only (cross-validation references) — never import it from `src/`.
  `inspect-ai` and `datasets` live behind optional extras with guarded imports.
- **Style:** `from __future__ import annotations`, type hints throughout,
  Google-style docstrings, ruff-clean.
- New estimators need a cross-validation test against a statsmodels (or equivalent
  reference) implementation, plus behavioral tests.
