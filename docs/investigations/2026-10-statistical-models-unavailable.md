# Statistical backends unavailable on fresh installs (v2.54.0)

**Status:** Fixed in v2.54.0 by pinning `pyarrow<26.0.0` in
`ml-forecast-lab/requirements.txt`.

## What happened

- `statsforecast` depends on `fugue`, which depends on `pyarrow`.
- `pyarrow` declares no NumPy requirement in its metadata, but 26.0.0
  (released 2026-10-09) refuses to import under NumPy 1.x:
  `ImportError: pyarrow requires NumPy 2.0 or newer, found 1.26.4`.
- `requirements.txt` pins `numpy<2.0`, so pip installed pyarrow 26 next to
  NumPy 1.26 without complaint.

Effects:

- **App.** `statsforecast` failed to import, so `statsforecast_backend`
  recorded it as missing, and `ARIMAModel()`, `ETSModel()` and
  `ThetaModel()` raised "statsforecast is not installed" on construction.
  - Benchmarks could not score AutoARIMA, AutoETS or AutoTheta.
  - A retrain whose production model was one of them could not build it.
  - This applies to any image whose pip layer was built after 2026-10-09.
    A host that reused a cached layer from an earlier build was unaffected
    until its next uncached build.
- **CI.** The unit and integration jobs died before collecting a test.
  pytest loads `fugue_test` as a plugin entry point, and that plugin imports
  pyarrow. `main` was red from at least run 376 (the v2.52.5 merge). The
  smoke job, which installs only the web stack, stayed green.

## Fix and checks

`pyarrow<26.0.0`, with no lower bound, so armv7 resolves as before.

- `uv pip compile requirements.txt` for amd64 and aarch64
  (`manylinux_2_28`), Python 3.11 and 3.12, resolves to
  - pyarrow 25.0.1,
  - NumPy 1.26.4,
  - statsforecast 1.7.8,
  - datasets 5.1.0 (which needs `pyarrow>=24`).
- With NumPy 1.26.4:
  - pyarrow 25.0.0 and 25.0.1 import;
  - `statsforecast` imports;
  - all three statsforecast backends construct.
- pyarrow 26.0.0 reproduces the error.

## When to lift it

The pin can go once `numpy<2.0` does. That is a separate change: pandas,
torch and the tree backends would all need checking against NumPy 2, and
the gap-free bit-identity invariant re-verified.
