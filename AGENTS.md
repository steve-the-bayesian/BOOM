# AGENTS.md

Guidance for AI coding agents working in BOOM. Read this before proposing any
change. Code style lives in [STYLES.md](STYLES.md); read that too before writing
or editing code.

## How to behave in this repo

- **Propose minimal diffs.** Show the changed lines in isolation and explain the
    fix *before* writing files. Never bury a small fix inside a large rewrite of
    the surrounding code.

- **No drive-by changes.** Don't reformat, rename, or restructure code you
    weren't asked to touch. When editing a legacy file, match its local style
    even where it deviates from the conventions in STYLES.md.

- **Explain before implementing.** For any non-trivial fix, state the bug, the
    cause, and the proposed change in a few sentences first, and wait for
    agreement.

## Repo layout

- C++ library at the root (`Models/`, `LinAlg/`, `Samplers/`, `distributions/`,
  `stats/`, `cpputil/`, `numopt/`, `TargetFun/`, `Bandits/`). Everything lives
  in `namespace BOOM`.

- `Interfaces/python/` — the BayesBoom Python namespace package (pybind11
  bindings plus pure-Python subpackages: `R`, `models`, `bandits`, `bsts`,
  `spikeslab`, ...).

- `Interfaces/R/` — R packages (`Boom` core, then `BoomSpikeSlab`, `bsts`, ...).

- `Eigen/` is vendored. Never modify it or add an external Eigen dependency.

- `python_package/` and `rpackage/` are throwaway staging directories recreated
  by the install scripts. Never edit files there; edit the originals under
  `Interfaces/`.

## Build and test

- **C++ build:** `bazel build boom` (add `-c opt` for optimized). On macOS the
    hardcoded `-lpthread` in the top-level `BUILD` may need removing locally.

- **C++ tests:** `./testall` (wraps `bazel test` over
    `Models/... cpputil/... LinAlg/... Samplers/... stats/... distributions/...`),
    or target a subtree directly: `bazel test //Models/Glm/...`.

- **Python package:** built only via `./install/pyboom` from the repo root,
    which stages C++ sources into `python_package/` and builds the wheel. `pip
    install` on `Interfaces/python/BayesBoom` alone will fail. The compile is
    long (the entire C++ library builds into one `_boom` extension).

- **Python tests:** stdlib `unittest` (`python -m unittest`), and they require
    the compiled `BayesBoom.boom` module to be installed first.

- **R packages:** `./install/create_boom_rpackage -i` first (the others
    `LinkingTo: Boom`), then `./install/boom_spike_slab -i`, `./install/bsts
    -i`. macOS needs `gsed`.
