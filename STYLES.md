# STYLES.md

How code is written in BOOM. These conventions are described from the existing code; when editing a legacy file that deviates, match the surrounding file.

## C++

### Files

- One class per `Foo.hpp`/`Foo.cpp` pair named after the class; free-function utilities are lowercase (`cpputil/report_error.hpp`). Tests are `snake_case_test.cc` in a `tests/` subdirectory next to the code.
- Every file starts with the standard copyright block (Google line + Steven L. Scott LGPL block). Copy it verbatim from a neighboring file onto new files.
- Include guards are `#ifndef BOOM_<NAME>_HPP_` style, never `#pragma once`. Close with `#endif  // BOOM_<NAME>_HPP_`.
- In-repo headers are included with root-relative quoted paths (`#include "Models/Glm/Glm.hpp"`); a `.cpp` includes its own header first.

### Naming

- `PascalCase` classes, `lower_snake_case` methods and free functions, `ALL_CAPS` enumerators and macros.
- Member variables end in a trailing underscore (`reference_count_`).

### Ownership and errors

- Heap objects use the intrusive `Ptr<T>` smart pointer with `RefCounted`, allocated via the `NEW(Type, var)(args)` macro. Never `std::shared_ptr`. Cast through `.dcast<>()` / `.scast<>()`, not raw `dynamic_cast`.
- Errors: build the message in a local `ostringstream err;` and call `report_error(err.str())`. Don't throw `std::runtime_error` directly.

### Documentation

Plain `//` comments in the header, directly above the declaration, using indented `Args:` / `Returns:` / `Effects:` sections. Never Doxygen markup. Example:

```cpp
// Log likelihood function.
// Args:
//   beta: The vector of included coefficients (i.e. the dimension
//     matches that of included_coefficients()).
//
// Returns:
//   The value of log likelihood at the supplied beta.
```

`Effects:` documents side effects of mutating or void methods.

### Formatting

`.clang-format` at the repo root (Google style, 80 columns, 2-space indent, namespace bodies indented). Run clang-format on new code; don't reformat untouched code.

### Tests

googletest fixtures that seed the global RNG in the constructor (`GlobalRng::rng.seed(8675309);`), wired as `cc_test` (`size = "small"`) against `//:boom`, `//:boom_test_utils`, and `@googletest//:gtest_main`. Use `MatrixEquals` / `VectorEquals` from `test_utils` for array comparisons.

## Python (BayesBoom)

### Object model

- **Two-tier pattern.** Every model/bandit/encoder is a pure-Python class that is the source of truth for its own state, holding lazily-built `self._boom_*` handles. A `boom()` method constructs the C++ object on demand. `__setstate__` resets the `_boom_*` handles to `None`.
- **Conversions go through the converters.** Use `R.to_boom_vector` / `R.to_boom_matrix` / `R.to_boom_spd` / `R.to_numpy` at the numpy/pandas ⇄ boom boundary; never construct `boom.Vector(...)` directly. Coerce scalars explicitly at the C++ boundary (`int(arm)`, `float(x)`).
- **Serialization:** user-facing classes implement `__getstate__`/`__setstate__` and a paired `<Thing>JsonEncoder` / `<Thing>JsonDecoder`, registered in the module-level registry where one exists. Tests round-trip both.

### Docstrings

Google-derived dialect with `Args:` / `Returns:` / `Effects:` / `Raises:` sections. `Effects:` documents side effects of mutating methods. Two-space hanging indent for argument descriptions; identifiers quoted with single quotes in prose. Every public function, method, and class gets one; `_`-private helpers may skip it. Class docstrings describe the concept; `__init__` carries the `Args:`. Example:

```python
def add_factor(self, factor_name: str, factor_levels, baseline_level: str = ""):
    """
    Add a factor to the experiment.

    Args:
      factor_name: The name of the experimental factor.  Think of this as
        the variable name in a data frame.
      factor_levels: The possible values the factor can assume.  A list of
        strings.
      baseline_level: The level of the factor to leave out when creating
        dummy variables.

    Effects:
      The internal structure is updated to reflect the additional factor and
      levels.
    """
```

### Naming and types

- `snake_case` functions and modules, `PascalCase` classes, `_`-prefixed private state, `UPPER_SNAKE` module constants.
- Type hints are sparse and deliberate: annotate scalar parameters (`arm: int`); numpy/pandas parameters are documented in the docstring instead.
- Imports are aliased canonically: `import BayesBoom.boom as boom`, `import BayesBoom.R as R`, `import BayesBoom.models as models`, `import numpy as np`, `import pandas as pd`.

### Tests

`unittest.TestCase` in `test_*.py`, `_make_*` factory helpers for fixtures, seeds set in `setUp` (`np.random.seed(8675309)`), and `np.testing.assert_array_almost_equal` for arrays.

### Formatting

80-column limit (flake8, the only configured lint rule), 4-space indent.
