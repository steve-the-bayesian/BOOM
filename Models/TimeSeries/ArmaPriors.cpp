#include "Models/TimeSeries/ArmaPriors.hpp"
#include "Models/TimeSeries/ArmaModel.hpp"
#include "cpputil/math_utils.hpp"
#include "distributions.hpp"

namespace BOOM {

  double UniformMaPrior::logp(const Vector &x) const {
    if (ArmaModel::is_causal(x)) {
      return 0.0;
    } else {
      return negative_infinity();
    }
  }

  Vector UniformMaPrior::sim(RNG &rng) const {
    Vector ans(dim_);
    for (int i = 0; i < 1000; ++i) {
      for (int j = 0; j < dim_; ++j) {
        ans[j] = runif_mt(rng, -1, 1);
      }
      if (logp(ans) > -1) {
        return ans;
      }
    }
    report_error(
        "Could not simulate from UniformMaPrior.  "
        "Maybe dimension is too high?");
    return Vector(0);
  }

  double UniformArPrior::logp(const Vector &x) const {
    if (ArmaModel::is_invertible(x)) {
      return 0.0;
    } else {
      return negative_infinity();
    }
  }

  Vector UniformArPrior::sim(RNG &rng) const {
    Vector ans(dim_);
    for (int i = 0; i < 1000; ++i) {
      for (int j = 0; j < dim_; ++j) {
        ans[j] = runif_mt(rng, -1, 1);
      }
      if (logp(ans) > -1) {
        return ans;
      }
    }
    report_error("Simulation failed.  Maybe dimension is too high?");
    return Vector(0);
  }

}  // namespace BOOM
