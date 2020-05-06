#include <cmath>
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"
#include "distributions/rng.hpp"

// An algorithm for drawing gamma random variables with small alpha
// parameters.  This code implements the algorithm described by Liu,
// Martin, and Syring http://arxiv.org/abs/1302.1884v3
namespace BOOM {
  namespace {
    // The algorithm in Liu et al. is implemented in terms of
    // constants they call w, h, r, and eta.  Those names are used
    // here for want of better ones, and so the reader can easily
    // verify that the code implemented here actually does what is in
    // the paper.
    inline double log_h(double z, double alpha) {
      return -z - exp(-z / alpha);
    }

   // The Liu, et al. the algorithm is implemented in terms of a
   // parameter eta.  We compute log eta instead.  This function takes
   // both lambda and log_lambda because it might be called repeatedly
   // in the rejection sampler, and we want to avoid taking the log of
   // lambda multiple times.
   inline double log_eta(double z, double log_w, double log_lambda,
                         double lambda) {
     if (z >= 0) {
       return -z;
     } else {
       return log_w + log_lambda + lambda * z;
     }
   }
  }  // namespace

  //======================================================================
  // Returns the log of a Gamma(alpha, 1) random variable.
  // Args:
  //   rng:  A U[0, 1) random number generator.
  //   alpha: The shape parameter of the gamma distribution.  Must
  //     satisfy 0 < alpha <= 0.3.
  // Returns:
  //   The log of a Gamma(alpha, 1) random variable.
  double rloggamma_small_alpha(RNG &rng, double alpha) {
    if (alpha <= 0) {
      report_error("alpha parameter must be positive.");
          }
    if (alpha > .3) {
      report_error("alpha parameter should be less than 0.3.  "
                   "Consider using rgamma() instead.");
    }
    // Compute a bunch of constants needed by the Liu et al. rejection sampler.
    static const double e = exp(1);
    const double w = alpha / (e * (1 - alpha));
    const double r = 1.0 / (1 + w);
    const double lambda = (1.0 / alpha) - 1.0;
    // Taking logs here means we don't need to repeatedly take logs of
    // the same numbers as part of the rejection sampling algorithm.
    const double log_w = log(w);
    const double log_lambda = log(lambda);
    const int max_number_of_attempts = 1000;
    for (int i = 0; i < max_number_of_attempts; ++i) {
      double u = rng();
      double z = u <= r ? -log(u / r) : log(rng()) / lambda;
      if (log_h(z, alpha) >=
          log(rng()) + log_eta(z, log_w, log_lambda, lambda)) {
        return -z / alpha;
      }
    }
    report_error("Max number of attempts exceeded.");
    // The following line will never be reached, but we need to avoid
    // falling off the end to prevent compiler warnings.
    return negative_infinity();
  }
}  // namespace BOOM
