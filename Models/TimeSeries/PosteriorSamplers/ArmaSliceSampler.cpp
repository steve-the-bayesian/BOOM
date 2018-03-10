/*
  Copyright (C) 2005-2018 Steven L. Scott

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2.1 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA
*/

#include "Models/TimeSeries/PosteriorSamplers/ArmaSliceSampler.hpp"

namespace BOOM {

  ArmaSliceSampler::ArmaSliceSampler(ArmaModel *model,
                                     const Ptr<VectorModel> &ar_prior,
                                     const Ptr<VectorModel> &ma_prior,
                                     const Ptr<DoubleModel> &precision_prior,
                                     RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        ar_prior_(ar_prior),
        ma_prior_(ma_prior),
        precision_prior_(precision_prior),
        sampler_(
            [this](const Vector &parameters) {
              return vectorized_log_posterior(parameters);
            },
            1.0, false, &rng()) {}

  void ArmaSliceSampler::draw() {
    Vector parameters = model_->vectorize_params();
    // The model is parameterized in terms of sigsq.  The log posterior is
    // parameterized in terms of 1.0 / sigsq.
    parameters.back() = 1.0 / parameters.back();
    parameters = sampler_.draw(parameters);
    parameters.back() = 1.0 / parameters.back();
    model_->unvectorize_params(parameters);
  }

  double ArmaSliceSampler::logpri() const {
    return ar_prior_->logp(model_->ar_coefficients()) +
           ma_prior_->logp(model_->ma_coefficients()) +
           precision_prior_->logp(1.0 / model_->sigsq());
  }

  double ArmaSliceSampler::log_posterior(const Vector &ar_coefficients,
                                         const Vector &ma_coefficients,
                                         double precision) const {
    double ans = ar_prior_->logp(ar_coefficients) +
                 ma_prior_->logp(ma_coefficients) +
                 precision_prior_->logp(precision);
    if (std::isfinite(ans)) {
      ans += model_->log_likelihood(ar_coefficients, ma_coefficients,
                                    1.0 / precision);
    }
    return ans;
  }

  double ArmaSliceSampler::vectorized_log_posterior(
      const Vector &parameters) const {
    if (parameters.size() !=
        model_->ar_dimension() + model_->ma_dimension() + 1) {
      report_error("Wrong size parameter vector.");
    }
    ConstVectorView ar(parameters, 0, model_->ar_dimension());
    ConstVectorView ma(parameters, model_->ar_dimension(),
                       model_->ma_dimension());
    double siginv = parameters.back();
    return log_posterior(ar, ma, siginv);
  }

}  // namespace BOOM
