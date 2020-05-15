/*
  Copyright (C) 2005-2020 Steven L. Scott

  This library is free software; you can redistribute it and/or modify it under
  the terms of the GNU Lesser General Public License as published by the Free
  Software Foundation; either version 2.1 of the License, or (at your option)
  any later version.

  This library is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
  details.

  You should have received a copy of the GNU Lesser General Public License along
  with this library; if not, write to the Free Software Foundation, Inc., 51
  Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
*/

#include "Models/Glm/PosteriorSamplers/LoglinearModelBipfSampler.hpp"
#include "distributions/trun_gamma.hpp"

namespace BOOM {

  namespace {
    using BIPF = LoglinearModelBipfSampler;
  }  // namespace

  BIPF::LoglinearModelBipfSampler(LoglinearModel *model,
                                  double prior_count,
                                  double min_scale,
                                  RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        prior_count_(prior_count),
        min_scale_(min_scale)
  {}

  void BIPF::draw() {
    for (int i = 0; i < model_->number_of_effects(); ++i) {
      draw_effect_parameters(i);
    }
    draw_intercept();
  }

  double BIPF::logpri() const {
    return negative_infinity();
  }

  void BIPF::draw_intercept() {
    double n = model_->suf()->sample_size();
    double b0 = rgamma_mt(rng(), n + prior_count_, n + prior_count_);
    model_->prm()->set_element(log(b0), 0);
  }

  void BIPF::draw_effect_parameters(int effect_index) {

    //
    // TODO: This needs to be redone with the GIG distribution.
    //
    report_error("The loglinear model posterior sampler needs to be"
                 " redone with draws from the GIG distribution.");

    const CategoricalDataEncoder &encoder(model_->encoder(effect_index));
    const std::vector<int> &which_variables(encoder.which_variables());

    Vector adjusted_counts(encoder.dim(), 0.0);
    const Array& counts(model_->suf()->margin(which_variables));
    double sample_size = model_->suf()->sample_size();
    std::vector<int> indices(model_->nvars(), 0);

    for (auto it = counts.abegin(); it != counts.aend(); ++it) {
      for (int i = 0; i < which_variables.size(); ++i) {
        indices[which_variables[i]] = it.position()[i];
      }
      adjusted_counts += *it * encoder.encode(indices);
    }

    // adjusted_counts now contains the count associated with each parameter.
    // The count starts off with the number of times the base level for that
    // parameter was observed.  It then subtracts off the counts for the
    // reference level of the next margin up, etc.

    // The count in adjusted_counts might be negative.
    Vector coefficients(adjusted_counts.size());

    for (size_t i = 0; i < adjusted_counts.size(); ++i) {
      coefficients[i] = log(rtrun_gamma_mt(
          rng(),
          adjusted_counts[i] + prior_count_,
          1 + sample_size,
          min_scale_));
    }

    model_->set_effect_coefficients(coefficients, effect_index);
  }

}  // namespace BOOM
