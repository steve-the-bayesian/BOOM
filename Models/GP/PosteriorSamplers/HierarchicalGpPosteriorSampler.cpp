/*
  Copyright (C) 2005-2023 Steven L. Scott

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

#include "Models/GP/PosteriorSamplers/HierarchicalGpPosteriorSampler.hpp"
#include "distributions.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  HierarchicalGpPosteriorSampler::HierarchicalGpPosteriorSampler(
      HierarchicalGpRegressionModel *model, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model)
  {}

  double HierarchicalGpPosteriorSampler::logpri() const {
    return negative_infinity();
  }

  void HierarchicalGpPosteriorSampler::draw() {

    for (int i = 0; i < model_->number_of_groups(); ++i) {
      // model_->data_model(i)->sample_posterior();
      // model_->data_model(i)->sample_function_values();
      // model_->adjust_data(i)
    }


  }


}  // namespace BOOM
