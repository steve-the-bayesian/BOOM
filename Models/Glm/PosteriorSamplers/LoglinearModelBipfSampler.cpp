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

namespace BOOM {

  namespace {
    using BIPF = LoglinearModelBipfSampler;
  }  // namespace

  BIPF::LoglinearModelBipfSampler(LoglinearModel *model,
                                  RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model)
  {}

  void BIPF::draw() {
    for (int i = 0; i < model_->number_of_effects(); ++i) {
      draw_effect_parameters(i);
    }

  }

  double BIPF::logpri() const {
    return negative_infinity();
  }

  void BIPF::draw_effect_parameters(int effect_index) {
    const CategoricalDataEncoder &encoder(model_->encoder(effect_index));
    const std::vector<int> &which_variables(encoder.which_variables());
    const Array& counts(model_->suf()->margin(which_variables));
  }

}  // namespace BOOM
