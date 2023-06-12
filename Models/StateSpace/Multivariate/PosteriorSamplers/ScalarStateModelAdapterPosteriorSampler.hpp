#ifndef BOOM_STATE_SPACE_MULTIVARIATE_SCALAR_ADAPTER_POSTERIOR_SAMPLER_HPP_
#define BOOM_STATE_SPACE_MULTIVARIATE_SCALAR_ADAPTER_POSTERIOR_SAMPLER_HPP_
/*
  Copyright (C) 2005-2022 Steven L. Scott

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

#include "Models/StateSpace/Multivariate/StateModels/ScalarStateModelAdapter.hpp"

namespace BOOM {

  class CiScalarStateAdapterPosteriorSampler
      : public PosteriorSampler {
   public:

    explicit CiScalarStateAdapterPosteriorSampler(
        ConditionallyIndependentScalarStateModelMultivariateAdapter *model,
        RNG &seeding_rng = GlobalRng::rng);

    double logpri() const override;
    void draw() override;

   private:
    ConditionallyIndependentScalarStateModelMultivariateAdapter *model_;
  };

}  // namespace BOOM


#endif //  BOOM_STATE_SPACE_MULTIVARIATE_SCALAR_ADAPTER_POSTERIOR_SAMPLER_HPP_
