// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2012 Steven L. Scott

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

#ifndef BOOM_INDEPENDENT_MVN_CONJ_SAMPLER_HPP_
#define BOOM_INDEPENDENT_MVN_CONJ_SAMPLER_HPP_

#include "Models/IndependentMvnModel.hpp"
#include "Models/PosteriorSamplers/GenericGaussianVarianceSampler.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "cpputil/math_utils.hpp"

namespace BOOM {

  // Posterior sampler for and MvnModel assuming the off-diagonal
  // elements of Sigma are all zero.  Draws each element of mu and
  // Sigma independently from the NormalInverseGamma distribution.
  class IndependentMvnConjSampler : public PosteriorSampler {
   public:
    IndependentMvnConjSampler(IndependentMvnModel *model,
                              const Vector &mean_guess,
                              const Vector &mean_sample_size,
                              const Vector &sd_guess,
                              const Vector &sd_sample_size,
                              const Vector &sigma_upper_limit,
                              RNG &seeding_rng = GlobalRng::rng);

    IndependentMvnConjSampler(IndependentMvnModel *model,
                              double mean_guess,
                              double mean_sample_size,
                              double sd_guess,
                              double sd_sample_size,
                              double sigma_upper_limit = infinity(),
                              RNG &seeding_rng = GlobalRng::rng);

    IndependentMvnConjSampler *clone_to_new_host(
        Model *new_host) const override;

    void draw() override;
    double logpri() const override;

   private:
    void check_sizes(const Vector &sigma_upper_limit);
    void check_vector_size(const Vector &v, const char *vector_name);

    IndependentMvnModel *model_;
    Vector mean_prior_guess_;
    Vector mean_prior_sample_size_;
    Vector prior_ss_;
    Vector prior_df_;
    std::vector<GenericGaussianVarianceSampler> sigsq_samplers_;
  };

}  // namespace BOOM
#endif  // BOOM_INDEPENDENT_MVN_CONJ_SAMPLER_HPP_
