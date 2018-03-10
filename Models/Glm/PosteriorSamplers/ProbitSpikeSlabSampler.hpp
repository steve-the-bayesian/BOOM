// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2009 Steven L. Scott

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
#ifndef BOOM_PROBIT_SPIKE_SLAB_SAMPLER_HPP_
#define BOOM_PROBIT_SPIKE_SLAB_SAMPLER_HPP_
#include "Models/Glm/PosteriorSamplers/ProbitRegressionSampler.hpp"
#include "Models/Glm/VariableSelectionPrior.hpp"
namespace BOOM {

  class ProbitSpikeSlabSampler : public ProbitRegressionSampler {
    // inheriting from ProbitRegressionSampler gives impute_latent_data
    // and access to complete data sufficient statistics
   public:
    ProbitSpikeSlabSampler(ProbitRegressionModel *model,
                           const Ptr<MvnBase> &prior,
                           const Ptr<VariableSelectionPrior> &vspri,
                           bool check_initial_condition = true,
                           RNG &seeding_rng = GlobalRng::rng);
    double logpri() const override;
    void draw() override;
    void limit_model_selection(uint n);
    void suppress_model_selection();
    void allow_model_selection();
    uint max_nflips() const;

    void draw_gamma();
    void draw_beta() override;

   private:
    bool keep_flip(double logp_new, double logp_old) const;
    double log_model_prob(const Selector &inc);

    ProbitRegressionModel *m_;
    Ptr<MvnBase> beta_prior_;
    Ptr<VariableSelectionPrior> gamma_prior_;

    SpdMatrix Ominv_;
    SpdMatrix iV_tilde_;
    uint max_nflips_;
    bool allow_selection_;
    Vector beta_, wsp_;
  };

}  // namespace BOOM
#endif  // BOOM_PROBIT_SPIKE_SLAB_SAMPLER_HPP_
