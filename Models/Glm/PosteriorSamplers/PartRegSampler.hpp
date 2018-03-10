// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2007 Steven L. Scott

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
#ifndef BOOM_PARTICLE_REGRESSION_SAMPLER_HPP
#define BOOM_PARTICLE_REGRESSION_SAMPLER_HPP

#include "Models/Glm/RegressionModel.hpp"
#include "distributions.hpp"
#include "stats/FreqDist.hpp"

namespace BOOM {

  class PartRegSampler {
    // assumes that the X's are included independently (no special
    // treatment for interactions.
   public:
    typedef std::map<Selector, double> LogpostTable;
    typedef std::map<Selector, uint> ModelCount;
    typedef std::pair<Selector, double> Mlike;

    // user supplies sufficient statistics
    PartRegSampler(uint Npart, const SpdMatrix &xtx, const Vector &xty,
                   double yty, const Vector &prior_mean,
                   const SpdMatrix &prior_ivar_, double prior_df,
                   double prior_sigma_guess, double inc_prob);

    // user supplies sufficient statistics
    PartRegSampler(uint Npart, const SpdMatrix &xtx, const Vector &xty,
                   double yty, const Vector &prior_mean,
                   const SpdMatrix &prior_ivar_, double prior_df,
                   double prior_sigma_guess, const Vector &inc_probs);

    void draw_model_indicators(uint ntimes = 1);
    void draw_params();  // to be called after "draw_models"

    void resample_models();
    void mcmc(uint niter);
    void set_number_of_mcmc_iterations(uint n);

    uint Nparticles() const;
    uint Nvars() const;
    const std::vector<Selector> &model_indicators() const;
    uint Nvisited() const;

    std::vector<Mlike> all_models() const;
    std::vector<Mlike> top_models(uint n = 100) const;
    Vector marginal_inclusion_probs() const;
    FrequencyDistribution model_sizes() const;

    double log_model_prob(const Selector &) const;
    double logprior(const Selector &g) const;
    double empirical_prob(const Selector &g) const;

   private:
    RNG rng_;
    const Ptr<NeRegSuf> suf_;

    const Vector prior_mean_;  // This stuff defines the prior
    SpdMatrix prior_ivar_;
    const double prior_df_;
    const double prior_ss_;
    const Vector inc_probs_;  // ind. prior: $prod_i Bern(\gamma_i | \pi_i)$

    std::vector<uint> indices_;
    uint Nmcmc_;

    mutable Vector beta_tilde_;     // this is work space for computing
    mutable SpdMatrix iV_tilde_;    // posterior model probs
    std::vector<Selector> models_;  // this is the actual
    std::vector<Vector> betas_;
    Vector sigsq_;  // one for each model
    Vector weights_;
    mutable LogpostTable logpost_;
    // logpost_ needs to be mutable because it might change when you
    // compute probabilities for a new model

    ModelCount model_counts_;

    double compute_log_model_prob(const Selector &g) const;
    void mcmc_one_var(Selector &mod);
    void mcmc_all_vars(Selector &mod);
    void mcmc_one_flip(Selector &mod, uint which_var);
    void draw_initial_particles(uint N);
    double set_reg_post_params(const Selector &g, const SpdMatrix &Ominv) const;
  };
}  // namespace BOOM
#endif  // BOOM_PARTICLE_REGRESSION_SAMPLER_HPP
