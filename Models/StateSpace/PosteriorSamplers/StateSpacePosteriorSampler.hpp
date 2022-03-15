// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2011 Steven L. Scott

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
#ifndef BOOM_STATE_SPACE_POSTERIOR_SAMPLER_HPP_
#define BOOM_STATE_SPACE_POSTERIOR_SAMPLER_HPP_

#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "Models/StateSpace/StateSpaceModelBase.hpp"

namespace BOOM {
  class StateSpacePosteriorSampler : public PosteriorSampler {
   public:
    // Args:
    //   model:  The state space model to be managed.
    //   seeding_rng: The random number generator used to set the seed
    //     for the RNG owned by this sampler.
    explicit StateSpacePosteriorSampler(StateSpaceModelBase *model,
                                        RNG &seeding_rng = GlobalRng::rng);

    StateSpacePosteriorSampler *clone_to_new_host(
        Model *new_host) const override;

    void draw() override;
    double logpri() const override;

    // Set the model parameters equal to the posterior mode.  Note
    // that some state models are not amenable to this method, such as
    // regression models with a spike and slab prior.  If the model
    // contains such a state model then an exception will be thrown.
    //
    // Args:
    //   epsilon: Convergence for optimization algorithm will be
    //     declared when consecutive values of (log_likelihood +
    //     log_prior) are observed with a difference of less than
    //     epsilon.
    //
    void find_posterior_mode(double epsilon = 1e-5) override;

    // Returns the log posterior density evaluated at the vector of
    // parameters.
    // Args:
    //   parameters: A vector of model parameters, in the order
    //     returned by model_->vectorize_params(true).
    double log_prior_density(const ConstVectorView &parameters) const override;

    // Args:
    //   parameters: A vector of model parameters, in the order
    //     returned by model_->vectorize_params(true).
    //   gradient: A vector of the same size as parameters.  The
    //     elements of the vector will be incremented by the gradient
    //     of the log prior density with respect to the parameters.
    //
    // Returns:
    //   The log prior density evaluated at the specified parameters.
    double increment_log_prior_gradient(const ConstVectorView &parameters,
                                        VectorView gradient) const override;

    void disable_threads() { pool_.set_number_of_threads(-1); }

   protected:
    // Samplers for models with observation equations that are
    // conditionally normal can override this function to impute the
    // mixing distribution.  For Gaussian observation models it is a
    // no-op.
    virtual void impute_nonstate_latent_data() {}

   private:
    // The M step in an EM algorithm for finding the posterior mode.
    // The Estep is provided by the model.  The Mstep is kept here
    // because it needs access to the prior.
    void Mstep();

    // Specific mode finding methods used to implement
    // find_posterior_mode.
    void find_posterior_mode_using_em(double epsilon, int max_steps);
    void find_posterior_mode_numerically(double epsilon);

    StateSpaceModelBase *model_;
    bool latent_data_initialized_;

    ThreadWorkerPool pool_;
  };
}  // namespace BOOM
#endif  // BOOM_STATE_SPACE_POSTERIOR_SAMPLER_HPP_
