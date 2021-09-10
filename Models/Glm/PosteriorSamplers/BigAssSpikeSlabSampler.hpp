#ifndef BOOM_GLM_BIG_ASS_SPIKE_SLAB_HPP_
#define BOOM_GLM_BIG_ASS_SPIKE_SLAB_HPP_

/*
  Copyright (C) 2005-2021 Steven L. Scott

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

#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "Models/Glm/RegressionModel.hpp"
#include "Models/Glm/RegressionSlabPrior.hpp"
#include "Models/Glm/VariableSelectionPrior.hpp"
#include "Models/Glm/PosteriorSamplers/BregVsSampler.hpp"

#include "Models/GammaModel.hpp"


namespace BOOM {

  class BigAssSpikeSlabSampler : public PosteriorSampler {
   public:
    // Args:
    //   model: The model to be sampled.
    //   global_spike: A prior over the inclusion/exclusion decisions for the
    //     global model.  This prior will be subdivided into smaller priors for
    //     the initial screen models.  Because of this subdivision, the prior
    //     must be independent across variables.
    //   slab_prototype: An EMPTY regression slab prior.  This serves as the
    //     prototype prior for the sub-models and for the global model once the
    //     relevant candidate predictor variables have been identified by the
    //     initial screen.
    //   residual_precision_prior: The prior distribution on the residual
    //     precision parameter.  This will be cloned across the initial screen
    //     models.
    //   seeding_rng: The random number used to seed the RNG held by the Sampler
    //     base class to this object.
    BigAssSpikeSlabSampler(BigRegressionModel *model,
                           const Ptr<VariableSelectionPrior> &global_spike,
                           const Ptr<RegressionSlabPrior> &slab_prototype,
                           const Ptr<GammaModelBase> &residual_precision_prior,
                           RNG &seeding_rng = GlobalRng::rng);

    void draw() override;
    double logpri() const override;

    void initial_screen(int niter, double threshold);

    ConstVectorView select_chunk(const Vector &v, int chunk) const;

   private:
    // The basic objects provided to the constructor.
    BigRegressionModel *model_;
    Ptr<VariableSelectionPrior> spike_;
    Ptr<RegressionSlabPrior> slab_prototype_;
    Ptr<GammaModelBase> residual_precision_prior_;

    // The posterior samplers assigned to the subordinate models held by model_.
    std::vector<Ptr<BregVsSampler>> intial_screen_samplers_;

    // Objects used to implement the MCMC for the restricted set of variables
    // identified by the initial screen.
    Ptr<RegressionSlabPrior> candidate_slab_;
    Ptr<BregVsSampler> candidate_sampler_;

    // Assigns posterior samplers to the subordinate models contained in model_.
    void assign_subordinate_samplers();

    // Assign data to workers in different threads.
    void run_parallel_initial_screen(int niter);

    // After the initial screen is completed, identify the candidate variables
    // to be sampled in the primary MCMC phase.
    void set_candidate_variables(double threshold);

    // Set up the posterior sampler for the restricted model (held in model_) if
    // it has not already been set.
    void ensure_restricted_model_sampler();
  };

}  // namespace BOOM


#endif  // BOOM_GLM_BIG_ASS_SPIKE_SLAB_HPP_
