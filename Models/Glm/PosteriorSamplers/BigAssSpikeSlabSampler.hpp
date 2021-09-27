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

    // Take one MCMC step for the restricted model, and map the draw to the
    // overall model.
    //
    // Preconditions:
    //   - This call will fail if "initial_screen" has not been called.
    //   - This call will fail if the data has not passed through
    //     "stream_data_for_restricted_model".
    void draw() override;

    double logpri() const override;

    // Run an MCMC algorithm over each sub-model to identify candidate predictor
    // variables.
    //
    // Args:
    //   niter:  The number of MCMC iterations to use.
    //   threshold: The variables whose 'marginal inclusion probabilities'
    //     exceed 'threshold' become candidates in the next round.
    //   use_threads: If 'true' then C++11 threads will be used to run the MCMC
    //     algorithms for the subordinate models.  If 'false' then the code path
    //     for doing the MCMC will not use threads.
    //
    // Preconditions:
    //   It is expected that the model being screened has passed the data
    //   through "stream_data_for_initial_screen".
    //
    // Postconditions:
    //   The "restricted_model_" and "predictor_candidates_" members of the
    //   managed model are set.
    void initial_screen(int niter, double threshold, bool use_threads);

    // Args:
    //   v: A Vector of objects to be selected.  The elements of v correspond to
    //     columns in the global predictor matrix.
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
    // Args:
    //   niter: The number of MCMC iterations to use in the initial screen on
    //     each worker.
    //   threshold: The inclusion probability that must be exceeded on a worker
    //     run for a predictor variable to be promomted to a candidate for the
    //     next round.
    //   use_threads: A debugging flag.  If false then a code path will be used
    //     that does not invoke threading tools.  This will be slow, but it
    //     makes gdb easy.
    //
    // Effects:
    //   The inclusion flags for each worker model are set according to whether
    //   each variables's marginal inclusion probability exceeds the specified
    //   threshold.
    void run_parallel_initial_screen(int niter, double threshold,
                                     bool use_threads);

    // After the initial screen is completed, identify the candidate variables
    // to be sampled in the primary MCMC phase.
    void set_candidate_variables();

    // Set up the posterior sampler for the restricted model (held in model_) if
    // it has not already been set.
    void ensure_restricted_model_sampler();
  };

}  // namespace BOOM


#endif  // BOOM_GLM_BIG_ASS_SPIKE_SLAB_HPP_
