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
    BigAssSpikeSlabSampler(BigRegressionModel *model,
                           const Ptr<VariableSelectionPrior> &spike,
                           const Ptr<RegressionSlabPrior> &slab,
                           const Ptr<GammaModelBase> &residual_precision_prior,
                           RNG &seeding_rng = GlobalRng::rng);

    void draw() override;
    double logpri() const override;

    void initial_screen(int niter, double threshold);

    // The largest number of predictors in a worker model.
    int max_worker_dim() const;

   private:
    BigRegressionModel *model_;
    Ptr<VariableSelectionPrior> spike_;
    Ptr<RegressionSlabPrior> slab_;
    Ptr<GammaModelBase> residual_precision_prior_;

    std::vector<Ptr<BregVsSampler>> subordinate_samplers_;

    // Assigns posterior samplers to the subordinate models contained in model_.
    void assign_subordinate_samplers();

    void run_parallel_initial_screen(int niter);

    void gather_candidate_predictors_from_workers(double threshold);
  };

}  // namespace BOOM


#endif  // BOOM_GLM_BIG_ASS_SPIKE_SLAB_HPP_
