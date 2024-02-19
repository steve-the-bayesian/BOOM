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

#ifndef BOOM_MULTINOMIAL_LOGIT_VARIABLE_SELECTION_HPP
#define BOOM_MULTINOMIAL_LOGIT_VARIABLE_SELECTION_HPP

#include "Models/Glm/PosteriorSamplers/MLVS_data_imputer.hpp"
#include "Models/Glm/VariableSelectionPrior.hpp"
#include "Models/PosteriorSamplers/Imputer.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"

namespace BOOM {

  class MultinomialLogitModel;
  class MvnBase;
  class ChoiceData;

  //------------------------------------------------------------
  // Draws the parameters of a multinomial logit model using the
  // approximate method from Fruewirth-Schnatter and Fruewirth,
  // Computational Statistics and Data Analysis 2007, 3508-3528.

  // This implementation only stores the complete data sufficient
  // statistics and some workspace.  It does not store the imputed
  // latent data.
  class MLVS : public PosteriorSampler,
               public LatentDataSampler<MlvsDataImputer> {
   public:
    // Args:
    //   model: The multinomial logit model to be sampled.
    //   slab: The conditional prior distribution for the coefficients of *model,
    //     given inclusion.  The dimension of this prior must match the
    //     dimension of model->beta(), which is: (number of choices - 1) * (number
    //     of subject level predictors) + (number of choice level predictors).
    //   spike: The prior distribution on the include/exclude pattern for
    //     model->beta().  See above for the required dimension.
    //   nthreads: The number of threads to use when imputing the latent data.
    //   check_initial_condition: If true an error will be reported if the
    //     initial value of model->beta() is inconsistent with the prior
    //     distribution.  The usual cause of something like this is if certain
    //     coefficients have been forced in by setting spike = 1 for that
    //     coefficient, but the coefficient is excluded by Mod.  If false then
    //     no error checking will take place on the initial condition.
    MLVS(MultinomialLogitModel *model,
         const Ptr<MvnBase> &slab,
         const Ptr<VariableSelectionPrior> &spike,
         uint nthreads = 1,
         bool check_initial_condition = true,
         RNG &seeding_rng = GlobalRng::rng);

    void draw() override;
    double logpri() const override;

    void clear_latent_data() override;
    Ptr<MlvsDataImputer> create_worker(std::mutex &m) override;
    void assign_data_to_workers() override;

    // Part of the implementation for draw().
    void draw_beta();

    // Functions to control Bayesian variable selection.  If
    // suppress_model_selection is called then the current
    // include/exclude state of the coefficients will not be modified
    // by the sampler.  A call to allow_model_selection() turns model
    // selection back on.  It is on by default.
    void suppress_model_selection();
    void allow_model_selection();

    // If the predictor space is very large, then you can save time by
    // not sampling the include/exclude decision for all possible
    // variables.  Calling limit_model_selection(nflips) will cause
    // the algorithm to only try to modify 'nflips' randomly chosen
    // locations each iteration.  By default the algorithm tries all
    // variables each time.
    void limit_model_selection(uint nflips);

    // Returns the number of variables that the model selection
    // algorithm will attempt to modify at each iteration.
    uint max_nflips() const;

   private:
    MultinomialLogitModel *model_;
    Ptr<MvnBase> slab_;
    Ptr<VariableSelectionPrior> spike_;

    typedef MultinomialLogit::CompleteDataSufficientStatistics LocalSuf;
    LocalSuf suf_;

    const Vector &log_sampling_probs_;
    const bool downsampling_;
    bool select_;
    uint max_nflips_;

    SpdMatrix Ominv;
    SpdMatrix iV_tilde_;
    virtual void draw_inclusion_vector();
    double log_model_prob(const Selector &inc);
  };

}  // namespace BOOM

#endif  // BOOM_MULTINOMIAL_LOGIT_VARIABLE_SELECTION_HPP
