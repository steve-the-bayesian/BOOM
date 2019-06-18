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

#ifndef BOOM_HIERARCHICAL_ZERO_INFLATED_POISSON_MODEL_HPP_
#define BOOM_HIERARCHICAL_ZERO_INFLATED_POISSON_MODEL_HPP_

#include "Models/BetaModel.hpp"
#include "Models/GammaModel.hpp"
#include "Models/Policies/CompositeParamPolicy.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/ZeroInflatedPoissonModel.hpp"

namespace BOOM {

  // A class that wraps a ZeroInflatedPoissonSuf, so that it can be
  // used as "Data" in the hierarchical model below.
  class ZeroInflatedPoissonData : public Data {
   public:
    ZeroInflatedPoissonData(double number_of_zero_trials,
                            double number_of_positive_trials,
                            double total_number_of_events);
    // Automatic conversions from ZeroInflatedPoissonSuf are allowed.
    explicit ZeroInflatedPoissonData(const ZeroInflatedPoissonSuf &suf);
    ZeroInflatedPoissonData(const ZeroInflatedPoissonData &rhs);
    ZeroInflatedPoissonData *clone() const override;
    std::ostream &display(std::ostream &out) const override;
    const ZeroInflatedPoissonSuf &suf() const;

   private:
    ZeroInflatedPoissonSuf suf_;
  };

  //======================================================================
  // Mathematical Comments:
  // A zero-inflated Poisson model for group-level data.  The data
  // from group i obeys the model
  //
  //    y[i, j] - p[i] * I(0) + (1-p[i]) * Poisson(lambda[i])
  //
  // The number p[i] is called the "zero probability" and lambda[i] is
  // the Poisson rate.  We assume
  //
  //  p[i] ~ Beta(zero_probability_a, zero_probability_b)
  //
  // and
  //
  //  lambda[i] ~ Gamma(lambda_a, lambda_b)
  //
  // Before seeing any data in a group, the expected zero probability
  // is zero_probability_a/(zero_probability_a + zero_probability_b),
  // and the variance is roughly mean*(1-mean) / (zero_probability_a +
  // zero_probability_b).  We call the mean
  // 'zero_probability_prior_mean' and we call zero_probability_a +
  // zero_probability_b the 'zero_probability_prior_sample_size'.
  //
  // The conditional mean event rate (given that it is not identically
  // zero) for a group, before seeing any data, is lambda_a/lambda_b,
  // with variance mean / lambda_b.  Thus we call lambda_a/lambda_b
  // the 'poisson_prior_mean' and we call lambda_b the
  // 'poisson_prior_sample_size'.
  //
  // ____________________________________________________________
  // Software Comments:
  // It really does not make sense for a hierarchical model to have a
  // data policy, because the data will be managed by the set of
  // data_level_models.  The three functions normally managed by a
  // DataPolicy are add_data, clear_data, and combine_data.  In the
  // context of hierarchical models, this is how they should behave.
  //
  // *) add_data: Will add a new data_level_model with the data
  //              assigned.  The parameters will not be initialized.
  // *) clear_data: Will remove all data_level_models (including the
  //                data and parameters that they manage).
  // *) combine_data: Will add the data_level_models (including
  //                  parameters and data) from the rhs argument to
  //                  the current model.
  class HierarchicalZeroInflatedPoissonModel : public CompositeParamPolicy,
                                               public PriorPolicy {
   public:
    // Convenience constructor.
    HierarchicalZeroInflatedPoissonModel(
        double lambda_prior_guess, double lambda_prior_sample_size,
        double zero_probability_prior_guess,
        double zero_probability_prior_sample_size);

    // Provides control over the priors.
    HierarchicalZeroInflatedPoissonModel(
        const Ptr<GammaModel> &prior_for_lambda,
        const Ptr<BetaModel> &prior_for_zero_probability);

    // Args:
    //   trials: A vector giving the number of trials represented by
    //     each data-level group.  Must have trials[i] >= 0.
    //   events: The total number of events generated in each
    //     data-level group, across all trials in that group.  Must
    //     have events.size() == trials.size(), and events[i] >= 0.
    //   zeros: The number of trials that produced zero events, for
    //     each data-level group.  Must have zeros[i] >= 0 and
    //     zeros[i] <= trials[i].
    // Details:
    //   Initializes the priors with rough estimates from the data.
    //   These are respectible starting values for an MCMC algorithm,
    //   but not particularly accurate otherwise.
    HierarchicalZeroInflatedPoissonModel(const BOOM::Vector &trials,
                                         const BOOM::Vector &events,
                                         const BOOM::Vector &number_of_zeros);

    HierarchicalZeroInflatedPoissonModel(
        const HierarchicalZeroInflatedPoissonModel &rhs);

    HierarchicalZeroInflatedPoissonModel *clone() const override;

    // Unless a separate pointer to data_level_model is kept, the data
    // for data_level_model should be set before calling this
    // function.  The posterior sampling_method should not be set.
    void add_data_level_model(
        const Ptr<ZeroInflatedPoissonModel> &data_level_model);

    // Removes all data_level_models and their associated parameters
    // and data.
    void clear_data() override;

    // Clear the data from all data_level_models, but does not delete
    // the models.
    void clear_client_data();

    // Clear the learning methods for each of the client models.
    void clear_methods();

    // Adds the data_level_models from rhs to this.
    void combine_data(const Model &rhs, bool just_suf = true) override;

    // Creates a new data_level_model with data assigned.
    void add_data(const Ptr<Data> &) override;

    // Returns the number of data_level_models managed by this model.
    int number_of_groups() const;

    ZeroInflatedPoissonModel *data_model(int which_group);
    GammaModel *prior_for_poisson_mean();
    BetaModel *prior_for_zero_probability();

    double poisson_prior_mean() const;
    double poisson_prior_sample_size() const;
    double zero_probability_prior_mean() const;
    double zero_probability_prior_sample_size() const;

    // Args:
    //   n:  The number of trials for a particular observation.  All trials will
    //     use the same zero probability and reward value distribution.
    // Returns:
    //   Aggregated data for the all the requested observations.
    ZeroInflatedPoissonData sim(int64_t n) const;

   private:
    void initialize();
    Ptr<GammaModel> prior_for_lambda_;
    Ptr<BetaModel> prior_for_zero_probability_;
    std::vector<Ptr<ZeroInflatedPoissonModel> > data_level_models_;
  };

}  // namespace BOOM

#endif  //  BOOM_HIERARCHICAL_ZERO_INFLATED_POISSON_MODEL_HPP_
