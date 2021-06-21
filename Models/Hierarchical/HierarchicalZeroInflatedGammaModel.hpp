// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2013 Steven L. Scott

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

#ifndef BOOM_HIERARCHICAL_ZERO_INFLATED_GAMMA_MODEL_HPP_
#define BOOM_HIERARCHICAL_ZERO_INFLATED_GAMMA_MODEL_HPP_

#include "Models/BetaModel.hpp"
#include "Models/GammaModel.hpp"
#include "Models/Policies/CompositeParamPolicy.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/ZeroInflatedGammaModel.hpp"

namespace BOOM {

  // The "data" class for a HierarchicalZeroInflatedGammaModel.  Each
  // data point contains the sufficient statistics for a group.
  class HierarchicalZeroInflatedGammaData : public Data {
   public:
    HierarchicalZeroInflatedGammaData(int n0, int n1, double sum,
                                      double sumlog);
    HierarchicalZeroInflatedGammaData *clone() const override;
    std::ostream &display(std::ostream &out) const override;
    int number_of_zeros() const;
    int number_of_positives() const;
    double sum() const;
    double sumlog() const;

   private:
    int number_of_zeros_;
    int number_of_positives_;
    double sum_;
    double sum_of_logs_of_positives_;
  };

  //======================================================================
  // Mathematical Comments:
  // A zero-inflated gamma model for group-level data.  The data
  // from group i obeys the model
  //
  //    y[i, j] - (1 - p[i]) * I(0) + p[i] * Gamma(mu[i], a[i])
  //
  // The number p[i] is called the "positive probability," and mu[i]
  // and a[i] are the parameters of the Gamma distribution with mean
  // mu[i] and shape parameter a[i].  A more common parameterization
  // of the Gamma distribution is Gamma(a, b) which maps to our
  // paramterization through mu[i] = a/b and a[i] = a.  At the next
  // level up the hierarchy, the group level parameters are assumed to
  // independently follow
  //
  //  p[i] ~ Beta(positive_probability_a, positive_probability_b)
  //
  // mu[i] ~ Gamma(mu_mean, mu_shape)
  //
  //  a[i] ~ Gamma(a_mean, a_shape)
  //
  // Before seeing any data in a group, the expected positive
  // probability is positive_probability_a/(positive_probability_a +
  // positive_probability_b), and the variance is roughly
  // mean*(1-mean) / (positive_probability_a +
  // positive_probability_b).  We call the mean
  // 'positive_probability_prior_mean' and we call
  // positive_probability_a + positive_probability_b the
  // 'positive_probability_prior_sample_size'.
  //
  // If the parameters had all been observed for a given day, then the
  // conditional mean reward would be p[i] * mu[i].  Because p and mu
  // are independent, averaging across groups means the expected value
  // of y[i, j] in a previously unseen group i is
  //
  // positive_probability_prior_mean * mu_mean.
  // ____________________________________________________________
  // Software Comments:
  // It really does not make sense for a hierarchical model to have a
  // data policy.  Each point of Data is really a data set (or set of
  // sufficient statistics) managed by one of the data_level_models.
  // The three functions normally managed by a DataPolicy are
  // add_data, clear_data, and combine_data.  In the context of
  // hierarchical models, this is how they should behave.
  //
  // *) add_data: Will add a new data_level_model with the data
  //              assigned.  The parameters will not be initialized.
  // *) clear_data: Will remove all data_level_models (including the
  //                data and parameters that they manage).
  // *) combine_data: Will add the data_level_models (including
  //                  parameters and data) from the rhs argument to
  //                  the current model.
  class HierarchicalZeroInflatedGammaModel : public CompositeParamPolicy,
                                             public PriorPolicy {
   public:
    // This is the constructor to be used when modeling data.
    // Args:
    //   Each argument is a vector with dimension equal to the number of groups.
    //   number_of_zeros_per_group: The number of observations in each
    //     group where the response was zero.
    //   number_of_positives_per_group: The number of observations in
    //     each group where the response was greater than zero.
    //   sum_of_positive_observations_per_group: The sum of all
    //     positive observations in each group.  This is the same as
    //     the sum of all observations, of course, because if an
    //     observation is not positive it is zero.
    //   sum_of_logs_of_positive_observations: The sum of the
    //     (natural) logs of the positive observations in each group.
    //     If a group has no positive observations then this is zero.
    //   seeding_rng: The random-number generator to use for seeding the model's
    //     internal posterior samplers.
    HierarchicalZeroInflatedGammaModel(
        const BOOM::Vector &number_of_zeros_per_group,
        const BOOM::Vector &number_of_positives_per_group,
        const BOOM::Vector &sum_of_positive_observations_per_group,
        const BOOM::Vector &sum_of_logs_of_positive_observations,
        BOOM::RNG &seeding_rng = BOOM::GlobalRng::rng);

    HierarchicalZeroInflatedGammaModel(
        const HierarchicalZeroInflatedGammaModel &rhs);

    HierarchicalZeroInflatedGammaModel *clone() const override;

    // TODO: Should have a HierarchicalPriorPolicy that
    // implements clear_methods() by calling it on the data-level
    // models.
    void clear_methods() override;

    // Removes all data_level_models and their associated parameters
    // and data.
    void clear_data() override;

    // Adds the data_level_models from rhs to this.
    void combine_data(const Model &rhs, bool just_suf = true) override;

    // Creates a new data_level_model with data assigned.
    void add_data(const Ptr<Data> &) override;

    // Returns the number of data_level_models managed by this model.
    int number_of_groups() const;

    BetaModel *prior_for_positive_probability();
    GammaModel *prior_for_mean_parameters();
    GammaModel *prior_for_shape_parameters();
    ZeroInflatedGammaModel *data_model(int i);

    double positive_probability_prior_mean() const;
    double positive_probability_prior_sample_size() const;
    double mean_parameter_prior_mean() const;
    double mean_parameter_prior_shape() const;
    double shape_parameter_prior_mean() const;
    double shape_parameter_prior_shape() const;

    // From the law of iterated expectation, the prior mean is the
    // positive_probability_prior_mean() times the
    // mean_parameter_prior_mean().
    double prior_mean() const;

    // Args:
    //   n:  The number of trials for a particular observation.  All trials will
    //     use the same positive probability and reward value distribution.
    // Returns:
    //   Aggregated data for the all the requested observations.
    HierarchicalZeroInflatedGammaData sim(int64_t n) const;

   private:
    void setup();
    Ptr<GammaModel> prior_for_mean_parameters_;
    Ptr<GammaModel> prior_for_shape_parameters_;
    Ptr<BetaModel> prior_for_positive_probability_;
    std::vector<Ptr<ZeroInflatedGammaModel> > data_models_;
  };

}  // namespace BOOM

#endif  //  BOOM_HIERARCHICAL_ZERO_INFLATED_POISSON_MODEL_HPP_
