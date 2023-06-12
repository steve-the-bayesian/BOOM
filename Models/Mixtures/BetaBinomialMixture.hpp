#ifndef BOOM_MIXTURES_BETA_BINOMIAL_MIXTURE_HPP_
#define BOOM_MIXTURES_BETA_BINOMIAL_MIXTURE_HPP_

/*
  Copyright (C) 2005-2022 Steven L. Scott

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

#include "Models/BinomialModel.hpp"
#include "Models/BetaBinomialModel.hpp"
#include "Models/MultinomialModel.hpp"
#include "Models/Policies/CompositeParamPolicy.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"

namespace BOOM {

  // Binomial data (number of trials and number of successes) agumented by the a count
  class AggregatedBinomialData
      : public BinomialData {
   public:
    AggregatedBinomialData(int64_t trials, int64_t successes, int64_t count);
    AggregatedBinomialData * clone() const override;

    int64_t count() const {return count_;}

   private:
    int64_t count_;
  };

  // ===========================================================================
  class BetaBinomialMixtureModel
      : public LatentVariableModel,
        public CompositeParamPolicy,
        public IID_DataPolicy<AggregatedBinomialData>,
        public PriorPolicy
  {
   public:
    BetaBinomialMixtureModel(
        const std::vector<Ptr<BetaBinomialModel>> &components,
        const Ptr<MultinomialModel> &mixing_weights);
    BetaBinomialMixtureModel(const BetaBinomialMixtureModel &rhs);
    BetaBinomialMixtureModel(BetaBinomialMixtureModel &&rhs) = default;
    BetaBinomialMixtureModel * clone() const override;

    Ptr<MultinomialModel> mixing_distribution() {return mixing_weight_model_;}
    const MultinomialModel *mixing_distribution() const {
      return mixing_weight_model_.get();
    }
    const Vector &mixing_weights() const {return mixing_weight_model_->pi();}

    int number_of_mixture_components() const {return components_.size();}
    Ptr<BetaBinomialModel> mixture_component(int s) {return components_[s];}
    const BetaBinomialModel *mixture_component(int s) const {
      return components_[s].get();
    }

    // Clear the data from all component models and recompute the latent mixture
    // indicators.
    //
    // Args:
    //   rng: The random number generator used to sample the missing cluster
    //     indicators.
    //
    // Effects:
    //   The sufficient statistics for the mixing distribution and the managed
    //   mixture components are updated
    void impute_latent_data(RNG &rng) override;

    // Assign the values in the given fully observed data to one or more of the
    // managed mixture components.
    //
    // Args:
    //   rng: The random number generator used to sample the missing cluster
    //     indicators.
    //   data_point: The data point containing the data to be apportioned among
    //     the components.
    //
    // Effects:
    //   The sufficient statistics for the mixing distribution and the managed
    //   mixture components are updated
    void impute_data_point(RNG &rng, const Ptr<AggregatedBinomialData> &data_point);

    // Remove data from all submodels.  Data assigned to directly to this model
    // is unaffected.
    void clear_component_data();

    // Args:
    //   weights: A discrete probability distribution giving the mixing weights
    //     of each mixture component.
    //   ab: A two column matrix with one row for each mixture component.  The
    //     'a' parameter for each component is in column 0.  The 'b' parameter
    //     is in column 1.
    //
    // Returns:
    //   The observed data log likelihood, averaging over the missing mixture
    //   indicators.
    double log_likelihood(const Vector &weights, const Matrix &ab) const;

    // Args:
    //   packed_parameters: A vector of parameters packed as if by a call to
    //     parameter_vector().
    //
    // Returns:
    //   The observed data log likelihood, averaging over the missing mixture
    //   indicators.
    // double log_likelihood(const Vector &packed_parameters) const;

   private:
    // A utility to call during construction.
    void add_models_to_param_policy();

    std::vector<Ptr<BetaBinomialModel>> components_;
    Ptr<MultinomialModel> mixing_weight_model_;
  };
}  // namespace BOOM


#endif  //  BOOM_MIXTURES_BETA_BINOMIAL_MIXTURE_HPP_
