// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2017 Steven L. Scott

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
#ifndef BOOM_STATE_SPACE_STUDENT_POSTERIOR_SAMPLER_HPP_
#define BOOM_STATE_SPACE_STUDENT_POSTERIOR_SAMPLER_HPP_

#include "Models/StateSpace/PosteriorSamplers/StateSpaceStudentPosteriorSampler.hpp"
#include "Models/StateSpace/PosteriorSamplers/SufstatManager.hpp"

namespace BOOM {

  namespace StateSpace {
    class StudentSufstatManager : public SufstatManagerBase {
     public:
      explicit StudentSufstatManager(StateSpaceStudentPosteriorSampler *sampler)
          : sampler_(sampler) {}
      void clear_complete_data_sufficient_statistics() override {
        sampler_->clear_complete_data_sufficient_statistics();
      }
      void update_complete_data_sufficient_statistics(int t) override {
        sampler_->update_complete_data_sufficient_statistics(t);
      }

     private:
      StateSpaceStudentPosteriorSampler *sampler_;
    };
  }  // namespace StateSpace

  namespace {
    typedef StateSpaceStudentPosteriorSampler SSSPS;
    typedef StateSpace::AugmentedStudentRegressionData AugmentedData;
  }  // namespace

  SSSPS::StateSpaceStudentPosteriorSampler(
      StateSpaceStudentRegressionModel *model,
      const Ptr<TRegressionSpikeSlabSampler> &observation_model_sampler,
      RNG &seeding_rng)
      : StateSpacePosteriorSampler(model, seeding_rng),
        model_(model),
        observation_model_sampler_(observation_model_sampler) {
    model_->register_data_observer(new StateSpace::StudentSufstatManager(this));
    observation_model_sampler_->fix_latent_data(true);
  }

  void SSSPS::impute_nonstate_latent_data() {
    const std::vector<Ptr<AugmentedData>> &data(model_->dat());
    for (int t = 0; t < data.size(); ++t) {
      Ptr<AugmentedData> dp = data[t];
      double state_contribution =
          model_->observation_matrix(t).dot(model_->state(t));
      if (dp->missing() == Data::observed) {
        double regression_contribution =
            model_->observation_model()->predict(dp->x());
        double weight = data_imputer_.impute(
            rng(),
            dp->y() - regression_contribution - state_contribution,
            model_->observation_model()->sigma(),
            model_->observation_model()->nu());
        dp->set_weight(weight);
      }
    }
  }

  void SSSPS::clear_complete_data_sufficient_statistics() {
    observation_model_sampler_->clear_complete_data_sufficient_statistics();
    // model_->observation_model()->clear_data();
  }

  void SSSPS::update_complete_data_sufficient_statistics(int t) {
    Ptr<AugmentedData> dp = model_->dat()[t];
    ensure_observation_model_data();
    if (dp->missing() == Data::observed) {
      double time_series_residual = dp->y() - dp->state_model_offset();
      observation_model_sampler_->update_complete_data_sufficient_statistics(
          time_series_residual, dp->x(), dp->weight());
      model_->observation_model()->dat()[t]->set_y(time_series_residual);  //////////////// dat() is not populated.
    }
  }

  void SSSPS::ensure_observation_model_data() {
    const std::vector<Ptr<AugmentedData>> &data(model_->dat());
    if (model_->observation_model()->dat().size() == data.size()) {
      return;
    } else {
      while (model_->observation_model()->dat().size() < data.size()) {
        size_t t = model_->observation_model()->dat().size();
        Ptr<AugmentedData> dp = data[t];
        NEW(DoubleData, proxy_response)(dp->y());
        NEW(RegressionData, proxy_data)(proxy_response, dp->Xptr());
        model_->observation_model()->add_data(proxy_data);
      }
    }
  }

}  // namespace BOOM

#endif  // BOOM_STATE_SPACE_STUDENT_POSTERIOR_SAMPLER_HPP_
