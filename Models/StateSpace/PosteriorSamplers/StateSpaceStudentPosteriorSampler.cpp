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
      for (int j = 0; j < dp->total_sample_size(); ++j) {
        const RegressionData &observation(dp->regression_data(j));
        if (observation.missing() == Data::observed) {
          double regression_contribution =
              model_->observation_model()->predict(observation.x());
          double weight = data_imputer_.impute(
              rng(),
              observation.y() - regression_contribution - state_contribution,
              model_->observation_model()->sigma(),
              model_->observation_model()->nu());
          dp->set_weight(weight, j);
        }
      }
    }
  }

  void SSSPS::clear_complete_data_sufficient_statistics() {
    observation_model_sampler_->clear_complete_data_sufficient_statistics();

    // The observation model needs access to the actual data.  Regression
    // coefficients and residual variance are drawn based on complete data
    // sufficient statistics, but the tail thickness parameter is drawn
    // conditional on the other parameters using a likelihood evaluation that
    // requires a loop over the data.
    if (model_->observation_model()->dat().size() !=
        model_->total_sample_size()) {
      model_->observation_model()->clear_data();
      subordinate_data_.clear();
      for (int i = 0; i < model_->time_dimension(); ++i) {
        std::vector<Ptr<RegressionData>> local_subordinate_data;
        Ptr<AugmentedData> real_data_point = model_->dat()[i];
        int local_sample_size = real_data_point->total_sample_size();
        for (int j = 0; j < local_sample_size; ++j) {
          const RegressionData &real_observation(
              real_data_point->regression_data(j));
          NEW(RegressionData, subordinate_data)
          (new DoubleData(real_observation.y()), real_observation.Xptr());
          local_subordinate_data.push_back(subordinate_data);
          if (real_observation.missing() == Data::observed) {
            model_->observation_model()->add_data(subordinate_data);
          }
        }
        subordinate_data_.push_back(local_subordinate_data);
      }
    }
  }

  void SSSPS::update_complete_data_sufficient_statistics(int t) {
    Ptr<AugmentedData> dp = model_->dat()[t];
    for (int i = 0; i < dp->total_sample_size(); ++i) {
      const RegressionData &observation(dp->regression_data(i));
      if (observation.missing() == Data::observed) {
        double time_series_residual =
            observation.y() - dp->state_model_offset();
        observation_model_sampler_->update_complete_data_sufficient_statistics(
            time_series_residual, observation.x(), dp->weight(i));
        subordinate_data_[t][i]->set_y(time_series_residual);
      }
    }
  }

}  // namespace BOOM

#endif  // BOOM_STATE_SPACE_STUDENT_POSTERIOR_SAMPLER_HPP_
