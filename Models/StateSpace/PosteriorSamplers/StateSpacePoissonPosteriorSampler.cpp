// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2015 Steven L. Scott

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
#include "Models/StateSpace/PosteriorSamplers/StateSpacePoissonPosteriorSampler.hpp"
#include "Models/StateSpace/PosteriorSamplers/SufstatManager.hpp"
#include "distributions.hpp"

namespace BOOM {
  namespace StateSpace {

    class PoissonSufstatManager : public SufstatManagerBase {
     public:
      explicit PoissonSufstatManager(StateSpacePoissonPosteriorSampler *sampler)
          : sampler_(sampler) {}

      void clear_complete_data_sufficient_statistics() override {
        sampler_->clear_complete_data_sufficient_statistics();
      }

      void update_complete_data_sufficient_statistics(int t) override {
        sampler_->update_complete_data_sufficient_statistics(t);
      }

     private:
      StateSpacePoissonPosteriorSampler *sampler_;
    };

  }  // namespace StateSpace

  namespace {
    typedef StateSpacePoissonPosteriorSampler SSPPS;
    typedef StateSpace::AugmentedPoissonRegressionData AugmentedData;
  }  // namespace

  SSPPS::StateSpacePoissonPosteriorSampler(
      StateSpacePoissonModel *model,
      const Ptr<PoissonRegressionSpikeSlabSampler> &observation_model_sampler,
      RNG &seeding_rng)
      : StateSpacePosteriorSampler(model, seeding_rng),
        model_(model),
        observation_model_sampler_(observation_model_sampler) {
    model_->register_data_observer(new StateSpace::PoissonSufstatManager(this));
    observation_model_sampler_->fix_latent_data(true);
  }

  SSPPS *SSPPS::clone_to_new_host(Model *new_host) const {
    StateSpacePoissonModel *new_model = dynamic_cast<StateSpacePoissonModel *>(
        new_host);
    Ptr<PoissonRegressionSpikeSlabSampler> new_observation_model_sampler;
    if (new_model->observation_model()->number_of_sampling_methods() == 0) {
      new_observation_model_sampler.reset(
          observation_model_sampler_->clone_to_new_host(
              new_model->observation_model()));
      new_model->observation_model()->set_method(new_observation_model_sampler);
    } else {
      new_observation_model_sampler.reset(
          dynamic_cast<PoissonRegressionSpikeSlabSampler *>(
              new_model->observation_model()->sampler(0)));
    }
    return new SSPPS(new_model, new_observation_model_sampler, rng());
  }


  void SSPPS::impute_nonstate_latent_data() {
    const std::vector<Ptr<AugmentedData> > &data(model_->dat());
    for (int t = 0; t < data.size(); ++t) {
      Ptr<AugmentedData> dp = data[t];
      if (dp->missing()) {
        continue;
      }
      double state_contribution =
          model_->observation_matrix(t).dot(model_->state(t));
      for (int j = 0; j < dp->total_sample_size(); ++j) {
        const PoissonRegressionData &observation(dp->poisson_data(j));
        if (observation.missing() == Data::observed) {
          double regression_contribution =
              model_->observation_model()->predict(observation.x());

          double internal_neglog_final_event_time = 0;
          double internal_mixture_mean = 0;
          double internal_mixture_precision = 0;
          double neglog_final_interarrival_time = 0;
          double external_mixture_mean = 0;
          double external_mixture_precision = 0;
          data_imputer_.impute(
              rng(),
              observation.y(),
              observation.exposure(),
              state_contribution + regression_contribution,
              &internal_neglog_final_event_time,
              &internal_mixture_mean,
              &internal_mixture_precision,
              &neglog_final_interarrival_time,
              &external_mixture_mean,
              &external_mixture_precision);

          double total_precision = external_mixture_precision;
          double precision_weighted_sum =
              neglog_final_interarrival_time - external_mixture_mean;
          precision_weighted_sum *= external_mixture_precision;
          if (observation.y() > 0) {
            precision_weighted_sum +=
                (internal_neglog_final_event_time - internal_mixture_mean) *
                internal_mixture_precision;
            total_precision += internal_mixture_precision;
          }
          dp->set_latent_data(precision_weighted_sum / total_precision,
                              total_precision, j);
        }
      }
      dp->set_state_model_offset(state_contribution);
    }
  }

  void SSPPS::clear_complete_data_sufficient_statistics() {
    observation_model_sampler_->clear_complete_data_sufficient_statistics();
  }

  void SSPPS::update_complete_data_sufficient_statistics(int t) {
    Ptr<AugmentedData> dp = model_->dat()[t];
    for (int j = 0; j < dp->total_sample_size(); ++j) {
      if (dp->poisson_data(j).missing() == Data::observed) {
        double precision_weighted_mean = dp->latent_data_value(j);

        precision_weighted_mean -= dp->state_model_offset();
        double precision = 1.0 / dp->latent_data_variance(j);
        observation_model_sampler_->update_complete_data_sufficient_statistics(
            precision_weighted_mean * precision, precision,
            model_->data(t, j).x());
      }
    }
  }

}  // namespace BOOM
