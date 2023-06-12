/*
  Copyright (C) 2018 Steven L. Scott

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

#include "Models/StateSpace/Multivariate/PosteriorSamplers/SharedLocalLevelPosteriorSampler.hpp"
#include "distributions.hpp"

namespace BOOM {

  const bool enforce_triangular_coefficients = true;

  namespace {
    using GSLLPS = GeneralSharedLocalLevelPosteriorSampler;

    void check_slabs(const std::vector<Ptr<MvnBase>> &slabs,
                     int nseries,
                     int state_dimension) {
      if (slabs.size() != nseries) {
        report_error("Number of slab priors does not match number of series.");
      }
      for (int i = 0; i < slabs.size(); ++i) {
        if (slabs[i]->dim() != state_dimension) {
          report_error("At least one slab prior expects the wrong state size.");
        }
      }
    }

    void check_spikes(const std::vector<Ptr<VariableSelectionPrior>> &spikes,
                      int nseries,
                      int state_dimension) {
      if (spikes.size() != nseries) {
        report_error("Number of spike priors does not match number of series.");
      }
      for (int i = 0; i < spikes.size(); ++i) {
        if (spikes[i]->potential_nvars() != state_dimension) {
          report_error("At least one spike prior expects the wrong state size.");
        }
      }
    }

    void set_unit_innovation_variances(SharedLocalLevelStateModelBase *model) {
      for (int i = 0; i < model->state_dimension(); ++i) {
        model->innovation_model(i)->set_sigsq(1.0);
      }
    }

    void build_samplers(std::vector<SpikeSlabSampler> &samplers,
                        const std::vector<Ptr<MvnBase>> &slabs,
                        const std::vector<Ptr<VariableSelectionPrior>> &spikes) {
      for (int i = 0; i < spikes.size(); ++i) {
        samplers.push_back(SpikeSlabSampler(nullptr, slabs[i], spikes[i]));
      }
    }

  }  // namespace

  GSLLPS::GeneralSharedLocalLevelPosteriorSampler(
      GeneralSharedLocalLevelStateModel *model,
      const std::vector<Ptr<MvnBase>> &slabs,
      const std::vector<Ptr<VariableSelectionPrior>> &spikes,
      RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        slabs_(slabs),
        spikes_(spikes)
  {
    // Check that spikes and slabs are the right size.
    check_slabs(slabs, model->nseries(), model->state_dimension());
    check_spikes(spikes, model->nseries(), model->state_dimension());

    // Use the spikes to enforce the constraint on the coefficients.
    if (enforce_triangular_coefficients) {
      Matrix coefficients = model_->coefficient_model()->Beta().transpose();
      for (int i = 0; i < spikes_.size(); ++i) {
        Selector inclusion_indicator(model_->state_dimension(), true);
        for (int j = i + 1; j < model_->state_dimension(); ++j) {
          spikes_[i]->set_prior_inclusion_probability(j, 0.0);
          coefficients(i, j) = 0.0;
          inclusion_indicator.drop(j);
        }
        inclusion_indicators_.push_back(inclusion_indicator);
      }
      model_->coefficient_model()->set_Beta(coefficients.transpose());
    } else {
      for (int i = 0; i < spikes_.size(); ++i) {
        Selector inclusion_indicator(model_->state_dimension(), true);
        inclusion_indicators_.push_back(inclusion_indicator);
      }
    }

    // Set the innovation variances to 1, for identifiability.
    set_unit_innovation_variances(model_);
    // Build the samplers.
    build_samplers(samplers_, slabs_, spikes_);
  }

  //---------------------------------------------------------------------------
  double GSLLPS::logpri() const {
    double ans = 0;
    const Matrix &transposed_coefficients(
        model_->coefficient_model()->Beta());

    for (int i = 0; i < spikes_.size(); ++i) {
      ans += spikes_[i]->logp(inclusion_indicators_[i]);
      if (!std::isfinite(ans)) {
        return ans;
      }
      ans += dmvn(
          inclusion_indicators_[i].select(transposed_coefficients.col(i)),
          inclusion_indicators_[i].select(slabs_[i]->mu()),
          inclusion_indicators_[i].select(slabs_[i]->siginv()),
          true);
    }
    return ans;
  }

  //---------------------------------------------------------------------------
  void GSLLPS::draw() {
    Matrix coefficients = model_->coefficient_model()->Beta().transpose();
    WeightedRegSuf suf(model_->number_of_factors());
    const MvRegSuf &mvsuf(*model_->coefficient_model()->suf());
    // Each time series corresponds to a row in 'coefficients'.  The elements of
    // that row describe the sensitivity of that series to the latent factors.
    //
    // There is one element of slabs_ for each time series.
    for (int i = 0; i < slabs_.size(); ++i) {
      suf.reset(mvsuf.xtx(),
                mvsuf.xty().col(i),
                mvsuf.yty()(i, i),
                mvsuf.n(),
                mvsuf.n(),
                0.0);

      samplers_[i].draw_inclusion_indicators(
          rng(), inclusion_indicators_[i], suf);
      Vector row = coefficients.row(i);
      samplers_[i].draw_coefficients_given_inclusion(
          rng(), row, inclusion_indicators_[i], suf, 1.0);
      coefficients.row(i) = row;
    }
    model_->coefficient_model()->set_Beta(coefficients.transpose());
  }

  //---------------------------------------------------------------------------
  void GSLLPS::limit_model_selection(int max_flips) {
    for (int i = 0; i < samplers_.size(); ++i) {
      samplers_[i].limit_model_selection(max_flips);
    }
  }

  //===========================================================================
  namespace {
    using CindSLLPS = ConditionallyIndependentSharedLocalLevelPosteriorSampler;
  }  // namespace

  CindSLLPS::ConditionallyIndependentSharedLocalLevelPosteriorSampler(
      ConditionallyIndependentSharedLocalLevelStateModel *model,
      const std::vector<Ptr<MvnBase>> &slabs,
      const std::vector<Ptr<VariableSelectionPrior>> &spikes,
      const std::vector<Ptr<UnivParams>> &sigsq,
      RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        slabs_(slabs),
        spikes_(spikes),
        sigsq_(sigsq)
  {
    check_spikes(spikes, model->nseries(), model->state_dimension());
    check_slabs(slabs, model->nseries(), model->state_dimension());

    if (enforce_triangular_coefficients) {
      for (int i = 0; i < spikes_.size(); ++i) {
        /////////
      }
    }

    // Set the innovation variances to 1.0, for identifiability.
    set_unit_innovation_variances(model_);
    build_samplers(samplers_, slabs_, spikes_);
  }

  void CindSLLPS::draw() {
    for (int i = 0; i < model_->nseries(); ++i) {
      double sigsq = sigsq_[i]->value();
      Selector inc = model_->raw_observation_coefficients(i)->inc();
      samplers_[i].draw_inclusion_indicators(
          rng(), inc, *model_->suf(i), sigsq);
      model_->raw_observation_coefficients(i)->set_inc(inc);

      Vector full_beta = model_->raw_observation_coefficients(i)->Beta();
      samplers_[i].draw_coefficients_given_inclusion(
          rng(), full_beta, inc, *model_->suf(i), sigsq);
      model_->raw_observation_coefficients(i)->set_Beta(full_beta);
    }
  }

  double CindSLLPS::logpri() const {
    double ans = 0;
    for (int i = 0; i < samplers_.size(); ++i) {
      ans += samplers_[i].log_prior(*model_->raw_observation_coefficients(i));
    }
    return ans;
  }


}  // namespace BOOM
