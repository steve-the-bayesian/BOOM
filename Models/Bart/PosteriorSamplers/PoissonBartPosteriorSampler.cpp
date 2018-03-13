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

#include "Models/Bart/PosteriorSamplers/PoissonBartPosteriorSampler.hpp"
#include "Models/Glm/PosteriorSamplers/poisson_mixture_approximation_table.hpp"
#include "cpputil/math_utils.hpp"
#include "distributions.hpp"

namespace BOOM {
  namespace Bart {
    PoissonResidualRegressionData::PoissonResidualRegressionData(
        const Ptr<PoissonRegressionData> &dp,
        double initial_predicted_log_lambda)
        : ResidualRegressionData(dp->Xptr().get()),
          observed_data_(dp.get()),
          neglog_final_event_time_minus_mu_(0),
          internal_weight_(0),
          neglog_final_interarrival_time_minus_mu_(0),
          external_weight_(0),
          log_lambda_(initial_predicted_log_lambda) {}

    //----------------------------------------------------------------------
    int PoissonResidualRegressionData::y() const { return observed_data_->y(); }

    //----------------------------------------------------------------------
    double PoissonResidualRegressionData::exposure() const {
      return observed_data_->exposure();
    }

    //----------------------------------------------------------------------
    void PoissonResidualRegressionData::add_to_residual(double value) {
      log_lambda_ -= value;
    }

    //----------------------------------------------------------------------
    void PoissonResidualRegressionData::add_to_poisson_suf(
        PoissonSufficientStatistics &suf) const {
      suf.update(*this);
    }

    //----------------------------------------------------------------------
    void PoissonResidualRegressionData::set_latent_data(
        double neglog_final_event_time_minus_mu, double internal_weight,
        double neglog_final_interarrival_time_minus_mu,
        double external_weight) {
      if (internal_weight < 0 || external_weight < 0) {
        report_error(
            "Negative weights in PoissonResidualRegressionData::"
            "set_residuals");
      }
      neglog_final_event_time_minus_mu_ = neglog_final_event_time_minus_mu;
      internal_weight_ = internal_weight;
      neglog_final_interarrival_time_minus_mu_ =
          neglog_final_interarrival_time_minus_mu;
      external_weight_ = external_weight;
    }

    double PoissonResidualRegressionData::internal_residual() const {
      return neglog_final_event_time_minus_mu_ - log_lambda_;
    }

    double PoissonResidualRegressionData::internal_weight() const {
      return internal_weight_;
    }

    double PoissonResidualRegressionData::external_residual() const {
      return neglog_final_interarrival_time_minus_mu_ - log_lambda_;
    }

    double PoissonResidualRegressionData::external_weight() const {
      return external_weight_;
    }

    //======================================================================
    PoissonSufficientStatistics *PoissonSufficientStatistics::clone() const {
      return new PoissonSufficientStatistics(*this);
    }

    //----------------------------------------------------------------------
    void PoissonSufficientStatistics::clear() {
      sum_of_weights_ = 0;
      weighted_sum_of_residuals_ = 0;
      weighted_sum_of_squared_residuals_ = 0;
    }

    //----------------------------------------------------------------------
    void PoissonSufficientStatistics::update(
        const ResidualRegressionData &data) {
      data.add_to_poisson_suf(*this);
    }

    //----------------------------------------------------------------------
    void PoissonSufficientStatistics::update(
        const PoissonResidualRegressionData &data) {
      double weight = data.external_weight();
      sum_of_weights_ += weight;
      double residual = data.external_residual();
      weighted_sum_of_residuals_ += weight * residual;
      weighted_sum_of_squared_residuals_ += weight * square(residual);
      if (data.y() > 0) {
        weight = data.internal_weight();
        sum_of_weights_ += weight;
        residual = data.internal_residual();
        weighted_sum_of_residuals_ += weight * residual;
        weighted_sum_of_squared_residuals_ += weight * square(residual);
      }
    }
  }  // namespace Bart

  //======================================================================

  PoissonBartPosteriorSampler::PoissonBartPosteriorSampler(
      PoissonBartModel *model, double total_prediction_sd,
      double prior_tree_depth_alpha, double prior_tree_depth_beta,
      const std::function<double(int)> &log_prior_on_number_of_trees,
      RNG &seeding_rng)
      : BartPosteriorSamplerBase(model, total_prediction_sd,
                                 prior_tree_depth_alpha, prior_tree_depth_beta,
                                 log_prior_on_number_of_trees, seeding_rng),
        model_(model),
        data_imputer_(new PoissonDataImputer) {}

  //----------------------------------------------------------------------
  void PoissonBartPosteriorSampler::draw() {
    impute_latent_data();
    BartPosteriorSamplerBase::draw();
  }

  //----------------------------------------------------------------------
  double PoissonBartPosteriorSampler::draw_mean(Bart::TreeNode *leaf) {
    const Bart::PoissonSufficientStatistics &suf(
        dynamic_cast<const Bart::PoissonSufficientStatistics &>(
            leaf->compute_suf()));
    double ivar = suf.sum_of_weights() + 1.0 / mean_prior_variance();
    double posterior_mean = suf.weighted_sum_of_residuals() / ivar;
    double posterior_sd = sqrt(1.0 / ivar);
    return rnorm_mt(rng(), posterior_mean, posterior_sd);
  }

  //----------------------------------------------------------------------
  double PoissonBartPosteriorSampler::log_integrated_likelihood(
      const Bart::SufficientStatisticsBase &abstract_suf) const {
    const Bart::PoissonSufficientStatistics &suf(
        dynamic_cast<const Bart::PoissonSufficientStatistics &>(abstract_suf));
    double prior_variance = mean_prior_variance();
    double ivar = suf.sum_of_weights() + (1.0 / prior_variance);
    double posterior_mean = suf.weighted_sum_of_residuals() / ivar;
    double posterior_variance = 1.0 / ivar;

    // We omit facors in the integrated likelihood that will cancel in
    // the MH ratio for splitting nodes in the same tree.  The omitted
    // is the log of (2 * pi)^(-n/2) * \prod(w[i] ^.5)
    double ans = .5 * (log(posterior_variance / prior_variance) +
                       (square(posterior_mean) / posterior_variance));
    return ans;
  }

  //----------------------------------------------------------------------
  double PoissonBartPosteriorSampler::complete_data_log_likelihood(
      const Bart::SufficientStatisticsBase &abstract_suf) const {
    const Bart::PoissonSufficientStatistics &suf(
        dynamic_cast<const Bart::PoissonSufficientStatistics &>(abstract_suf));
    return complete_data_poisson_log_likelihood(suf);
  }

  //----------------------------------------------------------------------
  // The complete data log likelihood is the weighted Gaussian log
  // likelihood.  Each observation is paired with either one (if y =
  // 0) or two (if y > 0) latent Gaussian variables with potentially
  // different means and variances.  These are called z0 and z1 in the
  // extended comments introducing PoissonBartPosteriorSampler.hpp.
  // The full complete data log likelihood is
  //
  // (2*pi)^-(n/2) \prod_i w[i]^.5
  // \exp(-.5 \sum_i w[i] (z[i] - mu[i] - eta[i])^2)
  //
  // where [i] indexes the complete data observation, whether z0 or
  // z1.  This function omits the factors before the exponential,
  // because they are omitted by log_integrated_likelihood.
  double PoissonBartPosteriorSampler::complete_data_poisson_log_likelihood(
      const Bart::PoissonSufficientStatistics &suf) const {
    double ans = -.5 * suf.weighted_sum_of_squared_residuals();
    return ans;
  }

  //----------------------------------------------------------------------
  void PoissonBartPosteriorSampler::clear_residuals() { residuals_.clear(); }

  //----------------------------------------------------------------------
  int PoissonBartPosteriorSampler::residual_size() const {
    return residuals_.size();
  }

  //----------------------------------------------------------------------
  Bart::PoissonResidualRegressionData *
  PoissonBartPosteriorSampler::create_and_store_residual(int i) {
    Ptr<PoissonRegressionData> dp = model_->dat()[i];
    double initial_prediction = model_->predict(dp->x());
    std::shared_ptr<Bart::PoissonResidualRegressionData> data(
        new Bart::PoissonResidualRegressionData(dp, initial_prediction));
    residuals_.push_back(data);
    return data.get();
  }

  //----------------------------------------------------------------------
  Bart::PoissonResidualRegressionData *PoissonBartPosteriorSampler::residual(
      int i) {
    return residuals_[i].get();
  }

  //----------------------------------------------------------------------
  Bart::PoissonSufficientStatistics *PoissonBartPosteriorSampler::create_suf()
      const {
    return new SufType;
  }

  //----------------------------------------------------------------------
  void PoissonBartPosteriorSampler::impute_latent_data() {
    check_residuals();
    for (int i = 0; i < residuals_.size(); ++i) {
      impute_latent_data_point(residuals_[i].get());
    }
  }

  //----------------------------------------------------------------------
  void PoissonBartPosteriorSampler::impute_latent_data_point(DataType *data) {
    double eta = data->predicted_log_lambda();
    double neglog_final_event_time = 0;
    double internal_mu = 0;
    double internal_weight = 0;
    double neglog_final_interarrival_time;
    double external_mu;
    double external_weight;
    data_imputer_->impute(rng(), data->y(), data->exposure(), eta,
                          &neglog_final_event_time, &internal_mu,
                          &internal_weight, &neglog_final_interarrival_time,
                          &external_mu, &external_weight);
    data->set_latent_data(
        neglog_final_event_time - internal_mu, internal_weight,
        neglog_final_interarrival_time - external_mu, external_weight);
  }

}  // namespace BOOM
