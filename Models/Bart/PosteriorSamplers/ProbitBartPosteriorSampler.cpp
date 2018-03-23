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

#include "Models/Bart/PosteriorSamplers/ProbitBartPosteriorSampler.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {
  namespace Bart {

    ProbitResidualData::ProbitResidualData(
        const Ptr<BinomialRegressionData> &dp, double original_prediction)
        : ResidualRegressionData(dp->Xptr().get()),
          original_data_(dp.get()),
          prediction_(original_prediction) {}

    void ProbitResidualData::add_to_probit_suf(
        ProbitSufficientStatistics &suf) const {
      suf.update(*this);
    }

    // add_to_residual is called when adjusting the tree values.  It
    // is not called when imputing the latent data.
    void ProbitResidualData::add_to_residual(double value) {
      sum_of_latent_probit_residuals_ += n() * value;
      prediction_ -= value;
    }

    double ProbitResidualData::sum_of_residuals() const {
      return sum_of_latent_probit_residuals_;
    }

    // set_sum_of_residuals is called when imputing the latent data.
    void ProbitResidualData::set_sum_of_residuals(double sum_of_residuals) {
      sum_of_latent_probit_residuals_ = sum_of_residuals;
    }

    //======================================================================
    ProbitSufficientStatistics::ProbitSufficientStatistics()
        : n_(0), sum_(0)
    {} 
    
    ProbitSufficientStatistics *ProbitSufficientStatistics::clone() const {
      return new ProbitSufficientStatistics(*this);
    }

    void ProbitSufficientStatistics::clear() {
      n_ = 0;
      sum_ = 0;
    }

    void ProbitSufficientStatistics::update(
        const ResidualRegressionData &data) {
      data.add_to_probit_suf(*this);
    }

    void ProbitSufficientStatistics::update(const ProbitResidualData &data) {
      n_ += data.n();
      sum_ += data.sum_of_residuals();
    }

    int ProbitSufficientStatistics::sample_size() const { return n_; }

    double ProbitSufficientStatistics::sum() const { return sum_; }

  }  // namespace Bart

  //======================================================================
  ProbitBartPosteriorSampler::ProbitBartPosteriorSampler(
      ProbitBartModel *model, double total_prediction_sd,
      double prior_tree_depth_alpha, double prior_tree_depth_beta,
      const std::function<double(int)> &log_prior_on_number_of_trees,
      RNG &seeding_rng)
      : BartPosteriorSamplerBase(model, total_prediction_sd,
                                 prior_tree_depth_alpha, prior_tree_depth_beta,
                                 log_prior_on_number_of_trees, seeding_rng),
        model_(model) {}

  //----------------------------------------------------------------------
  void ProbitBartPosteriorSampler::draw() {
    check_residuals();
    impute_latent_data();
    BartPosteriorSamplerBase::draw();
  }

  //----------------------------------------------------------------------
  double ProbitBartPosteriorSampler::draw_mean(Bart::TreeNode *leaf) {
    const Bart::ProbitSufficientStatistics &suf(
        dynamic_cast<const Bart::ProbitSufficientStatistics &>(
            leaf->compute_suf()));
    double prior_variance = mean_prior_variance();
    double ivar = suf.sample_size() + (1.0 / prior_variance);
    double posterior_mean = suf.sum() / ivar;
    double posterior_sd = sqrt(1.0 / ivar);
    return rnorm_mt(rng(), posterior_mean, posterior_sd);
  }

  //----------------------------------------------------------------------
  double ProbitBartPosteriorSampler::log_integrated_likelihood(
      const Bart::SufficientStatisticsBase &suf) const {
    return log_integrated_probit_likelihood(
        dynamic_cast<const Bart::ProbitSufficientStatistics &>(suf));
  }

  //----------------------------------------------------------------------
  // Omits a factor of (2*pi)^{N/2} \exp{-.5 * (N - 1) * s^2 } from
  // the integrated likelihood.
  double ProbitBartPosteriorSampler::log_integrated_probit_likelihood(
      const Bart::ProbitSufficientStatistics &suf) const {
    double n = suf.sample_size();
    if (n <= 0) {
      return negative_infinity();
    }
    double ybar = suf.sum() / n;  // handle n == 0;
    double prior_variance = mean_prior_variance();

    double ivar = n + (1.0 / prior_variance);
    double posterior_variance = 1.0 / ivar;
    double posterior_mean = suf.sum() / ivar;
    double ans = log(posterior_variance / prior_variance) - n * square(ybar) +
                 square(posterior_mean) / posterior_variance;
    return .5 * ans;
  }

  //----------------------------------------------------------------------
  double ProbitBartPosteriorSampler::complete_data_log_likelihood(
      const Bart::SufficientStatisticsBase &suf) const {
    return complete_data_probit_log_likelihood(
        dynamic_cast<const Bart::ProbitSufficientStatistics &>(suf));
  }

  //----------------------------------------------------------------------
  double ProbitBartPosteriorSampler::complete_data_probit_log_likelihood(
      const Bart::ProbitSufficientStatistics &suf) const {
    double n = suf.sample_size();
    double ybar = n > 0 ? (suf.sum() / n) : 0;
    return -0.5 * n * square(ybar);
  }

  //----------------------------------------------------------------------
  void ProbitBartPosteriorSampler::clear_residuals() { residuals_.clear(); }

  //----------------------------------------------------------------------
  int ProbitBartPosteriorSampler::residual_size() const {
    return residuals_.size();
  }

  //----------------------------------------------------------------------
  Bart::ProbitResidualData *
  ProbitBartPosteriorSampler::create_and_store_residual(int i) {
    Ptr<BinomialRegressionData> data_point = model_->dat()[i];
    double original_prediction = model_->predict(data_point->x());
    std::shared_ptr<Bart::ProbitResidualData> ans(
        new Bart::ProbitResidualData(data_point, original_prediction));
    residuals_.push_back(ans);
    return ans.get();
  }

  //----------------------------------------------------------------------
  Bart::ProbitResidualData *ProbitBartPosteriorSampler::residual(int i) {
    return residuals_[i].get();
  }

  //----------------------------------------------------------------------
  Bart::ProbitSufficientStatistics *ProbitBartPosteriorSampler::create_suf()
      const {
    return new Bart::ProbitSufficientStatistics;
  }

  //----------------------------------------------------------------------
  void ProbitBartPosteriorSampler::impute_latent_data() {
    for (int i = 0; i < residuals_.size(); ++i) {
      impute_latent_data_point(residuals_[i].get());
    }
  }

  //----------------------------------------------------------------------
  void ProbitBartPosteriorSampler::impute_latent_data_point(DataType *data) {
    double eta = data->prediction();
    int n = data->n();
    int number_positive = data->y();
    int number_negative = n - number_positive;

    double sum_of_probits = 0;
    if (number_positive > 5) {
      double mean = 0;
      double variance = 1;
      trun_norm_moments(eta, 1, 0, true, &mean, &variance);
      sum_of_probits += rnorm_mt(rng(), number_positive * mean,
                                 sqrt(number_positive * variance));
    } else {
      for (int i = 0; i < number_positive; ++i) {
        sum_of_probits += rtrun_norm_mt(rng(), eta, 1, 0, true);
      }
    }

    if (number_negative > 5) {
      double mean = 0;
      double variance = 1;
      trun_norm_moments(eta, 1, 0, false, &mean, &variance);
      sum_of_probits += rnorm_mt(rng(), number_negative * mean,
                                 sqrt(number_negative * variance));
    } else {
      for (int i = 0; i < number_negative; ++i) {
        sum_of_probits += rtrun_norm_mt(rng(), eta, 1, 0, false);
      }
    }
    data->set_sum_of_residuals(sum_of_probits - (n * eta));
  }

}  // namespace BOOM
