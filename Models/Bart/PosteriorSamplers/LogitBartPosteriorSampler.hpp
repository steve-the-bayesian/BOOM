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

#ifndef BOOM_LOGIT_BART_POSTERIOR_SAMPLER_HPP_
#define BOOM_LOGIT_BART_POSTERIOR_SAMPLER_HPP_
#include "Models/Bart/LogitBartModel.hpp"
#include "Models/Bart/PosteriorSamplers/BartPosteriorSampler.hpp"
#include "Models/Bart/ResidualRegressionData.hpp"
#include "Models/Glm/PosteriorSamplers/BinomialLogitDataImputer.hpp"

namespace BOOM {
  namespace Bart {
    class LogitSufficientStatistics;

    // One instance of LogitResidualData considers the local
    // sufficient statistics for N success failure trials where y()
    // successes were observed.  N == 1 is an important special case.
    // Each trial is associated with a latent variable z[i] ~
    // Logistic(sum_of_trees) = MixtureOfNormals(sum_of_trees,
    // variances).  The trial is a success if z[i] > 0 and a failure
    // otherwise.
    //
    // During data augmentation, each z[i] is associated with a
    // variance sigsq[i].  This class tracks the information weighted
    // sum of z's: sum_i z[i] / sigsq[i], and the sum of the
    // information: sum_i 1/sigsq[i].
    class LogitResidualData : public ResidualRegressionData {
     public:
      LogitResidualData(const Ptr<BinomialRegressionData> &data_point,
                        double original_prediction);
      double y() const { return original_data_->y(); }
      double n() const { return original_data_->n(); }

      void add_to_residual(double value) override;
      void add_to_logit_suf(LogitSufficientStatistics &suf) const override;

      double information_weighted_sum() const {
        return information_weighted_sum_;
      }
      double sum_of_information() const { return sum_of_information_; }

      // The predicted value is subtracted from the latent logit in
      // each Bernoulli trial in the
      // weighted_sum_of_latent_logit_residuals.  Because each
      // Bernoulli trial has the same mean, we simply subtract
      // (prediction * sum_of_information).
      void set_latent_data(double information_weighted_sum_of_latent_logits,
                           double sum_of_information);

      // The value of the sum-of-trees for this data point.  It is
      // cheaper to maintain it here and adjust it as needed than to
      // recompute it each time it is needed.  The value of the
      // prediction is adjusted each time add_to_residual or
      // subtract_from_residual is called.
      //
      // The return value is the log odds of success for a single
      // Bernoulli trial.
      double prediction() const { return prediction_; }
      void set_prediction(double value) { prediction_ = value; }

     private:
      const BinomialRegressionData *original_data_;

      // Let z[i] denote the latent logit random variable for
      // Bernoulli observation i.  Let \sigma^2_i be the variance of
      // z[i] in the normal mixture, and let w[i] = 1.0 / sigma^2_i.

      // sum_of_information = \sum_i w[i].
      double sum_of_information_;

      // information_weighted_sum_ = \sum_i w[i] * z[i].
      double information_weighted_sum_;

      // The log_odds of a success at this data point.
      double prediction_;
    };

    //======================================================================
    class LogitSufficientStatistics : public SufficientStatisticsBase {
     public:
      LogitSufficientStatistics();
      LogitSufficientStatistics *clone() const override;
      void clear() override;
      void update(const ResidualRegressionData &abstract_data) override;
      virtual void update(const LogitResidualData &data);

      double sum_of_information() const;
      double information_weighted_sum() const;
      double information_weighted_residual_sum() const;

      double information_weighted_cross_product() const;
      double information_weighted_sum_of_squared_predictions() const;

     private:
      // \sum_i \sum_j information_{ij} * latent_observation_{ij}
      double information_weighted_sum_;

      // \sum_i \sum_j information_{ij}
      double sum_of_information_;

      // \sum_i prediction{i}  * \sum_j information_{ij}
      double information_weighted_prediction_;

      // \sum_i prediction{i} \sum_j information_{ij} * latent_observation_{ij}
      double information_weighted_sum_of_observation_times_prediction_;

      // \sum_i prediction{i}^2 * \sum_j \information_{ij}
      double information_weighted_sum_of_squared_predictions_;
    };
  }  // namespace Bart

  //======================================================================
  // Posterior sampler for the LogitBartModel.
  class LogitBartPosteriorSampler : public BartPosteriorSamplerBase {
   public:
    typedef Bart::LogitResidualData DataType;
    LogitBartPosteriorSampler(
        LogitBartModel *model, double total_prediction_sd,
        double prior_tree_depth_alpha, double prior_tree_depth_beta,
        const std::function<double(int)> &log_prior_on_number_of_trees,
        RNG &seeding_rng = GlobalRng::rng);
    void draw() override;
    double draw_mean(Bart::TreeNode *leaf) override;
    double log_integrated_likelihood(
        const Bart::SufficientStatisticsBase &suf) const override;
    double log_integrated_logit_likelihood(
        const Bart::LogitSufficientStatistics &suf) const;

    double complete_data_log_likelihood(
        const Bart::SufficientStatisticsBase &suf) const override;
    double complete_data_logit_log_likelihood(
        const Bart::LogitSufficientStatistics &suf) const;

    void clear_residuals() override;
    int residual_size() const override;
    Bart::LogitResidualData *create_and_store_residual(int i) override;
    Bart::LogitResidualData *residual(int i) override;
    Bart::LogitSufficientStatistics *create_suf() const override;

    void impute_latent_data();
    void impute_latent_data_point(DataType *data);

   private:
    LogitBartModel *model_;
    std::vector<std::shared_ptr<DataType> > residuals_;
    std::shared_ptr<BinomialLogitDataImputer> data_imputer_;
  };

}  // namespace BOOM

#endif  //  BOOM_LOGIT_BART_POSTERIOR_SAMPLER_HPP_
