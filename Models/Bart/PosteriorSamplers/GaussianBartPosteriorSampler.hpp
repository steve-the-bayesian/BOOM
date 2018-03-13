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
#ifndef BOOM_GAUSSIAN_BART_POSTERIOR_SAMPLER_HPP_
#define BOOM_GAUSSIAN_BART_POSTERIOR_SAMPLER_HPP_

#include "Models/Bart/Bart.hpp"
#include "Models/Bart/GaussianBartModel.hpp"
#include "Models/Bart/PosteriorSamplers/BartPosteriorSampler.hpp"
#include "Models/Bart/ResidualRegressionData.hpp"
#include "Models/ChisqModel.hpp"
#include "Models/PosteriorSamplers/GenericGaussianVarianceSampler.hpp"

namespace BOOM {
  namespace Bart {
    class GaussianBartSufficientStatistics;

    // This is the internal data type managed by the
    // GaussianBartPosteriorSampler, and fed to the nodes of the
    // GaussianBartModel model being managed.
    class GaussianResidualRegressionData : public ResidualRegressionData {
     public:
      // The data_point argument retains ownership of the data it
      // manages.  It must remain in scope while the
      // GaussianBartPosteriorSampler does its thing.  This should be
      // fine, as data_point itself is owned by the GaussianBartModel
      // managed by the GaussianBartPosteriorSampler.
      //
      // At construction time, the residual is the same as the
      // original observed response.
      GaussianResidualRegressionData(const Ptr<RegressionData> &data_point,
                                     double original_prediction);
      double y() const { return observed_response_->y(); }
      double residual() const { return residual_; }
      void set_residual(double r) { residual_ = r; }
      void add_to_residual(double value) override { residual_ += value; }
      void add_to_gaussian_suf(
          GaussianBartSufficientStatistics &suf) const override;

     private:
      const RegressionData *observed_response_;
      double residual_;
    };

    class GaussianBartSufficientStatistics : public SufficientStatisticsBase {
     public:
      GaussianBartSufficientStatistics *clone() const override {
        return new GaussianBartSufficientStatistics(*this);
      }
      void clear() override { suf_.clear(); }
      void update(const ResidualRegressionData &abstract_data) override {
        abstract_data.add_to_gaussian_suf(*this);
      }
      virtual void update(const GaussianResidualRegressionData &data) {
        suf_.update_raw(data.residual());
      }
      double n() const { return suf_.n(); }
      double ybar() const { return suf_.ybar(); }
      double sum() const { return suf_.sum(); }
      double sumsq() const { return suf_.sumsq(); }
      double sample_var() const { return suf_.sample_var(); }

     private:
      GaussianSuf suf_;
    };

  }  // namespace Bart

  class GaussianBartPosteriorSampler : public BartPosteriorSamplerBase {
    // The prior is that the probability of a node at depth 'd'
    // splitting is a / (1 + d)^b.  Given a split, a variable is
    // chosen uniformly from the set of available variables, and a
    // cutpoint uniformly from the set of available cutpoints.  Note
    // that 'available' is influenced by a node's position in the
    // tree, because splits made by ancestors will make some splits
    // logically impossible, and impossible splits are not
    // 'available.'  For example, descendants cannot split on the same
    // dummy variable as an ancestor.  The conditional prior on the
    // mean parameters at the leaves is N(0, prediction_sd^2 / number
    // of trees), and the prior on the residual variance is 1/sigma^2
    // ~ Gamma( prior_residual_sd_weight / 2, prior_residual_sd_weight
    // * prior_residual_sd_guess^2 / 2).
   public:
    GaussianBartPosteriorSampler(
        GaussianBartModel *model, double prior_residual_sd_guess,
        double prior_residual_sd_weight, double prediction_sd,
        double prior_tree_depth_alpha, double prior_tree_depth_beta,
        const std::function<double(int)> &log_prior_on_number_of_trees,
        RNG &seeding_rng = GlobalRng::rng);
    //----------------------------------------------------------------------
    // Virtual function over-rides....

    void draw() override;
    double draw_mean(Bart::TreeNode *leaf) override;

    double log_integrated_likelihood(
        const Bart::SufficientStatisticsBase &suf) const override;
    double log_integrated_gaussian_likelihood(
        const Bart::GaussianBartSufficientStatistics &suf) const;

    double complete_data_log_likelihood(
        const Bart::SufficientStatisticsBase &suf) const override;
    double complete_data_gaussian_log_likelihood(
        const Bart::GaussianBartSufficientStatistics &suf) const;

    void clear_residuals() override;
    int residual_size() const override;

    Bart::GaussianResidualRegressionData *create_and_store_residual(
        int i) override;
    Bart::GaussianResidualRegressionData *residual(int i) override;

    Bart::GaussianBartSufficientStatistics *create_suf() const override {
      return new Bart::GaussianBartSufficientStatistics;
    }

    //----------------------------------------------------------------------
    // non-virtual functions start here.

    // Draw the residual variance given structure and mean parameters.
    void draw_residual_variance();

    const std::vector<const Bart::GaussianResidualRegressionData *> residuals()
        const;

    void set_residual(int i, double residual);

   private:
    static const double log_2_pi;
    GaussianBartModel *model_;
    GenericGaussianVarianceSampler sigsq_sampler_;

    // Residuals will be held by all the nodes in all the trees.
    // Local changes will be reflected in other trees, so they need to
    // be locally adjusted before they are used.  This makes the
    // algorithm thread-unsafe.
    std::vector<std::shared_ptr<Bart::GaussianResidualRegressionData> >
        residuals_;
  };

}  // namespace BOOM

#endif  //  BOOM_GAUSSIAN_BART_POSTERIOR_SAMPLER_HPP_
