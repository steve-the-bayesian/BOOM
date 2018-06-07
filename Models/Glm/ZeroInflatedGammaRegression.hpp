// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2016 Steven L. Scott

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

#ifndef BOOM_ZERO_INFLATED_GAMMA_REGRESSION_HPP_
#define BOOM_ZERO_INFLATED_GAMMA_REGRESSION_HPP_

#include "Models/Glm/BinomialLogitModel.hpp"
#include "Models/Glm/GammaRegressionModel.hpp"
#include "Models/Policies/CompositeParamPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"

namespace BOOM {

  // A model for semicontinuous non-negative data.  A mixture of a
  // point mass at zero and a regression model with Gamma errors.  The
  // probability of a nonzero (positive) observation is given by a
  // logistic regression.
  //
  // Let p(x) = 1 / (1 + exp(-logit_beta.dot(x))) be the probability
  // that y is nonzero, and let mu(x) = exp(gamma_beta.dot(x)) be the
  // conditional mean given that y > 0.  The distribution is a
  // mixture:
  //
  //   f(y) = (1 - p(x)) * I(0) + p(x) * Gamma(alpha, alpha / mu(x))
  //
  // The mean of this distribution is p(x) * mu(x).
  class ZeroInflatedGammaRegressionModel : public CompositeParamPolicy,
                                           public PriorPolicy,
                                           virtual public MLE_Model {
   public:
    explicit ZeroInflatedGammaRegressionModel(int xdim,
                                              double zero_threshold = 1e-5);
    ZeroInflatedGammaRegressionModel(
        const ZeroInflatedGammaRegressionModel &rhs);
    ZeroInflatedGammaRegressionModel *clone() const override;

    double expected_value(const Vector &x) const;
    double variance(const Vector &x) const;
    double standard_deviation(const Vector &x) const;
    double probability_nonzero(const Vector &x) const;
    double probability_zero(const Vector &x) const;

    Ptr<GlmCoefs> regression_coefficient_ptr() {
      return gamma_model_->coef_prm();
    }
    const GlmCoefs &regression_coefficients() const {
      return gamma_model_->coef();
    }

    Ptr<UnivParams> shape_prm() { return gamma_model_->shape_prm(); }
    double shape_parameter() const { return gamma_model_->shape_parameter(); }
    void set_shape_parameter(double alpha) {
      gamma_model_->set_shape_parameter(alpha);
    }

    Ptr<GlmCoefs> logit_coefficient_ptr() { return logit_model_->coef_prm(); }
    const GlmCoefs &logit_coefficients() const { return logit_model_->coef(); }

    double zero_threshold() const { return zero_threshold_; }
    double sim(const Vector &x, RNG &rng = BOOM::GlobalRng::rng) const;

    void add_data(const Ptr<RegressionData> &dp);
    void add_data(const Ptr<Data> &dp) override;

    void increment_sufficient_statistics(int number_zeros, int number_nonzero,
                                         double sum_of_nonzero,
                                         double sum_of_logs_of_nonzero,
                                         const Ptr<VectorData> &predictors);

    void clear_data() override;
    void combine_data(const Model &other_model, bool just_suf = true) override;

    GammaRegressionModelConditionalSuf *gamma_regression() {
      return gamma_model_.get();
    }
    const GammaRegressionModelConditionalSuf *gamma_regression() const {
      return gamma_model_.get();
    }
    BinomialLogitModel *logit_model() { return logit_model_.get(); }
    const BinomialLogitModel *logit_model() const { return logit_model_.get(); }

    void mle() override;

   private:
    Ptr<GammaRegressionModelConditionalSuf> gamma_model_;
    Ptr<BinomialLogitModel> logit_model_;
    double zero_threshold_;
  };

}  // namespace BOOM

#endif  //  BOOM_ZERO_INFLATED_GAMMA_REGRESSION_HPP_
