// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005 Steven L. Scott

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

#ifndef BOOM_T_REGRESSION_HPP
#define BOOM_T_REGRESSION_HPP

#include "Models/Glm/Glm.hpp"
#include "Models/Glm/WeightedRegressionModel.hpp"
#include "Models/Policies/CompositeParamPolicy.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/ParamPolicy_3.hpp"
#include "Models/Policies/PriorPolicy.hpp"

namespace BOOM {

  class WeightedRegSuf;

  class TRegressionModel
      : public GlmModel,
        public ParamPolicy_3<GlmCoefs, UnivParams, UnivParams>,
        public IID_DataPolicy<RegressionData>,
        public PriorPolicy,
        public NumOptModel
  {
   public:
    explicit TRegressionModel(uint xdim);  // dimension of beta
    TRegressionModel(const Vector &b, double Sigma, double nu = 30);
    TRegressionModel(const Matrix &X, const Vector &y);
    TRegressionModel *clone() const override;

    GlmCoefs &coef() override;
    const GlmCoefs &coef() const override;
    Ptr<GlmCoefs> coef_prm() override;
    const Ptr<GlmCoefs> coef_prm() const override;
    Ptr<UnivParams> Sigsq_prm();
    const Ptr<UnivParams> Sigsq_prm() const;
    Ptr<UnivParams> Nu_prm();
    const Ptr<UnivParams> Nu_prm() const;

    // beta() and Beta() inherited from GlmModel;
    const double &sigsq() const;
    double sigma() const;
    void set_sigsq(double s2);

    const double &nu() const;
    void set_nu(double Nu);

    // The variance of the T distribution with 'nu' degrees of freedom and
    // scatter parameter 'sigma'.
    double residual_variance() const {
      double nu = this->nu();
      if (nu > 2.0) {
        return sigsq() * nu / (nu - 2.0);
      } else {
        return infinity();
      }
    }

    // The argument to Loglike is a vector containing the included
    // regression coefficients, followed by the residual 'dispersion'
    // parameter sigsq, followed by the tail thickness parameter nu.
    double Loglike(const Vector &beta_sigsq_nu, Vector &g, Matrix &h,
                   uint nd) const override;

    // Args:
    //   full_beta: The full set of regression coefficients, including
    //     any that are set to zero.
    //   sigma:  The "residual standard deviation" parameter.
    //   nu:  The tail thickness parameter.
    double log_likelihood(const Vector &full_beta, double sigma,
                          double nu) const;

    double log_likelihood() const override {
      return log_likelihood(Beta(), sigma(), nu());
    }

    // The MLE is computed using an EM algorithm.
    void mle() override;

    double pdf(const Ptr<Data> &dp, bool) const;
    double pdf(const Ptr<DataType> &dp, bool) const;

    Ptr<RegressionData> sim(RNG &rng = GlobalRng::rng) const;
    Ptr<RegressionData> sim(const Vector &X,
                            RNG &rng = GlobalRng::rng) const;

   private:
    // Clear 'suf' and fill it with the expected complete data
    // sufficient statistics.
    void EStep(WeightedRegSuf &suf) const;

    // Take the contents of suf and use it to set model parameters to
    // their MLE's.  Estimate of nu is based on the observed data.
    // Return the observed data log likelihood given the new
    // parameters.
    double MStep(const WeightedRegSuf &suf);
  };

  //===========================================================================
  // A TRegressionModel that is an explicit mixture of normals.

  class CompleteDataStudentRegressionModel
      : public TRegressionModel,
        public LatentVariableModel {
   public:
    explicit CompleteDataStudentRegressionModel(int xdim)
        : TRegressionModel(xdim),
          suf_(new WeightedRegSuf(xdim)),
          latent_data_disabled_(false)
    {}

    CompleteDataStudentRegressionModel(
        const CompleteDataStudentRegressionModel &rhs);
    CompleteDataStudentRegressionModel(
        CompleteDataStudentRegressionModel &&rhs) = default;
    CompleteDataStudentRegressionModel &operator=(
        CompleteDataStudentRegressionModel &&rhs) = default;

    CompleteDataStudentRegressionModel * clone() const override;

    void clear_data() override {
      TRegressionModel::clear_data();
      suf_->clear();
      weights_.clear();
    }

    void add_data(double y, const Vector &x, double weight) {
      suf_->add_data(x, y, weight);
      weights_.push_back(weight);
    }

    void add_data(const Ptr<Data> &dp) override {
      Ptr<RegressionData> reg_data = dp.dcast<RegressionData>();
      add_data(reg_data);
    }

    void add_data(const Ptr<RegressionData> &dp, double weight) {
      suf_->add_data(dp->x(), dp->y(), weight);
      weights_.push_back(weight);
      DataPolicy::add_data(dp);
    }

    void add_data(RegressionData *dp) override {
      add_data(Ptr<RegressionData>(dp));
    }

    void add_data(const Ptr<RegressionData> &dp) override {
      TRegressionModel::add_data(dp);
      weights_.push_back(1.0);
      suf_->add_data(dp->x(), dp->y(), weights_.back());
    }

    void remove_data(const Ptr<Data> &dp) override {
      auto it = std::find(dat().begin(), dat().end(), dp);
      if (it != dat().end()) {
        int index = it - dat().begin();
        double weight = weights_[index];
        weights_.erase(weights_.begin() + index);
        Ptr<RegressionData> regression_data = dp.dcast<RegressionData>();
        suf_->remove_data(regression_data->x(), regression_data->y(), weight);
      }
    }

    void set_weight(size_t i, double value) {
      weights_[i] = value;
    }
    double weight(size_t i) const {return weights_[i];}

    WeightedRegSuf *suf() {return suf_.get();}
    const Ptr<WeightedRegSuf> &suf() const {return suf_;}

    void impute_latent_data(RNG &rng) override;

    // Calling this function changes impute_latent_data() into a no-op.  This
    // can be desirable if this model is used as a building block in a larger
    // model that wishes to take on the responsibility of imputing the latent
    // variables.
    void disable_imputation(bool disabled = true) {
      latent_data_disabled_ = disabled;
    }

    bool latent_data_disabled() const {return latent_data_disabled_;}

   private:
    Ptr<WeightedRegSuf> suf_;
    Vector weights_;
    bool latent_data_disabled_;
  };

}  // namespace BOOM

#endif  // BOOM_T_REGRESSION_HPP
