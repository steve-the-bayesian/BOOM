#ifndef BOOM_MODELS_GP_GAUSSIAN_PROCESS_REGRESSION_HPP_
#define BOOM_MODELS_GP_GAUSSIAN_PROCESS_REGRESSION_HPP_
/*
  Copyright (C) 2005-2023 Steven L. Scott

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

#include "Models/GP/GpMeanFunction.hpp"
#include "Models/GP/kernels.hpp"
#include "Models/Policies/ParamPolicy_3.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"

#include "LinAlg/Vector.hpp"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/SpdMatrix.hpp"

#include "Models/Glm/Glm.hpp"
#include "Models/MvnModel.hpp"
#include "Models/LowRankMvnModel.hpp"

namespace BOOM {

  // A Gaussian process regression is a nonparametric regression model that
  // assumes a collection of observations (Y | X), where Y is a vector and X is
  // an associated predictor matrix, has a multivariate normal distribution with
  // mean mu(X) and variance matrix K(X, X) + sigsq * I.
  //
  // The mean function m and "kernal function" K operate row-wise, so that if xi
  // is row i of X then mu(X) = (mu0(x1), mu0(x2), ...).  Likewise, element i, j
  // of K(X, X) is k(xi, xj).
  //
  // Both the mean function and the kernel function may depend on model
  // parameters.  Any such dependence is encapsulated by wrapping these
  // functions in FunctionParams and KernelParams objects.
  class GaussianProcessRegressionModel :
      public ParamPolicy_3<FunctionParams, KernelParams, UnivParams>,
      public IID_DataPolicy<RegressionData>,
      public PriorPolicy,
      public LoglikeModel
  {
   public:
    GaussianProcessRegressionModel(
        const Ptr<FunctionParams> &mean_function,
        const Ptr<KernelParams> &kernel_function,
        const Ptr<UnivParams> &residual_variance);

    GaussianProcessRegressionModel(const GaussianProcessRegressionModel &rhs);

    GaussianProcessRegressionModel * clone() const override;

    //----------- Parameter access

    Ptr<FunctionParams> mean_param() {
      return ParamPolicy::prm1();
    }

    const Ptr<FunctionParams> mean_param() const {
      return ParamPolicy::prm1();
    }

    Ptr<KernelParams> kernel_param() {
      return ParamPolicy::prm2();
    }
    const Ptr<KernelParams> kernel_param() const {
      return ParamPolicy::prm2();
    }

    Ptr<UnivParams> sigsq_param() {
      return ParamPolicy::prm3();
    }

    const Ptr<UnivParams> sigsq_param() const {
      return ParamPolicy::prm3();
    }

    //--------------- Using the parameters

    double mean_function(const ConstVectorView &x) const {
      return ParamPolicy::prm1_ref()(x);
    }

    double kernel(const ConstVectorView &x1, const ConstVectorView &x2) const {
      return prm2_ref()(x1, x2);
    }

    double sigma() const {
      return sqrt(sigsq());
    }

    double sigsq() const {
      return ParamPolicy::prm3_ref().value();
    }

    double residual_variance() const {
      return sigsq();
    }

    double residual_sd() const {
      return sigma();
    }

    void set_sigsq(double sigsq) {
      sigsq_param()->set(sigsq);
    }

    // The inverse of the kernel matrix (K(X) + sigsq) evaluated at the training
    // data.
    const SpdMatrix &inverse_kernel_matrix() const {
      refresh_kernel_matrix();
      return Kinv_;
    }

    //----------- Data access

    using DataPolicy::add_data;

    void add_data(const Ptr<RegressionData> &data_point) override {
      kernel_matrix_current_ = false;
      DataPolicy::add_data(data_point);
    }

    size_t sample_size() const {return dat().size();}
    size_t xdim() const {
      return dat().empty() ? 0 : dat()[0]->xdim();
    }

    //----------- Prediction
    double predict(const Vector &x) const;

    // Compute the predictive distribution of the data at specific X points.
    // This distribution incorporates the residual error around specific data
    // points.
    //
    // Args:
    //   X: The matrix of points (rows in the matrix) where the predictive
    //     distribution is desired.
    //   predict_data: If true then the predictive distribution is an MvnModel
    //     describing the distribution of individual data points at the
    //     locations in X.  If false then the predictive distribution is a
    //     LowRankMvnModel describing the function values a the locations in X.
    //
    // Returns:
    //   A MvnModel object giving the predictive distribution at the locations
    //   specified in X.
    Ptr<MvnBase> predict_distribution(
        const Matrix &X, bool predict_data = true) const;

    Vector posterior_residuals() const;

    double loglike(const Vector &theta) const override;
    double log_likelihood() const override {
      return evaluate_log_likelihood();
    }

   private:
    double evaluate_log_likelihood() const;

    mutable bool kernel_matrix_current_;

    // The inverse of the kernel matrix based on the training data.  This matrix
    // includes contributions from the residual variance, so it describes
    // individual data points.  This one is for "prediction intervals" not
    // "confidence intervals."
    mutable SpdMatrix Kinv_;

    // The kernel matrix based on the training data.  This matrix omits
    // contributions from the residual variance, so it describes covariance of
    // the posterior mean function.  As such, its values may be highly
    // correlated.  It should not be inverted directly.
    mutable SpdMatrix Kfunc_;

    // The residuals from the prior mean function.
    mutable Vector residuals_;

    // Put observers on the kernel and mean function parameters so if the
    // parameters change our kernel matrix will be invalidated.
    void add_observers();

    // Refresh the mutable parameters.  Fill a matrix K with K(X) + sigsq, where
    // X is the matrix of predictors in the training data.
    void refresh_kernel_matrix() const;
  };


}  // namespace BOOM


#endif  //  BOOM_MODELS_GP_GAUSSIAN_PROCESS_REGRESSION_HPP_
