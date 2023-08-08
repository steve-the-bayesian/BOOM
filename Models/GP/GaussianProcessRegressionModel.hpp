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

#include "Models/GP/kernels.hpp"
#include "Models/Policies/ParamPolicy_3.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"

#include "LinAlg/Vector.hpp"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/SpdMatrix.hpp"

#include "Models/Glm/Glm.hpp"
#include "Models/MvnModel.hpp"

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

    Ptr<KernelParams> kernel_param() {
      return ParamPolicy::prm2();
    }
    const Ptr<KernelParams> kernel_param() const {
      return ParamPolicy::prm2();
    }

    Ptr<FunctionParams> mean_param() {
      return ParamPolicy::prm1();
    }

    const Ptr<FunctionParams> mean_param() const {
      return ParamPolicy::prm1();
    }

    Ptr<UnivParams> sigsq_param() {
      return ParamPolicy::prm3();
    }

    const Ptr<UnivParams> sigsq_param() const {
      return ParamPolicy::prm3();
    }

    //--------------- Using the parameters

    double kernel(const ConstVectorView &x1, const ConstVectorView &x2) const {
      return prm2_ref()(x1, x2);
    }

    double mean_function(const ConstVectorView &x) const {
      return ParamPolicy::prm1_ref()(x);
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

    //----------- Data access

    using DataPolicy::add_data;

    void add_data(const Ptr<RegressionData> &data_point) override {
      kernel_matrix_current_ = false;
      DataPolicy::add_data(data_point);
    }

    //----------- Prediction
    double predict(const Vector &x) const;

    // Returns a (joint) distribution over the Y values of
    Ptr<MvnModel> predict_distribution(const Matrix &X) const;

    Vector posterior_residuals() const;

    double loglike(const Vector &theta) const override;

   private:
    double evaluate_log_likelihood() const;

    mutable bool kernel_matrix_current_;

    // The inverse of the kernel matrix based on the training data.
    mutable SpdMatrix Kinv_;

    // The residuals from the prior mean function.
    mutable Vector residuals_;

    // Put observers on the kernel and mean function parameters so if the
    // parameters change our kernel matrix will be invalidated.
    void add_observers();

    // Refresh the mutable parameters
    void refresh_kernel_matrix() const;
  };


}  // namespace BOOM


#endif  //  BOOM_MODELS_GP_GAUSSIAN_PROCESS_REGRESSION_HPP_
