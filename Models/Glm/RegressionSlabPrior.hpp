#ifndef BOOM_MODELS_GLM_REGRESSION_SLAB_PRIOR_HPP_
#define BOOM_MODELS_GLM_REGRESSION_SLAB_PRIOR_HPP_
/*
  Copyright (C) 2005-2021 Steven L. Scott

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

#include "Models/MvnBase.hpp"
#include "Models/Policies/ParamPolicy_2.hpp"
#include "Models/Policies/NullDataPolicy.hpp"
#include "Models/Policies/NullPriorPolicy.hpp"

namespace BOOM {

  // A conditional prior distribution for regression models.  Suitable as a
  // "slab" in a spike-and-slab prior.
  //
  // Let X be the design (predictor) matrix of the regression, with X'X the
  // cross-product matrix.  Let D be the diagonal matrix whose diagonal elements
  // match the diagonal of X'X.  And let M(mu, P) denote the multivariate normal
  // distribution with mean mu and variance matrix P^{-1}.
  //
  // The model here is beta ~ M(b, P0 / sigsq) where
  //   P0 = [(1-a) * X'X + a * D] * prior_sample_size / data_sample_size
  //
  // The first element of the vector b is the sample mean of the response
  // variable.  All other elements are 0.
  class RegressionSlabPrior
      : public MvnBase,
        public ParamPolicy_2<VectorParams, UnivParams>,
        public NullDataPolicy,
        public NullPriorPolicy
  {
   public:
    // Args:
    //   xtx:  The cross product matrix from regression sufficient statisitcs.
    //   sigsq:  The residual variance parameter from a regression model.
    //   sample_mean: The mean of the response (Y) variable from the regression
    //     model.  This becomes the default first element of the mean vector for
    //     the distribution.
    //   data_sample_size: The number of observations in the training data.  xtx
    //     is scaled down by this number so that it gives the "average
    //     information" in a single observation.
    //   prior_sample_size: The number of observations the prior is to carry.
    //   diagonal_shrinkage: A number between 0 and 1.  This is the 'a'
    //     parameter from the class documentation.  Diagonal shrinkage can help
    //     ensure the prior is proper when X'X is not full rank.
    RegressionSlabPrior(const SpdMatrix &xtx,
                        const Ptr<UnivParams> &sigsq,
                        double sample_mean,
                        double data_sample_size,
                        double prior_sample_size,
                        double diagonal_shrinkage = 0.0);

    ~RegressionSlabPrior();
    RegressionSlabPrior(const RegressionSlabPrior &rhs);
    RegressionSlabPrior * clone() const override;

    Ptr<VectorParams> Mu_prm() {
      return prm1();
    }

    Ptr<UnivParams> SampleSize_prm() {
      return prm2();
    }

    const Vector &mu() const override {
      return prm1_ref().value();
    }

    const SpdMatrix &Sigma() const override {
      ensure_sigma_current();
      return var_wsp_->var();
    }

    const SpdMatrix &siginv() const override {
      ensure_sigma_current();
      return var_wsp_->ivar();
    }

    double ldsi() const override {
      ensure_sigma_current();
      return var_wsp_->ldsi();
    }

    double prior_sample_size() const {
      return prm2_ref().value();
    }

    double diagonal_shrinkage() const {
      return diagonal_shrinkage_;
    }

    double data_sample_size() const {
      return data_sample_size_;
    }

    // Return  ((1-alpha) * xtx + alpha * diag(xtx)) / n
    // Args:
    //   xtx:  The base matrix to be modified.
    //   sample_size:  The 'n' parameter above.  n > 0
    //   diagonal_shrinkage:  The 'alpha' parameter above (0 <= alpha <= 1)
    static SpdMatrix scale_xtx(
        const SpdMatrix &xtx,
        double sample_size,
        double diagonal_shrinkage);

   private:
    // modified_xtx_ holds the xtx parameter supplied by the constructor, after
    // scaling by the data sample size and applying diagonal shrinkage.
    SpdMatrix modified_xtx_;

    // These aren't used after construction, but we keep them around so we can
    // recover the original construction parameters.
    double data_sample_size_;
    double diagonal_shrinkage_;

    // The residual variance parameter from the regression model described by
    // this prior.
    Ptr<UnivParams> sigsq_;

    mutable bool wsp_current_;
    mutable Ptr<SpdParams> var_wsp_;

    void ensure_sigma_current() const {
      if (wsp_current_) {
        return;
      }
      var_wsp_->set_ivar(modified_xtx_ * (prior_sample_size() / sigsq_->value()));
      wsp_current_ = true;
    }

    void set_observers();
    void remove_observers();
  };
}  // namespace BOOM

#endif  // BOOM_MODELS_GLM_REGRESSION_SLAB_PRIOR_HPP_
