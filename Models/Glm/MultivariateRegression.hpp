// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2006 Steven L. Scott

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
#ifndef BOOM_MVREG_HPP
#define BOOM_MVREG_HPP
#include "LinAlg/QR.hpp"
#include "Models/Glm/Glm.hpp"
#include "Models/Policies/ParamPolicy_2.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/Policies/SufstatDataPolicy.hpp"
#include "Models/SpdParams.hpp"
#include "Models/Sufstat.hpp"

namespace BOOM {

  // Sufficient statsitics for multivariate regression models.
  class MvRegSuf : virtual public SufstatDetails<MvRegData> {
   public:
    // Args:
    //   xdim:  The dimension of the x (predictor) variable.
    //   ydim:  The dimension of the y (response) variable.
    MvRegSuf(uint xdim, uint ydim);

    // Args:
    //   X:  The design matrix.
    //   Y:  The matrix of responses.
    MvRegSuf(const Matrix &X, const Matrix &Y);

    // Build an MvRegSuf from a sequence of smart or raw pointers to
    // MvRegData.
    template <class Fwd>
    MvRegSuf(Fwd b, Fwd e);

    MvRegSuf(const MvRegSuf &rhs);
    MvRegSuf *clone() const override;

    void clear() override;

    uint xdim() const {return xtx().nrow();}
    uint ydim() const {return yty().nrow();}

    // Add data to the sufficient statistics managed by this object.
    void Update(const MvRegData &data) override;

    // Add the individual data components to the sufficient statistics
    // managed by this object.
    // Args:
    //   Y:  The response for a single data point.
    //   X:  The predictor for a single data point.
    //   w:  A weight to apply to the data point.
    virtual void update_raw_data(const Vector &Y, const Vector &X,
                                 double w = 1.0);

    // Clear the sufficient statistics that depend on y, but not the ones that
    // depend on X.  This is a useful optimization in some latent variable
    // models.
    void clear_y_keep_x();

    // Update the sufficient statistics that depend on y, but not the ones that
    // depend only on X.  This is a useful optimization in some latent variable
    // models.
    void update_y_not_x(const Vector &y, const Vector &x, double w);

    // Returns the least squares estimate of beta given the current
    // sufficient statistics.
    Matrix beta_hat() const;
    Matrix conditional_beta_hat(const SelectorMatrix &included) const;

    // Returns the sum of squared errors assuming beta = B.
    SpdMatrix SSE(const Matrix &B) const;

    const SpdMatrix &yty() const;  // sum_i y_i * y_i.transpose()
    const Matrix &xty() const;     // sum_i y_i * x_i.transpose()
    const SpdMatrix &xtx() const;  // sum_i x_i * x_i.transpose();
    double n() const;              // number of observations
    double sumw() const;           // sum of weights

    // Add the sufficient statistics managed by the argument to *this.
    void combine(const Ptr<MvRegSuf> &);
    virtual void combine(const MvRegSuf &);
    MvRegSuf *abstract_combine(Sufstat *s) override;

    Vector vectorize(bool minimal = true) const override;
    Vector::const_iterator unvectorize(Vector::const_iterator &v,
                                       bool minimal = true) override;
    Vector::const_iterator unvectorize(const Vector &v,
                                       bool minimal = true) override;
    std::ostream &print(std::ostream &out) const override;

   private:
    SpdMatrix yty_;
    SpdMatrix xtx_;
    Matrix xty_;
    double sumw_;
    double n_;
  };

  // Implementation for the sequence constructor.  It is assumed that b and e
  template <class Fwd>
  MvRegSuf::MvRegSuf(Fwd b, Fwd e) {
    Ptr<MvRegData> dp = *b;
    const Vector &x(dp->x());
    const Vector &y(dp->y());

    uint xdim = x.size();
    uint ydim = y.size();
    xtx_ = SpdMatrix(xdim, 0.0);
    yty_ = SpdMatrix(ydim, 0.0);
    xty_ = Matrix(xdim, ydim, 0.0);
    n_ = 0;
    sumw_ = 0;

    while (b != e) {
      this->update(*b);
      ++b;
    }
  }

  //============================================================
  // Multivariate regression, where both y_i and x_i are vectors.
  class MultivariateRegressionModel
      : public ParamPolicy_2<MatrixGlmCoefs, SpdParams>,
        public SufstatDataPolicy<MvRegData, MvRegSuf>,
        public PriorPolicy,
        public LoglikeModel {
   public:
    // Args:
    //   xdim: The dimension of the predictor, including the intercept
    //     (if any).
    //   ydim:  The dimension of the response.
    MultivariateRegressionModel(uint xdim, uint ydim);

    // Args:
    //   X:  The design matrix.
    //   Y:  The matrix of responses.  The number of rows must match X.
    MultivariateRegressionModel(const Matrix &X, const Matrix &Y);

    // Args:
    //   B: The matrix of regression coefficients.  The number of rows
    //     defines the dimension of the predictor.  The number of
    //     columns defines the dimension of the response.
    //   Sigma: The residual variance matrix.  Its dimension must
    //     match ncol(B).
    MultivariateRegressionModel(const Matrix &B, const SpdMatrix &Sigma);

    using MvReg = MultivariateRegressionModel;
    MultivariateRegressionModel(
        const MultivariateRegressionModel &rhs) = default;
    MultivariateRegressionModel(
        MultivariateRegressionModel &&rhs) = default;
    MultivariateRegressionModel &operator=(
        const MultivariateRegressionModel &rhs) = default;
    MultivariateRegressionModel &operator=(
        MultivariateRegressionModel &&rhs) = default;
    MultivariateRegressionModel *clone() const override;

    // Dimension of the predictor (including the intercept, if any).
    uint xdim() const;

    // Dimension of the response variable.
    uint ydim() const;

    // Matrix of regression coefficients, with xdim() rows, ydim()
    // columns.
    const Matrix &Beta() const;
    void set_Beta(const Matrix &B);

    const SelectorMatrix &included_coefficients() const {
      return Beta_prm_ref().included_coefficients();
    }

    // Residual variance matrix.
    const SpdMatrix &Sigma() const;
    void set_Sigma(const SpdMatrix &V);

    // Matrix inverse of the residual variance matrix;
    const SpdMatrix &Siginv() const;
    void set_Siginv(const SpdMatrix &iV);

    // The Cholesky decomposition of Siginv.
    Matrix residual_precision_cholesky() const;

    // log determinant of Siginv().
    double ldsi() const;

    // Access to parameters.
    Ptr<MatrixGlmCoefs> Beta_prm();
    const Ptr<MatrixGlmCoefs> Beta_prm() const;
    const MatrixGlmCoefs &Beta_prm_ref() const {return prm1_ref();}

    Ptr<SpdParams> Sigma_prm();
    const Ptr<SpdParams> Sigma_prm() const;

    //--- estimation and probability calculations
    void mle() override;
    // The argument to loglike is a vector created by stacking the
    // columns of Beta, and the upper triangle of Sigma
    double loglike(const Vector &beta_sigma) const override;
    //
    double log_likelihood(const Matrix &Beta, const SpdMatrix &Sigma) const;
    double log_likelihood_ivar(const Matrix &Beta, const SpdMatrix &Siginv) const;
    double log_likelihood() const override;

    virtual double pdf(const Ptr<Data> &, bool) const;

    // Returns x * Beta();
    virtual Vector predict(const Vector &x) const;

    //---- simulate MV regression data ---
    virtual MvRegData *sim(RNG &rng = GlobalRng::rng) const;
    virtual MvRegData *sim(const Vector &X, RNG &rng = GlobalRng::rng) const;

    // no intercept
    Vector simulate_fake_x(RNG &rng = GlobalRng::rng) const;

  };
}  // namespace BOOM
#endif  // BOOM_MVREG_HPP
