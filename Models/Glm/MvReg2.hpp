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

  class MvRegSuf : virtual public Sufstat {
   public:
    typedef std::vector<Ptr<MvRegData> > dataset_type;
    typedef Ptr<dataset_type, false> dsetPtr;

    MvRegSuf *clone() const override = 0;

    uint xdim() const;
    uint ydim() const;
    virtual const SpdMatrix &yty() const = 0;
    virtual const Matrix &xty() const = 0;
    virtual const SpdMatrix &xtx() const = 0;
    virtual double n() const = 0;
    virtual double sumw() const = 0;

    virtual SpdMatrix SSE(const Matrix &B) const = 0;

    virtual Matrix beta_hat() const = 0;
    virtual void combine(const Ptr<MvRegSuf> &) = 0;
  };
  //------------------------------------------------------------
  class MvReg;
  class QrMvRegSuf : public MvRegSuf, public SufstatDetails<MvRegData> {
   public:
    QrMvRegSuf(const Matrix &X, const Matrix &Y, MvReg *);
    QrMvRegSuf(const Matrix &X, const Matrix &Y, const Vector &w, MvReg *);
    QrMvRegSuf *clone() const override;

    void Update(const MvRegData &) override;
    Matrix beta_hat() const override;
    SpdMatrix SSE(const Matrix &B) const override;
    void clear() override;

    const SpdMatrix &yty() const override;
    const Matrix &xty() const override;
    const SpdMatrix &xtx() const override;
    double n() const override;
    double sumw() const override;

    void refresh(const std::vector<Ptr<MvRegData> > &) const;
    void refresh(const Matrix &X, const Matrix &Y) const;
    void refresh(const Matrix &X, const Matrix &Y, const Vector &w) const;
    void refresh() const;
    void combine(const Ptr<MvRegSuf> &) override;
    virtual void combine(const MvRegSuf &);
    QrMvRegSuf *abstract_combine(Sufstat *s) override;

    Vector vectorize(bool minimal = true) const override;
    Vector::const_iterator unvectorize(Vector::const_iterator &v,
                                       bool minimal = true) override;
    Vector::const_iterator unvectorize(const Vector &v,
                                       bool minimal = true) override;
    ostream &print(ostream &out) const override;

   private:
    mutable QR qr;
    mutable Matrix y_;
    mutable Vector w_;

    MvReg *owner;

    mutable bool current;
    mutable SpdMatrix yty_;
    mutable SpdMatrix xtx_;
    mutable Matrix xty_;
    mutable double n_;
    mutable double sumw_;
  };

  //------------------------------------------------------------
  // Sufficient statistics for the multivariate regression model based
  // on the normal equations.
  class NeMvRegSuf : public MvRegSuf, public SufstatDetails<MvRegData> {
   public:
    // Args:
    //   xdim:  The dimension of the x (predictor) variable.
    //   ydim:  The dimension of the y (response) variable.
    NeMvRegSuf(uint xdim, uint ydim);

    // Args:
    //   X:  The design matrix.
    //   Y:  The matrix of responses.
    NeMvRegSuf(const Matrix &X, const Matrix &Y);

    // Build an NeMvRegSuf from a sequence of smart or raw pointers to
    // MvRegData.
    template <class Fwd>
    NeMvRegSuf(Fwd b, Fwd e);

    NeMvRegSuf(const NeMvRegSuf &rhs);
    NeMvRegSuf *clone() const override;

    void clear() override;

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

    // Returns the least squares estimate of beta given the current
    // sufficient statistics.
    Matrix beta_hat() const override;

    // Returns the sum of squared errors assuming beta = B.
    SpdMatrix SSE(const Matrix &B) const override;

    const SpdMatrix &yty() const override;  // sum_i y_i * y_i.transpose()
    const Matrix &xty() const override;     // sum_i y_i * x_i.transpose()
    const SpdMatrix &xtx() const override;  // sum_i x_i * x_i.transpose();
    double n() const override;              // number of observations
    double sumw() const override;           // sum of weights

    // Add the sufficient statistics managed by the argument to *this.
    void combine(const Ptr<MvRegSuf> &) override;
    virtual void combine(const MvRegSuf &);
    NeMvRegSuf *abstract_combine(Sufstat *s) override;

    Vector vectorize(bool minimal = true) const override;
    Vector::const_iterator unvectorize(Vector::const_iterator &v,
                                       bool minimal = true) override;
    Vector::const_iterator unvectorize(const Vector &v,
                                       bool minimal = true) override;
    ostream &print(ostream &out) const override;

   private:
    SpdMatrix yty_;
    SpdMatrix xtx_;
    Matrix xty_;
    double sumw_;
    double n_;
  };

  template <class Fwd>
  NeMvRegSuf::NeMvRegSuf(Fwd b, Fwd e) {
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
  class MvReg : public ParamPolicy_2<MatrixParams, SpdParams>,
                public SufstatDataPolicy<MvRegData, MvRegSuf>,
                public PriorPolicy,
                public LoglikeModel {
   public:
    // Args:
    //   xdim: The dimension of the predictor, including the intercept
    //     (if any).
    //   ydim:  The dimension of the response.
    MvReg(uint xdim, uint ydim);

    // Args:
    //   X:  The design matrix.
    //   Y:  The matrix of responses.  The number of rows must match X.
    MvReg(const Matrix &X, const Matrix &Y);

    // Args:
    //   B: The matrix of regression coefficients.  The number of rows
    //     defines the dimension of the predictor.  The number of
    //     columns defines the dimension of the response.
    //   Sigma: The residual variance matrix.  Its dimension must
    //     match ncol(B).
    MvReg(const Matrix &B, const SpdMatrix &Sigma);

    MvReg(const MvReg &rhs);
    MvReg *clone() const override;

    // Dimension of the predictor (including the intercept, if any).
    uint xdim() const;

    // Dimension of the response variable.
    uint ydim() const;

    // Matrix of regression coefficients, with xdim() rows, ydim()
    // columns.
    const Matrix &Beta() const;
    void set_Beta(const Matrix &B);

    // Residual variance matrix.
    const SpdMatrix &Sigma() const;
    void set_Sigma(const SpdMatrix &V);

    // Matrix inverse of the residual variance matrix;
    const SpdMatrix &Siginv() const;
    void set_Siginv(const SpdMatrix &iV);

    // log determinant of Siginv().
    double ldsi() const;

    // Access to parameters.
    Ptr<MatrixParams> Beta_prm();
    const Ptr<MatrixParams> Beta_prm() const;
    Ptr<SpdParams> Sigma_prm();
    const Ptr<SpdParams> Sigma_prm() const;

    //--- estimation and probability calculations
    void mle() override;
    // The argument to loglike is a vector created by stacking the
    // columns of Beta, and the upper triangle of Sigma
    double loglike(const Vector &beta_sigma) const override;
    virtual double pdf(const Ptr<Data> &, bool) const;

    // Returns x * Beta();
    virtual Vector predict(const Vector &x) const;

    //---- simulate MV regression data ---
    virtual MvRegData *simdat(RNG &rng = GlobalRng::rng) const;
    virtual MvRegData *simdat(const Vector &X, RNG &rng = GlobalRng::rng) const;

    // no intercept
    Vector simulate_fake_x(RNG &rng = GlobalRng::rng) const;
  };
}  // namespace BOOM
#endif  // BOOM_MVREG_HPP
