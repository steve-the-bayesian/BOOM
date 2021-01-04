// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2007 Steven L. Scott

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

#ifndef BOOM_MVN_MODEL_BASE_HPP
#define BOOM_MVN_MODEL_BASE_HPP
#include "LinAlg/Selector.hpp"
#include "Models/DataTypes.hpp"
#include "Models/ModelTypes.hpp"
#include "Models/Policies/ParamPolicy_2.hpp"
#include "Models/SpdParams.hpp"
#include "Models/Sufstat.hpp"
#include "Models/VectorModel.hpp"

namespace BOOM {

  class MvnSuf : public SufstatDetails<VectorData> {
   public:
    // If created using the default constructor, the MvnSuf will be
    // resized to the dimension of the first data point passed to it
    // in update().
    explicit MvnSuf(uint p = 0);
    MvnSuf(double n, const Vector &ybar, const SpdMatrix &sumsq);
    MvnSuf(const MvnSuf &sf);
    MvnSuf *clone() const override;

    void clear() override;
    void resize(uint p);  // clears existing data
    void Update(const VectorData &x) override;
    void update_raw(const Vector &x);
    void add_mixture_data(const Vector &x, double prob);
    void update_expected_value(double sample_size, const Vector &expected_sum,
                               const SpdMatrix &expected_sum_of_squares);

    // Remove the vector x from the set of sufficient statistics,
    // assuming that x was previously added.
    void remove_data(const Vector &x);

    Vector sum() const;
    SpdMatrix sumsq() const;  // Un-centered sum of squares
    double n() const;
    const Vector &ybar() const;
    SpdMatrix sample_var() const;  // divides by n-1
    SpdMatrix var_hat() const;     // divides by n
    SpdMatrix center_sumsq(const Vector &mu) const;
    const SpdMatrix &center_sumsq() const;

    void combine(const Ptr<MvnSuf> &);
    void combine(const MvnSuf &);
    MvnSuf *abstract_combine(Sufstat *s) override;

    Vector vectorize(bool minimal = true) const override;
    Vector::const_iterator unvectorize(Vector::const_iterator &v,
                                       bool minimal = true) override;
    Vector::const_iterator unvectorize(const Vector &v,
                                       bool minimal = true) override;

    std::ostream &print(std::ostream &) const override;

   private:
    Vector ybar_;
    Vector wsp_;
    mutable SpdMatrix sumsq_;  // centered at ybar
    double n_;                 // sample size
    mutable bool sym_;
    void check_symmetry() const;

    // resizes if empty, otherwise throws if dimension is wrong.
    void check_dimension(const Vector &y);
  };

  inline std::ostream &operator<<(std::ostream &out, const MvnSuf &s) {
    return s.print(out);
  }
  //------------------------------------------------------------

  class MvnBase : public DiffVectorModel {
   public:
    MvnBase *clone() const override = 0;
    virtual uint dim() const;
    double Logp(const Vector &x_subset, Vector &gradient, Matrix &Hessian,
                uint nderiv) const override;
    // Args:
    //   x_subset: A subset (determined by 'inclusion') of the vector of random
    //     variables measured by this model.
    //   gradient: If non-NULL then *gradient will be filled with the gradient
    //     of this function with respect to the dimensions of x determined by
    //     'inclusion.'  In this case the gradient should have dimension equal
    //     to the number of included variables.  A NULL 'gradient' signals that
    //     the gradient should not be computed.
    //   Hessian: If gradient and Hessian are non-NULL then Hessian be filled
    //     with the matrix of second derivatives with respect to the dimensions
    //     of x determined by 'inclusion.'  In this case the Hessian should have
    //     rows and columns equal to the number of included varaibles.
    //     Otherwise 'Hessian' is not used.
    //   inclusion: A Selector identifying which positions are 'included'.
    //   reset_derivatives: If true then gradient and Hessian are resized and
    //     set to zero before the derivatives are computed.  If false then the
    //     derivatives are added to whatever gradient and Hessian contain when
    //     they are passed in.
    //
    // Returns:
    //   The log of the normal density with mean mu[inclusion] and precision
    //   siginv[inclusion] evaluated at x_subset.
    virtual double logp_given_inclusion(const Vector &x_subset,
                                        Vector *gradient, Matrix *Hessian,
                                        const Selector &inclusion,
                                        bool reset_derivatives) const;

    // Returns the multivariate normal log likelihood.  Assumes all
    // variables are included.
    double log_likelihood(const Vector &mu, const SpdMatrix &siginv,
                          const MvnSuf &suf) const;

    virtual const Vector &mu() const = 0;
    virtual const SpdMatrix &Sigma() const = 0;
    virtual const SpdMatrix &siginv() const = 0;
    virtual double ldsi() const = 0;
    Vector sim(RNG &rng = GlobalRng::rng) const override;
  };

  //____________________________________________________________
  class MvnBaseWithParams : public MvnBase,
                            public ParamPolicy_2<VectorParams, SpdParams>,
                            public LocationScaleVectorModel {
   public:
    explicit MvnBaseWithParams(uint p, double mu = 0.0, double sig = 1.0);
    // N(mu,V)... if(ivar) then V is the inverse variance.
    MvnBaseWithParams(const Vector &mean, const SpdMatrix &V,
                      bool ivar = false);
    MvnBaseWithParams(const Ptr<VectorParams> &mu, const Ptr<SpdParams> &Sigma);
    MvnBaseWithParams(const MvnBaseWithParams &);
    MvnBaseWithParams *clone() const override = 0;

    Ptr<VectorParams> Mu_prm();
    const Ptr<VectorParams> Mu_prm() const;
    Ptr<SpdParams> Sigma_prm();
    const Ptr<SpdParams> Sigma_prm() const;

    const Vector &mu() const override;
    const SpdMatrix &Sigma() const override;
    const SpdMatrix &siginv() const override;
    double ldsi() const override;
    Matrix Sigma_chol() const;

    void set_mu(const Vector &);
    void set_Sigma(const SpdMatrix &);
    void set_siginv(const SpdMatrix &);
    void set_S_Rchol(const Vector &sd, const Matrix &L);
  };

}  // namespace BOOM

#endif  // BOOM_MVN_MODEL_BASE_HPP
