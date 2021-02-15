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

#ifndef WISHART_MODEL_H
#define WISHART_MODEL_H

#include "Models/ModelTypes.hpp"
#include "Models/Policies/ParamPolicy_2.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/Policies/SufstatDataPolicy.hpp"
#include "Models/SpdModel.hpp"
#include "Models/SpdParams.hpp"
#include "Models/Sufstat.hpp"

namespace BOOM {
  class WishartSuf : public SufstatDetails<SpdData> {
   public:
    explicit WishartSuf(uint dim);
    WishartSuf(const WishartSuf &sf);
    WishartSuf *clone() const override;

    void clear() override;
    void Update(const SpdData &d) override;
    double n() const { return n_; }
    double sumldw() const { return sumldw_; }
    const SpdMatrix &sumW() const { return sumW_; }
    void combine(const Ptr<WishartSuf> &);
    void combine(const WishartSuf &);
    WishartSuf *abstract_combine(Sufstat *s) override;
    Vector vectorize(bool minimal = true) const override;
    Vector::const_iterator unvectorize(Vector::const_iterator &v,
                                       bool minimal = true) override;
    Vector::const_iterator unvectorize(const Vector &v,
                                       bool minimal = true) override;
    std::ostream &print(std::ostream &out) const override;

   private:
    double n_;
    double sumldw_;
    SpdMatrix sumW_;
  };

  //======================================================================
  // The density for the Wishart model with prior sum of squares S,
  // dimension d, and prior degrees of freedom nu is
  //
  // p(Siginv) =  K * |Siginv|^{(nu - d - 1) / 2}
  //                * exp( -(1/2) * tr(Siginv S^{-1}))
  // K^{-1} = 2^{nu * d / 2} * mgamma(nu / 2, d) * |S|^{nu/2}
  //
  // Where mgamma(x, d) = pi^{d * (d-1) / 4}
  //                      * \prod_{i = 1}^d Gamma(x + (1 - i) / 2)
  // is the multivariate Gamma distribution.
  //
  // The distribution is defined for nu >= d.
  //
  // The most frequent use of the Wishart model is as a conjugate
  // prior for the precision matrix of a multivariate normal
  // distribution.  A 'precision' (aka 'information') matrix is the
  // matrix inverse of a variance matrix.
  class WishartModel : public ParamPolicy_2<UnivParams, SpdParams>,
                       public SufstatDataPolicy<SpdData, WishartSuf>,
                       public PriorPolicy,
                       public dLoglikeModel,
                       public SpdModel {
   public:
    // A Wishart model with a constant diagonal sum of squares parameter.
    //
    // Args:
    //   dim: The dimension (number of rows) of the symmetric matrix
    //     this distribution models.
    //   prior_df: This is 'nu' in the class comment above.  If
    //     prior_df < 0 then a default of dim + 1 will be used.
    //   diagonal_variance: An estimate of the variance to use for
    //     each parameter.  This will be converted to a sum of squares
    //     by multiplying it by prior_df.
    explicit WishartModel(uint dim, double prior_df = -1.0,
                          double diagonal_variance = 1.0);

    // Args:
    //    prior_df: This is 'nu' in the class comment above.  Note
    //      the rquirement that prior_df > dim - 1.
    //    prior_variance_estimate: An estimate of the variance.  This
    //      will be converted to a "sum of squares" by multiplying it
    //      by prior_df.
    WishartModel(double prior_df, const SpdMatrix &prior_variance_estimate);

    WishartModel(const WishartModel &m);
    WishartModel *clone() const override;

    // Set model parameters to default values.  The mean is set using
    // the method of moments.  nu is set to a default.
    void initialize_params() override;

    Ptr<UnivParams> Nu_prm();
    Ptr<SpdParams> Sumsq_prm();
    const Ptr<UnivParams> Nu_prm() const;
    const Ptr<SpdParams> Sumsq_prm() const;

    const double &nu() const;
    const SpdMatrix &sumsq() const;
    void set_nu(double);
    void set_sumsq(const SpdMatrix &);

    SpdMatrix sim(RNG &rng = GlobalRng::rng);
    int dim() const { return sumsq().nrow(); }

    // Experimental code for finding the MLE of the Wishart density.
    // mle0 finds the mode using no derivatives via the Nelder Mead
    // method.  mle1 uses first derivatives via bfgs.
    void mle_no_derivatives();
    void mle_first_derivatives();

    double logp(const SpdMatrix &W) const override;

    int number_of_observations() const override { return dat().size(); }

    // Evaluate the log likelihood of Wishart data.  The model
    // parameters are Sumsq and nu, passed as a vector with the upper
    // trianglular elements of Sumsq coming first (column-wise, as
    // would be produced by Sumsq.vectorize()), and then nu at the
    // end.
    double loglike(const Vector &sumsq_triangle_nu) const override;
    double dloglike(const Vector &sumsq_triangle_nu, Vector &g) const override;
    double Loglike(const Vector &sumsq_triangle_nu, Vector &g, uint nd) const;
  };

}  // namespace BOOM
#endif  // WISHART_MODEL_H
