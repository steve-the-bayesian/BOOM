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

#ifndef DIRICHLET_MODEL_H
#define DIRICHLET_MODEL_H

#include "Models/ModelTypes.hpp"
#include "Models/VectorModel.hpp"

#include "Models/ParamTypes.hpp"
#include "Models/Policies/ParamPolicy_1.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/Policies/SufstatDataPolicy.hpp"
#include "Models/Sufstat.hpp"

namespace BOOM {

  class DirichletSuf : public SufstatDetails<VectorData> {
    Vector sumlog_;  // sum_i(log pi_j);
    double n_;

   public:
    // constructor
    explicit DirichletSuf(uint S);
    DirichletSuf(const DirichletSuf &rhs);
    DirichletSuf *clone() const override;

    void clear() override;
    void Update(const VectorData &x) override;
    void add_mixture_data(const Vector &x, double prob);

    const Vector &sumlog() const;
    double n() const;
    DirichletSuf *abstract_combine(Sufstat *s) override;
    void combine(const DirichletSuf &);
    void combine(const Ptr<DirichletSuf> &);

    Vector vectorize(bool minimal = true) const override;
    Vector::const_iterator unvectorize(Vector::const_iterator &v,
                                       bool minimal = true) override;
    Vector::const_iterator unvectorize(const Vector &v,
                                       bool minimal = true) override;
    std::ostream &print(std::ostream &out) const override;
  };

  //======================================================================
  class DirichletModel : public ParamPolicy_1<VectorParams>,
                         public SufstatDataPolicy<VectorData, DirichletSuf>,
                         public PriorPolicy,
                         public DiffVectorModel,
                         public NumOptModel,
                         public MixtureComponent {
   public:
    explicit DirichletModel(uint S, double Nu = 1.0);
    explicit DirichletModel(const Vector &Nu);
    DirichletModel(const DirichletModel &rhs);
    DirichletModel *clone() const override;

    Ptr<VectorParams> Nu();
    const Ptr<VectorParams> Nu() const;

    uint dim() const;
    const Vector &nu() const;
    const double &nu(uint i) const;
    void set_nu(const Vector &);

    Vector pi() const;
    double pi(uint i) const;

    double pdf(const Ptr<Data> &dp, bool logscale) const;
    double pdf(const Data *, bool logscale) const override;
    double pdf(const Vector &pi, bool logscale) const;

    int number_of_observations() const override { return dat().size(); }

    // The first argument should have the first element of probs
    // omitted, so the sum of probs is <= 1.0.  The gradient and
    // Hessian are taken with respect to the free elements (i.e. not
    // the first one).
    double Logp(const Vector &probs, Vector &gradient, Matrix &Hessian,
                uint nderiv) const override;
    double Loglike(const Vector &nu, Vector &g, Matrix &h,
                   uint nd) const override;
    double log_likelihood() const override { return loglike(nu()); }
    void mle() override { return d2LoglikeModel::mle(); }

    double nu_loglike(const Vector &nu) const;

    Vector sim(RNG &rng = GlobalRng::rng) const override;
    virtual void add_mixture_data(const Ptr<Data> &, double prob);
  };

}  // namespace BOOM
#endif
