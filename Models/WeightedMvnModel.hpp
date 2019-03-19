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
#ifndef BOOM_WEIGHTED_MVN_MODEL_HPP
#define BOOM_WEIGHTED_MVN_MODEL_HPP

#include "Models/ModelTypes.hpp"
#include "Models/Policies/ParamPolicy_2.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/Policies/SufstatDataPolicy.hpp"
#include "Models/SpdParams.hpp"
#include "Models/Sufstat.hpp"
#include "Models/WeightedData.hpp"

namespace BOOM {

  class WeightedMvnSuf : public SufstatDetails<WeightedVectorData> {
   public:
    explicit WeightedMvnSuf(uint p);
    WeightedMvnSuf(const WeightedMvnSuf &rhs);
    WeightedMvnSuf *clone() const override;

    void clear() override;
    void Update(const WeightedVectorData &x) override;

    const Vector &sum() const;
    const SpdMatrix &sumsq() const;
    double n() const;
    double sumw() const;
    double sumlogw() const;

    Vector ybar() const;
    SpdMatrix var_hat() const;
    SpdMatrix center_sumsq(const Vector &mu) const;
    SpdMatrix center_sumsq() const;
    void combine(const Ptr<WeightedMvnSuf> &);
    void combine(const WeightedMvnSuf &);
    WeightedMvnSuf *abstract_combine(Sufstat *s) override;

    Vector vectorize(bool minimal = true) const override;
    Vector::const_iterator unvectorize(Vector::const_iterator &v,
                                       bool minimal = true) override;
    Vector::const_iterator unvectorize(const Vector &v,
                                       bool minimal = true) override;
    std::ostream &print(std::ostream &out) const override;

   private:
    Vector sum_;
    SpdMatrix sumsq_;
    double n_;
    double sumw_;
    double sumlogw_;
  };

  class WeightedMvnModel
      : public ParamPolicy_2<VectorParams, SpdParams>,
        public SufstatDataPolicy<WeightedVectorData, WeightedMvnSuf>,
        public PriorPolicy,
        public LoglikeModel {
   public:
    explicit WeightedMvnModel(uint p, double mu = 0.0, double sig = 1.0);
    WeightedMvnModel(const Vector &mean, const SpdMatrix &Var);  // N(mu, Var)
    WeightedMvnModel(const WeightedMvnModel &m);
    WeightedMvnModel *clone() const override;

    Ptr<VectorParams> Mu_prm();
    const Ptr<VectorParams> Mu_prm() const;
    Ptr<SpdParams> Sigma_prm();
    const Ptr<SpdParams> Sigma_prm() const;

    int dim() const { return mu().size(); }
    const Vector &mu() const;
    const SpdMatrix &Sigma() const;
    const SpdMatrix &siginv() const;
    double ldsi() const;

    void set_mu(const Vector &);
    void set_Sigma(const SpdMatrix &);
    void set_siginv(const SpdMatrix &);
    void mle() override;
    double loglike(const Vector &mu_siginv_triangle) const override;

    double pdf(const Ptr<Data> &dp, bool logscale) const;
    double pdf(const Ptr<DataType> &dp, bool logscale) const;
  };
}  // namespace BOOM
#endif  // BOOM_WEIGHTED_MVN_MODEL_HPP
