// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2010 Steven L. Scott

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
#include "Models/MvnGivenScalarSigma.hpp"
#include "distributions.hpp"

namespace BOOM {

  MvnGivenScalarSigmaBase::MvnGivenScalarSigmaBase(const Ptr<UnivParams> &sigsq)
      : sigsq_(sigsq) {}

  double MvnGivenScalarSigmaBase::sigsq() const { return sigsq_->value(); }

  typedef MvnGivenScalarSigma MGSS;

  MvnGivenScalarSigma::MvnGivenScalarSigma(const SpdMatrix &ominv,
                                           const Ptr<UnivParams> &sigsq)
      : MvnGivenScalarSigmaBase(sigsq),
        ParamPolicy(new VectorParams(nrow(ominv))),
        DataPolicy(new MvnSuf(nrow(ominv))),
        PriorPolicy(),
        omega_(ominv, true),
        wsp_(ominv) {}

  MvnGivenScalarSigma::MvnGivenScalarSigma(const Vector &mean,
                                           const SpdMatrix &ominv,
                                           const Ptr<UnivParams> &sigsq)
      : MvnGivenScalarSigmaBase(sigsq),
        ParamPolicy(new VectorParams(mean)),
        DataPolicy(new MvnSuf(mean.size())),
        PriorPolicy(),
        omega_(ominv, true),
        wsp_(mean.size()) {}

  MvnGivenScalarSigma::MvnGivenScalarSigma(const MGSS &rhs)
      : Model(rhs),
        VectorModel(rhs),
        MvnGivenScalarSigmaBase(rhs),
        LoglikeModel(rhs),
        ParamPolicy(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs),
        omega_(rhs.omega_),
        wsp_(rhs.wsp_) {}

  MGSS *MGSS::clone() const { return new MGSS(*this); }

  Ptr<VectorParams> MGSS::Mu_prm() { return ParamPolicy::prm(); }
  const Ptr<VectorParams> MGSS::Mu_prm() const { return ParamPolicy::prm(); }

  uint MGSS::dim() const { return nrow(wsp_); }
  const Vector &MGSS::mu() const { return Mu_prm()->value(); }

  const SpdMatrix &MGSS::Sigma() const {
    wsp_ = omega_.var() * sigsq();
    return wsp_;
  }

  const SpdMatrix &MGSS::siginv() const {
    wsp_ = omega_.ivar() / sigsq();
    return wsp_;
  }

  double MGSS::ldsi() const { return omega_.ldsi() - dim() * log(sigsq()); }

  const SpdMatrix &MGSS::Omega() const { return omega_.var(); }
  const SpdMatrix &MGSS::ominv() const { return omega_.ivar(); }
  double MGSS::ldoi() const { return omega_.ldsi(); }

  void MGSS::set_mu(const Vector &mu) { Mu_prm()->set(mu); }
  void MGSS::set_unscaled_precision(const SpdMatrix &omega_inverse) {
    omega_.set_ivar(omega_inverse);
  }

  void MGSS::mle() { set_mu(suf()->ybar()); }

  double MGSS::loglike(const Vector &mu_ominv) const {
    const ConstVectorView mu(mu_ominv, 0, dim());
    SpdMatrix siginv(dim());
    Vector::const_iterator b(mu_ominv.cbegin() + dim());
    siginv.unvectorize(b, true);
    siginv /= sigsq();
    return MvnBase::log_likelihood(Vector(mu), siginv, *suf());
  }

  double MGSS::pdf(const Ptr<Data> &dp, bool logscale) const {
    const Vector &y(DAT(dp)->value());
    return dmvn(y, mu(), siginv(), ldsi(), logscale);
  }

}  // namespace BOOM
