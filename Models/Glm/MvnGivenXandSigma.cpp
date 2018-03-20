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
#include "Models/Glm/MvnGivenXandSigma.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {

  namespace {
    typedef MvnGivenXandSigma MGXS;
  }  // namespace

  MGXS::MvnGivenXandSigma(RegressionModel *mod,
                          const Ptr<VectorParams> &prior_mean,
                          const Ptr<UnivParams> &prior_sample_size,
                          const Vector &additional_prior_precision,
                          double diagonal_weight)
      : ParamPolicy(prior_mean, prior_sample_size),
        sigsq_(mod->Sigsq_prm()),
        compute_xtx_([mod]() { return mod->xtx(); }),
        data_sample_size_([mod]() { return mod->suf()->n(); }),
        additional_prior_precision_(additional_prior_precision),
        diagonal_weight_(diagonal_weight),
        ivar_(new SpdParams(prior_mean->dim())),
        current_(false) {
    if (!additional_prior_precision_.empty() &&
        additional_prior_precision_.size() != prior_mean->dim()) {
      report_error("Dimensions do not match.");
    }
    if (diagonal_weight_ < 0 || diagonal_weight_ > 1) {
      report_error("Illegal value for diagonal_weight.");
    }
    std::function<void(void)> observer = [this]() { this->observe_changes(); };
    mod->Sigsq_prm()->add_observer(observer);

    // Observe changes to data (e.g. when the regression is a
    // component in a mixture model).
    mod->add_observer(observer);
  }

  MGXS::MvnGivenXandSigma(const Ptr<VectorParams> &prior_mean,
                          const Ptr<UnivParams> &prior_sample_size,
                          const Ptr<UnivParams> &sigsq, const SpdMatrix &XTX,
                          double sample_size,
                          const Vector &additional_prior_precision,
                          double diagonal_weight)
      : ParamPolicy(prior_mean, prior_sample_size),
        sigsq_(sigsq),
        additional_prior_precision_(additional_prior_precision),
        diagonal_weight_(diagonal_weight),
        ivar_(new SpdParams(prior_mean->dim())),
        current_(false) {
    compute_xtx_ = [XTX]() { return XTX; };
    data_sample_size_ = [sample_size]() { return sample_size; };
    sigsq_->add_observer([this]() { this->observe_changes(); });
  }

  MGXS::MvnGivenXandSigma(const MGXS &rhs)
      : Model(rhs),
        VectorModel(rhs),
        MvnBase(rhs),
        ParamPolicy(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs),
        sigsq_(rhs.sigsq_),
        compute_xtx_(rhs.compute_xtx_),
        additional_prior_precision_(rhs.additional_prior_precision_),
        diagonal_weight_(rhs.diagonal_weight_),
        ivar_(rhs.ivar_->clone()),
        current_(rhs.current_) {}

  MGXS *MGXS::clone() const { return new MGXS(*this); }

  const Vector &MGXS::mu() const { return Mu_prm()->value(); }

  double MGXS::prior_sample_size() const { return Kappa_prm()->value(); }

  double MGXS::data_sample_size() const { return data_sample_size_(); }

  const SpdMatrix &MGXS::Sigma() const {
    set_ivar();
    return ivar_->var();
  }

  const SpdMatrix &MGXS::siginv() const {
    set_ivar();
    return ivar_->ivar();
  }

  double MGXS::ldsi() const {
    set_ivar();
    return ivar_->ldsi();
  }

  void MGXS::set_ivar() const {
    if (current_) return;
    SpdMatrix precision = compute_xtx_();
    double sample_size = data_sample_size_();
    if (sample_size > 0) {
      double w = diagonal_weight_;
      if (w >= 1.0)
        precision.set_diag(precision.diag());
      else if (w > 0.0) {
        precision *= (1 - w);
        precision.diag() /= (1 - w);
      }
      precision *= prior_sample_size() / sample_size;
    }
    if (!additional_prior_precision_.empty()) {
      precision.diag() += additional_prior_precision_;
    }
    precision /= sigsq_->value();
    ivar_->set_ivar(precision);
    current_ = true;
  }

  Ptr<VectorParams> MGXS::Mu_prm() { return ParamPolicy::prm1(); }
  const Ptr<VectorParams> MGXS::Mu_prm() const { return ParamPolicy::prm1(); }
  Ptr<UnivParams> MGXS::Kappa_prm() { return ParamPolicy::prm2(); }
  const Ptr<UnivParams> MGXS::Kappa_prm() const { return ParamPolicy::prm2(); }

  Vector MGXS::sim(RNG &rng) const {
    const Matrix &L(ivar_->var_chol());
    uint p = dim();
    Vector ans(p);
    for (uint i = 0; i < p; ++i) ans[i] = rnorm_mt(rng);
    ans = L * ans;
    ans += mu();
    return ans;
  }
}  // namespace BOOM
