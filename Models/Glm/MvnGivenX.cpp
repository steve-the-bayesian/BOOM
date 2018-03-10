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

#include "Models/Glm/MvnGivenX.hpp"
#include "LinAlg/SubMatrix.hpp"
#include "Models/Glm/Glm.hpp"
#include "Models/Glm/LogisticRegressionModel.hpp"
#include "Models/Glm/RegressionModel.hpp"
#include "cpputil/nyi.hpp"
#include "distributions.hpp"

namespace BOOM {

  MvnGivenX::MvnGivenX(const Vector &Mu, double nobs, double diag)
      : ParamPolicy(new VectorParams(Mu), new UnivParams(nobs)),
        diagonal_weight_(diag),
        Lambda_(Mu.length(), 0),
        ivar_(new SpdParams(Mu.length(), 0.0)),
        xtwx_(Mu.length(), 0.0),
        sumw_(0) {}

  MvnGivenX::MvnGivenX(const Ptr<VectorParams> &Mu, const Ptr<UnivParams> &nobs,
                       double diag)
      : ParamPolicy(Mu, nobs),
        diagonal_weight_(diag),
        Lambda_(Mu->dim(), 0),
        ivar_(new SpdParams(Mu->dim(), 0.0)),
        xtwx_(Mu->dim(), 0.0),
        sumw_(0) {}

  MvnGivenX::MvnGivenX(const Ptr<VectorParams> &Mu, const Ptr<UnivParams> &nobs,
                       const Vector &Lambda, double diag)
      : ParamPolicy(Mu, nobs),
        diagonal_weight_(diag),
        Lambda_(Lambda),
        ivar_(new SpdParams(Mu->dim(), 0.0)),
        xtwx_(Mu->dim(), 0.0),
        sumw_(0) {
    assert(Lambda_.size() == Mu->dim());
  }

  MvnGivenX::MvnGivenX(const MvnGivenX &rhs)
      : Model(rhs),
        VectorModel(rhs),
        MvnBase(rhs),
        ParamPolicy(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs),
        diagonal_weight_(rhs.diagonal_weight_),
        Lambda_(rhs.Lambda_),
        ivar_(rhs.ivar_->clone()),
        xtwx_(rhs.xtwx_),
        sumw_(rhs.sumw_) {}

  MvnGivenX *MvnGivenX::clone() const { return new MvnGivenX(*this); }

  void MvnGivenX::add_x(const Vector &x, double w) {
    xtwx_.add_outer(x, w, false);
    sumw_ += w;
    current_ = false;
  }

  void MvnGivenX::clear_xtwx() {
    xtwx_ = 0;
    sumw_ = 0;
    current_ = false;
  }

  const SpdMatrix &MvnGivenX::xtwx() const { return xtwx_; }

  void MvnGivenX::initialize_params() {}

  const Ptr<VectorParams> MvnGivenX::Mu_prm() const { return prm1(); }
  const Ptr<UnivParams> MvnGivenX::Kappa_prm() const { return prm2(); }
  Ptr<VectorParams> MvnGivenX::Mu_prm() { return prm1(); }
  Ptr<UnivParams> MvnGivenX::Kappa_prm() { return prm2(); }
  double MvnGivenX::diagonal_weight() const { return diagonal_weight_; }
  uint MvnGivenX::dim() const { return mu().size(); }
  const Vector &MvnGivenX::mu() const { return Mu_prm()->value(); }
  double MvnGivenX::kappa() const { return Kappa_prm()->value(); }

  const SpdMatrix &MvnGivenX::Sigma() const {
    if (!current_) set_ivar();
    return ivar_->var();
  }

  const SpdMatrix &MvnGivenX::siginv() const {
    if (!current_) set_ivar();
    return ivar_->ivar();
  }

  double MvnGivenX::ldsi() const {
    if (!current_) set_ivar();
    return ivar_->ldsi();
  }

  void MvnGivenX::set_ivar() const {
    SpdMatrix ivar = xtwx_;
    if (sumw_ > 0.0) {
      ivar /= sumw_;
      double w = diagonal_weight_;

      if (w >= 1.0) {
        ivar.set_diag(ivar.diag());
      } else if (w > 0.0) {
        ivar *= (1 - w);
        ivar.diag() /= (1 - w);
      }
    } else
      ivar *= 0.0;

    ivar.diag() += Lambda_;

    ivar_->set_ivar(ivar);
    current_ = true;
  }

  Vector MvnGivenX::sim(RNG &rng) const { return rmvn_mt(rng, mu(), Sigma()); }

  //______________________________________________________________________

  MvnGivenXMultinomialLogit::MvnGivenXMultinomialLogit(
      const Vector &beta_prior_mean, double prior_sample_size,
      double diagonal_weight)
      : ParamPolicy(new VectorParams(beta_prior_mean),
                    new UnivParams(prior_sample_size)),
        diagonal_weight_(diagonal_weight) {}

  MvnGivenXMultinomialLogit::MvnGivenXMultinomialLogit(
      const Ptr<VectorParams> &beta_prior_mean,
      const Ptr<UnivParams> &prior_sample_size, double diagonal_weight)
      : ParamPolicy(beta_prior_mean, prior_sample_size),
        diagonal_weight_(diagonal_weight) {}

  MvnGivenXMultinomialLogit::MvnGivenXMultinomialLogit(
      const MvnGivenXMultinomialLogit &rhs)
      : Model(rhs),
        MvnBase(rhs),
        ParamPolicy(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs),
        diagonal_weight_(rhs.diagonal_weight_) {}

  MvnGivenXMultinomialLogit *MvnGivenXMultinomialLogit::clone() const {
    return new MvnGivenXMultinomialLogit(*this);
  }

  void MvnGivenXMultinomialLogit::set_x(
      const Matrix &subject_characeristics,
      const std::vector<Mat> &choice_characteristics, int number_of_choices) {
    bool have_choice_predictors = !choice_characteristics.empty();

    if (have_choice_predictors &&
        choice_characteristics.size() != nrow(subject_characeristics)) {
      report_error(
          "the sizes of subject_characeristics and "
          "choice_characteristics must match");
    }
    current_ = false;
    scaled_subject_xtx_.resize(ncol(subject_characeristics));
    scaled_subject_xtx_ = 0;
    int number_of_observations = nrow(subject_characeristics);
    int number_of_subject_predictors = ncol(subject_characeristics);
    scaled_subject_xtx_.add_inner(subject_characeristics,
                                  1.0 / number_of_observations);

    int number_of_choice_predictors = 0;
    if (have_choice_predictors) {
      int number_of_choice_observations = choice_characteristics.size();
      number_of_choice_predictors = ncol(choice_characteristics[0]);
      if (nrow(choice_characteristics[0]) != number_of_choices) {
        ostringstream err;
        err << "The number_of_choices argument to set_x must match the "
            << "number of rows in the first element of choice_characteristics."
            << endl;
        report_error(err.str());
      }

      double weight = 1.0 / (number_of_choices * number_of_observations);

      scaled_choice_xtx_.resize(number_of_choice_predictors);
      scaled_choice_xtx_ = 0;
      for (int i = 0; i < number_of_choice_observations; ++i) {
        ConstVectorView baseline_predictors(choice_characteristics[i].row(0));
        for (int m = 1; m < number_of_choices; ++m) {
          const ConstVectorView choice_predictors(
              choice_characteristics[i].row(m));
          scaled_choice_xtx_.add_outer(choice_predictors - baseline_predictors);
        }
      }
      scaled_choice_xtx_ *= weight;
    }

    // Build overall_xtx_ as a sequence of blocks of scaled_subject_xtx_ with
    // single scaled_choice_xtx_ block at the end.
    int overall_xtx_dimension =
        (number_of_choices - 1) * number_of_subject_predictors +
        number_of_choice_predictors;
    overall_xtx_.resize(overall_xtx_dimension);
    overall_xtx_ = 0;

    int lo = 0;
    for (int m = 1; m < number_of_choices; ++m) {
      int hi = lo + number_of_subject_predictors - 1;
      SubMatrix block(overall_xtx_, lo, hi, lo, hi);
      block = scaled_subject_xtx_;
      lo = hi + 1;
    }

    if (have_choice_predictors) {
      int hi = lo + number_of_choice_predictors - 1;
      SubMatrix block(overall_xtx_, lo, hi, lo, hi);
      block = scaled_choice_xtx_;
    }

    if (diagonal_weight_ > 0) {
      Vector d(overall_xtx_.diag());
      overall_xtx_ *= (1 - diagonal_weight_);
      overall_xtx_.set_diag(d, false);
    }
  }

  Ptr<VectorParams> MvnGivenXMultinomialLogit::Mu_prm() {
    return ParamPolicy::prm1();
  }
  const Ptr<VectorParams> MvnGivenXMultinomialLogit::Mu_prm() const {
    return ParamPolicy::prm1();
  }
  const Vector &MvnGivenXMultinomialLogit::mu() const {
    return prm1_ref().value();
  }
  void MvnGivenXMultinomialLogit::set_mu(const Vector &mu) {
    Mu_prm()->set(mu);
  }

  Ptr<UnivParams> MvnGivenXMultinomialLogit::Kappa_prm() {
    return ParamPolicy::prm2();
  }
  const Ptr<UnivParams> MvnGivenXMultinomialLogit::Kappa_prm() const {
    return ParamPolicy::prm2();
  }
  double MvnGivenXMultinomialLogit::kappa() const { return prm2_ref().value(); }
  void MvnGivenXMultinomialLogit::set_kappa(double kappa) {
    Kappa_prm()->set(kappa);
    current_ = false;
  }

  const SpdMatrix &MvnGivenXMultinomialLogit::Sigma() const {
    make_current();
    return Sigma_storage_->var();
  }

  const SpdMatrix &MvnGivenXMultinomialLogit::siginv() const {
    make_current();
    return Sigma_storage_->ivar();
  }

  double MvnGivenXMultinomialLogit::ldsi() const {
    report_error("MvnGivenXMultinomialLogit::ldsi not yet implemented]\n");
    return 0;
  }

  void MvnGivenXMultinomialLogit::make_current() const {
    if (!Sigma_storage_) {
      Sigma_storage_.reset(new SpdData(nrow(overall_xtx_)));
    }
    if (!current_) {
      Sigma_storage_->set_ivar(overall_xtx_ * kappa());
      current_ = true;
    }
  }

}  // namespace BOOM
