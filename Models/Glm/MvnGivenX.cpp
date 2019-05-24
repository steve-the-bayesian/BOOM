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

  MvnGivenXBase::MvnGivenXBase(const Ptr<VectorParams> &mean,
                               const Ptr<UnivParams> &prior_sample_size,
                               const Vector &diagonal,
                               double diagonal_weight)
      : ParamPolicy(mean, prior_sample_size),
        diagonal_weight_(diagonal_weight),
        diagonal_(diagonal),
        precision_(new SpdData(mean->dim())),
        current_(false)
  {}

  const SpdMatrix &MvnGivenXBase::Sigma() const {
    set_precision_matrix();
    return precision_->var();
  }

  const SpdMatrix &MvnGivenXBase::siginv() const {
    set_precision_matrix();
    return precision_->ivar();
  }

  double MvnGivenXBase::ldsi() const {
    set_precision_matrix();
    return precision_->ldsi();
  }

  Vector MvnGivenXBase::sim(RNG &rng) const {
    return rmvn_mt(rng, mu(), Sigma());
  }

  //===========================================================================
  MvnGivenX::MvnGivenX(const Ptr<VectorParams> &mean,
                       const Ptr<UnivParams> &prior_sample_size,
                       const Vector &diagonal,
                       double diagonal_weight)
      : MvnGivenXBase(mean, prior_sample_size, diagonal, diagonal_weight),
        xtwx_(mean->dim(), 0.0),
        sumw_(0.0)
  {}

  void MvnGivenX::add_x(const Vector &x, double w) {
    xtwx_.add_outer(x, w, false);
    sumw_ += w;
    mark_not_current();
  }

  void MvnGivenX::clear_xtwx() {
    xtwx_ = 0;
    sumw_ = 0;
    mark_not_current();
  }

  void MvnGivenX::set_precision_matrix() const {
    if (current()) return;
    SpdMatrix ivar = xtwx_;
    ivar.reflect();
    if (sumw_ > 0) {
      ivar /= sumw_;
    } else {
      ivar *= 0.0;
    }
    store_precision_matrix(std::move(ivar));
  }

  // Args:
  //   ivar: the precision matrix before averaging with its diagonal or
  //     multiplying by prior information weight.  In the canonical example ivar
  //     == X'X / n.
  void MvnGivenXBase::store_precision_matrix(SpdMatrix &&ivar) const {
    // Average ivar with the diagonal.
    if (diagonal_weight_ >= 1.0) {
      if (diagonal().empty()) {
        ivar.set_diag(ivar.diag(), true);
      } else {
        ivar.set_diag(diagonal(), true);
      }
    } else if (diagonal_weight_ > 0) {
      if (diagonal().empty()) {
        // ominv = a * D(X'X) + (1 - a) * X'X
        // The diagonal is unchanged.  The off-diagonal elements are multiplied by 1-a.
        ivar *= (1 - diagonal_weight_);
        ivar.diag() /= (1 - diagonal_weight_);
      } else {
        ivar *= (1 - diagonal_weight_);
        ivar.diag().axpy(diagonal(), diagonal_weight_);
      }
    } else {
      // diagonal_weight_ is zero.  Leave ivar alone.
    }
    
    precision_->set_ivar(ivar * kappa());
    current_ = true;
  }                     

  
  //===========================================================================
  MvnGivenXRegSuf::MvnGivenXRegSuf(
      const Ptr<VectorParams> &mean,
      const Ptr<UnivParams> &prior_sample_size,
      const Vector &precision_diagonal,
      double diagonal_weight,
      const Ptr<RegSuf> &suf)
      : MvnGivenXBase(mean, prior_sample_size, precision_diagonal,
                      diagonal_weight),
        suf_(suf)
  {}

  MvnGivenXRegSuf::MvnGivenXRegSuf(const MvnGivenXRegSuf &rhs)
      : Model(rhs),
        MvnGivenXBase(rhs),
        suf_(rhs.suf_->clone())
  {}

  void MvnGivenXRegSuf::set_precision_matrix() const {
    if (current()) return;
    if (!suf_) {
      report_error("Sufficient statistics must be set.");
    }
    SpdMatrix xtx = suf_->xtx();
    double n = suf_->n();
    if (n <= 0) {
      xtx *= 0;
      n = 1;
    }
    store_precision_matrix(xtx / n);
  }
 
  //===========================================================================
  MvnGivenXMvRegSuf::MvnGivenXMvRegSuf(
      const Ptr<VectorParams> &mean,
      const Ptr<UnivParams> &prior_sample_size,
      const Vector &precision_diagonal,
      double diagonal_weight,
      const Ptr<MvRegSuf> &suf)
      : MvnGivenXBase(mean, prior_sample_size, precision_diagonal, diagonal_weight),
        suf_(suf)
  {}

  MvnGivenXMvRegSuf::MvnGivenXMvRegSuf(const MvnGivenXMvRegSuf &rhs)
      : Model(rhs),
        MvnGivenXBase(rhs),
        suf_(rhs.suf_->clone())
  {}

  void MvnGivenXMvRegSuf::set_precision_matrix() const {
    if (current()) return;
    if (!suf_) {
      report_error("Sufficient statistics must be set.");
    }
    SpdMatrix xtx = suf_->xtx();
    double n = suf_->n();
    if (n <= 0) {
      xtx *= 0;
    } else {
      xtx /= n;
    }
    store_precision_matrix(std::move(xtx));
  }
  
  //===========================================================================
  MvnGivenXWeightedRegSuf::MvnGivenXWeightedRegSuf(
      const Ptr<VectorParams> &mean,
      const Ptr<UnivParams> &prior_sample_size,
      const Vector &precision_diagonal,
      double diagonal_weight,
      const Ptr<WeightedRegSuf> &suf)
      : MvnGivenXBase(mean, prior_sample_size, precision_diagonal, diagonal_weight),
        suf_(suf)
  {}

  MvnGivenXWeightedRegSuf::MvnGivenXWeightedRegSuf(
      const MvnGivenXWeightedRegSuf &rhs)
      : Model(rhs),
        MvnGivenXBase(rhs),
        suf_(rhs.suf_->clone())
  {}
  
  void MvnGivenXWeightedRegSuf::set_precision_matrix() const {
    if (current()) return;
    if (!suf_) {
      report_error("Sufficient statistics must be set.");
    }
    SpdMatrix xtwx = suf_->xtx();
    double n = suf_->sumw();
    if (n <= 0) {
      xtwx *= 0;
      n = 1;
    }
    store_precision_matrix(xtwx / n);
  }

  //===========================================================================

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
      const std::vector<Matrix> &choice_characteristics,
      int number_of_choices) {
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
