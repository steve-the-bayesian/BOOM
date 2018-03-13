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

#include "Models/Glm/PosteriorSamplers/DAFE_MLM.hpp"
#include <cmath>
#include "Models/Glm/MultinomialLogitModel.hpp"
#include "Models/MvnModel.hpp"
#include "Models/MvtModel.hpp"
#include "cpputil/lse.hpp"
#include "cpputil/math_utils.hpp"  // for lse
#include "distributions.hpp"       // for rlexp

#include "LinAlg/VectorView.hpp"
#include "TargetFun/LogPost.hpp"
#include "TargetFun/TargetFun.hpp"

namespace BOOM {

  typedef MultinomialLogitModel MLM;
  typedef MetropolisHastings MH;

  //------------------------------------------------------------
  DafeMlmBase::DafeMlmBase(MultinomialLogitModel *mod,
                           const Ptr<MvnModel> &SubjectPri,
                           const Ptr<MvnModel> &ChoicePri, bool draw_b0,
                           RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        mlm_(mod),
        subject_pri_(SubjectPri),
        choice_pri_(ChoicePri),
        mlo_(draw_b0 ? 0 : 1) {
    compute_xtx();
  }
  //------------------------------------------------------------
  double DafeMlmBase::logpri() const {
    uint M = mlm_->Nchoices();
    double ans(0);
    for (uint m = 1; m < M; ++m)
      ans += subject_pri_->logp(mlm_->beta_subject(m));
    uint pch = mlm_->choice_nvars();
    if (pch > 0) ans += choice_pri_->logp(mlm_->beta_choice());
    return ans;
  }
  //------------------------------------------------------------
  // to be called by the constructor
  void DafeMlmBase::compute_xtx() {
    std::vector<Ptr<ChoiceData> > &d(mlm_->dat());
    uint psub = d[0]->subject_nvars();
    uint pch = d[0]->choice_nvars();

    xtx_subject_.resize(psub);
    xtx_subject_ = 0;

    xtx_choice_.resize(pch);
    if (pch > 0) xtx_choice_ = 0;

    for (uint i = 0; i < d.size(); ++i) {
      Ptr<ChoiceData> dp = d[i];
      const Vector &xsub(dp->Xsubject());
      xtx_subject_.add_outer(xsub);
      if (pch > 0) {
        for (uint m = 0; m < mlm_->Nchoices(); ++m) {
          const Vector &xch(dp->Xchoice(m));
          xtx_choice_.add_outer(xch);
        }
      }
    }
  }

  const SpdMatrix &DafeMlmBase::xtx_subject() const { return xtx_subject_; }
  const SpdMatrix &DafeMlmBase::xtx_choice() const { return xtx_choice_; }
  Ptr<MvnModel> DafeMlmBase::subject_pri() const { return subject_pri_; }
  Ptr<MvnModel> DafeMlmBase::choice_pri() const { return choice_pri_; }

  // ======================================================================
  // Target function for use with Metropolis Hastings samplers
  class LesSubjectTarget : public TargetFun {
   public:
    LesSubjectTarget(uint Which, Matrix &bigU, MLM *mod)
        : which(Which), U(bigU), mlm_(mod) {}
    LesSubjectTarget *clone() const { return new LesSubjectTarget(*this); }
    double operator()(const Vector &b) const;

   private:
    uint which;
    Matrix &U;
    MLM *mlm_;
  };
  double LesSubjectTarget::operator()(const Vector &b) const {
    const VectorView Uvec(U.col(which));
    uint n = Uvec.size();
    const std::vector<Ptr<ChoiceData> > &dat(mlm_->dat());
    double ans = 0;
    for (uint i = 0; i < n; ++i) {
      double u = Uvec[i];
      Ptr<ChoiceData> d(dat[i]);
      double eta = b.affdot(d->Xsubject());
      eta += mlm_->predict_choice(*d, which);
      ans += dexv(u, eta, 1, true);
    }
    return ans;
  }
  // ======================================================================
  class LesChoiceTarget : public TargetFun {
   public:
    LesChoiceTarget(Matrix &bigU, MLM *mod) : U(bigU), mlm_(mod) {}
    LesChoiceTarget *clone() const { return new LesChoiceTarget(*this); }
    double operator()(const Vector &b) const;

   private:
    Matrix &U;
    MLM *mlm_;
  };
  double LesChoiceTarget::operator()(const Vector &b) const {
    const std::vector<Ptr<ChoiceData> > &dat(mlm_->dat());
    double n = dat.size();
    uint M = mlm_->Nchoices();
    double ans = 0;
    for (uint i = 0; i < n; ++i) {
      Ptr<ChoiceData> d(dat[i]);
      for (uint m = 0; m < M; ++m) {
        double eta = mlm_->predict_subject(*d, m);
        eta += b.affdot(d->Xchoice(m));
        double u = U(i, m);
        ans += dexv(u, eta, 1, true);
      }
    }
    return ans;
  }
  // ======================================================================
  DafeMlm::DafeMlm(MultinomialLogitModel *mod, const Ptr<MvnModel> &SubjectPri,
                   const Ptr<MvnModel> &ChoicePri, double Tdf, bool draw_b0)
      : DafeMlmBase(mod, SubjectPri, ChoicePri, draw_b0),
        mlm_(mod),
        mu(-0.577215664902),
        sigsq(1.64493406685),
        U(mod->dat().size(), mod->Nchoices()) {
    uint M = mod->Nchoices();
    uint psub = mod->subject_nvars();
    const SpdMatrix &Ominv(subject_pri()->siginv());
    Ominv_mu_subject = Ominv * subject_pri()->mu();
    for (uint m = 0; m < M; ++m) {
      // just need to get the dimensions right, for now
      NEW(MvtIndepProposal, prop)(Ominv_mu_subject, Ominv, Tdf);
      subject_proposals_.push_back(prop);

      LesSubjectTarget target(m, U, mlm_);
      NEW(MH, sam)(target, prop);
      subject_samplers_.push_back(sam);

      Vector tmp(psub);
      xtu_subject.push_back(tmp);
    }

    uint pch = mod->choice_nvars();
    if (pch > 0) {
      LesChoiceTarget target(U, mlm_);
      choice_proposal_ =
          new MvtIndepProposal(choice_pri()->mu(), choice_pri()->siginv(), Tdf);
      choice_sampler_ = new MH(target, choice_proposal_);
      Ominv_mu_choice = choice_pri()->siginv() * choice_pri()->mu();
      xtu_choice = Vector(pch);
    }
  }

  // ======================================================================
  void DafeMlm::draw() {
    impute_latent_data();
    uint M = subject_samplers_.size();
    for (uint m = mlo(); m < M; ++m) draw_subject(m);
    if (mlm_->choice_nvars() > 0) draw_choice();
  }

  // ======================================================================
  void DafeMlm::impute_latent_data() {
    std::vector<Ptr<ChoiceData> > &dat(mlm_->dat());
    uint n = dat.size();
    uint M = dat[0]->nchoices();

    U.resize(n, M);
    Vector eta(M);
    Vector u(M);
    Vector logz2(2);
    for (uint m = 0; m < M; ++m) xtu_subject[m] = 0;
    uint pch = mlm_->choice_nvars();
    if (pch > 0) xtu_choice = 0;

    for (uint i = 0; i < n; ++i) {
      Ptr<ChoiceData> dp = dat[i];
      mlm_->fill_eta(*dp, eta);
      uint y = dp->value();
      double loglam = lse(eta);
      double logzmin = rlexp(loglam);
      logz2[0] = logzmin;
      u[y] = mu - logzmin;
      const Vector &xsub(dp->Xsubject());
      for (uint m = 0; m < M; ++m) {
        if (m != y) {
          logz2[1] = rlexp(eta[m]);
          double logz = lse(logz2);
          u[m] = mu - logz;
        }
        xtu_subject[m].axpy(xsub, u[m]);
        if (pch > 0) {
          const Vector &xch(dp->Xchoice(m));
          xtu_choice.axpy(xch, u[m]);
        }
      }  // m
      U.row(i) = u;
    }  // i
  }    // impute_latent_data

  // ======================================================================

  inline void Breg(Vector &b, SpdMatrix &ivar, double sigsq, const Vector &xty,
                   const SpdMatrix &xtx, const Vector &Ominv_b,
                   const SpdMatrix &Ominv) {
    ivar = xtx / sigsq + Ominv;
    b = xty / sigsq + Ominv_b;
    b = ivar.solve(b);
  }
  // ======================================================================
  void DafeMlm::draw_subject(uint i) {
    Vector b;
    SpdMatrix Ivar;
    const SpdMatrix &Ominv(subject_pri()->siginv());

    Breg(b, Ivar, sigsq, xtu_subject[i], xtx_subject(), Ominv_mu_subject,
         Ominv);

    Ptr<MvtIndepProposal> prop = subject_proposals_[i];
    prop->set_ivar(Ivar);
    prop->set_mu(b);

    b = mlm_->beta_subject(i);
    b = subject_samplers_[i]->draw(b);
    mlm_->set_beta_subject(b, i);
  }
  // ======================================================================
  void DafeMlm::draw_choice() {
    Vector b;
    SpdMatrix Ivar;
    const SpdMatrix &Ominv(choice_pri()->siginv());
    Breg(b, Ivar, sigsq, xtu_choice, xtx_choice(), Ominv_mu_choice, Ominv);
    choice_proposal_->set_mu(b);
    choice_proposal_->set_ivar(Ivar);

    b = choice_sampler_->draw(mlm_->beta_choice());
    if (b != mlm_->beta_choice()) {
      mlm_->set_beta_choice(b);
    }
  }

  //______________________________________________________________________

  class DafeLoglike {
   public:
    DafeLoglike(MLM *, uint m, bool choice = false);
    double operator()(const Vector &Beta) const;
    //    DafeLoglike * clone() const override;
   private:
    mutable MLM *mlm_;
    mutable Vector x;
    uint m;
    bool choice;
  };

  DafeLoglike::DafeLoglike(MLM *mod, uint which_choice, bool is_choice)
      : mlm_(mod), m(which_choice), choice(is_choice) {}

  double DafeLoglike::operator()(const Vector &beta) const {
    Vector full_beta = mlm_->beta();
    int begin = 0;
    int length = choice ? mlm_->choice_nvars() : mlm_->subject_nvars();
    if (choice) {
      begin = (m - 1) * mlm_->choice_nvars();
    } else {
      begin = (mlm_->Nchoices() - 1) * mlm_->choice_nvars();
    }
    VectorView subset(full_beta, begin, length);
    subset = beta;
    return mlm_->loglike(full_beta);
  }

  struct Logp {
    Logp(const std::shared_ptr<DafeLoglike> &log_likelihood,
         const Ptr<MvnModel> &prior)
        : loglike_(log_likelihood), prior_(prior) {}
    double operator()(const Vector &x) const {
      return (*loglike_)(x) + prior_->logp(x);
    }
    std::shared_ptr<DafeLoglike> loglike_;
    Ptr<MvnModel> prior_;
  };

  //______________________________________________________________________

  DafeRMlm::DafeRMlm(MultinomialLogitModel *mod,
                     const Ptr<MvnModel> &SubjectPri,
                     const Ptr<MvnModel> &ChoicePri, double Tdf)
      : DafeMlmBase(mod, SubjectPri, ChoicePri), mlm_(mod) {
    uint M = mod->Nchoices();
    for (uint m = 0; m < M; ++m) {
      std::shared_ptr<DafeLoglike> loglike(new DafeLoglike(mlm_, m));
      Logp logpost(loglike, subject_pri());
      Ptr<MH> sam = new MH(logpost, subject_proposals_[m]);
      subject_samplers_.push_back(sam);
    }

    uint pch = mlm_->choice_nvars();
    if (pch > 0) {
      std::shared_ptr<DafeLoglike> choice_loglike(
          new DafeLoglike(mlm_, 0, true));
      Logp choice_logpost(choice_loglike, choice_pri());
      choice_sampler_ = new MH(choice_logpost, choice_proposal_);
    }
  }
  // ======================================================================
  void DafeRMlm::draw() {
    // no need to draw beta 0
    for (uint m = mlo(); m < mlm_->Nchoices(); ++m) draw_subject(m);
    if (mlm_->choice_nvars() > 0) draw_choice();
  }

  // ======================================================================
  void DafeRMlm::draw_subject(uint i) {
    Vector b = mlm_->beta_subject(i);
    b = subject_samplers_[i]->draw(b);
    mlm_->set_beta_subject(b, i);
  }

  void DafeRMlm::draw_choice() {
    if (mlm_->choice_nvars() == 0) return;
    Vector b = mlm_->beta_choice();
    b = choice_sampler_->draw(b);
    mlm_->set_beta_choice(b);
  }

}  // namespace BOOM
