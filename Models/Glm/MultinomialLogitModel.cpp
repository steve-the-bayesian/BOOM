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
#include "Models/Glm/MultinomialLogitModel.hpp"

#include <cmath>
#include <functional>

#include "LinAlg/VectorView.hpp"
#include "Models/Glm/PosteriorSamplers/MLVS.hpp"
#include "Models/MvnBase.hpp"
#include "TargetFun/LogPost.hpp"
#include "TargetFun/Loglike.hpp"
#include "cpputil/lse.hpp"
#include "cpputil/math_utils.hpp"
#include "distributions.hpp"
#include "numopt.hpp"
#include "stats/FreqDist.hpp"

namespace BOOM {

  typedef MultinomialLogitModel MLM;

  inline Vector make_vector(const Matrix &beta_subject,
                            const Vector &beta_choice) {
    Vector b(beta_subject.begin(), beta_subject.end());
    b.concat(beta_choice);
    return b;
  }
  //------------------------------------------------------------
  MLM::MultinomialLogitModel(int num_choices,
                             int subject_xdim,
                             int choice_xdim)
      : num_choices_(num_choices),
        subject_xdim_(subject_xdim),
        choice_xdim_(choice_xdim)
  {
    setup();
  }
  //------------------------------------------------------------
  MLM::MultinomialLogitModel(const Matrix &beta_subject,
                             const Vector &beta_choice)
      : num_choices_(1 + beta_subject.ncol()),
        subject_xdim_(beta_subject.nrow()),
        choice_xdim_(beta_choice.size()) {
    setup();
    set_beta(make_vector(beta_subject, beta_choice));
  }
  //------------------------------------------------------------
  MLM::MultinomialLogitModel(
      const std::vector<Ptr<CategoricalData> > &responses,
      const Matrix &Xsubject, const std::vector<Matrix> &Xchoice)
      : num_choices_(responses[0]->nlevels()),
        subject_xdim_(Xsubject.ncol()),
        choice_xdim_(0)
  {
    int n = responses.size();
    if ((nrow(Xsubject) > 0 && nrow(Xsubject) != n) ||
        (!Xchoice.empty() && Xchoice.size() != n)) {
      ostringstream err;
      err << "Predictor sizes do not match in MultinomialLogitModel "
          << "constructor" << endl
          << "responses.size() = " << n << endl
          << "nrow(Xsubject)   = " << nrow(Xsubject) << endl;
      if (!Xchoice.empty()) {
        err << "Xchoice.size()   = " << Xchoice.size() << endl;
      }
      report_error(err.str());
    }

    for (int i = 0; i < n; ++i) {
      Ptr<VectorData> subject_predictors;
      if (nrow(Xsubject) > 0) {
        subject_predictors.reset(new VectorData(Xsubject.row(i)));
      } else {
        subject_predictors.reset(new VectorData(Vector(0)));
      }
      std::vector<Ptr<VectorData> > choice_predictors;
      if (!Xchoice.empty()) {
        const Matrix &choice_matrix(Xchoice[i]);
        if (choice_xdim_ == 0) {
          choice_xdim_ = ncol(choice_matrix);
        } else if (choice_xdim_ != ncol(choice_matrix)) {
          ostringstream err;
          err << "The number of columns in the choice matrix for observation "
              << i << " did not match previous observations." << endl
              << "ncol(Xsubject[i]) = " << ncol(choice_matrix) << endl
              << "previously:         " << choice_xdim_ << endl;
          report_error(err.str());
        }

        if (nrow(choice_matrix) != num_choices_) {
          ostringstream err;
          err << "The number of rows in choice matrix does not match the "
              << "number of choices available in the response." << endl
              << "response:  " << num_choices_ << endl
              << "Xchoice[" << i << "]: " << nrow(choice_matrix) << endl;
          report_error(err.str());
        }
        for (int j = 0; j < num_choices_; ++j) {
          NEW(VectorData, ch)(choice_matrix.row(j));
          choice_predictors.push_back(ch);
        }
      }

      NEW(ChoiceData, dp)(*responses[i], subject_predictors, choice_predictors);
      add_data(dp);
    }
    setup();
  }
  //------------------------------------------------------------
  MLM::MultinomialLogitModel(const MultinomialLogitModel &rhs)
      : Model(rhs),
        ParamPolicy(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs),
        NumOptModel(rhs),
        wsp_(rhs.wsp_),
        num_choices_(rhs.num_choices_),
        subject_xdim_(rhs.subject_xdim_),
        choice_xdim_(rhs.choice_xdim_),
        log_sampling_probs_(rhs.log_sampling_probs_) {
    // setup_observers();
  }
  //------------------------------------------------------------
  MLM *MLM::clone() const { return new MLM(*this); }
  //------------------------------------------------------------
  const Vector &MLM::beta() const { return coef().Beta(); }
  //------------------------------------------------------------
  // Vector MLM::beta_with_zeros() const {
  //   // if (!beta_with_zeros_current_) fill_extended_beta();
  //   // return beta_with_zeros_;
  // }

  //------------------------------------------------------------
  Vector MLM::beta_subject(int choice) const {
    int p = subject_nvars();
    if (choice == 0) return Vector(p, 0.0);
    const Vector &b(beta());
    Vector::const_iterator it = b.begin() + ((choice - 1) * p);
    return Vector(it, it + p);
  }

  //------------------------------------------------------------
  Vector MLM::beta_choice() const {
    Vector::const_iterator it = beta().begin();
    it += (Nchoices() - 1) * subject_nvars();
    return Vector(it, beta().end());
  }
  //------------------------------------------------------------
  void MLM::set_beta(const Vector &b) { coef().set_Beta(b); }
  //------------------------------------------------------------
  void MLM::set_beta_subject(const Vector &b, int m) {
    if (m == 0 || m >= Nchoices()) index_out_of_bounds(m);
    int p = subject_nvars();
    Vector beta(this->beta());
    Vector::iterator it = beta.begin() + (m - 1) * p;
    std::copy(b.begin(), b.end(), it);
    set_beta(beta);
  }
  //------------------------------------------------------------
  void MLM::set_beta_choice(const Vector &b) {
    int pos = (Nchoices() - 1) * subject_nvars();
    Vector beta(this->beta());
    std::copy(b.begin(), b.end(), beta.begin() + pos);
    set_beta(beta);
  }
  //------------------------------------------------------------
  GlmCoefs &MLM::coef() { return ParamPolicy::prm_ref(); }
  const GlmCoefs &MLM::coef() const { return ParamPolicy::prm_ref(); }
  Ptr<GlmCoefs> MLM::coef_prm() { return ParamPolicy::prm(); }
  const Ptr<GlmCoefs> MLM::coef_prm() const { return ParamPolicy::prm(); }

  //------------------------------------------------------------
  const Selector &MLM::inc() const { return coef().inc(); }

  //------------------------------------------------------------
  double MLM::log_likelihood(const Vector &beta, Vector &g, SpdMatrix &h,
                             int nd) const {
    const std::vector<Ptr<ChoiceData>> &d(dat());
    double ans = 0;
    int nobs = d.size();
    bool downsampling = log_sampling_probs().size() == Nchoices();
    Selector inc(this->inc());
    int beta_dim = inc.nvars();
    if (nd > 0) {
      g.resize(beta_dim);
      g = 0;
      if (nd > 1) {
        // h.resize(beta_dim, beta_dim);
        h.resize(beta_dim);
        h = 0;
      }
    }

    // Matrix h_xbar_buffer;
    // Matrix h_tmpx_buffer;

    // if (nd > 1) {
    //   h_xbar_buffer.resize(nobs, beta_dim);
    //   h_tmpx_buffer.resize(nobs * Nchoices(), beta_dim);
    // }

    // int h_xbar_index = 0;
    // int h_tmpx_index = 0;

    for (int i = 0; i < nobs; ++i) {
      Ptr<ChoiceData> dp = d[i];
      int y = dp->value();
      fill_eta(*dp, wsp_, beta);
      if (downsampling) {
        wsp_ += log_sampling_probs();
      }
      double lognc = lse(wsp_);
      ans += wsp_[y] - lognc;
      if (nd > 0) {
        int M = dp->nchoices();
        Matrix X = inc.select_cols(dp->X(false));
        Vector probs = exp(wsp_ - lognc);
        Vector xbar = probs * X;
        g += X.row(y) - xbar;

        if (nd > 1) {
          for (int m = 0; m < M; ++m) {
            Vector tmpx = X.row(m);
            // h_tmpx_buffer.row(h_tmpx_index++) = tmpx * sqrt(probs[m]);
            h.add_outer(tmpx, -probs[m]);
          }
          h.add_outer(xbar);
          // h_xbar_buffer.row(h_xbar_index++) = xbar;
        }
      }

      // if (nd > 1) {
      //   h.add_inner(h_xbar_buffer);
      //   h.add_inner(h_tmpx_buffer, -1);
      // }
    }
    return ans;
  }

  //------------------------------------------------------------
  double MLM::Loglike(const Vector &beta, Vector &g, Matrix &h, uint nd) const {
    SpdMatrix hessian;
    double ans = log_likelihood(beta, g, hessian, nd);
    if (nd > 1) {
      h = hessian;
    }
    return ans;
  }

  //----------------------------------------------------------------------
  void MLM::add_all_slopes() { coef().add_all(); }

  //----------------------------------------------------------------------
  void MLM::drop_all_slopes(bool keep_intercepts) {
    coef().drop_all();
    if (keep_intercepts) {
      int psub = subject_nvars();
      int nch = Nchoices();
      for (int m = 1; m < nch; ++m) {
        int pos = (m - 1) * psub;
        coef().add(pos);
      }
    }
  }

  //------------------------------------------------------------
  double MLM::predict_choice(const ChoiceData &dp, int m) const {
    int pch = choice_nvars();
    if (pch == 0) return 0;
    int psub = subject_nvars();
    int M = Nchoices();
    ConstVectorView b(beta(), (M - 1) * psub);
    assert(b.size() == dp.choice_nvars());
    return b.dot(dp.Xchoice(m));
  }

  //------------------------------------------------------------
  double MLM::predict_subject(const ChoiceData &dp, int m) const {
    if (m == 0) return 0;
    int psub = subject_nvars();
    assert(m < Nchoices());
    ConstVectorView b(beta(), (m - 1) * psub, psub);
    return b.dot(dp.Xsubject());
  }

  //------------------------------------------------------------
  Vector &MLM::fill_eta(const ChoiceData &dp,
                        Vector &ans,
                        const Vector &beta) const {
    ans.resize(Nchoices());;
    const Selector &included(inc());
    const Matrix &X(dp.X(false));
    if (included.nvars_excluded() == 0) {
      ans = X * beta;
    } else {
      included.sparse_multiply(X, beta, VectorView(ans));
    }
    // TODO: handle restricted choice sets and include an
    // offset.
    return ans;
  }

  Vector &MLM::fill_eta(const ChoiceData &dp, Vector &ans) const {
    return fill_eta(dp, ans, beta());
  }

  //------------------------------------------------------------
  double MLM::pdf(const Ptr<Data> &dp, bool logscale) const {
    double ans = logp(*DAT(dp));
    return logscale ? ans : exp(ans);
  }

  double MLM::pdf(const Data *dp, bool logscale) const {
    double ans = logp(*DAT(dp));
    return logscale ? ans : exp(ans);
  }

  double MLM::logp(const ChoiceData &dp) const {
    // For right now...  this assumes all choices are available to
    //    everyone int n = dp->n_avail();
    wsp_.resize(num_choices_);
    fill_eta(dp, wsp_);
    int y = dp.value();
    double ans = wsp_[y] - lse(wsp_);
    return ans;
  }

  //------------------------------------------------------------

  int MLM::beta_size(bool include_zeros) const {
    int nch(num_choices_);
    if (!include_zeros) --nch;
    return nch * subject_xdim_ + choice_xdim_;
  }

  int MLM::sim(const Ptr<ChoiceData> &dp, Vector &probs, RNG &rng) const {
    predict(dp, probs);
    return rmulti_mt(rng, probs);
  }

  int MLM::sim(const Ptr<ChoiceData> &dp, RNG &rng) const {
    Vector prob = predict(dp);
    return rmulti_mt(rng, prob);
  }

  Vector &MLM::predict(const Ptr<ChoiceData> &dp,
                       Vector &ans,
                       bool probscale) const {
    fill_eta(*dp, ans);
    if (probscale) {
      ans = exp(ans - lse(ans));
    }
    return ans;
  }

  Vector MLM::predict(const Ptr<ChoiceData> &dp, bool probscale) const {
    Vector ans(num_choices_);
    return predict(dp, ans, probscale);
  }

  //------------------------------------------------------------
  int MLM::subject_nvars() const { return subject_xdim_; }
  int MLM::choice_nvars() const { return choice_xdim_; }
  int MLM::Nchoices() const { return num_choices_; }
  //------------------------------------------------------------

  void MLM::set_sampling_probs(const Vector &probs) {
    assert(probs.size() == num_choices_);
    log_sampling_probs_ = log(probs);
  }

  const Vector &MLM::log_sampling_probs() const { return log_sampling_probs_; }

  //------------------------------------------------------------
  //  void MLM::watch_beta() { beta_with_zeros_current_ = false; }

  //------------------------------------------------------------
  void MLM::setup() {
    ParamPolicy::set_prm(new GlmCoefs(beta_size(false)));
    //    setup_observers();
    //beta_with_zeros_current_ = false;
  }

  //------------------------------------------------------------
  // void MLM::setup_observers() {
  //   GlmCoefs &b(coef());
  //   try {
  //     b.add_observer(this, [this]() { this->watch_beta(); });
  //   } catch (const std::exception &e) {
  //     report_error(e.what());
  //   } catch (...) {
  //     report_error(
  //         "unknown exception caught by "
  //         "MultinomialLogitModel::setup_observer");
  //   }
  // }

  //------------------------------------------------------------
  // void MLM::fill_extended_beta() const {
  //   int p = subject_nvars();
  //   Vector &b(beta_with_zeros_);
  //   b.resize(beta_size(true));
  //   const Vector &Beta(beta());
  //   std::fill(b.begin(), b.begin() + p, 0);
  //   std::copy(Beta.begin(), Beta.end(), b.begin() + p);
  //   beta_with_zeros_current_ = true;
  // }

  //------------------------------------------------------------
  void MLM::index_out_of_bounds(int m) const {
    ostringstream err;
    err << "index " << m << " outside the allowable range (" << 1 << ", "
        << Nchoices() - 1 << ") in MultinomialLogitModel::set_beta_subject."
        << endl;
    report_error(err.str());
  }

}  // namespace BOOM
