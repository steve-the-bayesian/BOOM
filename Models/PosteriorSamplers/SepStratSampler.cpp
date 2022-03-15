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

#include "Models/PosteriorSamplers/SepStratSampler.hpp"
#include <ctime>
#include "LinAlg/Cholesky.hpp"
#include "LinAlg/Matrix.hpp"
#include "Models/UniformCorrelationModel.hpp"
#include "Samplers/ScalarSliceSampler.hpp"
#include "Samplers/SliceSampler.hpp"
#include "TargetFun/TargetFun.hpp"
#include "cpputil/math_utils.hpp"
#include "distributions.hpp"

namespace BOOM {

  SepStratSampler::SepStratSampler(MvnModel *mod,
                                   const std::vector<Ptr<GammaModel> > &ivar,
                                   RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        mod_(mod),
        Rpri_(new UniformCorrelationModel(mod->dim())),
        sinv_pri_(ivar) {
    setup();
  }

  SepStratSampler::SepStratSampler(MvnModel *mod,
                                   const Ptr<CorrelationModel> &cor,
                                   const std::vector<Ptr<GammaModel> > &ivar,
                                   RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        mod_(mod),
        Rpri_(cor),
        sinv_pri_(ivar),
        fast_count_(0),
        stable_count_(0) {
    setup();
  }

  SepStratSampler *SepStratSampler::clone_to_new_host(Model *new_host) const {
    std::vector<Ptr<GammaModel>> marginal_precision_priors;
    for (const auto &el : sinv_pri_) {
      marginal_precision_priors.push_back(el->clone());
    }

    SepStratSampler *ans = new SepStratSampler(
        dynamic_cast<MvnModel *>(new_host),
        Rpri_->clone(),
        marginal_precision_priors,
        rng());

    ans->set_max_tries(max_tries_);
    ans->set_polar_frac(polar_frac_);
    ans->set_alpha(alpha_);
    return ans;
  }

  void SepStratSampler::setup() {
    max_tries_ = square(mod_->dim());
    alpha_ = .25;
    fast_count_ = 0;
    stable_count_ = 0;
    polar_count_ = 0;
    fast_time_ = 0;
    stable_time_ = 0;
    wasted_time_ = 0;
    polar_time_ = 0;
  }

  //----------------------------------------------------------------------
  // sets the maximum number of slice sampling attempts that will be
  // made in fast_draw
  void SepStratSampler::set_max_tries(int mt) { max_tries_ = mt; }
  //----------------------------------------------------------------------
  // sets the fraction of backup draws that will use polar slice
  // sampling instead of scalar updates
  void SepStratSampler::set_polar_frac(double f) { polar_frac_ = f; }
  //----------------------------------------------------------------------
  // sets the fraction of data lent to the prior during fast_draw
  void SepStratSampler::set_alpha(double a) { alpha_ = a; }
  //----------------------------------------------------------------------
  // required virtual method
  void SepStratSampler::draw() {
    n_ = mod_->suf()->n();
    sumsq_ = mod_->suf()->center_sumsq(mod_->mu());
    sumsq_upper_chol_ = sumsq_.chol();
    sumsq_upper_chol_.transpose_inplace_square();
    clock_t start = clock();
    bool ok = fast_draw();
    if (ok) {
      //      cout << "fast draw" << endl;
      ++fast_count_;
      clock_t done = clock();
      double fast_time = done - start;
      fast_time_ += fast_time / CLOCKS_PER_SEC;
      return;
    }
    clock_t bail = clock();
    double wasted_time = bail - start;
    wasted_time_ += wasted_time / CLOCKS_PER_SEC;
    if (runif_mt(rng()) < polar_frac_) {
      // backup is a polar draw
      //      cout << "polar draw" << endl;
      polar_draw();
      ++polar_count_;
      clock_t done = clock();
      double polar_time = done - bail;
      polar_time_ += polar_time / CLOCKS_PER_SEC;
    } else {
      // backup is a one-at-a-time draw
      //      cout << "stable draw" << endl;
      stable_draw();
      ++stable_count_;
      clock_t done = clock();
      double stable_time = done - bail;
      stable_time_ += stable_time / CLOCKS_PER_SEC;
    }
  }
  //----------------------------------------------------------------------
  // for use by fast_draw.  logposterior = log(p0 * p1), where p0 is
  // the prior times a fraction alpha_ of the likelihood.  The
  // likelihood assumes that n_ and the cholesky factor of sumsq_ have
  // been precomputed.
  double SepStratSampler::logp0(const SpdMatrix &Sigma, double alpha) const {
    Cholesky L(Sigma);
    if (!L.is_pos_def()) return BOOM::negative_infinity();

    double ans = logprior(Sigma);
    if (ans == BOOM::negative_infinity()) return ans;
    ans += -.5 * (n_ * alpha * L.logdet() +
                  L.solve(sqrt(alpha) * sumsq_upper_chol_.transpose()).sumsq());
    return ans;
  }
  //----------------------------------------------------------------------
  // log of the prior distribution at the model's current value of Sigma
  double SepStratSampler::logpri() const { return logprior(mod_->Sigma()); }
  //----------------------------------------------------------------------
  // log of the prior distribution at an arbitrary value of Sigma
  double SepStratSampler::logprior(const SpdMatrix &Sigma) const {
    // Prior is with respect to Sigma

    int d = Sigma.nrow();
    R_ = CorrelationMatrix(var2cor(Sigma));
    sd_ = sqrt(diag(Sigma));

    double ans = Rpri_->logp(R_);
    if (ans == BOOM::negative_infinity()) return ans;
    for (int i = 0; i < sd_.size(); ++i) {
      double sd = sd_[i];
      if (sd <= 0) return BOOM::negative_infinity();
      ans += sinv_pri_[i]->logp(1.0 / square(sd));
      ans += (d - 3) * log(sd);
      // d-3 comes from the Jacobian of two transformations:
      // Sigma -> (S,R) gives s^d
      // s -> 1/s^2 gives s^-3
    }
    return (ans);
  }
  //----------------------------------------------------------------------
  // log posterior if R(i_,j_) is replaced by r.  performs work needed
  // for slice sampling in draw_R(i,j)
  double SepStratSampler::logp_slice_R(double r) {
    set_R(r);
    fill_siginv(false);  // false means we need to compute Rinv
    const SpdMatrix &Siginv(cand_);
    double ans = .5 * n_ * logdet(Siginv);  // positive .5
    ans += -.5 * traceAB(Siginv, sumsq_);
    ans += Rpri_->logp(R_);
    // skip the jacobian because it only has products of sigma^2's in it
    return ans;
  }
  //----------------------------------------------------------------------
  // returns the log posterior distribution if element i_ of 1/sd_^2
  // is replaced by ivar.  supports slice sampler in draw_sigsq(i_);
  double SepStratSampler::logp_slice_ivar(double ivar) {
    double ans = sinv_pri_[i_]->logp(ivar);  // prior on ivar
    ans += .5 * n_ * log(ivar);              // determinant

    sd_[i_] = 1.0 / sqrt(ivar);
    fill_siginv(true);
    const SpdMatrix &Siginv(cand_);
    ans += -.5 * traceAB(Siginv, sumsq_);  // exponential

    double logsd = log(sd_[i_]);  // jacobian Sigma -> S
    int dim = sd_.size();
    ans += dim * logsd;  // see notes in "logprior" member function

    ans += -3 * logsd;  // jacobian S -> ivar
    return ans;
  }
  //----------------------------------------------------------------------
  // sets element (i_,j_) of R_ to r while maintaining symmetry
  void SepStratSampler::set_R(double r) {
    R_(i_, j_) = r;
    R_(j_, i_) = r;
  }
  //----------------------------------------------------------------------
  // attempts to draw Sigma using a slice sampling scheme.  returns
  // true if draw succeeds
  bool SepStratSampler::fast_draw() {
    count_ = 0;
    double d = mod_->dim();
    double slice = logp0(mod_->Sigma(), alpha_) - rexp_mt(rng(), 1);

    while (count_++ < max_tries_) {
      double a = 1 - alpha_;
      double df = a * n_ - d - 1;
      if (df <= 1) {
        ostringstream err;
        err << "the 'alpha' parameter is set too small in SepStratSampler, "
            << "causing the resulting  degrees of freedom to be less than "
            << "the dimension of the matrix." << endl
            << "dim           = " << d << endl
            << "n             = " << n_ << endl
            << "alpha         = " << alpha_ << endl
            << "(1-alpha) * n = " << a * n_;
        report_error(err.str());
      }
      cand_ = rWishChol(df, sqrt(a) * sumsq_upper_chol_, true);
      if (logp0(cand_, alpha_) > slice) {
        mod_->set_Sigma(cand_);
        return true;
      }
    }
    return false;
  }
  //----------------------------------------------------------------------
  // draws Sigma one element at a time using regular slice sampling
  // based on the separation strategy in Barnard, McCulloch, and Meng
  // (2000 statistica sinica).
  void SepStratSampler::stable_draw() {
    int dim = nrow(sumsq_);

    cand_ = mod_->Sigma();
    sd_ = sqrt(diag(cand_));
    R_ = var2cor(cand_);
    Rinv_ = R_.inv();

    for (int i = 0; i < dim; ++i) {
      draw_sigsq(i);
    }

    for (int i = 0; i < dim; ++i) {
      for (int j = 0; j < i; ++j) {
        draw_R(i, j);
      }
    }
    fill_sigma();
    mod_->set_Sigma(cand_);
  }
  //----------------------------------------------------------------------
  // class to be passed to the slice sampler in draw_sigsq
  class SigmaTarget : public ScalarTargetFun {
   public:
    explicit SigmaTarget(SepStratSampler *s) : s_(s) {}
    double operator()(double ivar) const { return s_->logp_slice_ivar(ivar); }

   private:
    mutable SepStratSampler *s_;
  };
  //----------------------------------------------------------------------
  SpdMatrix SepStratSampler::Sigma() {
    fill_sigma();
    return cand_;
  }
  //----------------------------------------------------------------------
  class SigmaPolarTarget : public TargetFun {
   public:
    explicit SigmaPolarTarget(SepStratSampler *s) : sam(s), Sigma_(s->Sigma()) {}
    double operator()(const Vector &x) const {
      Sigma_.unvectorize(x, true);
      return sam->logp0(Sigma_, 1.0);
    }

   private:
    SepStratSampler *sam;
    mutable SpdMatrix Sigma_;
  };

  //----------------------------------------------------------------------
  void SepStratSampler::polar_draw() {
    cand_ = mod_->Sigma();
    sd_ = cand_.vectorize(true);  // true means minimal, only upper triangle
    SigmaPolarTarget target(this);
    SliceSampler sam(target);
    sd_ = sam.draw(sd_);
    cand_.unvectorize(sd_, true);
    mod_->set_Sigma(cand_);
  }
  //----------------------------------------------------------------------
  // driver function to draw a scalar variance parameter conditional
  // on the correlations and other variances.
  void SepStratSampler::draw_sigsq(int i) {
    i_ = i;
    j_ = i;

    SigmaTarget target(this);
    ScalarSliceSampler sam(target);
    sam.set_lower_limit(0);
    double ivar = 1.0 / square(sd_[i]);
    ivar = sam.draw(ivar);
    sd_[i] = 1.0 / sqrt(ivar);
  }
  //----------------------------------------------------------------------
  // driver function to draw a single element of the correlation
  // matrix conditional on the variances.
  void SepStratSampler::draw_R(int i, int j) {
    i_ = i;
    j_ = j;

    double oldr = R_(i, j);
    double slice = logp_slice_R(oldr) - rexp_mt(rng());
    find_limits();
    double rcand = runif_mt(rng(), lo_, hi_);
    while (logp_slice_R(rcand) < slice && hi_ > lo_) {
      if (rcand > oldr)
        hi_ = rcand;
      else
        lo_ = rcand;
      rcand = runif_mt(rng(), lo_, hi_);
    }
    set_R(rcand);
  }
  //----------------------------------------------------------------------
  // returns the determinant of R_ if R_(i_,j_) is set to r
  double SepStratSampler::detR(double r) {
    R_(i_, j_) = r;
    R_(j_, i_) = r;
    return det(R_);
  }
  //----------------------------------------------------------------------
  // sets lo_ and hi_ to the smallest and largest values that
  // R_(i_,j_) can assume and still be positive definite
  void SepStratSampler::find_limits() {
    double f1 = detR(1);
    double f0 = detR(0);
    double fn = detR(-1);

    double a = .5 * (f1 + fn - 2 * f0);
    double b = .5 * (f1 - fn);
    double c = f0;

    double d2 = b * b - 4 * a * c;
    if (d2 <= 0) {
      lo_ = hi_ = 0;
      return;
    }
    double d = sqrt(d2);
    double a2 = 2 * a;
    lo_ = (-b - d) / a2;
    hi_ = (-b + d) / a2;
    if (hi_ < lo_) std::swap(lo_, hi_);
  }
  //----------------------------------------------------------------------
  // sets cand_ = S.inv() * Rinv_ * S.inv(), where S = diag(sd_)
  void SepStratSampler::fill_siginv(bool have_rinv) {
    if (!have_rinv) Rinv_ = R_.inv();
    cand_ = Rinv_;
    int d = Rinv_.nrow();
    for (int i = 0; i < d; ++i) {
      cand_.row(i) /= sd_[i];
      cand_.col(i) /= sd_[i];
    }
  }

  void SepStratSampler::fill_sigma() {
    cand_ = R_;
    int d = nrow(R_);
    for (int i = 0; i < d; ++i) {
      cand_.row(i) *= sd_[i];
      cand_.col(i) *= sd_[i];
    }
  }
}  // namespace BOOM
