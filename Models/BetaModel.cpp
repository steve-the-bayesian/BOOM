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
#include "Models/BetaModel.hpp"
#include <cmath>
#include <sstream>

#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "Models/SufstatAbstractCombineImpl.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {

  namespace {
    using BS = BOOM::BetaSuf;
    using BM = BOOM::BetaModel;
  }  // namespace

  BS::BetaSuf() : n_(0), sumlog_(0), sumlogc_(0) {}

  BS::BetaSuf(const BetaSuf &rhs)
      : SufstatDetails<DoubleData>(rhs),
        n_(rhs.n_),
        sumlog_(rhs.sumlog_),
        sumlogc_(rhs.sumlogc_) {}

  BS *BS::clone() const { return new BS(*this); }

  void BS::Update(const DoubleData &d) {
    double p = d.value();
    update_raw(p);
  }

  void BetaSuf::update_raw(double p) {
    ++n_;
    sumlog_ += log(p);
    sumlogc_ += log(1 - p);
  }

  BetaSuf *BS::abstract_combine(Sufstat *s) {
    return abstract_combine_impl(this, s);
  }

  void BS::combine(const Ptr<BS> &s) {
    n_ += s->n_;
    sumlog_ += s->sumlog_;
    sumlogc_ += s->sumlogc_;
  }

  void BS::combine(const BS &s) {
    n_ += s.n_;
    sumlog_ += s.sumlog_;
    sumlogc_ += s.sumlogc_;
  }

  Vector BS::vectorize(bool) const {
    Vector ans(3);
    ans[0] = n_;
    ans[1] = sumlog_;
    ans[2] = sumlogc_;
    return ans;
  }

  Vector::const_iterator BS::unvectorize(Vector::const_iterator &v, bool) {
    n_ = *v;
    ++v;
    sumlog_ = *v;
    ++v;
    sumlogc_ = *v;
    ++v;
    return v;
  }

  Vector::const_iterator BS::unvectorize(const Vector &v, bool minimal) {
    Vector::const_iterator it = v.begin();
    return unvectorize(it, minimal);
  }

  std::ostream &BS::print(std::ostream &out) const {
    out << n_ << " " << sumlog_ << " " << sumlogc_;
    return out;
  }

  BM::BetaModel(double a, double b)
      : ParamPolicy(new UnivParams(a), new UnivParams(b)), DataPolicy(new BS) {
    set_params(a, b);
  }

  BM::BetaModel(double mean, double sample_size, int)
      : ParamPolicy(new UnivParams(mean * sample_size),
                    new UnivParams((1 - mean) * sample_size)),
        DataPolicy(new BS) {
    if (mean <= 0 || mean >= 1.0 || sample_size <= 0) {
      report_error(
          "mean must be in (0, 1), and sample_size must "
          "be positive in BetaModel(mean, sample_size, int) "
          "constructor");
    }
  }

  BM::BetaModel(const BM &rhs)
      : Model(rhs),
        ParamPolicy(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs),
        NumOptModel(rhs),
        DiffDoubleModel(rhs) {}

  BM *BM::clone() const { return new BM(*this); }

  Ptr<UnivParams> BM::Alpha() { return ParamPolicy::prm1(); }
  Ptr<UnivParams> BM::Beta() { return ParamPolicy::prm2(); }
  const Ptr<UnivParams> BM::Alpha() const { return ParamPolicy::prm1(); }
  const Ptr<UnivParams> BM::Beta() const { return ParamPolicy::prm2(); }

  const double &BM::a() const { return ParamPolicy::prm1_ref().value(); }
  const double &BM::b() const { return ParamPolicy::prm2_ref().value(); }

  void BM::set_a(double alpha) {
    if (alpha <= 0) {
      ostringstream err;
      err << "The alpha parameter must be positive in BetaModel::set_a()."
          << endl
          << "Called with alpha = " << alpha << endl;
      report_error(err.str());
    }
    ParamPolicy::prm1_ref().set(alpha);
  }

  void BM::set_b(double beta) {
    if (beta <= 0) {
      ostringstream err;
      err << "The beta parameter must be positive in BetaModel::set_a()."
          << endl
          << "Called with beta = " << beta << endl;
      report_error(err.str());
    }
    ParamPolicy::prm2_ref().set(beta);
  }

  void BM::set_params(double a, double b) {
    set_a(a);
    set_b(b);
  }

  double BM::mean() const { return a() / sample_size(); }
  double BM::sample_size() const { return a() + b(); }
  double BM::variance() const {
    double p = mean();
    return p * (1 - p) / (1 + sample_size());
  }

  void BM::set_sample_size(double a_plus_b) {
    double mu = mean();
    double a = mu * a_plus_b;
    double b = (1 - mu) * a_plus_b;
    set_params(a, b);
  }

  void BM::set_mean(double a_over_a_plus_b) {
    double n = sample_size();
    double a = a_over_a_plus_b * n;
    double b = (1 - a_over_a_plus_b) * n;
    set_params(a, b);
  }

  double BM::Loglike(const Vector &ab, Vector &g, Matrix &h, uint nd) const {
    if (ab.size() != 2) {
      report_error("Wrong size argument.");
    }
    double alpha = ab[0];
    double beta = ab[1];
    if (alpha <= 0 || beta <= 0) {
      if (nd > 0) {
        g[0] = (alpha <= 0) ? 1.0 : 0.0;
        g[1] = (beta <= 0) ? 1.0 : 0.0;
        if (nd > 1) {
          h = 0.0;
          h.diag() = -1.0;
        }
      }
      return negative_infinity();
    }

    double n = suf()->n();
    double sumlog = suf()->sumlog();
    double sumlogc = suf()->sumlogc();

    double ans = n * (lgamma(alpha + beta) - lgamma(alpha) - lgamma(beta));
    ans += (alpha - 1) * sumlog + (beta - 1) * sumlogc;

    if (nd > 0) {
      double psisum = digamma(alpha + beta);
      g[0] = n * (psisum - digamma(alpha)) + sumlog;
      g[1] = n * (psisum - digamma(beta)) + sumlogc;

      if (nd > 1) {
        double trisum = trigamma(alpha + beta);
        h(0, 0) = n * (trisum - trigamma(alpha));
        h(0, 1) = h(1, 0) = n * trisum;
        h(1, 1) = n * (trisum - trigamma(beta));
      }
    }
    return ans;
  }

  double BM::log_likelihood(double a, double b) const {
    return beta_log_likelihood(a, b, *suf());
  }

  double BM::Logp(double x, double &d1, double &d2, uint nd) const {
    if (x < 0 || x > 1) {
      return BOOM::negative_infinity();
    }
    double inf = BOOM::infinity();
    double a = this->a();
    double b = this->b();
    if (a == inf || b == inf) {
      return Logp_degenerate(x, d1, d2, nd);
    }

    double ans = dbeta(x, a, b, true);

    double A = a - 1;
    double B = b - 1;
    double y = 1 - x;

    if (nd > 0) {
      d1 = A / x - B / y;
      if (nd > 1) {
        d2 = -A / square(x) - B / square(y);
      }
    }
    return ans;
  }

  double BM::Logp_degenerate(double x, double &d1, double &d2, uint nd) const {
    double inf = BOOM::infinity();
    double a_inf = static_cast<double>(a() == inf);
    double b_inf = static_cast<double>(b() == inf);
    if ((a_inf != 0.0) && (b_inf != 0.0)) {
      report_error("both a and b are finite in BetaModel::Logp");
    }
    if (nd > 0) {
      d1 = 0;
      if (nd > 1) {
        d2 = 0;
      }
    }
    if (b_inf != 0.0) {
      x = 1 - x;
    }
    return x == 1.0 ? 0.0 : BOOM::negative_infinity();
  }

  double BM::sim(RNG &rng) const { return rbeta_mt(rng, a(), b()); }

  double beta_log_likelihood(double a, double b, const BetaSuf &suf) {
    if (a <= 0 || b <= 0) {
      return negative_infinity();
    }

    double n = suf.n();
    double sumlog = suf.sumlog();
    double sumlogc = suf.sumlogc();

    double ans = n * (lgamma(a + b) - lgamma(a) - lgamma(b));
    ans += (a - 1) * sumlog + (b - 1) * sumlogc;
    return ans;
  }
}  // namespace BOOM
