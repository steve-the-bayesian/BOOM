// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2011 Steven L. Scott

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

#include "Models/BetaBinomialModel.hpp"
#include "Bmath/Bmath.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"
#include "stats/moments.hpp"

namespace BOOM {
  using Rmath::lgammafn;

  BetaBinomialModel::BetaBinomialModel(double a, double b)
      : ParamPolicy(new UnivParams(a), new UnivParams(b)),

        lgamma_n_y_(0.0) {
    check_positive(a, "BetaBinomialModel");
    check_positive(b, "BetaBinomialModel");
  }

  BetaBinomialModel::BetaBinomialModel(const BOOM::Vector &trials,
                                       const BOOM::Vector &successes)
      : ParamPolicy(new UnivParams(1.0), new UnivParams(1.0)),

        lgamma_n_y_(0.0) {
    if (trials.size() != successes.size()) {
      ostringstream err;
      err << "Vectors of trials and counts have different sizes in "
          << "BetaBinomialModel constructor";
      report_error(err.str());
    }
    for (int i = 0; i < trials.size(); ++i) {
      NEW(BinomialData, dp)(trials[i], successes[i]);
      add_data(dp);
    }
    if (trials.size() > 1) {
      mle();
      if (!mle_success()) {
        method_of_moments();
      }
      // Make sure a and b don't get set to absurdly small values in
      // the constructor.
      if (a() < .1) {
        set_a(.1);
      }
      if (b() < .1) {
        set_b(.1);
      }
    }
  }

  BetaBinomialModel::BetaBinomialModel(const BetaBinomialModel &rhs)
      : Model(rhs),
        ParamPolicy(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs),
        lgamma_n_y_(0.0) {}

  BetaBinomialModel *BetaBinomialModel::clone() const {
    return new BetaBinomialModel(*this);
  }

  void BetaBinomialModel::clear_data() {
    lgamma_n_y_ = 0.0;
    IID_DataPolicy<BinomialData>::clear_data();
  }

  void BetaBinomialModel::add_data(const Ptr<Data> &dp) {
    return this->add_data(DAT(dp));
  }

  void BetaBinomialModel::add_data(const Ptr<BinomialData> &data) {
    IID_DataPolicy<BinomialData>::add_data(data);
    const int64_t n = data->n();
    const int64_t y = data->y();
    lgamma_n_y_ += lgammafn(n + 1) - lgammafn(y + 1) - lgammafn(n - y + 1);
  }

  double BetaBinomialModel::loglike() const { return loglike(a(), b()); }

  double BetaBinomialModel::loglike(const Vector &ab) const {
    Vector g;
    Matrix h;
    return Loglike(ab, g, h, 0);
  }

  double BetaBinomialModel::Loglike(const Vector &ab, Vector &g, Matrix &h,
                                    uint nd) const {
    if (ab.size() != 2) {
      report_error("Wrong size argument.");
    }
    double a = ab[0];
    double b = ab[1];
    if (a <= 0 || b <= 0) {
      return BOOM::negative_infinity();
    }
    const std::vector<Ptr<BinomialData> > &data(dat());
    int nobs = data.size();
    double ans =
        nobs * (lgammafn(a + b) - lgammafn(a) - lgammafn(b)) + lgamma_n_y_;
    if (nd > 0) {
      g[0] = nobs * (digamma(a + b) - digamma(a));
      g[1] = nobs * (digamma(a + b) - digamma(b));
      if (nd > 1) {
        h(0, 0) = nobs * (trigamma(a + b) - trigamma(a));
        h(1, 1) = nobs * (trigamma(a + b) - trigamma(b));
        h(0, 1) = h(1, 0) = nobs * trigamma(a + b);
      }
    }

    for (int i = 0; i < nobs; ++i) {
      int64_t y = data[i]->y();
      int64_t n = data[i]->n();
      ans -= lgammafn(n + a + b) - lgammafn(a + y) - lgammafn(b + n - y);
      if (nd > 0) {
        double psin = digamma(a + b + n);
        g[0] += digamma(a + y) - psin;
        g[1] += digamma(b + n - y) - psin;
        if (nd > 1) {
          double trigamma_n = trigamma(a + b + n);
          h(0, 0) += trigamma(a + y) - trigamma_n;
          h(1, 1) += trigamma(b + n - y) - trigamma_n;
          h(0, 1) -= trigamma_n;
          h(1, 0) -= trigamma_n;
        }
      }
    }
    return ans;
  }

  double BetaBinomialModel::logp(int64_t n, int64_t y, double a,
                                 double b) const {
    if (a <= 0 || b <= 0) {
      return BOOM::negative_infinity();
    }
    double ans = lgammafn(n + 1) - lgammafn(y + 1) - lgammafn(n - y + 1);
    ans += lgammafn(a + b) - lgammafn(a) - lgammafn(b);
    ans -= lgammafn(n + a + b) - lgammafn(a + y) - lgammafn(b + n - y);
    return ans;
  }

  double BetaBinomialModel::loglike(double a, double b) const {
    if (a <= 0 || b <= 0) {
      return BOOM::negative_infinity();
    }
    const std::vector<Ptr<BinomialData> > &data(dat());
    int nobs = data.size();
    double ans =
        nobs * (lgammafn(a + b) - lgammafn(a) - lgammafn(b)) + lgamma_n_y_;
    for (int i = 0; i < nobs; ++i) {
      int64_t y = data[i]->y();
      int64_t n = data[i]->n();
      ans -= lgammafn(n + a + b) - lgammafn(a + y) - lgammafn(b + n - y);
    }
    return ans;
  }

  int64_t BetaBinomialModel::sim(RNG &rng, int64_t n) const {
    double rate = rbeta_mt(rng, a(), b());
    return rbinom_mt(rng, n, rate);
  }

  // Set a/(a+b) and a+b using a very rough method of moments
  // estimator.  The estimator can fail if either the sample mean or
  // the sample variance is zero, in which case this function will
  // exit without changing the model.
  void BetaBinomialModel::method_of_moments() {
    const std::vector<Ptr<BinomialData> > &data(dat());
    Vector p_hat;
    p_hat.reserve(data.size());
    for (int i = 0; i < data.size(); ++i) {
      int64_t trials = data[i]->trials();
      if (trials > 0) {
        double successes = data[i]->successes();
        p_hat.push_back(successes / trials);
      }
    }

    double sample_mean = mean(p_hat);
    double sample_variance = var(p_hat);
    if (sample_variance == 0.0 || sample_mean == 0.0 || sample_mean == 1.0) {
      return;
    }
    set_prior_mean(sample_mean);
    // v = (mean) * (1-mean) / (a+b+1)
    // =>
    // a+b+1 = mean * (1-mean) / v
    set_prior_sample_size(sample_mean * (1 - sample_mean) / sample_variance);
  }

  Ptr<UnivParams> BetaBinomialModel::SuccessPrm() {
    return ParamPolicy::prm1();
  }
  const Ptr<UnivParams> BetaBinomialModel::SuccessPrm() const {
    return ParamPolicy::prm1();
  }
  Ptr<UnivParams> BetaBinomialModel::FailurePrm() {
    return ParamPolicy::prm2();
  }
  const Ptr<UnivParams> BetaBinomialModel::FailurePrm() const {
    return ParamPolicy::prm2();
  }

  double BetaBinomialModel::a() const { return SuccessPrm()->value(); }
  void BetaBinomialModel::set_a(double a) {
    check_positive(a, "set_a");
    SuccessPrm()->set(a);
  }
  double BetaBinomialModel::b() const { return FailurePrm()->value(); }
  void BetaBinomialModel::set_b(double b) {
    check_positive(b, "set_b");
    FailurePrm()->set(b);
  }

  double BetaBinomialModel::prior_mean() const {
    double a = this->a();
    double n = this->b() + a;
    return a / n;
  }
  void BetaBinomialModel::set_prior_mean(double prob) {
    check_probability(prob, "set_prior_mean");
    double n = a() + b();
    double a = prob * n;
    double b = n - a;
    set_a(a);
    set_b(b);
  }

  double BetaBinomialModel::prior_sample_size() const { return a() + b(); }
  void BetaBinomialModel::set_prior_sample_size(double sample_size) {
    check_positive(sample_size, "set_prior_sample_size");
    double prob = prior_mean();
    double a = prob * sample_size;
    double b = sample_size - a;
    set_a(a);
    set_b(b);
  }

  void BetaBinomialModel::check_positive(double arg,
                                         const char *function_name) const {
    if (arg > 0) {
      return;
    }
    ostringstream err;
    err << "Illegal argument (" << arg << ") passed to "
        << "BetaBinomialModel::" << function_name
        << ".  Argument must be srictly positive." << endl;
    report_error(err.str());
  }

  void BetaBinomialModel::check_probability(double arg,
                                            const char *function_name) const {
    if (arg > 0 && arg < 1) {
      return;
    }
    ostringstream err;
    err << "Illegal argument (" << arg << ") passed to "
        << "BetaBinomialModel::" << function_name
        << ".  Argument must be srictly positive and strictly less than 1."
        << endl;
    report_error(err.str());
  }

  std::ostream &BetaBinomialModel::print_model_summary(
      std::ostream &out) const {
    using std::endl;
    out << "Model parameters:  " << endl
        << "a = " << a() << endl
        << "b = " << b() << endl
        << "Data consists of " << dat().size() << " observations: " << endl;
    for (size_t i = 0; i < dat().size(); ++i) {
      out << *(dat()[i]) << endl;
    }
    return out;
  }

}  // namespace BOOM
