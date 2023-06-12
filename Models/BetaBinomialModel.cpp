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
#include "Models/SufstatAbstractCombineImpl.hpp"

namespace BOOM {
  using Rmath::lgammafn;

  BetaBinomialSuf::BetaBinomialSuf()
      : sample_size_(0),
        sum_log_normalizing_constants_(0.0)
  {}

  BetaBinomialSuf * BetaBinomialSuf::clone() const {
    return new BetaBinomialSuf(*this);
  }

  void BetaBinomialSuf::Update(const BinomialData &dp) {
    add_data(dp.trials(), dp.successes(), 1);
  }

  void BetaBinomialSuf::add_data(int64_t trials, int64_t successes, int64_t counts) {
    if (counts < 0) {
      report_error("Negative 'counts' arugment.");
    }
    if (trials < 0) {
      report_error("Negative 'trials' argument.");
    }
    if (successes < 0) {
      report_error("Negative 'successes' argument.");
    }
    if (successes > trials) {
      report_error("'successes' cannot exceed 'trials'.");
    }
    int64_t failures = trials - successes;
    sum_log_normalizing_constants_ +=
        counts * (lgammafn(trials + 1) - lgammafn(successes + 1) - lgammafn(failures + 1));
    data_[std::pair<int64_t, int64_t>(trials, successes)] += counts;
    sample_size_ += counts;
  }

  void BetaBinomialSuf::clear() {
    data_.clear();
    sample_size_ = 0;
  }

  void BetaBinomialSuf::combine(const Ptr<BetaBinomialSuf> &suf) {
    return combine(*suf);
  }

  void BetaBinomialSuf::combine(const BetaBinomialSuf &rhs) {
    for (const auto &el : rhs.data_) {
      auto it = this->data_.find(el.first);
      if (it != this->data_.end()) {
        it->second += el.second;
      } else {
        this->data_[el.first] = el.second;
      }
    }
    sample_size_ += rhs.sample_size_;
    sum_log_normalizing_constants_ += rhs.sum_log_normalizing_constants_;
  }

  BetaBinomialSuf *BetaBinomialSuf::abstract_combine(Sufstat *s) {
    return abstract_combine_impl(this, s);
  }

  Vector BetaBinomialSuf::vectorize(bool minimal) const {
    Vector ans;
    ans.push_back(sample_size_);
    ans.push_back(sum_log_normalizing_constants_);
    ans.push_back(data_.size());
    for (const auto &el : data_) {
      ans.push_back(el.first.first);
      ans.push_back(el.first.second);
      ans.push_back(el.second);
    }
    return ans;
  }

  Vector::const_iterator BetaBinomialSuf::unvectorize(
      Vector::const_iterator &v, bool) {
    sample_size_ = static_cast<int64_t>(*v); ++v;
    sum_log_normalizing_constants_ = *v; ++v;

    size_t data_size = static_cast<size_t>(*v);
    for (size_t i = 0; i < data_size; ++i) {
      int64_t trials = static_cast<int64_t>(*v); ++v;
      int64_t successes = static_cast<int64_t>(*v); ++v;
      int64_t counts = static_cast<int64_t>(*v); ++v;
      data_[std::pair<int64_t, int64_t>(trials, successes)] = counts;
    }
    return v;
  }

  Vector::const_iterator BetaBinomialSuf::unvectorize(
      const Vector &v, bool minimal) {
    Vector::const_iterator it = v.begin();
    return unvectorize(it, minimal);
  }

  std::ostream & BetaBinomialSuf::print(std::ostream &out) const {
    for (const auto &el : data_) {
      out << std::setw(12) << el.first.first << ' '
          << std::setw(12) << el.first.second << ' '
          << std::setw(12) << el.second << '\n';
    }
    return out;
  }

  //===========================================================================
  BetaBinomialModel::BetaBinomialModel(double a, double b)
      : ParamPolicy(new UnivParams(a), new UnivParams(b)),
        DataPolicy(new BetaBinomialSuf)
  {
    check_positive(a, "BetaBinomialModel");
    check_positive(b, "BetaBinomialModel");
  }

  BetaBinomialModel::BetaBinomialModel(const BOOM::Vector &trials,
                                       const BOOM::Vector &successes)
      : ParamPolicy(new UnivParams(1.0), new UnivParams(1.0)),
        DataPolicy(new BetaBinomialSuf)
  {
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
        PriorPolicy(rhs)
  {}

  BetaBinomialModel *BetaBinomialModel::clone() const {
    return new BetaBinomialModel(*this);
  }

  void BetaBinomialModel::clear_data() {
    DataPolicy::clear_data();
  }

  void BetaBinomialModel::add_data(const Ptr<Data> &dp) {
    return this->add_data(DAT(dp));
  }

  void BetaBinomialModel::add_data(const Ptr<BinomialData> &data) {
    const int64_t n = data->n();
    const int64_t y = data->y();
    suf()->add_data(n, y, 1);
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

    const Ptr<BetaBinomialSuf> sufstat_ptr(suf());
    const BetaBinomialSuf &sufstat(*sufstat_ptr);
    auto &data(sufstat.count_table());
    int64_t nobs = sufstat.sample_size();

    double ans = nobs * (lgammafn(a + b) - lgammafn(a) - lgammafn(b))
        + sufstat.log_normalizing_constant();

    // Initialize the gradient and hessian if they were requested.
    if (nd > 0) {
      g[0] = nobs * (digamma(a + b) - digamma(a));
      g[1] = nobs * (digamma(a + b) - digamma(b));
      if (nd > 1) {
        h(0, 0) = nobs * (trigamma(a + b) - trigamma(a));
        h(1, 1) = nobs * (trigamma(a + b) - trigamma(b));
        h(0, 1) = h(1, 0) = nobs * trigamma(a + b);
      }
    }

    for (const auto &el : data) {
      int64_t trials = el.first.first;
      int64_t successes = el.first.second;
      int64_t failures = trials - successes;
      int64_t counts = el.second;
      ans += counts * (lgammafn(a + successes) + lgammafn(b + failures)
                                   -lgammafn(trials + a + b));
      if (nd > 0) {
        double psin = digamma(a + b + trials);
        g[0] += counts * (digamma(a + successes) - psin);
        g[1] += counts * (digamma(b + failures) - psin);
        if (nd > 1) {
          double trigamma_n = trigamma(a + b + trials);
          h(0, 0) += counts * (trigamma(a + successes) - trigamma_n);
          h(1, 1) += counts * (trigamma(b + failures) - trigamma_n);
          h(0, 1) -= counts * trigamma_n;
          h(1, 0) = h(0, 1);
        }
      }
    }
    return ans;
  }

  double BetaBinomialModel::logp(int64_t n, int64_t y, double a,
                                 double b) {
    if (a <= 0 || b <= 0 || n < 0 || y < 0 || y > n) {
      return BOOM::negative_infinity();
    }
    double ans = lgammafn(n + 1) - lgammafn(y + 1) - lgammafn(n - y + 1);
    ans += lgammafn(a + b) - lgammafn(a) - lgammafn(b);
    ans -= lgammafn(n + a + b) - lgammafn(a + y) - lgammafn(b + n - y);
    return ans;
  }

  double BetaBinomialModel::logp(int64_t n, int64_t y) const {
    return logp(n, y, a(), b());
  }

  double BetaBinomialModel::loglike(double a, double b) const {
    if (a <= 0 || b <= 0) {
      return BOOM::negative_infinity();
    }
    Vector ab = {a, b};
    Vector g;
    Matrix h;
    return Loglike(ab, g, h, 0);
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
    Vector p_hat;
    Vector counts;
    const auto &data(suf()->count_table());
    for (const auto &el : data) {
      double trials = el.first.first;
      double successes = el.first.second;
      p_hat.push_back(successes / trials);
      counts.push_back(static_cast<double>(el.second));
    }
    double sample_mean = BOOM::mean(p_hat);
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
