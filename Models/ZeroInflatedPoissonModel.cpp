// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2012 Steven L. Scott

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

#include "Models/ZeroInflatedPoissonModel.hpp"
#include <functional>
#include "Models/SufstatAbstractCombineImpl.hpp"
#include "cpputil/lse.hpp"
#include "distributions.hpp"

namespace BOOM {
  ZeroInflatedPoissonSuf::ZeroInflatedPoissonSuf()
      : number_of_zeros_(0), number_of_positives_(0), sum_of_positives_(0) {}

  ZeroInflatedPoissonSuf::ZeroInflatedPoissonSuf(double nzero, double npos,
                                                 double sum_of_positives)
      : number_of_zeros_(nzero),
        number_of_positives_(npos),
        sum_of_positives_(sum_of_positives) {}

  ZeroInflatedPoissonSuf *ZeroInflatedPoissonSuf::clone() const {
    return new ZeroInflatedPoissonSuf(*this);
  }

  void ZeroInflatedPoissonSuf::clear() {
    number_of_zeros_ = 0;
    number_of_positives_ = 0;
    sum_of_positives_ = 0;
  }

  void ZeroInflatedPoissonSuf::Update(const IntData &data) {
    int y = data.value();
    if (y == 0) {
      ++number_of_zeros_;
    } else {
      ++number_of_positives_;
      sum_of_positives_ += y;
    }
  }

  void ZeroInflatedPoissonSuf::add_mixture_data(double y, double prob) {
    if (lround(y) == 0) {
      number_of_zeros_ += prob;
    } else {
      number_of_positives_ += prob;
      sum_of_positives_ += prob * y;
    }
  }

  ZeroInflatedPoissonSuf *ZeroInflatedPoissonSuf::abstract_combine(Sufstat *s) {
    return abstract_combine_impl(this, s);
  }

  void ZeroInflatedPoissonSuf::combine(const ZeroInflatedPoissonSuf &rhs) {
    number_of_zeros_ += rhs.number_of_zeros_;
    number_of_positives_ += rhs.number_of_positives_;
    sum_of_positives_ += rhs.sum_of_positives_;
  }

  void ZeroInflatedPoissonSuf::combine(const Ptr<ZeroInflatedPoissonSuf> &rhs) {
    combine(*rhs);
  }

  Vector ZeroInflatedPoissonSuf::vectorize(bool) const {
    Vector ans(3);
    ans[0] = number_of_zeros_;
    ans[1] = number_of_positives_;
    ans[2] = sum_of_positives_;
    return ans;
  }

  Vector::const_iterator ZeroInflatedPoissonSuf::unvectorize(
      Vector::const_iterator &v, bool) {
    number_of_zeros_ = *v;
    ++v;
    number_of_positives_ = *v;
    ++v;
    sum_of_positives_ = *v;
    ++v;
    return v;
  }

  Vector::const_iterator ZeroInflatedPoissonSuf::unvectorize(const Vector &v,
                                                             bool minimal) {
    Vector::const_iterator it = v.begin();
    return unvectorize(it, minimal);
  }

  std::ostream &ZeroInflatedPoissonSuf::print(std::ostream &out) const {
    out << "number of zeros:     " << number_of_zeros_ << endl
        << "number of positives: " << number_of_positives_ << endl
        << "sum of positives:    " << sum_of_positives_ << endl;
    return out;
  }

  double ZeroInflatedPoissonSuf::number_of_zeros() const {
    return number_of_zeros_;
  }
  double ZeroInflatedPoissonSuf::number_of_positives() const {
    return number_of_positives_;
  }
  double ZeroInflatedPoissonSuf::sum_of_positives() const {
    return sum_of_positives_;
  }
  double ZeroInflatedPoissonSuf::mean_of_positives() const {
    if (number_of_positives_ == 0) return 0.0;
    return sum_of_positives_ * 1.0 / number_of_positives_;
  }

  void ZeroInflatedPoissonSuf::set_values(double nzero, double npos,
                                          double sum) {
    number_of_zeros_ = nzero;
    number_of_positives_ = npos;
    sum_of_positives_ = sum;
  }

  void ZeroInflatedPoissonSuf::add_values(double nzero, double npos,
                                          double sum) {
    number_of_zeros_ += nzero;
    number_of_positives_ += npos;
    sum_of_positives_ += sum;
  }

  //======================================================================

  ZeroInflatedPoissonModel::ZeroInflatedPoissonModel(double lambda,
                                                     double zero_prob)
      : ParamPolicy(new UnivParams(lambda), new UnivParams(zero_prob)),
        DataPolicy(new ZeroInflatedPoissonSuf),
        log_zero_prob_current_(false) {
    if (lambda <= 0) {
      report_error(
          "lambda must be positive in constructor for "
          "ZeroInflatedPoissonModel.");
    }

    if (zero_prob < 0 || zero_prob > 1) {
      report_error(
          "zero_prob must be between 0 and 1 in constructor for "
          "ZeroInflatedPoissonModel.");
    }
  }

  ZeroInflatedPoissonModel::ZeroInflatedPoissonModel(
      const ZeroInflatedPoissonModel &rhs)
      : Model(rhs),
        MixtureComponent(rhs),
        ParamPolicy(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs),
        log_zero_prob_current_(false) {}

  ZeroInflatedPoissonModel *ZeroInflatedPoissonModel::clone() const {
    return new ZeroInflatedPoissonModel(*this);
  }

  Ptr<UnivParams> ZeroInflatedPoissonModel::Lambda_prm() { return prm1(); }

  const UnivParams *ZeroInflatedPoissonModel::Lambda_prm() const {
    return prm1().get();
  }

  double ZeroInflatedPoissonModel::lambda() const {
    return Lambda_prm()->value();
  }

  void ZeroInflatedPoissonModel::set_lambda(double lambda) {
    Lambda_prm()->set(lambda);
  }

  Ptr<UnivParams> ZeroInflatedPoissonModel::ZeroProbability_prm() {
    return prm2();
  }

  const UnivParams *ZeroInflatedPoissonModel::ZeroProbability_prm() const {
    return prm2().get();
  }

  double ZeroInflatedPoissonModel::zero_probability() const {
    return ZeroProbability_prm()->value();
  }

  void ZeroInflatedPoissonModel::set_zero_probability(double zp) {
    ZeroProbability_prm()->set(zp);
  }

  void ZeroInflatedPoissonModel::set_sufficient_statistics(
      const ZeroInflatedPoissonSuf &s) {
    clear_data();
    suf()->combine(s);
  }

  double ZeroInflatedPoissonModel::pdf(const Ptr<Data> &dp,
                                       bool logscale) const {
    double ans = logp(DAT(dp)->value());
    return logscale ? ans : exp(ans);
  }

  double ZeroInflatedPoissonModel::pdf(const Data *dp, bool logscale) const {
    double ans = logp(DAT(dp)->value());
    return logscale ? ans : exp(ans);
  }

  double ZeroInflatedPoissonModel::logp(int y) const {
    check_log_probabilities();
    double logp1 = log_poisson_prob_ + dpois(y, lambda(), true);
    return y == 0 ? lse2(log_zero_prob_, logp1) : logp1;
  }

  double ZeroInflatedPoissonModel::sim(RNG &rng) const {
    if (runif_mt(rng) < zero_probability()) {
      return 0;
    }
    return rpois_mt(rng, lambda());
  }

  ZeroInflatedPoissonSuf ZeroInflatedPoissonModel::sim(int64_t n) const {
    double number_of_zeros = rbinom(n, zero_probability());
    double number_of_positives = n - number_of_zeros;
    double sum_of_positives = rpois(number_of_positives * lambda());
    return ZeroInflatedPoissonSuf(number_of_zeros, number_of_positives,
                                  sum_of_positives);
  }

  void ZeroInflatedPoissonModel::check_log_probabilities() const {
    if (log_zero_prob_current_) return;
    double p = zero_probability();
    log_zero_prob_ = log(p);
    log_poisson_prob_ = log(1 - p);
    log_zero_prob_current_ = true;
  }

  void ZeroInflatedPoissonModel::observe_zero_probability() {
    log_zero_prob_current_ = false;
  }

  std::function<void(void)>
  ZeroInflatedPoissonModel::create_zero_probability_observer() {
    return [this]() { this->observe_zero_probability(); };
  }

}  // namespace BOOM
