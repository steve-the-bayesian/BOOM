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

#include "Models/BinomialModel.hpp"

#include <cassert>
#include <cmath>

#include "Bmath/Bmath.hpp"

#include "Models/PosteriorSamplers/BetaBinomialSampler.hpp"
#include "Models/SufstatAbstractCombineImpl.hpp"
#include "cpputil/math_utils.hpp"
#include "distributions.hpp"

namespace BOOM {

  namespace {
    using BS = BOOM::BinomialSuf;
    using BM = BOOM::BinomialModel;
  }  // namespace

  BinomialData::BinomialData(int64_t n, int64_t y) : trials_(n), successes_(y) {
    check_size(n, y);
  }

  BinomialData *BinomialData::clone() const { return new BinomialData(*this); }

  uint BinomialData::size(bool) const { return 2; }

  std::ostream &BinomialData::display(std::ostream &out) const {
    out << "(" << trials_ << ", " << successes_ << ")";
    return out;
  }

  int64_t BinomialData::trials() const { return trials_; }
  int64_t BinomialData::n() const { return trials_; }
  void BinomialData::set_n(int64_t trials) {
    check_size(trials, successes_);
    trials_ = trials;
  }

  int64_t BinomialData::successes() const { return successes_; }
  int64_t BinomialData::y() const { return successes_; }
  void BinomialData::set_y(int64_t successes) {
    check_size(trials_, successes);
    successes_ = successes;
  }

  void BinomialData::increment(int64_t more_trials, int64_t more_successes) {
    if (more_trials < 0 || more_successes < 0 || more_successes > more_trials) {
      report_error("Illegal values passed to increment.");
    }
    trials_ += more_trials;
    successes_ += more_successes;
  }

  void BinomialData::check_size(int64_t n, int64_t y) const {
    if (n < 0 || y < 0) {
      ostringstream err;
      err << "Number of trials and successes must both be non-negative "
          << "in BetaBinomialModel.  You supplied " << endl
          << "trials = " << trials_ << endl
          << "successes = " << successes_ << endl;
      report_error(err.str());
    }
    if (y > n) {
      ostringstream err;
      err << "Number of successes must be less than or equal to the number "
          << "of trials. in BetaBinomialModel.  You supplied" << endl
          << "trials = " << trials_ << endl
          << "successes = " << successes_ << endl;
      report_error(err.str());
    }
  }

  //======================================================================

  BS::BinomialSuf() : sum_(0), nobs_(0) {}

  BS *BS::clone() const { return new BS(*this); }

  void BS::set(double sum, double observation_count) {
    nobs_ = observation_count;
    sum_ = sum;
  }

  void BS::clear() { nobs_ = sum_ = 0; }

  void BS::Update(const BinomialData &d) {
    sum_ += d.successes();
    nobs_ += d.trials();
  }

  void BS::update_raw(double y) {
    sum_ += y;
    nobs_ += 1;
  }

  void BS::batch_update(double n, double y) {
    sum_ += y;
    nobs_ += n;
  }

  void BS::remove(const BinomialData &d) {
    sum_ -= d.y();
    nobs_ -= d.n();
    if (sum_ < 0 || nobs_ < 0) {
      report_error("Removing data caused illegal sufficient statistics.");
    }
  }

  void BS::add_mixture_data(double y, double n, double prob) {
    sum_ += y * prob;
    nobs_ += n * prob;
  }

  double BS::sum() const { return sum_; }
  double BS::nobs() const { return nobs_; }

  void BS::combine(const Ptr<BS> &s) {
    sum_ += s->sum_;
    nobs_ += s->nobs_;
  }
  void BS::combine(const BS &s) {
    sum_ += s.sum_;
    nobs_ += s.nobs_;
  }

  BinomialSuf *BS::abstract_combine(Sufstat *s) {
    return abstract_combine_impl(this, s);
  }

  Vector BS::vectorize(bool) const {
    Vector ans(2);
    ans[0] = sum_;
    ans[1] = nobs_;
    return ans;
  }

  Vector::const_iterator BS::unvectorize(Vector::const_iterator &v, bool) {
    sum_ = *v;
    ++v;
    nobs_ = *v;
    ++v;
    return v;
  }

  Vector::const_iterator BS::unvectorize(const Vector &v, bool minimal) {
    Vector::const_iterator begin = v.begin();
    auto ans = unvectorize(begin, minimal);
    return ans;
  }

  std::ostream &BS::print(std::ostream &out) const {
    return out << sum_ << " " << nobs_;
  }

  BM::BinomialModel(double p)
      : ParamPolicy(new UnivParams(p)), DataPolicy(new BS) {
    observe_prob();
  }

  BM::BinomialModel(const BM &rhs)
      : Model(rhs),
        ParamPolicy(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs),
        NumOptModel(rhs) {
    observe_prob();
  }

  BM &BinomialModel::operator=(const BinomialModel &rhs) {
    if (&rhs != this) {
      Model::operator=(rhs);
      ParamPolicy::operator=(rhs);
      DataPolicy::operator=(rhs);
      PriorPolicy::operator=(rhs);
      NumOptModel::operator=(rhs);
      observe_prob();
    }
    return *this;
  }

  void BM::observe_prob() {
    Prob_prm()->add_observer([this]() {
      log_prob_ = log(prob());
      log_failure_prob_ = ::std::log1p(-prob());
    });
    set_prob(prob());
  }

  BM *BM::clone() const { return new BM(*this); }

  void BM::mle() {
    double n = suf()->nobs();
    set_prob(n > 0 ? suf()->sum() / n : 0.5);
  }

  double BM::prob() const { return Prob_prm()->value(); }

  void BM::set_prob(double p) {
    if (p < 0 || p > 1) {
      std::ostringstream err;
      err << "The argument to BinomialModel::set_prob was " << p
          << ", but a probability must be in the range [0, 1]." << endl;
      report_error(err.str());
    }
    Prob_prm()->set(p);
  }

  Ptr<UnivParams> BM::Prob_prm() { return ParamPolicy::prm(); }
  const Ptr<UnivParams> BM::Prob_prm() const { return ParamPolicy::prm(); }

  double BM::Loglike(const Vector &probvec, Vector &g, Matrix &h,
                     uint nd) const {
    if (probvec.size() != 1) {
      report_error("Wrong size argument.");
    }
    double p = probvec[0];
    if (p < std::numeric_limits<double>::min()
        || (1-p) < std::numeric_limits<double>::min()) {
      return negative_infinity();
    }
    double logp = log(p);
    double logp2 = log(1 - p);

    double ntrials = suf()->nobs();
    double success = suf()->sum();
    double fail = ntrials - success;

    double ans = success * logp + fail * logp2;

    if (nd > 0) {
      double q = 1 - p;
      g[0] = (success - p * ntrials) / (p * q);
      if (nd > 1) {
        h(0, 0) = -1 * (success / (p * p) + fail / (q * q));
      }
    }
    return ans;
  }

  double BM::pdf(double trials, double successes, bool logscale) const {
    if (successes > trials || successes < 0 || trials < 0) {
      return logscale ? BOOM::negative_infinity() : 0;
    }
    return dbinom(successes, trials, prob(), logscale);
  }

  double BM::pdf(const Data *dp, bool logscale) const {
    const BinomialData *data_point = dynamic_cast<const BinomialData *>(dp);
    return pdf(data_point->trials(), data_point->successes(), logscale);
  }

  unsigned int BM::sim(int n, RNG &rng) const {
    return rbinom_mt(rng, n, prob());
  }

  void BM::add_mixture_data(const Ptr<Data> &dp, double prob) {
    Ptr<BinomialData> data_point = DAT(dp);
    suf()->add_mixture_data(data_point->successes(), data_point->trials(),
                            prob);
  }

  void BM::remove_data(const Ptr<Data> &dp) {
    DataPolicy::remove_data(dp);
    suf()->remove(*DAT(dp));
  }

  std::set<Ptr<Data>> BM::abstract_data_set() const {
    return std::set<Ptr<Data>>(dat().begin(), dat().end());
  }

}  // namespace BOOM
