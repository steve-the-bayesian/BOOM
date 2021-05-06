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

#include "Models/ZeroInflatedLognormalModel.hpp"
#include <functional>
#include <sstream>
#include "Models/PosteriorSamplers/ZeroInflatedLognormalPosteriorSampler.hpp"
#include "Models/SufstatAbstractCombineImpl.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {
  namespace {
    typedef ZeroInflatedLognormalModel ZILM;
  }  // namespace

  ZILM::ZeroInflatedLognormalModel()
      : gaussian_(new GaussianModel),
        binomial_(new BinomialModel),
        precision_(1e-8),
        log_probabilities_are_current_(false) {
    ParamPolicy::add_model(gaussian_);
    ParamPolicy::add_model(binomial_);
    binomial_->Prob_prm()->add_observer(create_binomial_observer());
  }

  ZILM::ZeroInflatedLognormalModel(const ZeroInflatedLognormalModel &rhs)
      : DoubleModel(rhs),
        ParamPolicy(rhs),
        PriorPolicy(rhs),
        EmMixtureComponent(rhs),
        gaussian_(rhs.gaussian_->clone()),
        binomial_(rhs.binomial_->clone()),
        precision_(rhs.precision_) {
    ParamPolicy::add_model(gaussian_);
    ParamPolicy::add_model(binomial_);
    binomial_->Prob_prm()->add_observer(create_binomial_observer());
  }

  ZeroInflatedLognormalModel *ZILM::clone() const {
    return new ZeroInflatedLognormalModel(*this);
  }

  double ZILM::pdf(const Ptr<Data> &dp, bool logscale) const {
    double ans = logp(DAT(dp)->value());
    return logscale ? ans : exp(ans);
  }

  double ZILM::pdf(const Data *dp, bool logscale) const {
    double ans = logp(dynamic_cast<const DoubleData *>(dp)->value());
    return logscale ? ans : exp(ans);
  }

  double ZILM::logp(double x) const {
    check_log_probabilities();
    if (x < precision_) return log_probability_of_zero_;
    return log_probability_of_positive_ + dlnorm(x, mu(), sigma(), true);
  }

  double ZILM::sim(RNG &rng) const {
    if (runif_mt(rng) < positive_probability()) {
      return exp(rnorm_mt(rng, mu(), sigma()));
    }
    return 0;
  }

  void ZILM::add_data(const Ptr<Data> &dp) {
    if (dp->missing()) return;
    Ptr<DoubleData> d = DAT(dp);
    double y = d->value();
    add_data_raw(y);
  }

  void ZILM::add_data_raw(double y) {
    if (y < precision_) {
      binomial_->suf()->update_raw(0.0);
    } else {
      binomial_->suf()->update_raw(1.0);
      gaussian_->suf()->update_raw(log(y));
    }
  }

  void ZILM::add_mixture_data(const Ptr<Data> &dp, double prob) {
    if (dp->missing()) return;
    double y = DAT(dp)->value();
    add_mixture_data_raw(y, prob);
  }

  void ZILM::add_mixture_data_raw(double y, double prob) {
    if (y > precision_) {
      gaussian_->suf()->add_mixture_data(log(y), prob);
      binomial_->suf()->add_mixture_data(1.0, 1.0, prob);
    } else {
      binomial_->suf()->add_mixture_data(0, 1.0, prob);
    }
  }

  void ZILM::clear_data() {
    gaussian_->clear_data();
    binomial_->clear_data();
  }

  void ZILM::combine_data(const Model &rhs, bool just_suf) {
    const ZeroInflatedLognormalModel *rhsp =
        dynamic_cast<const ZeroInflatedLognormalModel *>(&rhs);
    if (!rhsp) {
      ostringstream err;
      err << "ZILM::combine_data was called "
          << "with an argument "
          << "that was not coercible to ZeroInflatedLognormalModel." << endl;
      report_error(err.str());
    } else {
      gaussian_->combine_data(*(rhsp->gaussian_), true);
      binomial_->combine_data(*(rhsp->binomial_), true);
    }
  }

  void ZILM::mle() {
    gaussian_->mle();
    binomial_->mle();
  }

  double ZILM::mu() const { return gaussian_->mu(); }
  void ZILM::set_mu(double mu) { gaussian_->set_mu(mu); }

  double ZILM::sigma() const { return gaussian_->sigma(); }
  void ZILM::set_sigma(double sigma) { gaussian_->set_sigsq(sigma * sigma); }
  void ZILM::set_sigsq(double sigsq) { gaussian_->set_sigsq(sigsq); }

  double ZILM::positive_probability() const { return binomial_->prob(); }
  void ZILM::set_positive_probability(double prob) {
    return binomial_->set_prob(prob);
  }

  double ZILM::mean() const {
    return positive_probability() * exp(mu() + .5 * gaussian_->sigsq());
  }

  double ZILM::variance() const {
    double sigsq = gaussian_->sigsq();
    return expm1(sigsq) * exp(2 * mu() + sigsq);
  }

  double ZILM::sd() const { return sqrt(variance()); }

  Ptr<GaussianModel> ZILM::Gaussian_model() { return gaussian_; }

  Ptr<BinomialModel> ZILM::Binomial_model() { return binomial_; }

  Ptr<DoubleData> ZILM::DAT(const Ptr<Data> &dp) const {
    if (!!dp) return dp.dcast<DoubleData>();
    return Ptr<DoubleData>();
  }

  std::function<void(void)> ZILM::create_binomial_observer() {
    return [this]() { this->observe_binomial_probability(); };
  }

  void ZILM::observe_binomial_probability() {
    log_probabilities_are_current_ = false;
  }

  void ZILM::check_log_probabilities() const {
    if (log_probabilities_are_current_) return;
    log_probability_of_positive_ = log(positive_probability());
    log_probability_of_zero_ = log(1 - positive_probability());
    log_probabilities_are_current_ = true;
  }
}  // namespace BOOM
