// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2013 Steven L. Scott

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

#include "Models/ZeroInflatedGammaModel.hpp"
#include <functional>
#include <sstream>
#include "Models/PosteriorSamplers/ZeroInflatedGammaPosteriorSampler.hpp"
#include "Models/SufstatAbstractCombineImpl.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {
  namespace {
    typedef ZeroInflatedGammaModel ZIGM;
  }  // namespace

  ZIGM::ZeroInflatedGammaModel()
      : gamma_(new GammaModel),
        binomial_(new BinomialModel),
        zero_threshold_(1e-8),
        log_probabilities_are_current_(false) {
    setup();
  }

  ZIGM::ZeroInflatedGammaModel(const Ptr<BinomialModel> &positive_probability,
                               const Ptr<GammaModel> &positive_density)
      : gamma_(positive_density),
        binomial_(positive_probability),
        zero_threshold_(1e-8),
        log_probabilities_are_current_(false) {
    setup();
  }

  ZIGM::ZeroInflatedGammaModel(int number_of_zeros, int number_of_positives,
                               double sum_of_positives,
                               double sum_of_logs_of_positives)
      : gamma_(new GammaModel),
        binomial_(new BinomialModel),
        zero_threshold_(1e-8),
        log_probabilities_are_current_(false) {
    if (sum_of_positives == 0 &&
        (sum_of_logs_of_positives != 0 || number_of_positives != 0)) {
      report_error(
          "If sum_of_positives is zero, then sum_of_log_positives and "
          "number_of_positives must also be zero.");
    }
    gamma_->suf()->set(sum_of_positives, sum_of_logs_of_positives,
                       number_of_positives);
    binomial_->suf()->set(number_of_positives,
                          number_of_positives + number_of_zeros);

    if (number_of_positives > 0 && number_of_zeros > 0) {
      // The binomial model has a closed form MLE.
      binomial_->mle();
    }
    if (number_of_positives > 1) {
      try {
        gamma_->mle();
      } catch (...) {
        report_warning("Warning:  failed to set gamma model to its MLE.");
      }
    }
  }

  ZIGM::ZeroInflatedGammaModel(const ZeroInflatedGammaModel &rhs)
      : DoubleModel(rhs),
        ParamPolicy(rhs),
        PriorPolicy(rhs),
        gamma_(rhs.gamma_->clone()),
        binomial_(rhs.binomial_->clone()),
        zero_threshold_(rhs.zero_threshold_),
        log_probabilities_are_current_(false) {
    setup();
  }

  ZeroInflatedGammaModel *ZIGM::clone() const {
    return new ZeroInflatedGammaModel(*this);
  }

  double ZIGM::pdf(const Ptr<Data> &dp, bool logscale) const {
    double ans = logp(DAT(dp)->value());
    return logscale ? ans : exp(ans);
  }

  double ZIGM::pdf(const Data *dp, bool logscale) const {
    double ans = logp(dynamic_cast<const DoubleData *>(dp)->value());
    return logscale ? ans : exp(ans);
  }

  double ZIGM::logp(double x) const {
    check_log_probabilities();
    if (x < zero_threshold_) return log_probability_of_zero_;
    return log_probability_of_positive_ + gamma_->logp(x);
  }

  double ZIGM::sim(RNG &rng) const {
    if (runif_mt(rng) < positive_probability()) {
      return gamma_->sim(rng);
    }
    return 0;
  }

  void ZIGM::add_data(const Ptr<Data> &dp) {
    if (dp->missing()) return;
    Ptr<DoubleData> d = DAT(dp);
    double y = d->value();
    add_data_raw(y);
  }

  void ZIGM::add_data_raw(double y) {
    if (y < zero_threshold_) {
      // The binomial "success" is a positive value, so this is a
      // failure.
      binomial_->suf()->update_raw(0.0);
    } else {
      // The binomial "success" is a positive value, so this is a
      // success.
      binomial_->suf()->update_raw(1.0);
      gamma_->suf()->update_raw(y);
    }
  }

  void ZIGM::add_mixture_data_raw(double y, double prob) {
    if (y < zero_threshold_) {
      binomial_->suf()->add_mixture_data(0, 1, prob);
    } else {
      binomial_->suf()->add_mixture_data(1.0, 1, prob);
      gamma_->suf()->add_mixture_data(y, prob);
    }
  }

  void ZIGM::clear_data() {
    gamma_->clear_data();
    binomial_->clear_data();
  }

  void ZIGM::combine_data(const Model &rhs, bool just_suf) {
    const ZeroInflatedGammaModel *rhsp =
        dynamic_cast<const ZeroInflatedGammaModel *>(&rhs);
    if (!rhsp) {
      ostringstream err;
      err << "ZIGM::combine_data was called "
          << "with an argument "
          << "that was not coercible to ZeroInflatedGammaModel." << endl;
      report_error(err.str());
    } else {
      gamma_->combine_data(*(rhsp->gamma_), true);
      binomial_->combine_data(*(rhsp->binomial_), true);
    }
  }

  void ZIGM::mle() {
    gamma_->mle();
    binomial_->mle();
  }

  double ZIGM::positive_probability() const { return binomial_->prob(); }

  void ZIGM::set_positive_probability(double prob) {
    binomial_->set_prob(prob);
    log_probability_of_positive_ = log(prob);
    log_probability_of_zero_ = log(1 - prob);
    log_probabilities_are_current_ = true;
  }

  double ZIGM::mean_parameter() const { return gamma_->mean(); }

  void ZIGM::set_mean_parameter(double mu) {
    gamma_->set_shape_and_mean(gamma_->alpha(), mu);
  }

  double ZIGM::shape_parameter() const { return gamma_->alpha(); }

  void ZIGM::set_shape_parameter(double a) { gamma_->set_alpha(a); }

  double ZIGM::scale_parameter() const { return gamma_->beta(); }

  double ZIGM::mean() const {
    return positive_probability() * mean_parameter();
  }

  double ZIGM::variance() const {
    double p = positive_probability();
    return p * square(mean_parameter()) * (1 - p + (1 / shape_parameter()));
  }

  double ZIGM::sd() const { return sqrt(variance()); }

  Ptr<GammaModel> ZIGM::Gamma_model() { return gamma_; }

  Ptr<BinomialModel> ZIGM::Binomial_model() { return binomial_; }

  Ptr<DoubleData> ZIGM::DAT(const Ptr<Data> &dp) const {
    if (!dp) return Ptr<DoubleData>();
    return dp.dcast<DoubleData>();
  }

  std::function<void(void)> ZIGM::create_binomial_observer() {
    return [this]() { this->observe_binomial_probability(); };
  }

  void ZIGM::observe_binomial_probability() {
    log_probabilities_are_current_ = false;
  }

  void ZIGM::check_log_probabilities() const {
    if (log_probabilities_are_current_) return;
    log_probability_of_positive_ = log(positive_probability());
    log_probability_of_zero_ = log(1 - positive_probability());
    log_probabilities_are_current_ = true;
  }

  void ZIGM::setup() {
    ParamPolicy::add_model(gamma_);
    ParamPolicy::add_model(binomial_);
    binomial_->Prob_prm()->add_observer(create_binomial_observer());
  }
}  // namespace BOOM
