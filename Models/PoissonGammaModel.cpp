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

#include "Models/PoissonGammaModel.hpp"
#include "Bmath/Bmath.hpp"
#include "cpputil/report_error.hpp"
#include "stats/moments.hpp"

namespace BOOM {
  using Rmath::digamma;
  using Rmath::trigamma;

  PoissonData::PoissonData(int trials, int events)
      : trials_(trials), events_(events) {
    check_legal_values();
  }

  PoissonData::PoissonData(const PoissonData &rhs) : Data(rhs) {
    trials_ = rhs.trials_;
    events_ = rhs.events_;
  }

  PoissonData *PoissonData::clone() const { return new PoissonData(*this); }

  PoissonData &PoissonData::operator=(const PoissonData &rhs) {
    if (&rhs != this) {
      Data::operator=(rhs);
      trials_ = rhs.trials_;
      events_ = rhs.events_;
    }
    return *this;
  }

  bool PoissonData::operator==(const PoissonData &rhs) const {
    return (trials_ == rhs.trials_) && (events_ == rhs.events_);
  }

  bool PoissonData::operator!=(const PoissonData &rhs) const {
    return !((*this) == rhs);
  }

  uint PoissonData::size(bool) const { return 2; }

  std::ostream &PoissonData::display(std::ostream &out) const {
    out << "[" << trials_ << ", " << events_ << "]";
    return out;
  }

  int PoissonData::number_of_trials() const { return trials_; }

  int PoissonData::number_of_events() const { return events_; }

  void PoissonData::set_number_of_trials(int n) {
    trials_ = n;
    check_legal_values();
  }

  void PoissonData::set_number_of_events(int n) {
    events_ = n;
    check_legal_values();
  }

  void PoissonData::check_legal_values() {
    if (trials_ < 0 || events_ < 0) {
      report_error(
          "Both 'trials' and 'events' must be non-negative in "
          "the PoissonData constructor.");
    }
    if (trials_ == 0 && events_ != 0) {
      report_error("If you have zero trials, you must also have zero events.");
    }
  }

  //======================================================================

  PoissonGammaModel::PoissonGammaModel(double a, double b)
      : ParamPolicy(new UnivParams(a), new UnivParams(b)) {}

  PoissonGammaModel::PoissonGammaModel(const std::vector<int> &number_of_trials,
                                       const std::vector<int> &number_of_events)
      : ParamPolicy(new UnivParams(1.0), new UnivParams(1.0)) {
    if (number_of_events.size() != number_of_trials.size()) {
      report_error(
          "The number_of_trials and number_of_events arguments must "
          "have the same size.");
    }
    int n = number_of_events.size();
    for (int i = 0; i < n; ++i) {
      NEW(PoissonData, dp)(number_of_trials[i], number_of_events[i]);
      add_data(dp);
    }
    try {
      mle();
    } catch (...) {
      method_of_moments();
    }
    if (a() < .1) {
      set_a(.1);
    }
    if (b() < .1) {
      set_b(.1);
    }
  }

  PoissonGammaModel::PoissonGammaModel(const PoissonGammaModel &rhs)
      : Model(rhs),
        ParamPolicy(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs),
        NumOptModel(rhs) {}

  PoissonGammaModel *PoissonGammaModel::clone() const {
    return new PoissonGammaModel(*this);
  }

  double PoissonGammaModel::loglike() const { return this->loglike(a(), b()); }

  double PoissonGammaModel::loglike(const Vector &ab) const {
    return loglike(ab[0], ab[1]);
  }

  double PoissonGammaModel::loglike(double a, double b) const {
    const std::vector<Ptr<PoissonData> > &data(dat());
    int nobs = data.size();
    double ans = nobs * (a * log(b) - lgamma(a));
    for (int i = 0; i < nobs; ++i) {
      double apy = a + data[i]->number_of_events();
      double npb = b + data[i]->number_of_trials();
      ans += lgamma(apy) - apy * log(npb);
    }
    return ans;
  }

  double PoissonGammaModel::Loglike(const Vector &ab, Vector &g, Matrix &H,
                                    uint nd) const {
    if (ab.size() != 2) {
      report_error("Wrong size argument.");
    }
    double a = ab[0];
    double b = ab[1];
    const std::vector<Ptr<PoissonData> > &data(dat());
    int nobs = data.size();

    // Initialize ans with the part that does not depend on the data.
    double ans = nobs * (a * log(b) - lgamma(a));

    // If derivatives are requested fill in g and H with the parts
    // that don't depend on the data.
    if (nd > 0) {
      g[0] = nobs * (-digamma(a) + log(b));
      g[1] = nobs * a / b;
      if (nd > 1) {
        H(0, 0) = -nobs * trigamma(a);
        H(1, 0) = nobs / b;
        H(0, 1) = nobs / b;
        H(1, 1) = -nobs * a / (b * b);
      }
    }

    for (int i = 0; i < nobs; ++i) {
      double apy = a + data[i]->number_of_events();
      double npb = b + data[i]->number_of_trials();
      ans += lgamma(apy) - apy * log(npb);
      if (nd > 0) {
        g[0] += digamma(apy) - log(npb);
        g[1] -= apy / (npb);
        if (nd > 1) {
          H(0, 0) += trigamma(apy);
          H(1, 0) -= 1.0 / npb;
          H(0, 1) -= 1.0 / npb;
          H(1, 1) += apy / (npb * npb);
        }
      }
    }
    return ans;
  }

  Ptr<UnivParams> PoissonGammaModel::Alpha_prm() { return ParamPolicy::prm1(); }

  Ptr<UnivParams> PoissonGammaModel::Beta_prm() { return ParamPolicy::prm2(); }

  double PoissonGammaModel::a() const {
    return ParamPolicy::prm1_ref().value();
  }

  double PoissonGammaModel::b() const {
    return ParamPolicy::prm2_ref().value();
  }

  void PoissonGammaModel::set_a(double a) {
    if (a <= 0) {
      report_error("Argument must be positive in PoissonGammaModel::set_a.");
    }
    Alpha_prm()->set(a);
  }

  void PoissonGammaModel::set_b(double b) {
    if (b <= 0) {
      report_error("Argument must be positive in PoissonGammaModel::set_b.");
    }
    Beta_prm()->set(b);
  }

  double PoissonGammaModel::prior_mean() const { return a() / b(); }

  double PoissonGammaModel::prior_sample_size() const { return b(); }

  void PoissonGammaModel::set_prior_mean_and_sample_size(double prior_mean,
                                                         double sample_size) {
    double b = sample_size;
    double a = sample_size * prior_mean;
    set_a(a);
    set_b(b);
  }

  // An estimate of the prior mean is the grand mean ybar.  An
  // estimate of the sample size is the variance of empirical means
  // around ybar.
  void PoissonGammaModel::method_of_moments() {
    Vector lambda;
    const std::vector<Ptr<PoissonData> > &data(dat());
    int nobs = data.size();
    lambda.reserve(nobs);
    for (int i = 0; i < nobs; ++i) {
      double yi = data[i]->number_of_events();
      int ni = data[i]->number_of_trials();
      if (ni > 0) {
        lambda.push_back(yi / ni);
      }
    }
    if (lambda.size() > 1) {
      double sample_mean = mean(lambda);
      double sample_variance = var(lambda);
      if (sample_variance == 0.0 || sample_mean == 0.0) {
        return;
      }
      set_prior_mean_and_sample_size(sample_mean,
                                     sample_mean / sample_variance);
    }
  }

}  // namespace BOOM
