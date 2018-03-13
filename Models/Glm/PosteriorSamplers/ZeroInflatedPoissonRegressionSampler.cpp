// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2015 Steven L. Scott

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

#include "Models/Glm/PosteriorSamplers/ZeroInflatedPoissonRegressionSampler.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {

  typedef ZeroInflatedPoissonRegressionSampler ZIPRS;

  ZIPRS::ZeroInflatedPoissonRegressionSampler(
      ZeroInflatedPoissonRegressionModel *model,
      const Ptr<VariableSelectionPrior> &poisson_spike,
      const Ptr<MvnBase> &poisson_slab,
      const Ptr<VariableSelectionPrior> &logit_spike,
      const Ptr<MvnBase> &logit_slab, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        poisson_(new PoissonRegressionModel(model_->poisson_coefficient_ptr())),
        logit_(new BinomialLogitModel(model_->logit_coefficient_ptr())),
        poisson_sampler_(new PoissonRegressionSpikeSlabSampler(
            poisson_.get(), poisson_slab, poisson_spike,
            1,  // number of threads
            seeding_rng)),
        logit_sampler_(new BinomialLogitCompositeSpikeSlabSampler(
            logit_.get(), logit_slab, logit_spike,
            5,    // clt threshold
            3,    // tdf
            100,  // tim chunk size
            1,    // rwm chunk size
            1.0,  // rwm scale factor
            seeding_rng)),
        posterior_mode_found_(false) {}

  double ZIPRS::logpri() const {
    double ans = poisson_sampler_->logpri();
    if (ans == negative_infinity()) return ans;
    ans += logit_sampler_->logpri();
    return ans;
  }

  void ZIPRS::draw() {
    impute_forced_zeros(true);
    poisson_sampler_->draw();
    logit_sampler_->draw();
  }

  // Normally one would return the value of the complete data
  // posterior here, but that is expensive to calculate.  Instead we
  // return a bool indicating whether the algorithm converged within a
  // specified number of iterations.
  void ZIPRS::find_posterior_mode(double) {
    try {
      double epsilon = 1e-5;
      double criterion = 1 + epsilon;
      int function_count = 0;
      int function_count_limit = 500;
      Vector logit_coefficients =
          model_->logit_coefficients().included_coefficients();
      Vector poisson_coefficients =
          model_->poisson_coefficients().included_coefficients();
      while (criterion > epsilon && function_count++ < function_count_limit) {
        // E-step of EM
        impute_forced_zeros(false);
        // M-step of EM
        poisson_sampler_->find_posterior_mode();
        logit_sampler_->find_posterior_mode();

        criterion = compute_convergence_criterion(logit_coefficients,
                                                  poisson_coefficients);
        logit_coefficients =
            model_->logit_coefficients().included_coefficients();
        poisson_coefficients =
            model_->poisson_coefficients().included_coefficients();
      }
      posterior_mode_found_ = criterion < epsilon;
    } catch (...) {
      posterior_mode_found_ = false;
    }
  }

  double ZIPRS::compute_convergence_criterion(
      const Vector &logit_coefficients,
      const Vector &poisson_coefficients) const {
    Vector new_logit_coefficients =
        model_->logit_coefficients().included_coefficients();
    Vector percent_change = abs((new_logit_coefficients - logit_coefficients));
    for (int i = 0; i < new_logit_coefficients.size(); ++i) {
      percent_change[i] = logit_coefficients[i] != 0
                              ? percent_change[i] / logit_coefficients[i]
                              : infinity();
    }
    double max_percent_change = max(percent_change);
    if (!std::isfinite(max_percent_change)) {
      return infinity();
    }

    Vector new_poisson_coefficients =
        model_->poisson_coefficients().included_coefficients();
    percent_change = abs((new_poisson_coefficients - poisson_coefficients) /
                         poisson_coefficients);
    max_percent_change =
        std::max(max_percent_change, BOOM::max(percent_change));
    if (!std::isfinite(max_percent_change)) {
      return infinity();
    }
    return max_percent_change;
  }

  void ZIPRS::impute_forced_zeros(bool stochastic) {
    const std::vector<Ptr<ZeroInflatedPoissonRegressionData>> &data(
        model_->dat());
    ensure_latent_data();
    const std::vector<Ptr<PoissonRegressionData>> &poisson_data(
        poisson_->dat());
    const std::vector<Ptr<BinomialRegressionData>> &logit_data(logit_->dat());
    for (int i = 0; i < data.size(); ++i) {
      int64_t total_number_of_zeros = lround(data[i]->number_of_zero_trials());
      if (total_number_of_zeros > 0) {
        // pbinomial will eventually be the probability that this zero
        // is a forced zero.
        //                         p(0 | forced) * pforced
        // pforced | 0 =  --------------------------------------------
        //                 p(0 | forced) * pforced + p(0 | free) * pfree
        //
        // where p(0|forced) = 1.0
        double pforced = model_->probability_forced_to_zero(data[i]->x());
        double pfree = 1.0 - pforced;
        double lambda = model_->poisson_mean(data[i]->x());
        double pforced_given_0 = pforced / (pforced + pfree * dpois(0, lambda));
        if (stochastic) {
          int64_t number_of_binomial_zeros =
              rbinom_mt(rng(), total_number_of_zeros, pforced_given_0);

          // The number of trials for the logit data is the total number
          // of zeros.  This should be set in ensure_latent_data, as it
          // does not vary with the data augmentation.  Note that this
          // might be zero.
          //
          // The notion of 'success' for the binomial model is an
          // observation that is not forced to zero.  Note that the
          // nonzero trials contribute to the binomial nonzeros too.
          int64_t number_of_poisson_observations =
              data[i]->total_number_of_trials() - number_of_binomial_zeros;
          logit_data[i]->set_y(number_of_poisson_observations);

          // The exposure for the poisson data is the total number of
          // trials minus the number of trials that are forced to be
          // zero.  The 'y' for the Poisson data is the sum of the event
          // values, which does not change across the data augmentation,
          // and so should be set in ensure_latent_data.
          poisson_data[i]->set_exposure(number_of_poisson_observations);
        } else {
          // If we're not imputing stochastically, then we're not
          // going to impute an integer number of draws.  Otherwise
          // this branch is the same as the preceding branch, with
          // draws replaced by expectations.
          double expected_number_of_binomial_zeros =
              total_number_of_zeros * pforced_given_0;
          double expected_number_of_poisson_observations =
              data[i]->total_number_of_trials() -
              expected_number_of_binomial_zeros;
          logit_data[i]->set_y(expected_number_of_poisson_observations);
          poisson_data[i]->set_exposure(
              expected_number_of_poisson_observations);
        }
      }
    }
  }

  void ZIPRS::allow_model_selection(bool tf) {
    poisson_sampler_->allow_model_selection(tf);
    logit_sampler_->allow_model_selection(tf);
  }

  void ZIPRS::ensure_latent_data() {
    int64_t number_of_observations = model_->dat().size();
    bool okay = true;
    if (number_of_observations != poisson_->dat().size() ||
        number_of_observations != logit_->dat().size()) {
      okay = false;
    }
    if (okay) {
      // If the sizes match, then pick a random sample of 5
      // observations to see if the X's match.
      for (int j = 0; j < std::min<int>(number_of_observations, 5); ++j) {
        int pos = random_int_mt(rng(), 0, number_of_observations - 1);
        if (model_->dat()[pos]->Xptr() != poisson_->dat()[pos]->Xptr() ||
            model_->dat()[pos]->Xptr() != logit_->dat()[pos]->Xptr()) {
          okay = false;
          break;
        }
      }
    }
    if (!okay) {
      // Either the sizes or X values didn't match, so blow away and
      // regenerate the latent data.
      refresh_latent_data();
    }
  }

  // Replace the data assigned to the poisson_ and logit_ models.
  void ZIPRS::refresh_latent_data() {
    poisson_->clear_data();
    logit_->clear_data();
    const std::vector<Ptr<ZeroInflatedPoissonRegressionData>> &data(
        model_->dat());
    for (int i = 0; i < data.size(); ++i) {
      Ptr<ZeroInflatedPoissonRegressionData> data_point = data[i];
      // The exposure for poisson data will be set by the posterior
      // sampler, unless there are zero trials with zero observations.
      // In that case it is important to set the poisson exposure
      // parameter correctly here as the total number of trials.
      NEW(PoissonRegressionData, poisson_data)
      (data_point->y(), data_point->Xptr(),
       data_point->total_number_of_trials());
      poisson_->add_data(poisson_data);

      // The number of unconstrained trials will be set by the
      // posterior sampler, unless the number of 0's is zero.  In that
      // case, we know there are no constrained observations, so the
      // number_of_unconstrained_trials is equal to the total number
      // of trials.
      int64_t number_of_unconstrained_trials =
          data_point->total_number_of_trials();
      NEW(BinomialRegressionData, logit_data)
      (number_of_unconstrained_trials, data_point->total_number_of_trials(),
       data_point->Xptr());
      logit_->add_data(logit_data);
    }
    poisson_sampler_->assign_data_to_workers();
    logit_sampler_->assign_data_to_workers();
  }

}  // namespace BOOM
