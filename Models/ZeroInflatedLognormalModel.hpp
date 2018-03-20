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

#ifndef BOOM_ZERO_INFLATED_LOGNORMAL_MODEL_HPP_
#define BOOM_ZERO_INFLATED_LOGNORMAL_MODEL_HPP_

#include "Models/BinomialModel.hpp"
#include "Models/DoubleModel.hpp"
#include "Models/EmMixtureComponent.hpp"
#include "Models/GaussianModel.hpp"
#include "Models/Policies/CompositeParamPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"

namespace BOOM {

  class ZeroInflatedLognormalModel : public CompositeParamPolicy,
                                     public PriorPolicy,
                                     public LocationScaleDoubleModel,
                                     public EmMixtureComponent {
   public:
    ZeroInflatedLognormalModel();
    ZeroInflatedLognormalModel(const ZeroInflatedLognormalModel &rhs);
    ZeroInflatedLognormalModel *clone() const override;

    double pdf(const Ptr<Data> &dp, bool logscale) const override;
    double pdf(const Data *, bool logscale) const override;
    double logp(double x) const override;
    double sim(RNG &rng = GlobalRng::rng) const override;

    // This model does not keep copies of the original data set, it
    // uses the sufficient statistics for its component
    // models. instead.
    void add_data(const Ptr<Data> &) override;
    void add_data_raw(double y);
    void add_mixture_data(const Ptr<Data> &dp, double prob) override;
    void add_mixture_data_raw(double y, double prob);
    void clear_data() override;
    void combine_data(const Model &, bool just_suf = true) override;

    void mle() override;

    // Mean and standard deviation of log of the positive observations.
    double mu() const;
    void set_mu(double mu);

    double sigma() const;
    void set_sigma(double sigma);
    void set_sigsq(double sigsq);

    // The probability that an event is greater than zero.
    double positive_probability() const;
    void set_positive_probability(double prob);

    // Moments of the actual observations, including zeros.
    double mean() const override;
    double variance() const override;
    double sd() const;

    Ptr<GaussianModel> Gaussian_model();
    Ptr<BinomialModel> Binomial_model();

    int number_of_observations() const override {
      return lround(binomial_->suf()->nobs());
    }

   private:
    Ptr<GaussianModel> gaussian_;
    Ptr<BinomialModel> binomial_;
    double precision_;

    mutable double log_probability_of_positive_;
    mutable double log_probability_of_zero_;
    mutable bool log_probabilities_are_current_;
    std::function<void(void)> create_binomial_observer();
    void observe_binomial_probability();
    void check_log_probabilities() const;
    Ptr<DoubleData> DAT(const Ptr<Data> &dp) const;
  };
}  // namespace BOOM
#endif  // BOOM_ZERO_INFLATED_LOGNORMAL_MODEL_HPP_
