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

#ifndef BOOM_ZERO_INFLATED_POISSON_MODEL_HPP_
#define BOOM_ZERO_INFLATED_POISSON_MODEL_HPP_

#include "Models/BinomialModel.hpp"
#include "Models/PoissonModel.hpp"
#include "Models/Policies/ParamPolicy_2.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/Policies/SufstatDataPolicy.hpp"

namespace BOOM {

  class ZeroInflatedPoissonSuf : public SufstatDetails<IntData> {
   public:
    ZeroInflatedPoissonSuf();
    ZeroInflatedPoissonSuf(double number_of_zero_trials,
                           double number_of_positive_trials,
                           double total_number_of_events);
    ZeroInflatedPoissonSuf *clone() const override;

    // Required virtual functions..
    void clear() override;
    void Update(const IntData &) override;
    ZeroInflatedPoissonSuf *abstract_combine(Sufstat *s) override;
    Vector vectorize(bool minimal = true) const override;
    Vector::const_iterator unvectorize(Vector::const_iterator &v,
                                       bool minimal = true) override;
    Vector::const_iterator unvectorize(const Vector &v,
                                       bool minimal = true) override;
    std::ostream &print(std::ostream &) const override;

    void combine(const Ptr<ZeroInflatedPoissonSuf> &);
    void combine(const ZeroInflatedPoissonSuf &rhs);
    void add_mixture_data(double y, double prob);

    double number_of_zeros() const;
    double number_of_positives() const;
    double sum_of_positives() const;
    double mean_of_positives() const;

    void set_values(double nzero, double npos, double sum_of_positives);
    void add_values(double nzero, double npos, double sum_of_positives);

   private:
    // These can be fractional if this model is used as a mixture
    // component in an EM algorith that assigns fractional
    // observations.
    double number_of_zeros_;
    double number_of_positives_;
    double sum_of_positives_;
  };

  class ZeroInflatedPoissonModel
      : virtual public MixtureComponent,
        public ParamPolicy_2<UnivParams, UnivParams>,
        public SufstatDataPolicy<IntData, ZeroInflatedPoissonSuf>,
        public PriorPolicy {
   public:
    explicit ZeroInflatedPoissonModel(double lambda = 1.0,
                                      double zero_prob = 0.5);
    ZeroInflatedPoissonModel(const ZeroInflatedPoissonModel &rhs);
    ZeroInflatedPoissonModel *clone() const override;

    Ptr<UnivParams> Lambda_prm();
    const UnivParams *Lambda_prm() const;
    double lambda() const;
    void set_lambda(double lambda);

    Ptr<UnivParams> ZeroProbability_prm();
    const UnivParams *ZeroProbability_prm() const;
    double zero_probability() const;
    void set_zero_probability(double zp);

    void set_sufficient_statistics(const ZeroInflatedPoissonSuf &suf);

    virtual double pdf(const Ptr<Data> &dp, bool logscale) const;
    double pdf(const Data *dp, bool logscale) const override;
    double logp(int y) const;
    double sim(RNG &rng = GlobalRng::rng) const;
    int number_of_observations() const override { return dat().size(); }

    // Simulates the specified number of trials and returns a structure
    // containing the a summary of the results.
    //
    // Args:
    //   n:  The number of trials to simulate.
    // Returns:
    //   Aggregated data for the all the requested observations.
    ZeroInflatedPoissonSuf sim(int64_t n) const;

   private:
    mutable double log_zero_prob_;
    mutable double log_poisson_prob_;
    mutable bool log_zero_prob_current_;
    std::function<void(void)> create_zero_probability_observer();
    void observe_zero_probability();
    void check_log_probabilities() const;
  };

}  // namespace BOOM

#endif  // BOOM_ZERO_INFLATED_POISSON_MODEL_HPP_
