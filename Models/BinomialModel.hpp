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
#ifndef BOOM_BINOMIAL_MODEL_HPP
#define BOOM_BINOMIAL_MODEL_HPP

#include <cstdint>
#include "Models/EmMixtureComponent.hpp"
#include "Models/ModelTypes.hpp"
#include "Models/Policies/ParamPolicy_1.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/Policies/SufstatDataPolicy.hpp"
#include "Models/Sufstat.hpp"

namespace BOOM {

  // Forward declaration of the conjugate sampler for this model.
  class BetaBinomialSampler;

  class BinomialData : public Data {
   public:
    BinomialData() : trials_(0), successes_(0) {}
    BinomialData(int64_t n, int64_t y);
    BinomialData *clone() const override;

    virtual uint size(bool minimal = true) const;
    std::ostream &display(std::ostream &) const override;

    int64_t trials() const;
    int64_t n() const;
    void set_n(int64_t trials);

    int64_t y() const;
    int64_t successes() const;
    void set_y(int64_t successes);

    // Add to the success and trial counts.
    // Args:
    //   more_trials:  The number of additional trials.
    //   more_successes:  The number of additional successes.
    void increment(int64_t more_trials, int64_t more_successes);

   private:
    int64_t trials_;
    int64_t successes_;

    void check_size(int64_t n, int64_t y) const;
  };

  class BinomialSuf : public SufstatDetails<BinomialData> {
   public:
    BinomialSuf();
    BinomialSuf *clone() const override;
    void set(double sum, double observation_count);

    double sum() const;
    double nobs() const;
    void clear() override;
    void Update(const BinomialData &) override;
    void update_raw(double y);
    void batch_update(double n, double y);

    void remove(const BinomialData &d);

    void add_mixture_data(double y, double n, double prob);

    BinomialSuf *abstract_combine(Sufstat *s) override;
    void combine(const Ptr<BinomialSuf> &);
    void combine(const BinomialSuf &);

    Vector vectorize(bool minimal = true) const override;
    Vector::const_iterator unvectorize(Vector::const_iterator &v,
                                       bool minimal = true) override;
    Vector::const_iterator unvectorize(const Vector &v,
                                       bool minimal = true) override;
    std::ostream &print(std::ostream &out) const override;

   private:
    double sum_, nobs_;
  };

  class BinomialModel : public ParamPolicy_1<UnivParams>,
                        public SufstatDataPolicy<BinomialData, BinomialSuf>,
                        public PriorPolicy,
                        public NumOptModel,
                        public EmMixtureComponent,
                        public ConjugateDirichletProcessMixtureComponent {
   public:
    explicit BinomialModel(double p = .5);
    BinomialModel(const BinomialModel &rhs);
    BinomialModel(BinomialModel &&rhs) = default;
    BinomialModel &operator=(const BinomialModel &rhs);
    BinomialModel &operator=(BinomialModel &&rhs) = default;
    BinomialModel *clone() const override;

    void mle() override;
    double Loglike(const Vector &probvec, Vector &g, Matrix &h,
                   uint nd) const override;

    double log_prob() const { return log_prob_; }
    double log_failure_prob() const { return log_failure_prob_; }
    double prob() const;
    void set_prob(double p);

    double pdf(const Data *dp, bool logscale) const override;
    double pdf(double trials, double successes, bool logscale) const;

    Ptr<UnivParams> Prob_prm();
    const Ptr<UnivParams> Prob_prm() const;
    unsigned int sim(int n, RNG &rng = GlobalRng::rng) const;

    void add_mixture_data(const Ptr<Data> &, double prob) override;

    void remove_data(const Ptr<Data> &dp) override;
    std::set<Ptr<Data>> abstract_data_set() const override;

    int number_of_observations() const override { return dat().size(); }

    double log_likelihood() const override {
      return LoglikeModel::log_likelihood();
    }

   private:
    double log_prob_;
    double log_failure_prob_;

    // Sets an observer on the Prob_prm parameter, so that log_prob_ and
    // log_failure_prob_ are updated whenever Prob_prm is changed.
    void observe_prob();
  };

}  // namespace BOOM

#endif  // BOOM_BINOMIAL_MODEL_HPP
