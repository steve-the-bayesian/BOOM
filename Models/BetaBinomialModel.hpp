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
#ifndef BOOM_BETA_BINOMIAL_MODEL_HPP
#define BOOM_BETA_BINOMIAL_MODEL_HPP

#include "Models/BinomialModel.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/ParamPolicy_2.hpp"
#include "Models/Policies/PriorPolicy.hpp"

namespace BOOM {
  // BetaBinomialModel describes a setting were binomial data occurs
  // within groups.  Each group has its own binomial success
  // probability drawn from a beta(a, b) distribution.  If the group
  // size is 1 then this is simply the BetaBinomial distribution.

  // The sufficient statistics for the beta binomial model are a collection of
  // BinomialData objects, with a set of counts for each one.
  class BetaBinomialSuf : public SufstatDetails<BinomialData> {
   public:
    using DataTableType = std::map<std::pair<int64_t, int64_t>, int64_t>;

    BetaBinomialSuf();

    BetaBinomialSuf * clone() const override;

    void Update(const BinomialData &dp) override;
    void add_data(int64_t trials, int64_t successes, int64_t counts);

    void clear() override;

    // Returns the sum over all data points of log n[i] choose y[i].
    double log_normalizing_constant() const {return sum_log_normalizing_constants_;}
    int64_t sample_size() const {return sample_size_;}

    const DataTableType &count_table() const {return data_;}

    BetaBinomialSuf *abstract_combine(Sufstat *suf) override;
    void combine(const Ptr<BetaBinomialSuf> &rhs);
    void combine(const BetaBinomialSuf &rhs);

    Vector vectorize(bool minimal = true) const override;
    Vector::const_iterator unvectorize(Vector::const_iterator &v, bool minimal = true) override;
    Vector::const_iterator unvectorize(const Vector &v, bool minimal = true) override;
    std::ostream & print(std::ostream &out) const override;

   private:
    DataTableType data_;
    int64_t sample_size_;

    // Sum of log n[i] choose y[i].  This is the part of the normalizing
    // constant that does not depend on model parameters.
    double sum_log_normalizing_constants_;
  };


  class BetaBinomialModel : public ParamPolicy_2<UnivParams, UnivParams>,
                            public SufstatDataPolicy<BinomialData, BetaBinomialSuf>,
                            public PriorPolicy,
                            public NumOptModel {
   public:
    BetaBinomialModel(double a, double b);

    // Using this constructor will initialize the model with one of
    // three sets of parameters.
    // a) If a call to mle() succeeds then the parameters will be set
    //    using maximum likelihood estimates.
    // b) If the call to mle() fails then the parameters will be set
    //    using a call to method_of_moments().
    // c) If the call to method_of_moments() fails then a and b will
    //    both be set to 1.0.
    // Args:
    //   trials:  The number of trials observed, per group.
    //   successes: The number of successes observed per group (must
    //     be <= the number of trials.)
    BetaBinomialModel(const BOOM::Vector &trials,
                      const BOOM::Vector &successes);
    BetaBinomialModel(const BetaBinomialModel &rhs);
    BetaBinomialModel *clone() const override;

    void clear_data() override;

    // Marking virtual functions as final so they can be called in the
    // constructor.
    void add_data(const Ptr<Data> &dp) final;
    void add_data(const Ptr<BinomialData> &data) final;
    void mle() final { NumOptModel::mle(); }

    // The likelihood contribution for observation i is
    // int Pr(y_i | theta_i, n_i) p(theta_i) dtheta_i
    virtual double loglike() const;
    double loglike(const Vector &ab) const override;
    double loglike(double a, double b) const;
    double Loglike(const Vector &ab, Vector &g, Matrix &h,
                   uint nd) const override;
    static double logp(int64_t n, int64_t y, double a, double b);
    double logp(int64_t n, int64_t y) const;

    // Args:
    //   n: The number of trials for a particular observation.  All trials will
    //     have the same success probability.
    // Returns:
    //   The number of successes for the observation in question.
    int64_t sim(RNG &rng, int64_t n) const;

    Ptr<UnivParams> SuccessPrm();
    Ptr<UnivParams> FailurePrm();
    const Ptr<UnivParams> SuccessPrm() const;
    const Ptr<UnivParams> FailurePrm() const;
    double a() const;
    void set_a(double a);
    double b() const;
    void set_b(double b);

    double prior_mean() const;  // a / a+b
    double mean() const {return prior_mean();}
    void set_prior_mean(double prob);

    double prior_sample_size() const;  // a+b
    void set_prior_sample_size(double sample_size);

    // Set a/(a+b) and a+b using a very rough method of moments
    // estimator.  The estimator can fail if either the sample mean or
    // the sample variance is zero, in which case this function will
    // exit without changing the model.
    void method_of_moments();

    // Print a summary of the model on the stream 'out', and return
    // 'out'.
    std::ostream &print_model_summary(std::ostream &out) const;

   private:
    void check_positive(double arg, const char *function_name) const;
    void check_probability(double arg, const char *function_name) const;
  };

}  // namespace BOOM

#endif  //  BOOM_BETA_BINOMIAL_MODEL_HPP
