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

#ifndef BOOM_POISSON_GAMMA_MODEL_HPP_
#define BOOM_POISSON_GAMMA_MODEL_HPP_

#include "Models/DataTypes.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/ParamPolicy_2.hpp"
#include "Models/Policies/PriorPolicy.hpp"

namespace BOOM {

  class PoissonData : public Data {
   public:
    explicit PoissonData(int trials = 0, int events = 0);
    PoissonData(const PoissonData &rhs);
    PoissonData *clone() const override;
    PoissonData &operator=(const PoissonData &rhs);
    bool operator==(const PoissonData &rhs) const;
    bool operator!=(const PoissonData &rhs) const;

    virtual uint size(bool minimal = true) const;
    std::ostream &display(std::ostream &out) const override;

    int number_of_trials() const;
    int number_of_events() const;

    void set_number_of_trials(int n);
    void set_number_of_events(int n);

   private:
    int trials_;
    int events_;
    void check_legal_values();
  };

  // The PoissonGammaModel describes a series of observations from a
  // series of Poisson distributions.  Each group of observations
  // (call it group i) has an externally defined sample size (i.e. the
  // sample size is not modeled here) and a number of events y[i]
  // drawn from a Poisson distribution with mean lambda[i].  The
  // sequence of lambda[i]'s is IID from a Gamma(a, b) distribution
  // (with mean a/b and variance a/b^2).  The parameters can be
  // interpreted as 'prior_mean = a/b' and a 'prior_sample_size = b'.
  class PoissonGammaModel : public ParamPolicy_2<UnivParams, UnivParams>,
                            public IID_DataPolicy<PoissonData>,
                            public PriorPolicy,
                            public NumOptModel {
   public:
    explicit PoissonGammaModel(double a = 1.0, double b = 1.0);

    // This constructor will attempt to initialize the model at the
    // MLE.  If that fails then it will attempt to initialize using a
    // method of moments estimate.  If that fails it will initialize
    // using parameter values from the default constructor.
    PoissonGammaModel(const std::vector<int> &number_of_trials,
                      const std::vector<int> &number_of_events);
    PoissonGammaModel(const PoissonGammaModel &rhs);
    PoissonGammaModel *clone() const override;

    virtual double loglike() const;
    double loglike(const Vector &ab) const override;
    double loglike(double a, double b) const;
    double Loglike(const Vector &ab, Vector &g, Matrix &H,
                   uint nd) const override;

    Ptr<UnivParams> Alpha_prm();
    Ptr<UnivParams> Beta_prm();
    double a() const;
    double b() const;
    void set_a(double a);
    void set_b(double b);

    double prior_mean() const;         //    a/b
    double prior_sample_size() const;  //    b

    void set_prior_mean_and_sample_size(double prior_mean,
                                        double prior_sample_size);

    // Set a/b and b using a very rough method of moments estimator.
    // The estimator can fail if either the sample mean or the sample
    // variance is zero, in which case the function will exit without
    // changing the model.
    void method_of_moments();
  };

}  // namespace BOOM

#endif  //  BOOM_POISSON_GAMMA_MODEL_HPP_
