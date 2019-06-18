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

#ifndef BOOM_NONZERO_MEAN_AR1_MODEL_HPP_
#define BOOM_NONZERO_MEAN_AR1_MODEL_HPP_

#include "Models/ParamTypes.hpp"
#include "Models/Policies/ParamPolicy_3.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/Policies/SufstatDataPolicy.hpp"
#include "Models/Sufstat.hpp"

namespace BOOM {

  class Ar1Suf : public SufstatDetails<DoubleData> {
   public:
    Ar1Suf();
    Ar1Suf *clone() const override;
    void clear() override;
    void Update(const DoubleData &d) override;
    void update_raw(double y);
    Ar1Suf *abstract_combine(Sufstat *s) override;
    void combine(const Ptr<Ar1Suf> &);
    void combine(const Ar1Suf &rhs);
    Vector vectorize(bool minimal = true) const override;
    Vector::const_iterator unvectorize(Vector::const_iterator &v,
                                       bool minimal = true) override;
    Vector::const_iterator unvectorize(const Vector &v,
                                       bool minimal = true) override;
    std::ostream &print(std::ostream &out) const override;

    double n() const;

    // (y[0] - mu)^2 + sum( (y[t] - phi*(y[t-1] - mu) - mu))^2
    double model_sumsq(double mu, double phi) const;
    double centered_lag_sumsq(double mu) const;  // sum( (y[t-1] - mu)^2 )
    double lag_sumsq() const;                    // sum(y[t-1]^2)
    double lag_sum() const;                      // sum(y[t-1])
    double sum_excluding_first() const;          // sum(y[t]) - y[0]
    double sumsq_excluding_first() const;        // sum(y[t]^2) - y[0]^2
    double cross() const;                        // sum(y[t] * y[t-1])
    double centered_cross(double mu) const;      // sum((y[t]-mu)(y[t-1]-mu))
    double first_value() const;
    double last_value() const;

   private:
    // All the following sums are over the full data
    double sumsq_;        // sum y[t]^2
    double sum_;          // sum y[t]
    double cross_;        // sum y[t]*y[t-1], assumes y[-1] = 0
    double n_;            // number of observations
    double first_value_;  // y[0]
    double last_value_;   // y[n_-1]
  };

  // This model assumes y[t] ~ N(mu + phi * (y[t-1] - mu), sigsq)
  // with y[0] ~ N(mu, sigsq)
  // The model can also be parameterized as a regression on a lag:
  //      y[t] ~ N( (1-phi)*mu + phi*y[t-1], sigsq)
  // where (1-phi)*mu is the intercept and phi is the slope.
  class NonzeroMeanAr1Model
      : virtual public MLE_Model,
        public ParamPolicy_3<UnivParams, UnivParams, UnivParams>,
        public SufstatDataPolicy<DoubleData, Ar1Suf>,
        public PriorPolicy {
   public:
    explicit NonzeroMeanAr1Model(double mu = 0, double phi = 0,
                                 double sigma = 1);
    explicit NonzeroMeanAr1Model(const Vector &y);
    NonzeroMeanAr1Model(const NonzeroMeanAr1Model &rhs);
    NonzeroMeanAr1Model *clone() const override;

    void mle() override;
    virtual double pdf(const Ptr<Data> &, bool logscale) const;

    double sigma() const;
    double sigsq() const;  // Var(y[t+1] | y[t], phi, mean);
    double phi() const;    // AR1 coefficient
    double mu() const;     // Long run mean, assuming |phi| < 1.

    void set_sigsq(double sigsq);
    void set_phi(double phi);
    void set_mu(double mu);

    Ptr<UnivParams> Mu_prm();
    Ptr<UnivParams> Phi_prm();
    Ptr<UnivParams> Sigsq_prm();
    const Ptr<UnivParams> Mu_prm() const;
    const Ptr<UnivParams> Phi_prm() const;
    const Ptr<UnivParams> Sigsq_prm() const;
  };

}  // namespace BOOM

#endif  // BOOM_NONZERO_MEAN_AR1_MODEL_HPP_
