// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2008 Steven L. Scott

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

#ifndef BOOM_GAUSSIAN_MODEL_BASE_HPP
#define BOOM_GAUSSIAN_MODEL_BASE_HPP

#include "Models/DataTypes.hpp"
#include "Models/DoubleModel.hpp"
#include "Models/EmMixtureComponent.hpp"
#include "Models/ModelTypes.hpp"
#include "Models/Policies/SufstatDataPolicy.hpp"
#include "Models/Sufstat.hpp"

namespace BOOM {

  class GaussianSuf : public SufstatDetails<DoubleData> {
   public:
    explicit GaussianSuf(double Sum = 0, double Sumsq = 0, double N = 0);
    GaussianSuf(const GaussianSuf &);
    GaussianSuf *clone() const override;

    void clear() override;
    void Update(const DoubleData &X) override;
    void update_raw(double y);

    void update_expected_value(double expected_sample_size,
                               double expected_sum,
                               double expected_sum_of_squares);

    // Remove the effect of observation y from the sufficient
    // statistics, as if it were dropped from the data set.
    void remove(double y);

    // Increment n by prob, sum by prob * y, and sumsq by prob * y^2.
    void add_mixture_data(double y, double prob);

    double sum() const;

    // sumsq returns the uncentered (raw) sum of squared y's: sum(y^2)
    double sumsq() const;

    // centered_sumsq returns sum((y - mu)^2).
    double centered_sumsq(double mu) const;

    double n() const;

    // The sample mean. If there is no data then ybar == 0.
    double ybar() const;
    // The sample variance.  If the sample size is less than 2 then sample_var == 0.
    double sample_var() const;

    GaussianSuf *abstract_combine(Sufstat *s) override;
    void combine(const Ptr<GaussianSuf> &);
    void combine(const GaussianSuf &);
    Vector vectorize(bool minimal = true) const override;
    Vector::const_iterator unvectorize(Vector::const_iterator &v,
                                       bool minimal = true) override;
    Vector::const_iterator unvectorize(const Vector &v,
                                       bool minimal = true) override;
    std::ostream &print(std::ostream &out) const override;

   private:
    double sum_, sumsq_, n_;
  };
  //======================================================================
  class GaussianModelBase
      : public SufstatDataPolicy<DoubleData, GaussianSuf>,
        public DiffDoubleModel,           // promises  Logp(x,g,h,nd);
        public LocationScaleDoubleModel,  // mean and variance.
        public NumOptModel,  // promises Loglike(g,h,nd), and mle();
        virtual public EmMixtureComponent,  // promises add_mixture_data
        virtual public DirichletProcessMixtureComponent  // remove_data
  {
   public:
    GaussianModelBase();
    explicit GaussianModelBase(const std::vector<double> &y);
    GaussianModelBase *clone() const override = 0;

    // Returns the mean of the distribution.
    virtual double mu() const = 0;

    // Variance of the distribution.
    virtual double sigsq() const = 0;

    // Standard deviation of the distribution.
    virtual double sigma() const;

    double mean() const override { return mu(); }
    double variance() const override { return sigsq(); }
    double sd() const {return sqrt(variance());}

    double pdf(const Ptr<Data> &dp, bool logscale) const override;
    double pdf(const Data *dp, bool logscale) const override;
    double Logp(double x, double &g, double &h, uint nd) const override;
    double Logp(const Vector &x, Vector &g, Matrix &h, uint nd) const;

    int number_of_observations() const override { return dat().size(); }

    // Sample moments of data assigned to the model.
    double ybar() const;
    double sample_var() const;

    void add_mixture_data(const Ptr<Data> &, double prob) override;
    double sim(RNG &rng = GlobalRng::rng) const override;

    void add_data_raw(double x);
    void remove_data(const Ptr<Data> &dp) override;
    std::set<Ptr<Data>> abstract_data_set() const override;

    // Log of the integrated Gaussian likelihood, conditional on sigma^2,
    // assuming mu ~ N(mu0, tausq).
    //
    // Args:
    //   suf: The sufficient statistics for the vector of y's over which to
    //     evaluate likelihood.
    //   mu0:  The mean of the prior distribution for mu.
    //   tausq:  The variance fo the prior distribution for mu.
    //   sigsq:  The variance of the data.
    //
    // Returns:
    //   The log of \int p(y | \mu, \sigma^2) p(\mu |\mu_0, \tausq) d \mu.
    static double log_integrated_likelihood(const GaussianSuf &suf,
                                            double mu0,
                                            double tausq,
                                            double sigsq);

    // Log of the integrated Gaussian likelihood, assuming y ~ N(mu, sigma^2)
    // with mu|sigma ~ N(mu0, sigma^2 / kappa) and 1/sigma^2 ~ Ga(df/2, ss/2).
    //
    // Args:
    //   suf: The sufficient statistics for the vector of y's over which to
    //     evaluate likelihood.
    //   mu0:  The mean of the prior distribution for mu.
    //   kappa:  The 'prior sample size' in the prior distribution for mu.
    //   df:  The prior degrees of freedom for sigsq.
    //   ss:  The prior sum of squares for sigsq.
    //
    // Returns:
    //  The log of \int p(y | \mu, \sigma^2) p(\mu | \sigma^2) p(1/ \sigma^2) 
    //     d \mu  d 1/sigma^2
    static double log_integrated_likelihood(const GaussianSuf &suf,
                                            double mu0,
                                            double kappa,
                                            double df,
                                            double ss);

    static double log_likelihood(const GaussianSuf &suf, double mu,
                                 double sigsq);
  };

}  // namespace BOOM
#endif  // BOOM_GAUSSIAN_MODEL_BASE_HPP
