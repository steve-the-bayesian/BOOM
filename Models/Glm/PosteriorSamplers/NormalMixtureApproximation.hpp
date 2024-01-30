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

#ifndef BOOM_NORMAL_MIXTURE_APPROXIMATION_HPP_
#define BOOM_NORMAL_MIXTURE_APPROXIMATION_HPP_

#include "LinAlg/Vector.hpp"
#include "distributions/Rmath_dist.hpp"
#include "distributions/rng.hpp"
#include "numopt.hpp"
#include "math/Permutation.hpp"

namespace BOOM {

  // A NormalMixtureApproximation is a finite mixture approximation to
  // a specific distribution.  The mixture approximation is determined
  // by a set of mixing weights w (non-negative numbers that sum to
  // 1), a vector of means mu, and a vector of standard deviations
  // sigma.  Then the distribution is approximated by
  //
  //       f(x) \approx sum_k w[k] N(x | mu[k], sigma[k]).
  class NormalMixtureApproximation {
   public:
    // Use this constructor for an empty NormalMixtureApproximation
    // with n mixture components.
    explicit NormalMixtureApproximation(int n);

    // Use this constructor for a NormalMixtureApproximation with a
    // specified set of mixture components and mixing weights.
    // Args:
    //   mu:  Means of the mixture components.
    //   sigma:  Standard deviations of the mixture components.
    //   weights: A discrete probability distribution giving the mixing weights
    //     for the mixture components.
    //
    // The lengths of mu, sigma, and weights must all be the same.
    NormalMixtureApproximation(const Vector &mu, const Vector &sigma,
                               const Vector &weights);

    // Use this constructor to find the approximation that best fits a
    // specific target distribution.
    // Args:
    //   log_target_density: The log of the distribution to be approximated.
    //   initial_mu: A vector of initial values to use for the means of the
    //     approximating mixture components.
    //   initial_sigma: A vector of standard deviations to use as the initial
    //     values for the approximating mixture components.
    //   initial_weights: A vector to use as the initial values of the mixing
    //     weights.
    //   precision: The fitting algorithm will stop if the Kullback-Leibler
    //     divergence between target and the approximating mixture is less than
    //     this number.
    //   max_evals: The maximum number of trials allowed for the fitting
    //     algorithm.
    //   initial_stepsize: A parameter passed to the NEWUOA fitting algorithm.
    //   force_zero_mu: If true then then all mixture components will be forced
    //     to have zero mean.
    NormalMixtureApproximation(
        const ScalarTarget &log_target_density, const Vector &initial_mu,
        const Vector &initial_sigma, const Vector &initial_weights,
        double precision = 1e-6,  // the precision of the final KL divergence
        int max_evals = 20000,    // max number of times to evaluate logf
        double initial_stepsize = 10.0, bool force_zero_mu = false);

    // Set the mixing weights, means, and standard deviations to the
    // specified values.
    void set(const Vector &mu, const Vector &sigma, const Vector &weights);

    // If the dimension of the approximation is k, then the first k elements of
    // theta are the k mu's, then the k values of log(sigma), then the k-1
    // values of log(weights / weights[0]).
    void set(const Vector &theta);

    // The number of mixture components used in the approximation.
    int dim() const { return mu_.size(); }

    const Vector &mu() const { return mu_; }
    const Vector &sigma() const { return sigma_; }
    const Vector &weights() const { return weights_; }
    const Vector &log_weights() const { return log_weights_; }

    // Return the log of the approximating normal mixture density at x.
    double logp(double x) const;

    // For a particular observation u drawn from the distribution being
    // approximated, take a random draw of the mixture component that generated
    // it.
    void unmix(RNG &rng, double u, double *mu, double *sigsq) const;

    // The Kullback Leibler distance to the distribution being approximated.
    double kullback_leibler() const;

    // Compute the Kullback Leibler distance to the specified target.  Records
    // the answer in state.
    double kullback_leibler(const ScalarTarget &target);

    // If the approximation was initialized by the optimization constructor,
    // this method returns the number of function evaluations necessary to
    // arrive at the current approximation.  If initialized using another
    // constructor then -1 is returned.
    int number_of_function_evaluations() const;

    std::ostream &print(std::ostream &out) const;

    Vector serialize() const;
    Vector::const_iterator deserialize(Vector::const_iterator begin);

   private:
    // Ensures that mu, sigma, and weights are all the correct size.
    // Calls report_error if they are not.
    void check_sizes();

    // Ensures:
    //   (1) that all values of mu, sigma, and weights are finite,
    //   (2) that all values of sigma and weights are positive,
    //   (3) that weights sums to 1.
    // If the sum of the weights is off by less than .001 the weights
    // are corrected and the log weights are recomputed..  Otherwise
    // If any of these is not satisfied then report_error is called.
    void check_values();

    // Apply the given permutation to the mixture components.
    // Args:
    //   permutation: An arrangement of the numbers 0, 1, 2,
    //   ... dim()-1 representing the new order of the mixture
    //   components.
    void set_order(const Permutation<Int> &permutation);
    void order_by_mu();
    void order_by_sigma();
    Vector mu_;
    Vector sigma_;
    Vector weights_;
    Vector log_weights_;
    bool force_zero_mu_;

    // The kl distance between the approximation and the target.  This
    // is set automatically by the constructor that takes the target.
    // It is set to -infinity by the other constructors and there is
    // no way to modify it.
    double kullback_leibler_;
    int number_of_function_evaluations_;
  };

  inline std::ostream &operator<<(std::ostream &out,
                             const NormalMixtureApproximation &approximation) {
    return approximation.print(out);
  }

  //======================================================================
  // A finite normal mixture approximation to the standard logistic distribution.
  class LogitMixtureApproximation : public NormalMixtureApproximation {
   public:
    LogitMixtureApproximation();
  };

  //======================================================================

  // A table of NormalMixtureApproximations.
  class NormalMixtureApproximationTable {
   public:
    NormalMixtureApproximationTable();
    ~NormalMixtureApproximationTable() {}
    NormalMixtureApproximationTable(const NormalMixtureApproximationTable &rhs);
    NormalMixtureApproximationTable &operator=(
        const NormalMixtureApproximationTable &rhs);

    // Add an entry into the table.
    void add(int index, const NormalMixtureApproximation &spec);

    // Return the smallest and lagest indices contained in the table.
    int smallest_index() const;
    int largest_index() const;

    // If a numerical approximation exists at index_value, then return it.
    // Otherwise return an interpolation between the two nearest indices.
    NormalMixtureApproximation &approximate(int index_value);

    // Save the current state of the table to a numeric vector that can be
    // stored somewhere.
    Vector serialize() const;

    // Clear the current contents of the table, and replace it with the values
    // read from serialized_state.
    void deserialize(const Vector &serialized_state);

   private:
    std::vector<int> index_;

    // These should all be of the same dimension.
    std::vector<NormalMixtureApproximation> approximations_;
  };

  // The density for -1 times the log of a gamma(nu, 1) random variable.
  class NegLogGamma {
   public:
    explicit NegLogGamma(double nu) : nu_(nu) {}
    double operator()(double y) const {
      return -nu_ * y - exp(-y) - lgamma(nu_);
    }

   private:
    double nu_;
  };

  //===========================================================================
  // A base class for a distance metric for measuring the closeness between the
  // NormalMixtureApproximation and the target function.
  class ApproximationDistance {
   public:
    ApproximationDistance(const ScalarTarget &logf,
                          const NormalMixtureApproximation &approximation,
                          double lower_limit, double upper_limit,
                          double guess_at_mode);

    virtual ~ApproximationDistance() {}

    // The distance metric is a function of a set of parameters contained in the
    // vector theta.  A default method is provided, but it can be over-ridden.
    virtual double operator()(const Vector &theta) const;

    double current_distance() const;

    // The integrand
    virtual double integrand(double x) const = 0;

    double logf(double x) const;
    double approximation(double x) const;
    double lower_limit() const;
    double upper_limit() const;

   private:
    ScalarTarget logf_;
    mutable NormalMixtureApproximation approx_;
    double lower_limit_;
    double upper_limit_;
    double guess_at_mode_;
  };

  class KullbackLeiblerDivergence : public ApproximationDistance {
   public:
    KullbackLeiblerDivergence(const ScalarTarget &logf,
                              const NormalMixtureApproximation &approx,
                              double lower_limit, double upper_limit,
                              double guess_at_mode);
    double integrand(double x) const override;
  };

  class AbsNormDistance : public ApproximationDistance {
   public:
    AbsNormDistance(const ScalarTarget &logf,
                    const NormalMixtureApproximation &approx,
                    double lower_limit, double upper_limit,
                    double guess_at_mode);
    double integrand(double x) const override;
  };
  
}  // namespace BOOM

#endif  //  BOOM_NORMAL_MIXTURE_APPROXIMATION_HPP_
