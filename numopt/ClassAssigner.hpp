#ifndef BOOM_NUMOPT_CLASS_ASSIGNER_HPP_
#define BOOM_NUMOPT_CLASS_ASSIGNER_HPP_

/*
  Copyright (C) 2005-2024 Steven L. Scott

  This library is free software; you can redistribute it and/or modify it under
  the terms of the GNU Lesser General Public License as published by the Free
  Software Foundation; either version 2.1 of the License, or (at your option)
  any later version.

  This library is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
  details.

  You should have received a copy of the GNU Lesser General Public License along
  with this library; if not, write to the Free Software Foundation, Inc., 51
  Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
*/

#include <vector>
#include "LinAlg/Matrix.hpp"
#include "LinAlg/Vector.hpp"
#include "distributions/rng.hpp"
#include "stats/FreqDist.hpp"

namespace BOOM {

  // Assigns a collection of objects to classes guided by their individual
  // posterior distributions as well as a global target.
  //
  // A collection of objects i = 0..n-1 must each be assigned to one of K
  // classes k = 0, ..., K-1.  Object i comes with a marginal distribution over
  // its class pi_i.  The empirical distribution of the collection of
  // assignments must closely match a global target distribution f.
  //
  // The assignment of object i to class k_i is done by minimizing a cost
  // function of the form
  //
  // \sum_i=1^n (log (pi_i(k_i_star) / pi_i(k_i))) / n + alpha * KL(f, pi_star)
  //
  // where k_i_star is the MAP estimate of user i's class, KL(p1, p2) is the
  // Kullback Liebler divergence from p1 to p2, and pi_star is the empirical
  // distribution of the assignments.  The minimization is done by a simulated
  // annealing algorithm.
  //
  // The user inputs a maximum allowed value of KL (a default is available).
  // The value of alpha is gradually increased until the acceptable tolerance
  // limit is reached.
  class ClassAssigner {
   public:
    ClassAssigner();
    void set_initial_temperature(double temp) {initial_temperature_ = temp;}
    void set_max_kl(double kl) {max_tolerable_kl_ = kl;}
    void set_max_iterations(int niter) {niter_ = niter;}

    std::vector<int> assign(const Matrix &marginal_posteriors,
                            const Vector &global_target,
                            RNG &rng);

    // The Kullback-Liebler divergence between the target and empirical
    // distributions.
    double kl() const;

   private:

    // Row i of marginal_posteriors_ contains the discrete probability
    // distribution for the class of observation i.
    Matrix marginal_posteriors_;

    // global_target_[i] is the population level proportion of category level i.
    Vector global_target_;

    // assignment_[i] is the category assigned to user i.
    std::vector<int> assignment_;

    // The empirical distribution of the values in assignment_.
    FrequencyDistribution empirical_distribution_;

    // The maximum number of simulated annealing iterations.
    int niter_;

    // A weight placed on the KL divergence between empirical_distribution_ and
    // global_target_ when assigning values to individuals.  The larger this
    // weight the greater the emphasis on distributional agreement.  The smaller
    // the weight the less emphasis on individuals matching their individual
    // marginal posterior distributions.
    double distribution_scale_factor_;

    // The temperature value used to initialize a simulated annealing run.
    double initial_temperature_;

    // The current temperature value in a simulated annealing run.
    double temperature_;

    // The largest acceptable KL divergence between empirical_distribution_ and
    // global_target_.
    double max_tolerable_kl_;

    // Check that the inputs to 'assign' are valid.
    void check_inputs(const Matrix &marginal_posteriors,
                      const Vector &global_target) const;

    // Perform a simulated annealing run on the assignments.
    void simulated_annealing(RNG &rng);

    // Run a single step of simulated_annealing.
    // Args:
    //   rng: A random number generator used to drive the proposals and random
    //     acceptances.
    //
    // Returns:
    //   The number of inputs that were changed as a result of simulated
    //   annealing proposals.
    Int simulated_annealing_step(RNG &rng);

    // This isn't actually used.
    double cost_function(const std::vector<int> &assignment) const;

    // Decide whether a candidate assignment should be accepted.
    //
    // Args:
    //   candidate:  A proposed value to assign to an object.
    //   index: The index (observation numb) of the object receiving the
    //     assignment.
    //   empirical_distribution: Current distribution of assignments (before
    //     candidate is proposed).
    //   rng:  Random number generator used to make the decision.
    //
    // Returns:
    //   If true then the candidate is to be accepted.  If false then it is to
    //   be rejected.
    bool accept_candidate(int candidate,
                          size_t index,
                          FrequencyDistribution &empirical_distribution,
                          RNG &rng) const;

  };

} // namespace BOOM

#endif  // BOOM_NUMOPT_CLASS_ASSIGNER_HPP_
