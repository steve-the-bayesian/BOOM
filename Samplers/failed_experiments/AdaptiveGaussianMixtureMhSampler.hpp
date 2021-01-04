#ifndef BOOM_ADAPTIVE_GAUSSIAN_MIXTURE_MH_SAMPLER_HPP_
#define BOOM_ADAPTIVE_GAUSSIAN_MIXTURE_MH_SAMPLER_HPP_

/*
  Copyright (C) 2005-2018 Steven L. Scott

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

#include <functional>
#include <vector>
#include "LinAlg/SpdMatrix.hpp"
#include "LinAlg/Vector.hpp"
#include "Models/MvnBase.hpp"
#include "Samplers/Sampler.hpp"
#include "distributions/rng.hpp"

namespace BOOM {

  // An 'independence Metropolis-Hastings' sampler, where the proposal
  // distribution is a mixture of Gaussians selected to approximate the
  // target density.
  class AdaptiveGaussianMixtureMhSampler : public Sampler {
   public:
    typedef std::function<double(const Vector &)> LogDensity;
    explicit AdaptiveGaussianMixtureMhSampler(
        const LogDensity &log_density,
        RNG *rng = nullptr);
    Vector draw(const Vector &old) override;

    double log_proposal_density(const Vector &x) const;
    Vector simulate_proposal();

    //
    void set_initial_candidate_points(const std::vector<Vector> &candidates);

    int number_of_components() const { return mixing_weights_.size(); }

    const Vector &mixing_weights() const { return mixing_weights_; }

    void set_shared_variance(const SpdMatrix &variance) {
      shared_variance_ = variance;
      shared_precision_ = variance.inv();
      ldsi_ = shared_precision_.logdet();
    }

   private:
    // Recompute the mixing weights for the proposal distribution, leaving the
    // variance and means unchanged.  This also recomputes the estimated log
    // normalizing constant.
    void recompute_proposal_distribution_weights();

    // Args:
    //   log_density:  The un-normalized log target density at the candidate.
    //   log_proposal_density: The normalized proposal density at the candidate.
    //
    // Returns:
    //   The approximate normalizing constant is applied to the log density (to
    //   normalize it), and the normalized density is compared to the proposal
    //   density.  If the proposal density is very much smaller than the target
    //   density then the candidate is said to be poorly supported.  "Very much
    //   smaller" means the difference on the log scale exceeds the value of
    //   "density_envelope_".
    bool candidate_poorly_supported(double log_density,
                                    double log_proposal_density) const;

    // Start with a single candidate.  Assign a new cluster when logp(candidate)
    // - logp(mode) exceeds logq(candidate) - logq(mode) (to within a fudge
    // factor).
    void update_proposal_distribution(const Vector &x, double log_density_value,
                                      double log_proposal_density_value);

    // Initialize an empty proposal distribution as a single component normal
    // mixture with mean set to the starting point, and variance proportional to
    // the identity.
    //
    // TODO: Improve the starting variance.  Might need help from the user.
    void initialize_proposal_distribution(const Vector &starting_point);

    // Add a new cluster to the proposal distribution.
    void add_new_cluster(const Vector &starting_point);

    // The log density of the target distribution for the sampler.
    LogDensity log_density_;

    // The largest density value observed thus far, the value that generated it,
    // and the log density of this point under the proposal distribution.
    Vector mode_;
    double log_density_value_at_mode_;
    double log_proposal_density_value_at_mode_;
    double log_normalizing_constant_;

    // History of the candidates that have been proposed by this algorithm, and
    // their associated log target densities.
    std::vector<Vector> candidates_;
    Vector log_density_values_;

    // The amount by which the target density is allowed to exceed the proposal
    // density without requiring a new mixture to be added.
    double density_envelope_;

    std::vector<MvnSuf> mixture_sufficient_statistics_;
    std::vector<Vector> mixture_means_;
    SpdMatrix shared_precision_;
    SpdMatrix shared_variance_;
    double ldsi_;
    Vector mixing_weights_;
    Vector log_mixing_weights_;
  };

}  // namespace BOOM

#endif  //  BOOM_ADAPTIVE_GAUSSIAN_MIXTURE_MH_SAMPLER_HPP_
