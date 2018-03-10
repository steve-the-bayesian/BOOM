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

#include "Samplers/AdaptiveGaussianMixtureMhSampler.hpp"
#include "cpputil/lse.hpp"
#include "cpputil/math_utils.hpp"
#include "distributions.hpp"
#include "numopt/Powell.hpp"
#include "stats/logit.hpp"

namespace BOOM {
  using std::endl;
  namespace {
    using AGMS = AdaptiveGaussianMixtureMhSampler;

    // A target function for numerical optimization of the weights.
    class MixingWeightTargetFunction {
     public:
      MixingWeightTargetFunction(const std::vector<Vector> &candidates,
                                 const Vector &log_density_values,
                                 const SpdMatrix &shared_precision, double ldsi,
                                 const std::vector<Vector> &means)
          : candidates_(candidates),
            log_density_values_(log_density_values),
            shared_precision_(shared_precision),
            ldsi_(ldsi),
            means_(means) {}

      // Returns the sum of squared errors in the comparison of the log density
      // of the observed candidates to the log density of the proposal
      // distribution.
      //
      // Args:
      //   parameters: The logit of the mixing weights used in the mixture
      //     approximation.
      //     - The first element is the log normalizing constant.
      //     - Remaining elements are the logits of the mixing weights.
      double operator()(const Vector &parameters) const {
        double log_normalizing_constant = parameters[0];
        Vector log_mixing_weights =
            log(multinomial_logit_inverse(ConstVectorView(parameters, 1)));

        double ans = 0;
        for (int i = 0; i < candidates_.size(); ++i) {
          double err = log_density_values_[i] - log_normalizing_constant -
                       log_proposal_density(candidates_[i], log_mixing_weights);
          ans += err * err;
        }
        return ans;
      }

      double log_proposal_density(const Vector &candidate,
                                  const Vector &log_mixing_weights) const {
        int dim = means_.size();
        Vector workspace(dim);
        for (int i = 0; i < dim; ++i) {
          workspace[i] =
              log_mixing_weights[i] +
              dmvn(candidate, means_[i], shared_precision_, ldsi_, true);
        }
        return lse(workspace);
      }

     private:
      const std::vector<Vector> &candidates_;
      const Vector &log_density_values_;
      const SpdMatrix &shared_precision_;
      double ldsi_;
      const std::vector<Vector> &means_;
    };

  }  // namespace

  ///---------------------------------------------------------------------------
  AGMS::AdaptiveGaussianMixtureMhSampler(const LogDensity &log_density,
                                         RNG *rng)
      : Sampler(rng),
        log_density_(log_density),
        mode_(0),
        log_density_value_at_mode_(negative_infinity()),
        log_proposal_density_value_at_mode_(negative_infinity()),
        log_normalizing_constant_(0),
        density_envelope_(2.0) {}

  //---------------------------------------------------------------------------
  double AGMS::log_proposal_density(const Vector &x) const {
    Vector workspace = log_mixing_weights_;
    for (int i = 0; i < workspace.size(); ++i) {
      workspace[i] +=
          dmvn(x, mixture_means_[i], shared_precision_, ldsi_, true);
    }
    return lse(workspace);
  }

  //---------------------------------------------------------------------------
  Vector AGMS::simulate_proposal() {
    int i = rmulti_mt(rng(), mixing_weights_);
    return rmvn_mt(rng(), mixture_means_[i], shared_variance_);
  }

  //---------------------------------------------------------------------------
  Vector AGMS::draw(const Vector &old) {
    if (mixing_weights_.empty()) {
      initialize_proposal_distribution(old);
    }
    Vector candidate = simulate_proposal();
    double candidate_log_density = log_density_(candidate);
    double old_log_density = log_density_(old);
    double candidate_log_proposal_density = log_proposal_density(candidate);
    double old_log_proposal_density = log_proposal_density(old);
    if (candidate_log_density > log_density_value_at_mode_) {
      log_density_value_at_mode_ = candidate_log_density;
      mode_ = candidate;
    }
    candidates_.push_back(candidate);
    log_density_values_.push_back(candidate_log_density);
    double log_mh_ratio = candidate_log_density -
                          candidate_log_proposal_density -
                          (old_log_density - old_log_proposal_density);
    double log_u = log(runif_mt(rng()));
    if (log_u < log_mh_ratio) {
      return candidate;
    } else {
      if (candidate_poorly_supported(candidate_log_density,
                                     candidate_log_proposal_density)) {
        add_new_cluster(candidate);
      }
      return old;
    }
  }

  //---------------------------------------------------------------------------
  void AGMS::initialize_proposal_distribution(const Vector &initial_value) {
    mixture_means_.push_back(initial_value);
    mixing_weights_.push_back(1.0);
    log_mixing_weights_.push_back(0);
  }

  bool AGMS::candidate_poorly_supported(double log_density,
                                        double log_proposal_density) const {
    double density_delta =
        log_density - log_normalizing_constant_ - log_proposal_density;
    return density_delta < density_envelope_;
  }

  //---------------------------------------------------------------------------
  // Add a new cluster to the proposal distribution.
  // Args:
  //   starting_point: The initial cluster center.
  // Effects:
  //   A new cluster is created, centered at starting_point, and starting_point
  //   is assigned as data to the cluster.  The variance of the new cluster is
  //   set to the initial cluster variance, and all proposal distribution
  //   parameters are modified.
  void AGMS::add_new_cluster(const Vector &starting_point) {
    mixture_means_.push_back(starting_point);

    mixing_weights_.push_back(1.0);
    mixing_weights_ /= mixing_weights_.sum();
    log_mixing_weights_ = log(mixing_weights_);
    recompute_proposal_distribution_weights();
    log_proposal_density_value_at_mode_ = log_proposal_density(mode_);
  }

  //---------------------------------------------------------------------------
  void AGMS::recompute_proposal_distribution_weights() {
    Vector parameters =
        concat(log_normalizing_constant_, multinomial_logit(mixing_weights_));

    MixingWeightTargetFunction target(candidates_, log_density_values_,
                                      shared_precision_, ldsi_, mixture_means_);
    PowellMinimizer powell(target);
    powell.minimize(parameters);
    parameters = powell.minimizing_value();
    log_normalizing_constant_ = parameters[0];
    mixing_weights_ = multinomial_logit_inverse(ConstVectorView(parameters, 1));
    log_mixing_weights_ = log(mixing_weights_);
    log_proposal_density_value_at_mode_ = log_proposal_density(mode_);
  }

}  // namespace BOOM
