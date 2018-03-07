#ifndef THIRD_PARTY_BOOM_SRC_INTERFACES_R_BOOMSPIKESLAB_SRC_UTILS_H_
#define THIRD_PARTY_BOOM_SRC_INTERFACES_R_BOOMSPIKESLAB_SRC_UTILS_H_

#include "Models/Glm/Glm.hpp"

namespace BOOM {

namespace spikeslab {

// Set the coefficients equal to their initial values, and determine
// which coefficients are initially excluded (i.e. forced to zero).
// Args:
//   initial_beta:  Vector containing initial coefficients.
//   prior_inclusion_probabilities: Prior probabilities that each
//     coefficient is nonzero.
//   model:  The model that owns the coefficients.
//   sampler:  The sampler that will make posterior draws for the model.
template<class SAMPLER>
void InitializeCoefficients(
    const BOOM::Vector &initial_beta,
    const BOOM::Vector &prior_inclusion_probabilities,
    Ptr<GlmModel>  model,
    BOOM::Ptr<SAMPLER> sampler) {
  model->set_Beta(initial_beta);
  if (min(prior_inclusion_probabilities) >= 1.0) {
    // Ensure all coefficients are included if you're not going to
    // do model averaging.
    sampler->allow_model_selection(false);
    model->coef().add_all();
  } else {
    // Model averaging is desired.  "Small" coefficients start off
    // excluded from the model.  Large ones start off included.
    // Adding or dropping is idempotent, so no need to worry about
    // dropping an already excluded coefficient.
    for (int i = 0; i < initial_beta.size(); ++i) {
      if (fabs(initial_beta[i]) < 1e-8) {
        model->coef().drop(i);
      } else {
        model->coef().add(i);
      }

      // Respect absolute prior opinions about coefficients,
      // regardless of whether the initial coefficient is large or
      // small,
      if (prior_inclusion_probabilities[i] >= 1.0) {
        model->add(i);
      } else if (prior_inclusion_probabilities[i] <= 0.0) {
        model->drop(i);
      }
    }
  }
}

}  // namespace spikeslab

}  // namespace BOOM

#endif  // THIRD_PARTY_BOOM_SRC_INTERFACES_R_BOOMSPIKESLAB_SRC_UTILS_H_
