#ifndef BOOM_MODELS_FACTOR_MODELS_POISSON_FACTOR_MODEL_POSTERIOR_SAMPLER_HPP
#define BOOM_MODELS_FACTOR_MODELS_POISSON_FACTOR_MODEL_POSTERIOR_SAMPLER_HPP
/*
  Copyright (C) 2005-2022 Steven L. Scott

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

#include "Models/FactorModels/PoissonFactorModel.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "distributions/rng.hpp"

namespace BOOM {

  class PoissonFactorModelPosteriorSampler
      : public PosteriorSampler
  {
   public:

    // Args:
    //   model:  The model to be posterior sampled.

    //   default_prior_class_probabilities: The default prior to use
    //     for visitors whose membership probabilities are not otherwise
    //     decalred.
    PoissonFactorModelPosteriorSampler(
        PoissonFactorModel *model,
        const Vector &default_prior_class_probabilities,
        RNG &seeding_rng = GlobalRng::rng);

    void draw() override;

    double logpri() const override {
      // Just to get things compiled.
      return negative_infinity();
    }

    void impute_visitors();
    void draw_site_parameters();

    Vector prior_class_probabilities(
        const std::string &visitor_id) const;

    void set_prior_class_probabilities(
        const std::string &visitor_id,
        const Vector &probs);

   private:
    // Initialize the sum_of_lambdas_ data element by looping over the model
    // values.  This only needs to be called once, because the "draw" method
    // will keep sum_of_lambdas_ current on an ongoing basis.
    void initialize_sum_of_lambdas();

    PoissonFactorModel *model_;

    Vector default_prior_class_probabilities_;
    std::map<std::string, Vector> prior_class_probabilities_;

    Vector exposure_counts_;
    Vector sum_of_lambdas_;
  };

}


#endif // BOOM_MODELS_FACTOR_MODELS_POISSON_FACTOR_MODEL_POSTERIOR_SAMPLER_HPP
