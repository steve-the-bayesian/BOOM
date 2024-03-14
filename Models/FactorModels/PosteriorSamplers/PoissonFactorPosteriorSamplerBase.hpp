#ifndef BOOM_POISSON_FACTOR_POSTERIOR_SAMPLER_BASE_HPP_
#define BOOM_POISSON_FACTOR_POSTERIOR_SAMPLER_BASE_HPP_

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

#include "Models/FactorModels/PoissonFactorModel.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "Models/FactorModels/PosteriorSamplers/VisitorPriorManager.hpp"
#include "distributions/rng.hpp"

namespace BOOM {

  class PoissonFactorPosteriorSamplerBase
      : public PosteriorSampler {
   public:
    PoissonFactorPosteriorSamplerBase(
        PoissonFactorModel *model,
        const Vector &default_prior_class_probabilities,
        RNG &seeding_rng = GlobalRng::rng);

    int number_of_classes() const {return model_->number_of_classes();}
    
    void impute_visitors();

    void set_prior_class_probabilities(const std::string &visitor_id,
                                       const Vector &probs) {
      visitor_prior_.set_prior_class_probabilities(visitor_id, probs);
    }

    const Vector &prior_class_probabilities(
        const std::string &visitor_id) const {
      return visitor_prior_.prior_class_probabilities(visitor_id);
    }

    const Vector &exposure_counts() const {return exposure_counts_;}

    // Return the number of visits to the site from visitors in each imputed
    // latent category.
    Vector compute_visit_counts(const FactorModels::PoissonSite &site) const;

   protected:
    PoissonFactorModel *model() {return model_;}
    const PoissonFactorModel *model() const {return model_;}

   private:

    // Raise an exception if logprob contains non-finite values.
    //
    // Args:
    //   logprob: The vector to check.
    //   visit_counts: The number of visits by this visitor to the site
    //     currently adding to logprob.
    //   site:  The site currently being added to logprob.
    void check_logprob(const Vector &logprob,
                       int visit_counts,
                       const Ptr<FactorModels::PoissonSite> &site) const;

    PoissonFactorModel *model_;
    VisitorPriorManager visitor_prior_;

    // The total number of users of each class.
    Vector exposure_counts_;
  };


}  // namespace BOOM


#endif  //  BOOM_POISSON_FACTOR_POSTERIOR_SAMPLER_BASE_HPP_
