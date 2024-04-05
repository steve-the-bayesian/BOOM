#ifndef BOOM_MULTINOMIAL_FACTOR_MODEL_POSTERIOR_SAMPLER_HPP_
#define BOOM_MULTINOMIAL_FACTOR_MODEL_POSTERIOR_SAMPLER_HPP_

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

#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "Models/FactorModels/MultinomialFactorModel.hpp"
#include "Models/FactorModels/PosteriorSamplers/VisitorPriorManager.hpp"

namespace BOOM {

  class MultinomialFactorModelPosteriorSampler
      : public PosteriorSampler
  {
    using Visitor = FactorModels::MultinomialVisitor;
    using Site = FactorModels::MultinomialSite;
    
   public:
    MultinomialFactorModelPosteriorSampler(
        MultinomialFactorModel *model,
        const Vector &default_prior_class_probabilities,
        RNG & seeding_rng = GlobalRng::rng);

    void draw() override;
    double logpri() const override;

    int number_of_classes() const {return model_->number_of_classes();}
    void impute_visitors();
    void impute_visitor(Visitor &visitor);

    void draw_site_parameters();
    
    void set_prior_class_probabilities(
        const std::string &visitor_id,
        const Vector &probs) {
      visitor_prior_.set_prior_class_probabilities(visitor_id, probs);
    }

    const Vector &prior_class_probabilities(
        const std::string &visitor_id) const {
      return visitor_prior_.prior_class_probabilities(visitor_id);
    }
    
   private:
    MultinomialFactorModel *model_;
    VisitorPriorManager visitor_prior_;

    // Raise an exception if any elements of logprob are non-finite.
    void check_logprob(const Vector &logprob) const;
  };

  
}  // namespace BOOM

#endif  // BOOM_MULTINOMIAL_FACTOR_MODEL_POSTERIOR_SAMPLER_HPP_
