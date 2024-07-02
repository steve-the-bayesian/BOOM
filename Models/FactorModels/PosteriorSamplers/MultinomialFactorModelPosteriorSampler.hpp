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

#include "cpputil/ThreadTools.hpp"
#include "Samplers/MoveAccounting.hpp"

namespace BOOM {

  namespace MfmThreading {

    //=========================================================================
    // A worker to help implement multi-threading.  Each thread in the pro
    class VisitorImputer {
      using Visitor = FactorModels::MultinomialVisitor;

     public:
      VisitorImputer(RNG &rng,
                     VisitorPriorManager *prior_manager)
          : rng_(seed_rng(rng)),
            prior_manager_(prior_manager)
      {}

      size_t num_visitors() const {
        return visitors_.size();
      }

      void add_visitor(const Ptr<Visitor> &visitor) {
        visitors_.push_back(visitor);
      }

      void clear_visitors() {
        visitors_.clear();
      }

      void impute_visitors();

      void impute_visitor(Visitor &visitor);

     private:
      RNG rng_;
      const VisitorPriorManager *prior_manager_;
      std::vector<Ptr<Visitor>> visitors_;
    };

  }  // namespace MfmThreading

  
  //===========================================================================
  // A posterior sampler MulinomialFactorModel objects.  The sampler alternates
  // between imputing class membership and drawing site-level parameters.  The
  // imputation step is parallelizable because (large number of) users are
  // conditionally independent given site-level parameters.
  //
  // The same is not true of Sites.  For each level of the latent category in
  // the MultinomialFactorModel, the collection of all site parameters is one
  // giant Dirichlet random variable.  Of course, a draw from a Dirichlet is a
  // draw from a bunch of independent gamma's, normalized by their sum.  That
  // might work.
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

    void set_num_threads(int num_threads);

    void draw() override;
    double logpri() const override;

    int number_of_classes() const {return model_->number_of_classes();}
    void impute_visitors();
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
    // void check_logprob(const Vector &logprob) const;

    std::vector<MfmThreading::VisitorImputer> visitor_imputers_;
    ThreadWorkerPool pool_;

    // Fill the unknown_visitors_ object and recompute known_site_visit_counts_.
    //
    // There is a subtlety here
    void fill_unknown_visitors();

    // Visitors with unknown categories.
    std::set<Ptr<Visitor>> unknown_visitors_;

    // Site visit counts from visitors with known categories, as defined by
    // having a prior category probability above a high threshold (e.g. .999)
    // for a single category.
    Matrix known_site_visit_counts_;

    MoveAccounting accounting_;
  };

}  // namespace BOOM

#endif  // BOOM_MULTINOMIAL_FACTOR_MODEL_POSTERIOR_SAMPLER_HPP_
