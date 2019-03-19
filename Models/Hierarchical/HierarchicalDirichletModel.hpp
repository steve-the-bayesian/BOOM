// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2015 Steven L. Scott

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

#ifndef BOOM_HIERARCHICAL_DIRICHLET_MODEL_HPP_
#define BOOM_HIERARCHICAL_DIRICHLET_MODEL_HPP_

#include "Models/DirichletModel.hpp"
#include "Models/Hierarchical/HierarchicalModel.hpp"
#include "Models/MultinomialModel.hpp"
#include "Models/PosteriorSamplers/MultinomialDirichletSampler.hpp"

namespace BOOM {

  // The HierarchicalDirichletModel is a model for groups of
  // multinomial data.  Each group has a different multinomial
  // distribution, with success probabilities drawn from a common
  // Dirichlet distribution with parameters nu = alpha * pi (where
  // alpha is a positive real, and pi is a discrete probability
  // distribution).

  class HierarchicalDirichletData : public Data {
   public:
    explicit HierarchicalDirichletData(uint dimension);
    explicit HierarchicalDirichletData(const MultinomialSuf &suf);
    HierarchicalDirichletData *clone() const override;
    std::ostream &display(std::ostream &out) const override;
    const MultinomialSuf &suf() { return counts_; }

   private:
    MultinomialSuf counts_;
  };

  class HierarchicalDirichletModel
      : public HierarchicalModelBase<MultinomialModel, DirichletModel> {
   public:
    // The Dirichlet parameters are alpha * pi, where alpha is a
    // postive scalar and pi is a discrete probability distribution.
    // The larger alpha, the closer the draws from the prior are to
    // pi.  Alpha acts like a prior number of observations when the
    // Dirichlet distribution is used as a prior for multinomial
    // observations.
    // Args:
    //   prior_sample_size: Denoted alpha above.
    //   mean: Denoted pi above.  All elements must be non-negative,
    //     and elements must sum to 1.  The dimension of pi will
    //     determine the acceptable dimension of
    //     HierarchicalDirichletData.
    HierarchicalDirichletModel(double prior_sample_size, const Vector &mean);
    explicit HierarchicalDirichletModel(const Ptr<DirichletModel> &prior);
    HierarchicalDirichletModel *clone() const override;
    void add_data(const Ptr<Data> &) override;

    // Mean of the Dirichlet distribution in the prior.  This is the
    // same as the vector of prior sample sizes, normalized to sum to
    // 1.
    Vector prior_mean() const;

    // Sample size, or content, parameter of the Dirichlet
    // distribution in the prior.  This is the sum of the elements in
    // the vector of sample sizes.
    double prior_sampler_size() const;
  };

}  // namespace BOOM

#endif  // BOOM_HIERARCHICAL_DIRICHLET_MODEL_HPP_
