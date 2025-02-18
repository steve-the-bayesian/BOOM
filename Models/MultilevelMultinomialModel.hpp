#ifndef BOOM_MODELS_MULTILEVEL_MULTINOMIAL_MODEL_HPP_
#define BOOM_MODELS_MULTILEVEL_MULTINOMIAL_MODEL_HPP_

/*
  Copyright (C) 2005-2025 Steven L. Scott

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

#include "CategoricalData.hpp"
#include "MultinomialModel.hpp"

#include "Models/ModelTypes.hpp"
#include "Models/ParamTypes.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/Policies/CompositeParamPolicy.hpp"
#include "Models/Policies/DeferredDataPolicy.hpp"

#include "cpputil/SortedVector.hpp"

namespace BOOM {

  // Need a composite sufstat data policy?

  // A MultilevelMultinomialModel describes multilevel categorical data.  That
  // is, data from a taxonomy of the form L1/L2/L3/...  Traditional categorical
  // data assumes one of K categories.  Multilevel categorical data is a
  // hierarchical data structure where the first level describes the most
  // general category, the second level describes a more specific subcategory,
  // etc.
  //
  // A MultilevelMultinomialModel describes multilevel categorical data using
  // conditional probabilities.  If v = (v1, v2, v3) are the three levels of a
  // data point (each v_i is an integer in {0, 1, ...}), then Pr(v) = Pr(v1) *
  // Pr(v2 | v1) * Pr(v3 | v1, v2).  The model for each layer is a traditional
  // Multinomial model.  Thus this model is implemented using a tree of
  // MultinomialModel objects.
  class MultilevelMultinomialModel
      : public CompositeParamPolicy,
        public DeferredDataPolicy,
        public PriorPolicy,
        virtual public MixtureComponent
  {
   public:
    MultilevelMultinomialModel(const Ptr<Taxonomy> &tax);

    MultilevelMultinomialModel(const MultilevelMultinomialModel &rhs);
    MultilevelMultinomialModel &operator=(const MultilevelMultinomialModel &rhs);

    MultilevelMultinomialModel(MultilevelMultinomialModel &&rhs) = default;
    MultilevelMultinomialModel &operator=(MultilevelMultinomialModel &&rhs) = default;
    
    MultilevelMultinomialModel *clone() const override;

    double logp(const MultilevelCategoricalData &data_point) const;
    double pdf(const Data *dp, bool logscale) const override;
    int number_of_observations() const override;

    void add_data(const Ptr<Data> &dp) override;
    void add_data(const Ptr<MultilevelCategoricalData> &dp);
    void clear_data() override;
    void combine_data(const Model &other_model, bool just_suf = true) override;

    MultinomialModel *conditional_model(const std::string &value);
    const MultinomialModel *conditional_model(const std::string &value) const;
    MultinomialModel *conditional_model(const TaxonomyNode *node);
    const MultinomialModel *conditional_model(const TaxonomyNode *node) const;
    
    MultinomialModel *top_level_model() {return top_level_model_.get();}
    const MultinomialModel *top_level_model() const {return top_level_model_.get();}
    
   private:
    Ptr<Taxonomy> taxonomy_;

    // The top_level_model_ is the model for the first level in the taxonomy.
    Ptr<MultinomialModel> top_level_model_;

    // conditional_models_ 
    std::map<const TaxonomyNode *, Ptr<MultinomialModel>> conditional_models_;

    std::vector<Ptr<MultilevelCategoricalData>> data_;
    bool only_keep_suf_;

    // Populate top_level_model_ and conditional_models_ using 
    void create_models();
  };

}  // namespace BOOM

#endif  //  BOOM_MODELS_MULTILEVEL_MULTINOMIAL_MODEL_HPP_
