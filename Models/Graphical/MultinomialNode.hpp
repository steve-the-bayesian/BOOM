#ifndef BOOM_MODELS_GRAPHICAL_MULTINOMIAL_NODE_HPP_
#define BOOM_MODELS_GRAPHICAL_MULTINOMIAL_NODE_HPP_

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

#include "Models/Graphical/Node.hpp"
#include "LinAlg/Array.hpp"
#include "Models/MultinomialModel.hpp"

namespace BOOM {
  namespace Graphical {

    // A node in a DirectedGraphicalModel that describes ao categorical variable
    // using a collection of MultinomialModel objects.
    //
    // A MultinomialNode must have parents that are all discrete.
    class MultinomialNode : public Node {
     public:
      // Build a MultinomialNode by searching a DataTable for a variable with a
      // given name.
      //
      // Args:
      //   table:  The data table to search.
      //   variable_name:  The variable_name in 'table' to search for.
      //
      // Exceptions:
      //   Throws if variable_name is not present in DataTable or if it is not a
      //   CategoricalVariable.
      MultinomialNode(const DataTable &table, const std::string &variable_name);

      MultinomialNode(int node_id, const std::string &name, int variable_index,
                      const Ptr<CatKey> &key);

      NodeType node_type() const override {
        return NodeType::CATEGORICAL;
      }

      Int dim() const override {return categorical_key_->max_levels();}

      // The conditional distribution of this node's outcome variable given the
      // node's parents.  Because Gaussian nodes cannot be parents of
      // categorical nodes, all parents are categorical.
      //
      // Returns:
      //   A K + 1 dimensional array, where K is the number of this node's
      //   parents.  The order of the dimensions is the same as the order of
      //   nodes in this->parents().  The final dimension corresponds to levels
      //   of this node's target variable.
      Array conditional_probability_table() const;

      // Returns the conditional distribution of this node's variable given its
      // ancestors.
      double logp(const MixedMultivariateData &dp) const override;

      // Returns the output dimension of each parent.  Note that parents of
      // categorical nodes must also be categorical.  If this method finds a
      // non-categorical parent of this object it will throw an exception.
      std::vector<int> parent_dims() const;

      // The value of this node's target variable in 'data_point', or -1 if the
      // variable is missing.
      int categorical_value(
          const MixedMultivariateData &data_point) const override;

      // Syntactic sugar for categorical_value.
      int value(const MixedMultivariateData &data_point) {
        return categorical_value(data_point);
      }

     private:
      // The structure of models_ depends on the number of parents this node
      // has.  If there are no parents then models_[{0}] contains the marginal
      // distribution of the variable.  Otherwise models_[{i, j, k}] contains
      // the conditional distribution of the target variable given that parent 1
      // has value i, parent 2 has value j, and parent 3 has value k.
      //
      // If we are learning model structure then the structure in models_ may
      // change as we try new structures.
      mutable bool models_current_;
      mutable GenericArray<Ptr<MultinomialModel>> models_;

      // If models_current_ is false then rebuild models_ and set
      // models_current_ to true.
      void ensure_models() const;

      Ptr<CatKey> categorical_key_;
    };

  }  // namespace Graphical
}  // namespace BOOM


#endif  // BOOM_MODELS_GRAPHICAL_MULTINOMIAL_NODE_HPP_
