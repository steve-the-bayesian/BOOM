#ifndef BOOM_GRAPHICAL_MODELS_CLIQUE_HPP_
#define BOOM_GRAPHICAL_MODELS_CLIQUE_HPP_

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

#include <sstream>

#include "Models/Graphical/Node.hpp"
#include "Models/Graphical/NodeSet.hpp"
#include "LinAlg/Array.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {
  namespace Graphical {

    // A clique is NodeSet of nodes that are all neighbors of one another.
    template <class NODETYPE>
    class Clique : public NodeSet<NODETYPE> {
     public:
      // An empty Clique.
      Clique() {}

      // Create a Clique from a NodeSet.  All nodes in node_set must be
      // neighbors of one another.  Otherwise an exception is generated.
      Clique(const NodeSet<NODETYPE> &node_set) {
        for (const auto &el : node_set.elements()) {
          bool ok = try_add(el);
          if (!ok) {
            std::ostringstream err;
            err << "Could not add " << el->name()
                << " to the clique " << name();
            report_error(err.str());
          }
        }
      }

      const std::string &name() const override {
        return NodeSet<NODETYPE>::name();
      }

      // Attempt to add a node, which may or may not belong, to the Clique.
      // Return true iff the addition was successful.
      //
      // Args:
      //   node:  A node to add to the Clique.
      //
      // Effects:
      //   If 'node' is a neighbor of all current elements, then 'node' is added
      //   to the set of elements_.
      //
      // Returns:
      //   A flag indicating whether 'node' was added to this object's elements.
      bool try_add(const Ptr<NODETYPE> &node){
        for (const auto &el : elements()) {
          if (!node->is_neighbor(el)) {
            return false;
          }
        }
        NodeSet<NODETYPE>::add(node);
        return true;
      }

      // Two cliques are equal if their elements_ are equal.
      bool operator==(const Clique &rhs) const {
        return elements() == rhs.elements();
      }

      // The underlying set of elements comprising the Clique.
      const SortedVector<Ptr<NODETYPE>> &elements() const {
        return NodeSet<NODETYPE>::elements();
      }

      size_t size() const {
        return elements().size();
      }

      // Return true iff *this and other have at least one element in common.
      bool shares_node_with(const Ptr<Clique> &other) const {
        return !elements().disjoint_from(other->elements());
      }

    };

    //===========================================================================


    template <class NODETYPE>
    class CliqueFinder {
     public:
      bool add_node(const Ptr<NODETYPE> &node) {
        int membership_count = 0;
        for (size_t i = 0; i < cliques_.size(); ++i) {
          membership_count += cliques_[i]->try_add(node);
        }

        if (membership_count == 0) {
          NEW(Clique<NODETYPE>, clique)();
          clique->try_add(node);
          cliques_.push_back(clique);
          return true;
        } else {
          return false;
        }
      }

      void find_cliques(const std::vector<Ptr<NODETYPE>> &nodes) {
        for (size_t i = 0; i < nodes.size(); ++i) {
          const Ptr<NODETYPE> &node(nodes[i]);
          bool added_new_clique = add_node(node);
          if (added_new_clique) {
            Ptr<Clique<NODETYPE>> &clique(cliques_.back());
            for (size_t j = 0; j < i; ++j) {
              const Ptr<NODETYPE> &old_node(nodes[j]);
              clique->try_add(old_node);
            }
          }
        }
      }

      void establish_links(std::vector<Ptr<Clique<NODETYPE>>> &cliques) const {
        for (size_t i = 0; i < cliques.size(); ++i ){
          Ptr<Clique<NODETYPE>> &first(cliques[i]);
          for (size_t j = i + 1; j < cliques.size(); ++j) {
            Ptr<Clique<NODETYPE>> &second(cliques[j]);
            if (first->shares_node_with(second)) {
              first->add_neighbor(second);
            }
          }
        }
      }

      const std::vector<Ptr<Clique<NODETYPE>>> &cliques() const {
        return cliques_;
      }

     private:
      std::vector<Ptr<Clique<NODETYPE>>> cliques_;
    };


    // A mix of categorical and conditionally Gaussian distributions.
    // Some values are marked as known.
    class CliqueMarginalDistribution {
     public:
      void marginalize();
     private:
      std::map<Ptr<Node>, int> known_discrete_variables_;
      std::map<Ptr<Node>, double> known_gaussian_variables_;

      Array unknown_discrete_distribution_;
      Vector unknown_gaussian_potential_means_;
      SpdMatrix unknown_gaussian_potential_precisions_;
    };

    template <class NODETYPE>
    std::vector<Ptr<Clique<NODETYPE>>> find_cliques(
        const std::vector<Ptr<NODETYPE>> &nodes) {
      CliqueFinder<NODETYPE> clique_finder;
      clique_finder.find_cliques(nodes);
       return clique_finder.cliques();
    }


  } // namespace Graphical
}  // namespace BOOM

#endif  //  BOOM_GRAPHICAL_MODELS_CLIQUE_HPP_
