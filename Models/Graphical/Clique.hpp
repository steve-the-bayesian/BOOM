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
#include "Models/Graphical/UndirectedGraph.hpp"
#include "LinAlg/Array.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {
  namespace Graphical {

    // A clique is NodeSet of MoralNode's that are all neighbors of one another.
    class Clique : public NodeSet<MoralNode> {
     public:
      // An empty Clique.
      Clique() {}

      // Create a Clique from a NodeSet.  All nodes in node_set must be
      // neighbors of one another.  Otherwise an exception is generated.
      Clique(const NodeSet<MoralNode> &node_set) {
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

      // const std::string &name() const override {
      //   return NodeSet<MoralNode>::name();
      // }

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
      bool try_add(const Ptr<MoralNode> &node){
        for (const auto &el : elements()) {
          if (!node->is_neighbor(el)) {
            return false;
          }
        }
        NodeSet<MoralNode>::add(node);
        return true;
      }

      // Two cliques are equal if their elements_ are equal.
      bool operator==(const Clique &rhs) const {
        return elements() == rhs.elements();
      }

      // The underlying set of elements comprising the Clique.
      // const SortedVector<Ptr<MoralNode>> &elements() const {
      //   return NodeSet<MoralNode>::elements();
      // }

      // size_t size() const {
      //   return elements().size();
      // }

      // Return true iff *this and other have at least one element in common.
      bool shares_node_with(const Ptr<Clique> &other) const {
        return !elements().disjoint_from(other->elements());
      }

      bool contains(const Ptr<DirectedNode> &node) const {
        for (const auto &el : elements()) {
          if (el->base_node() == node) {
            return true;
          }
        }
        return false;
      }
    };

    //===========================================================================
    class CliqueFinder {
     public:
      Ptr<Clique> add_node(const Ptr<MoralNode> &node) {
        int membership_count = 0;
        for (auto &clique : cliques_) {
          membership_count += clique->try_add(node);
        }

        if (membership_count == 0) {
          NEW(Clique, clique)();
          clique->try_add(node);
          cliques_.add_element(clique);
          return clique;
        } else {
          return nullptr;
        }
      }

      void find_cliques(const std::vector<Ptr<MoralNode>> &nodes) {
        for (size_t i = 0; i < nodes.size(); ++i) {
          Ptr<Clique> clique = add_node(nodes[i]);
          if (!!clique) {
            for (size_t j = 0; j < i; ++j) {
              const Ptr<MoralNode> &old_node(nodes[j]);
              clique->try_add(old_node);
            }
          }
        }
      }

      void establish_links(UndirectedGraph<Ptr<Clique>> &cliques) const {
        using CIT = UndirectedGraph<Ptr<Clique>>::const_iterator;
        for (CIT it1 = cliques.begin(); it1 != cliques.end(); ++it1) {
          const Ptr<Clique> &first(*it1);
          CIT it2 = it1;
          ++it2;
          for (; it2 != cliques.end(); ++it2) {
            const Ptr<Clique> &second(*it2);
            if (first->shares_node_with(second)) {
              cliques.add_neighbor(first, second);
            }
          }
        }
      }

      const UndirectedGraph<Ptr<Clique>> &cliques() const {
        return cliques_;
      }

     private:
      UndirectedGraph<Ptr<Clique>> cliques_;
    };

    //==========================================================================
    // A mix of categorical and conditionally Gaussian distributions.
    // Some values are marked as known.
    class CliqueMarginalDistribution : private RefCounted {
      friend void intrusive_ptr_add_ref(CliqueMarginalDistribution *d) {
        d->up_count();
      }
      friend void intrusive_ptr_release(CliqueMarginalDistribution *d) {
        d->down_count();
        if (d->ref_count() == 0) delete d;
      }

     public:
      void marginalize();

     private:
      std::map<Ptr<Node>, int> known_discrete_variables_;
      std::map<Ptr<Node>, double> known_gaussian_variables_;

      Array unknown_discrete_distribution_;
      Vector unknown_gaussian_potential_means_;
      SpdMatrix unknown_gaussian_potential_precisions_;
    };

    inline UndirectedGraph<Ptr<Clique>> find_cliques(
        const std::vector<Ptr<MoralNode>> &nodes) {
      CliqueFinder clique_finder;
      clique_finder.find_cliques(nodes);
      return clique_finder.cliques();
    }

  } // namespace Graphical
}  // namespace BOOM

#endif  //  BOOM_GRAPHICAL_MODELS_CLIQUE_HPP_
