#ifndef BOOM_MODELS_GRAPHICAL_CLIQUE_FINDER_HPP_
#define BOOM_MODELS_GRAPHICAL_CLIQUE_FINDER_HPP_

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

#include "Models/Graphical/Clique.hpp"
#include "Models/Graphical/UndirectedGraph.hpp"

namespace BOOM {
  namespace Graphical {

    class CliqueFinder {
     public:
      Ptr<Clique> add_node(const Ptr<Node> &node) {
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

      void find_cliques(const std::vector<Ptr<Node>> &nodes) {
        for (size_t i = 0; i < nodes.size(); ++i) {
          Ptr<Clique> clique = add_node(nodes[i]);
          if (!!clique) {
            for (size_t j = 0; j < i; ++j) {
              const Ptr<Node> &old_node(nodes[j]);
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


    inline UndirectedGraph<Ptr<Clique>> find_cliques(
        const std::vector<Ptr<Node>> &nodes) {
      CliqueFinder clique_finder;
      clique_finder.find_cliques(nodes);
      return clique_finder.cliques();
    }

  }  // namespace Graphical
}  // namespace BOOM

#endif  // BOOM_MODELS_GRAPHICAL_CLIQUE_FINDER_HPP_
