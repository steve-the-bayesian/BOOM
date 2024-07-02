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
#include "Models/Graphical/Node.hpp"

#include <sstream>

namespace BOOM {
  namespace Graphical {

    const std::string &Clique::name() const {
      std::ostringstream name_maker;
      for (int i = 0; i < elements_.size(); ++i) {
        name_maker << elements_[i]->name();
        if (i + 1 < elements_.size()) {
          name_maker << ":";
        }
      }
      Node::set_name(name_maker.str());
      return Node::name();
    }

    bool Clique::try_add(const Ptr<Node> &node) {
      for (const auto &el : elements_) {
        if (!node->is_neighbor(el)) {
          return false;
        }
      }
      elements_.insert(node);
      return true;
    }

    class CliqueFinder {
     public:
      bool add_node(const Ptr<Node> &node) {
        int membership_count = 0;
        for (size_t i = 0; i < cliques_.size(); ++i) {
          membership_count += cliques_[i]->try_add(node);
        }

        if (membership_count == 0) {
          NEW(Clique, clique)(cliques_.size());
          clique->try_add(node);
          cliques_.push_back(clique);
          return true;
        } else {
          return false;
        }
      }

      void find_cliques(const std::vector<Ptr<Node>> &nodes) {
        for (size_t i = 0; i < nodes.size(); ++i) {
          const Ptr<Node> &node(nodes[i]);
          bool added_new_clique = add_node(node);
          if (added_new_clique) {
            Ptr<Clique> &clique(cliques_.back());
            for (size_t j = 0; j < i; ++j) {
              const Ptr<Node> &old_node(nodes[j]);
              clique->try_add(old_node);
            }
          }
        }
      }

      void establish_links(std::vector<Ptr<Clique>> &cliques) const {
        for (size_t i = 0; i < cliques.size(); ++i ){
          Ptr<Clique> &first(cliques[i]);
          for (size_t j = i + 1; j < cliques.size(); ++j) {
            Ptr<Clique> &second(cliques[j]);
            if (first->shares_node_with(second)) {
              first->add_neighbor(second);
            }
          }
        }
      }

      const std::vector<Ptr<Clique>> &cliques() const {
        return cliques_;
      }

     private:
      std::vector<Ptr<Clique>> cliques_;
    };

    std::vector<Ptr<Clique>> find_cliques(const std::vector<Ptr<Node>> &nodes) {
      CliqueFinder clique_finder;
      clique_finder.find_cliques(nodes);
      return clique_finder.cliques();
    }

  }  // namespace Graphical
}  // namespace BOOM
