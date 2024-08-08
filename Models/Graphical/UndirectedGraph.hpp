#ifndef BOOM_MODELS_GRAPHICAL_UNDIRECTED_GRAPH_HPP_
#define BOOM_MODELS_GRAPHICAL_UNDIRECTED_GRAPH_HPP_

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

#include <map>
#include <set>
#include <ostream>
#include "cpputil/report_error.hpp"

namespace BOOM {

  //===========================================================================
  // An iterator over node in an UndirectedGraph.
  template <class ELEMENT>
  class UndirectedGraphIterator {
   public:
    using element_type=ELEMENT;

    using BaseIteratorType = typename std::map<
      element_type, std::set<element_type>>::iterator;

    UndirectedGraphIterator(BaseIteratorType it)
        : base_iterator_(it)
    {}

    const ELEMENT &operator*() const {
      return base_iterator_->first;
    }

    bool operator==(const UndirectedGraphIterator &rhs) const {
      return base_iterator_ == rhs.base_iterator_;
    }

    bool operator!=(const UndirectedGraphIterator &rhs) const {
      return base_iterator_ != rhs.base_iterator_;
    }

    bool operator>(const UndirectedGraphIterator &rhs) const {
      return base_iterator_ > rhs.base_iterator_;
    }

    bool operator>=(const UndirectedGraphIterator &rhs) const {
      return base_iterator_ >= rhs.base_iterator_;
    }

    bool operator<(const UndirectedGraphIterator &rhs) const {
      return base_iterator_ < rhs.base_iterator_;
    }

    bool operator<=(const UndirectedGraphIterator &rhs) const {
      return base_iterator_ <= rhs.base_iterator_;
    }

    UndirectedGraphIterator &operator++() {
      ++base_iterator_;
      return *this;
    }

    UndirectedGraphIterator operator++(int) {
      UndirectedGraphIterator ans(*this);
      ++base_iterator_;
      return ans;
    }

    UndirectedGraphIterator operator--(int) {
      UndirectedGraphIterator ans(*this);
      --base_iterator_;
      return *this;
    }

    const BaseIteratorType &base() const {
      return base_iterator_;
    }

   private:
    BaseIteratorType base_iterator_;
  };

  //===========================================================================
  // A const iterator over node in an UndirectedGraph.
  template <class ELEMENT>
  class UndirectedGraphConstIterator {
   public:
    using BaseIteratorType = typename std::map<
     ELEMENT, std::set<ELEMENT>>::const_iterator;

    UndirectedGraphConstIterator(BaseIteratorType it)
        : base_iterator_(it)
    {}

    // Allow implicit conversion.
    UndirectedGraphConstIterator(
        const UndirectedGraphIterator<ELEMENT> &non_const)
        : base_iterator_(non_const.base())
    {}

    const ELEMENT &operator*() const {
      return base_iterator_->first;
    }

    bool operator==(const UndirectedGraphConstIterator &rhs) const {
      return base_iterator_ == rhs.base_iterator_;
    }

    bool operator!=(const UndirectedGraphConstIterator &rhs) const {
      return base_iterator_ != rhs.base_iterator_;
    }

    bool operator>(const UndirectedGraphConstIterator &rhs) const {
      return base_iterator_ > rhs.base_iterator_;
    }

    bool operator>=(const UndirectedGraphConstIterator &rhs) const {
      return base_iterator_ >= rhs.base_iterator_;
    }

    bool operator<(const UndirectedGraphConstIterator &rhs) const {
      return base_iterator_ < rhs.base_iterator_;
    }

    bool operator<=(const UndirectedGraphConstIterator &rhs) const {
      return base_iterator_ <= rhs.base_iterator_;
    }

    UndirectedGraphConstIterator &operator++() {
      ++base_iterator_;
      return *this;
    }

    UndirectedGraphConstIterator operator++(int) {
      UndirectedGraphConstIterator ans(*this);
      ++base_iterator_;
      return ans;
    }

    UndirectedGraphConstIterator operator--(int) {
      UndirectedGraphConstIterator ans(*this);
      --base_iterator_;
      return *this;
    }

    const BaseIteratorType &base() const {
      return base_iterator_;
    }

   private:
    BaseIteratorType base_iterator_;
  };

  //===========================================================================
  // A generic undirected graph that stores
  template <class ELEMENT>
  class UndirectedGraph {
   public:
    using iterator = UndirectedGraphIterator<ELEMENT>;
    using const_iterator = UndirectedGraphConstIterator<ELEMENT>;
    using size_type = size_t;
    using value_type = ELEMENT;

    std::set<ELEMENT> &neighbors(const ELEMENT &element) {
      auto it = neighbors_.find(element);
      if (it == neighbors_.end()) {
        report_error("Element not found.");
      }
      return it->second;
    }

    const std::set<ELEMENT> &neighbors(const ELEMENT &element) const {
      auto it = neighbors_.find(element);
      if (it == neighbors_.end()) {
        report_error("Element not found.");
      }
      return it->second;
    }

    void add_element(const ELEMENT &element) {
      auto it = neighbors_.find(element);
        if (it == neighbors_.end()) {
          std::set<ELEMENT> empty;
          neighbors_[element] = empty;
        }
    }

    void add_neighbor(const ELEMENT &element,
                      const ELEMENT &neighbor) {
      neighbors_[element].insert(neighbor);
      neighbors_[neighbor].insert(element);
    }

    // STL Container interface
    UndirectedGraphIterator<ELEMENT> begin() {
      return UndirectedGraphIterator<ELEMENT>(neighbors_.begin());
    }

    UndirectedGraphConstIterator<ELEMENT> begin() const {
      return UndirectedGraphConstIterator<ELEMENT>(neighbors_.cbegin());
    }

    UndirectedGraphConstIterator<ELEMENT> cbegin() const {
      return UndirectedGraphIterator<ELEMENT>(neighbors_.cbegin());
    }

    UndirectedGraphIterator<ELEMENT> end() {
      return UndirectedGraphIterator<ELEMENT>(neighbors_.end());
    }

    UndirectedGraphConstIterator<ELEMENT> end() const {
      return UndirectedGraphConstIterator<ELEMENT>(neighbors_.cend());
    }

    UndirectedGraphConstIterator<ELEMENT> cend() const {
      return UndirectedGraphIterator<ELEMENT>(neighbors_.cend());
    }

    // Common interface with set and map.
    bool empty() const {
      return neighbors_.empty();
    }

    size_t size() const {
      return neighbors_.size();
    }

    void clear() {
      neighbors_.clear();
    }

    UndirectedGraphConstIterator<ELEMENT> find(const ELEMENT &element) const {
      return UndirectedGraphConstIterator<ELEMENT>(neighbors_.find(element));
    }

    void erase(const UndirectedGraphConstIterator<ELEMENT> &it) {
      neighbors_.erase(it.base());
    }

    // TODO: there are a few more functions in std::set that would be nice to
    // have here for completeness.

    std::ostream &print(std::ostream &out) const {
      for (const auto &el : neighbors_) {
        out << el.first << " |";
        for (const auto &neighbor : el.second) {
          out << " " << neighbor;
        }
        out <<"\n";
      }
      return out;
    }

   private:
    std::map<ELEMENT, std::set<ELEMENT>> neighbors_;
  };

  template <class ELEMENT>
  std::ostream & operator<<(std::ostream &out,
                            const UndirectedGraph<ELEMENT> &graph) {
    return graph.print(out);
  }

}  // namespace BOOM

#endif  // BOOM_MODELS_GRAPHICAL_UNDIRECTED_GRAPH_HPP_
