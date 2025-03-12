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

#include "Models/MultilevelCategoricalData.hpp"
#include "cpputil/report_error.hpp"
#include "cpputil/StringSplitter.hpp"

namespace BOOM {

  namespace {
    template <class IT>
    std::ostream &print_iterator_stack_fun(
        std::ostream &out,
        const std::vector<IT> &stack) {
      if (stack.empty()) {
        out << "Iterator stack is empty.\n";
      } else {
        out << "------\n";
        for (const auto &el : stack) {
          out << "   " << (*el)->path_from_root() << std::endl;
        }
        out << "------\n";
      }
      return out;
    }
  }  // namespace

  //======================================================================
  bool TaxNodeLess::operator()(const Ptr<TaxonomyNode> &lhs,
                               const Ptr<TaxonomyNode> &rhs) const {
    return lhs->value() < rhs->value();
  }

  bool TaxNodeStringLess::operator()(const Ptr<TaxonomyNode> &lhs,
                                     const std::string &rhs) const {
    return lhs->value() < rhs;
  }

  //======================================================================
  namespace {

    // Set 'current_end' to an iterator value corresponding to the 'end'
    // iterator for the last element in the iterator stack.
    //
    // A TaxonomyIterator moves through a taxonmy in a depth-first traversal.
    // If a node has a child, the next node will be that node's child.  If not
    // the next node will be that node's sibling.  If there are no further
    // siblings, then the next node is the parent's next sibling, and so forth.
    //
    // The algorithm needs an 'end' for the top level of the stack.  Otherwise,
    // the
    template <class IT>
    void set_current_end(const std::vector<IT> &stack,
                         const IT &top_level_end,
                         IT &current_end) {
      int n = stack.size();
      if (n < 2) {
        // If we are at the top level of the stack, then current_end =
        // top_level_end, sort of by definition.
        current_end = top_level_end;
      } else {
        // Otherwise, the ending element for the final iterator in the stack can
        // be obtained by asking its parent.
        current_end = (*stack[n-2])->end();
      }
    }

    template <class IT>
    void taxonomy_iterator_increment(std::vector<IT> &stack,
                                     IT &top_level_end,
                                     IT &current_end) {
      if (stack.empty()) {
        return;
      }
      // 'node' is the TaxonomyNode we're pointing to before we iterate.
      auto &node(*stack.back());
      if (!node->is_leaf()) {
        // Search depth first.  Unless the node is a leaf, step down to its
        // children.
        stack.push_back(node->begin());
        set_current_end(stack, top_level_end, current_end);
      } else {
        // If the node is a leaf, step to its next sibling.
        ++stack.back();
      }

      while(!stack.empty() && stack.back() == current_end) {
        // If stepping took us as far as we can go, step back up, then step over
        // to the next sibling of the parent.
        stack.pop_back();
        set_current_end(stack, top_level_end, current_end);
        if (!stack.empty()) {
          ++stack.back();
          // The 'while' loop handles the case of this being a step to the 'end'.
        }
      }
    }
  }  // namespace

  //======================================================================
  TaxonomyIterator::TaxonomyIterator(const std::vector<base_iterator> &stack,
                                     const base_iterator &top_level_end)
        : iterator_stack_(stack),
          top_level_end_(top_level_end)
  {
    set_current_end(
        iterator_stack_,
        top_level_end_,
        current_end_);
  }

  TaxonomyConstIterator::TaxonomyConstIterator(
      const std::vector<base_iterator> &stack,
      const base_iterator &top_level_end)
      : iterator_stack_(stack),
        top_level_end_(top_level_end)
  {
    set_current_end(iterator_stack_, top_level_end_, current_end_);
  }


  std::ostream &TaxonomyIterator::print_iterator_stack(
      std::ostream &out) const {
    return print_iterator_stack_fun(out, iterator_stack_);
  }

  std::ostream &TaxonomyConstIterator::print_iterator_stack(
      std::ostream &out) const {
    return print_iterator_stack_fun(out, iterator_stack_);
  }

  //---------------------------------------------------------------------------
  // This iterator is supposed to iterate over the whole tree, not just the
  // leaves.
  TaxonomyIterator &TaxonomyIterator::operator++() {
    // if (iterator_stack_.empty()) {
    //   return *this;
    // }

    // // 'node' is the TaxonomyNode we're pointing to before we iterate.
    // Ptr<TaxonomyNode> &node(*iterator_stack_.back());
    // if (!node->is_leaf()) {
    //   // Search depth first.  Unless the node is a leaf, step down to its
    //   // children.
    //   iterator_stack_.push_back(node->begin());
    //   set_current_end();
    // } else {
    //   // If the node is a leaf, step to its next sibling.
    //   ++iterator_stack_.back();
    // }

    // while(!iterator_stack_.empty() && iterator_stack_.back() == current_end_) {
    //   // If stepping took us as far as we can go, step back up, then step over
    //   // to the next sibling of the parent.
    //   iterator_stack_.pop_back();
    //   set_current_end();
    //   if (!iterator_stack_.empty()) {
    //     ++iterator_stack_.back();
    //     // The 'while' loop handles the case of this being a step to the 'end'.
    //   }
    // }
    taxonomy_iterator_increment(iterator_stack_, top_level_end_, current_end_);
    return *this;
  }

  TaxonomyConstIterator &TaxonomyConstIterator::operator++() {
    taxonomy_iterator_increment(iterator_stack_, top_level_end_, current_end_);
    return *this;
  }

  //======================================================================
  TaxonomyNode::TaxonomyNode(const std::string &value)
      : value_(value),
        position_(-1),
        parent_(nullptr)
  {}


  TaxonomyNode *TaxonomyNode::add_child(const std::string &level) {
    NEW(TaxonomyNode, child)(level);
    children_.insert(child);
    child->set_parent(this);
    return child.get();
  }

  TaxonomyNode *TaxonomyNode::find_child(const std::string &value) const {
    // auto it = children_.find(value, TaxNodeStringLess());
    // if (it == children_.end()) {
    //   return nullptr;
    // } else {
    //   return (*it).get();
    // }
    for (const Ptr<TaxonomyNode> &child : children_) {
      if (child->value() == value) {
        return child.get();
      }
    }
    return nullptr;
  }

  TaxonomyNode *TaxonomyNode::find_child(int level) const {
    if (level < 0 || level >= children_.size()) {
      std::ostringstream err;
      err << "Level " << level << " is not present under taxonomy node "
          << path_from_root();
      report_error(err.str());
    }
    return children_[level].get();
  }

  void TaxonomyNode::set_parent(TaxonomyNode *parent) {
    parent_ = parent;
  }

  bool TaxonomyNode::operator==(const TaxonomyNode &rhs) const {
      if (value_ != rhs.value_) {
        return false;
      } else if (parent_) {
        return *parent_ == *rhs.parent_;
      } else {
        return true;
      }
    }

  bool TaxonomyNode::recursive_equals(const TaxonomyNode &rhs) const {
    bool ans = true;
    if (value_ == rhs.value_ && children_.size() == rhs.children_.size()) {
      for (int i = 0; i < children_.size(); ++i) {
        ans = ans && children_[i]->recursive_equals(*rhs.children_[i]);
        if (!ans) {
          return false;
        }
      }
    } else {
      ans = false;
    }
    return ans;
  }

  void TaxonomyNode::fill_position_error(const std::vector<std::string> &values,
                                         int step) const {
    const std::string &value(values[step]);
    std::ostringstream err;
    err << value << " was not found in level " << step
        << " of the taxonomy with taxonomy element ";
    for (int i = 0; i < values.size(); ++i) {
      err << values[i];
      if (i + 1 < values.size()) {
        err << "/";
      }
    }
    err << "." << std::endl;

    err << "The available children are: \n";
    for (const auto &child : children_) {
      err << "   " << child->value() << "\n";
    }
    report_error(err.str());
  }

  void TaxonomyNode::fill_position(const std::vector<std::string> &values,
                                   std::vector<int> &output,
                                   const TaxNodeStringLess &less) const {
    if (output.size() == values.size()) {
      return;
    }
    int step = output.size();
    const std::string &value(values[step]);

    auto it = children_.find(value, less);
    if (it == children_.end()) {
      fill_position_error(values, step);
    } else {
      const Ptr<TaxonomyNode> &node(*it);
      if (node->value() == value) {
        output.push_back(node->position());
        node->fill_position(values, output, less);
      } else {
        fill_position_error(values, step);
      }
    }
  }

  std::vector<std::string> TaxonomyNode::leaf_names(char sep) const {
    std::vector<std::string> ans;
    if (children_.empty()) {
      ans.push_back(value_);
    } else {
      for (const auto &el : children_) {
        std::vector<std::string> child_levels = el->leaf_names(sep);
        for (const auto &level : child_levels) {
          ans.push_back(value_ + sep + level);
        }
      }
    }
    return ans;
  }

  std::vector<std::string> TaxonomyNode::node_names(char sep) const {
    std::vector<std::string> ans;
    ans.push_back(value_);
    for (const auto &el : children_) {
      std::vector<std::string> child_levels = el->node_names(sep);
      for (const auto &level : child_levels) {
        ans.push_back(value_ + sep + level);
      }
    }
    return ans;
  }

  std::string TaxonomyNode::path_from_root() const {
    if (!parent_) {
      return value_;
    } else {
      return parent_->path_from_root() + "/" + value_;
    }
  }

  Int TaxonomyNode::tree_size() const {
    Int ans = 1;
    for (const auto &el : children_) {
      ans += el->tree_size();
    }
    return ans;
  }

  Int TaxonomyNode::number_of_leaves() const {
    if (children_.empty()) {
      return 1;
    } else {
      Int ans = 0;
      for (const auto &el : children_) {
        ans += el->number_of_leaves();
      }
      return ans;
    }
  }

  void TaxonomyNode::finalize(int position) {
    position_ = position;
    for (size_t i = 0; i < children_.size(); ++i) {
      children_[i]->finalize(i);
    }
  }

  TaxonomyNode *TaxonomyNode::find_node(std::list<std::string> &child_levels) {
    if (child_levels.empty()) {
      return this;
    } else {
      std::string child = child_levels.front();
      child_levels.pop_front();
      TaxonomyNode *child_node = find_child(child);
      if (!child_node) {
        std::ostringstream err;
        err << "child node at level '" << child << "' not found.";
        report_error(err.str());
      }
      return child_node->find_node(child_levels);
    }
  }

  const TaxonomyNode *TaxonomyNode::find_node(std::list<std::string> &child_levels) const {
    if (child_levels.empty()) {
      return this;
    } else {
      std::string child = child_levels.front();
      child_levels.pop_front();
      const TaxonomyNode *child_node = find_child(child);
      if (!child_node) {
        std::ostringstream err;
        err << "child node at level '" << child << "' not found.";
        report_error(err.str());
      }
      return child_node->find_node(child_levels);
    }
  }

  //======================================================================
  Taxonomy::Taxonomy(const std::vector<std::vector<std::string>> &values) {
    create(values);
  }

  void Taxonomy::create(const std::vector<std::vector<std::string>> &values) {
    for (const auto &entry : values) {
      add(entry);
    }
    finalize();
  }

  Taxonomy::Taxonomy(const std::vector<std::string> &values, char sep) {
    std::string delim(1, sep);
    std::vector<std::vector<std::string>> unpacked;
    StringSplitter splitter(delim);
    for (const auto &el : values) {
      std::vector<std::string> split = splitter(el);
      unpacked.push_back(split);
    }
    create(unpacked);
  }

  bool Taxonomy::operator==(const Taxonomy &rhs) const {
    if (tree_size() != rhs.tree_size()) {
      return false;
    }

    if (top_levels_.size() != rhs.top_levels_.size()) {
      return false;
    }

    for (Int i = 0; i < top_levels_.size(); ++i) {
      if (!top_levels_[i]->recursive_equals(*rhs.top_levels_[i])) {
        return false;
      }
    }

    return true;
  }

  void Taxonomy::add(const std::vector<std::string> &element) {
    if (element.empty()) {
      return;
    }

    std::vector<std::string>::const_iterator it = element.begin();
    while (it->empty() && it != element.end()) {
      ++it;
    }
    if (it == element.end()) {
      // All entries in 'element' are empty strings.
      return;
    }

    TaxonomyNode *node = nullptr;

    for (Ptr<TaxonomyNode> &top_level : top_levels_) {
      if (top_level->value() == *it) {
        node = top_level.get();
        break;
      }
    }

    if (!node) {
      NEW(TaxonomyNode, new_node)(*it);
      top_levels_.insert(new_node);
      node = new_node.get();
    }

    // Now 'node' points to the top level node, and 'it' points to the first
    // non-empty string in 'element'.
    while (++it != element.end()) {
      TaxonomyNode *next_level = node->find_child(*it);
      if (!next_level) {
        next_level = node->add_child(*it);
      }
      node = next_level;
    }
  }

  void Taxonomy::finalize() {
    for (int i = 0; i  < top_levels_.size(); ++i) {
      top_levels_[i]->finalize(i);
    }
  }

  std::vector<int> Taxonomy::index(const std::vector<std::string> &levels) const {
    std::vector<int> ans;
    if (levels.empty()) {
      return ans;
    }
    auto it = top_levels_.find(levels[0], level_less_);
    if (it == top_levels_.end()){
      std::ostringstream err;
      err << "Initial level " << levels[0] << " not present in taxonomy.";
      report_error(err.str());
    } else {
      const Ptr<TaxonomyNode> &top(*it);
      ans.push_back(top->position());
      if (ans[0] == -1) {
        report_error("Taxonomy was never finalized.");
      }
      top->fill_position(levels, ans, level_less_);
    }
    return ans;
  }

  std::string Taxonomy::name(const std::vector<int> &values) const {
    std::ostringstream name_builder;
    if (values.empty()) {
      return "";
    }
    TaxonomyNode *node = top_levels_[values[0]].get();
    name_builder << node->value();
    for (int level = 1; level < values.size(); ++level) {
      node = node->find_child(values[level]);
      name_builder << '/' << node->value();
    }
    return name_builder.str();
  }

  void Taxonomy::ensure_valid(const std::vector<int> &values) const {
    if (values.empty()) {
      return;
    }
    TaxonomyNode *node = top_levels_[values[0]].get();
    for (int level = 1; level < values.size(); ++level) {
      node = node->find_child(values[level]);
    }
  }

  std::vector<std::string> Taxonomy::leaf_names(char sep) const {
    std::vector<std::string> ans;
    for (const auto &top : top_levels_) {
      auto entries = top->leaf_names(sep);
      std::copy(entries.begin(), entries.end(), std::back_inserter(ans));
    }
    return ans;
  }

  std::vector<std::string> Taxonomy::node_names(char sep) const {
    std::vector<std::string> ans;
    for (const auto &top : top_levels_) {
      auto entries = top->node_names(sep);
      std::copy(entries.begin(), entries.end(), std::back_inserter(ans));
    }
    return ans;
  }

  Int Taxonomy::tree_size() const {
    Int ans = 0;
    for (const auto &top : top_levels_) {
      ans += top->tree_size();
    }
    return ans;
  }

  Int Taxonomy::number_of_leaves() const {
    Int ans = 0;
    for (const auto &top : top_levels_) {
      ans += top->number_of_leaves();
    }
    return ans;
  }

  TaxonomyNode *Taxonomy::node(const std::string &level, char sep) {
    StringSplitter split(sep);
    return node(split(level));
  }

  const TaxonomyNode *Taxonomy::node(const std::string &level, char sep) const {
    StringSplitter split(sep);
    return node(split(level));
  }

  TaxonomyNode *Taxonomy::node(const std::vector<std::string> &levels) {
    if (levels.empty()) {
      return nullptr;
    }
    auto it = top_levels_.find(levels[0], level_less_);
    if (it == top_levels_.end()){
      std::ostringstream err;
      err << "Initial level " << levels[0] << " not present in taxonomy.";
      report_error(err.str());
    } else {
      const Ptr<TaxonomyNode> &top(*it);
      if (levels.size() == 1) {
        return top.get();
      } else {
        std::list<std::string> child_levels(levels.begin() + 1, levels.end());
        return top->find_node(child_levels);
      }
    }
    return nullptr;
  }

  const TaxonomyNode *Taxonomy::node(const std::vector<std::string> &levels) const {
    if (levels.empty()) {
      return nullptr;
    }
    auto it = top_levels_.find(levels[0], level_less_);
    if (it == top_levels_.end()){
      std::ostringstream err;
      err << "Initial level " << levels[0] << " not present in taxonomy.";
      report_error(err.str());
    } else {
      const Ptr<TaxonomyNode> &top(*it);
      if (levels.size() == 1) {
        return top.get();
      } else {
        std::list<std::string> child_levels(levels.begin() + 1, levels.end());
        return top->find_node(child_levels);
      }
    }
    return nullptr;
  }

  TaxonomyIterator Taxonomy::begin() {
    std::vector<TaxonomyIterator::base_iterator> iterator_stack;
    if (!top_levels_.empty()) {
      iterator_stack.push_back(top_levels_.begin());
    }
    return TaxonomyIterator(iterator_stack, top_levels_.end());
  }

  TaxonomyConstIterator Taxonomy::begin() const {
    std::vector<TaxonomyConstIterator::base_iterator> iterator_stack;
    if (!top_levels_.empty()) {
      iterator_stack.push_back(top_levels_.begin());
    }
    return TaxonomyConstIterator(iterator_stack, top_levels_.end());
  }

  TaxonomyIterator Taxonomy::end() {
    std::vector<TaxonomyIterator::base_iterator> iterator_stack;
    return TaxonomyIterator(iterator_stack, top_levels_.end());
  }

  TaxonomyConstIterator Taxonomy::end() const {
    std::vector<TaxonomyConstIterator::base_iterator> iterator_stack;
    return TaxonomyConstIterator(iterator_stack, top_levels_.end());
  }

  std::ostream & Taxonomy::print(std::ostream &out) const {
    out << "[\n";
    for (auto it = begin(); it != end(); ++it) {
      out << "   " << (*it)->path_from_root() << ",\n";
    }
    out << "]" << std::endl;
    return out;
  }

  std::string Taxonomy::to_string() const {
    std::ostringstream out;
    print(out);
    return out.str();
  }

  //======================================================================

  MultilevelCategoricalData::MultilevelCategoricalData(
      const Ptr<Taxonomy> &taxonomy)
      : taxonomy_(taxonomy)
  {}

  MultilevelCategoricalData::MultilevelCategoricalData(
      const Ptr<Taxonomy> &taxonomy,
      const std::vector<int> &values)
      : taxonomy_(taxonomy)
  {
    set(values);
  }

  MultilevelCategoricalData::MultilevelCategoricalData(
      const Ptr<Taxonomy> &taxonomy,
      const std::vector<std::string> &values)
      : taxonomy_(taxonomy)
  {
    set(values);
  }

  MultilevelCategoricalData::MultilevelCategoricalData(
      const Ptr<Taxonomy> &taxonomy,
      const std::string &value,
      char sep)
      : taxonomy_(taxonomy)
  {
    StringSplitter split(std::string(1, sep));
    set(split(value));
  }

  MultilevelCategoricalData *MultilevelCategoricalData::clone() const {
    return new MultilevelCategoricalData(*this);
  }

  std::ostream &MultilevelCategoricalData::display(std::ostream &out) const {
    out << taxonomy_->name(values_);
    return out;
  }

  void MultilevelCategoricalData::set(const std::vector<int> &values) {
    taxonomy_->ensure_valid(values);
    values_ = values;
  }

  void MultilevelCategoricalData::set(
      const std::vector<std::string> &value_names) {
    values_ = taxonomy_->index(value_names);
  }


}  // namespace BOOM
