// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005 Steven L. Scott

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

#include "Models/CategoricalData.hpp"
#include <algorithm>
#include <fstream>
#include <set>
#include <sstream>
#include <utility>
#include "cpputil/report_error.hpp"
#include "cpputil/StringSplitter.hpp"


namespace BOOM {

  void CatKeyBase::Register(CategoricalData *dp) {
    observers_.insert(dp);
    dp->set_key(this);
  }

  void CatKeyBase::Remove(CategoricalData *dat) {
    observers_.erase(dat);
    if (dat->key().get() == this) {
      dat->set_key(nullptr);
    }
  }

  std::ostream &CatKeyBase::print(uint value, std::ostream &out) const {
    out << value;
    return out;
  }

  uint CatKeyBase::findstr(const std::string &s) const {
    report_error(
        "A string value was used with a categorical variable that "
        "does not support string operations.");
    return 0;
  }
  //======================================================================
  UnboundedIntCatKey *UnboundedIntCatKey::clone() const {
    return new UnboundedIntCatKey(*this);
  }

  std::ostream &UnboundedIntCatKey::print(std::ostream &out) const {
    out << "Numeric data with no upper bound.";
    return out;
  }
  //======================================================================
  FixedSizeIntCatKey *FixedSizeIntCatKey::clone() const {
    return new FixedSizeIntCatKey(*this);
  }

  std::ostream &FixedSizeIntCatKey::print(std::ostream &out) const {
    out << "Numeric data with upper bound " << max_levels();
    return out;
  }
  //======================================================================
  CatKey::CatKey() : grow_(true) {}

  CatKey::CatKey(int number_of_levels) : labs_(number_of_levels), grow_(false) {
    for (int i = 0; i < number_of_levels; ++i) {
      std::ostringstream label;
      label << "level_" << i;
      labs_[i] = label.str();
    }
  }

  CatKey::CatKey(const std::vector<std::string> &labels)
      : labs_(labels), grow_(false) {}

  CatKey *CatKey::clone() const {
    return new CatKey(*this);
  }

  void CatKey::allow_growth(bool allow) { grow_ = allow; }

  void CatKey::Register(CategoricalData *dp) {
    CatKeyBase::Register(dp);
    if (dp->value() >= labs_.size()) {
      report_error("Illegal value passed to CatKey::Register");
    }
  }

  void CatKey::RegisterWithLabel(CategoricalData *dp,
                                 const std::string &label) {
    CatKeyBase::Register(dp);
    bool found = true;
    uint numeric_value_for_label = findstr_safe(label, found);
    if (found) {
      dp->set(numeric_value_for_label);
    } else {
      if (grow_) {
        add_label(label);
        dp->set(findstr_safe(label, found));
      } else {
        report_error("illegal label passed to CatKey::Register");
      }
    }
  }

  const std::vector<std::string> &CatKey::labels() const { return labs_; }

  uint CatKey::findstr_safe(const std::string &label, bool &found) const {
    std::vector<std::string>::const_iterator it =
        std::find(labs_.begin(), labs_.end(), label);
    if (it == labs_.end()) {
      found = false;
      return labs_.size();
    }
    found = true;
    return distance(labs_.begin(), it);
  }

  uint CatKey::findstr(const std::string &lab) const {
    bool found(true);
    uint ans = findstr_safe(lab, found);
    if (!found) {
      ostringstream out;
      out << "label " << lab << " not found in CatKey::findstr";
      report_error(out.str());
    }
    return ans;
  }

  int CatKey::findstr_or_neg(const std::string &lab) const {
    bool found = true;
    int position = findstr_safe(lab, found);
    return found ? position : -1;
  }

  void CatKey::add_label(const std::string &lab) { labs_.push_back(lab); }

  void CatKey::reorder(const std::vector<std::string> &new_ordering) {
    if (labs_ == new_ordering) {
      return;
    }
    assert(new_ordering.size() == labs_.size());
    std::vector<uint> new_vals(labs_.size());
    for (uint i = 0; i < labs_.size(); ++i) {
      std::string lab = labs_[i];
      for (uint j = 0; j < new_ordering.size(); ++j) {
        if (lab == new_ordering[j]) {
          new_vals[i] = j;
          break;
        }
      }
    }
    for (auto &el : observers()) {
      el->set(new_vals[el->value()]);
    }
    labs_ = new_ordering;
  }

  void CatKey::relabel(const std::vector<std::string> &new_labels) {
    if (labs_ == new_labels) {
      return;
    }
    assert(new_labels.size() == labs_.size());
    labs_ = new_labels;
  }

  std::ostream &CatKey::print(std::ostream &out) const {
    uint nlab = labs_.size();
    for (uint i = 0; i < nlab; ++i) {
      out << "level " << i << " = " << labs_[i] << std::endl;
    }
    return out;
  }

  std::ostream &CatKey::print(uint value, std::ostream &out) const {
    if (value >= labs_.size()) {
      out << "NA";
    } else {
      out << labs_[value];
    }
    return out;
  }

  void CatKey::set_levels(const std::vector<std::string> &new_ordering) {
    uint nobs = observers().size();
    if (!labs_.empty() && nobs > 0) {
      std::vector<uint> new_pos = map_levels(new_ordering);
      for (auto &dp : observers()) {
        uint old = dp->value();
        uint y = new_pos[old];
        dp->set(y);
      }
    }
    labs_ = new_ordering;
  }

  std::vector<uint> CatKey::map_levels(
      const std::vector<std::string> &sv) const {
    std::vector<uint> new_pos(labs_.size());
    for (uint i = 0; i < labs_.size(); ++i) {
      std::string s = labs_[i];
      for (uint j = 0; j < sv.size(); ++j) {
        if (sv[j] == s) {
          new_pos[i] = j;
          break;
        } else {
          ostringstream err;
          err << "CatKey::map_levels:  the replacement set of category "
              << "labels is not a superset of the original labels." << endl
              << "Could not find level: " << labs_[i]
              << " in replacement labels." << endl;
          report_error(err.str());
        }
      }
    }
    return new_pos;
  }

  //======================================================================
  CategoricalData::~CategoricalData() { key_->Remove(this); }

  CategoricalData::CategoricalData(uint val, uint Nlevels)
      : val_(val), key_(new FixedSizeIntCatKey(Nlevels)) {
    key_->Register(this);
  }

  CategoricalData::CategoricalData(uint val, const Ptr<CatKeyBase> &key)
      : val_(val), key_(key) {
    key_->Register(this);
  }

  CategoricalData::CategoricalData(uint val, CategoricalData &other)
      : val_(val), key_(other.key_) {
    key_->Register(this);
  }

  CategoricalData::CategoricalData(const std::string &label,
                                   const Ptr<CatKey> &key)
      : key_(key) {
    key->RegisterWithLabel(this, label);
  }

  CategoricalData *CategoricalData::clone() const {
    return new CategoricalData(*this);
  }

  //------------------------------------------------------------
  void CategoricalData::set(const uint &value, bool signal_observers) {
    if (key_->max_levels() > 0 && value >= key_->max_levels()) {
      ostringstream msg;
      msg << "CategoricalData::set() argument " << value
          << " exceeds "
             "maximum number of levels.";
      report_error(msg.str());
    }
    val_ = value;
    if (signal_observers) {
      signal();
    }
  }

  //------------------------------------------------------------
  bool CategoricalData::operator==(uint rhs) const { return val_ == rhs; }

  bool CategoricalData::operator==(const CategoricalData &rhs) const {
    return val_ == rhs.val_;
  }

  //------------------------------------------------------------

  uint CategoricalData::nlevels() const { return key_->max_levels(); }
  // note:  key_ must be set in order for this to work

  const uint &CategoricalData::value() const { return val_; }

  bool CategoricalData::comparable(const CategoricalData &rhs) const {
    return key_ == rhs.key_;
  }

  inline void incompatible_category_error() {
    report_error("comparison between incompatible categorical variables");
  }
  //------------------------------------------------------------
  std::ostream &CategoricalData::display(std::ostream &out) const {
    return key_->print(value(), out);
  }

  void CategoricalData::print_key(std::ostream &out) const {
    out << *key_ << endl;
  }

  //======================================================================
  LabeledCategoricalData::LabeledCategoricalData(
      const std::string &value,
      const Ptr<CatKey> &key)
      : CategoricalData(key->findstr(value), key),
        catkey_(key)
  {}

  LabeledCategoricalData::LabeledCategoricalData(
      uint value, const Ptr<CatKey> &key)
      : CategoricalData(value, key),
        catkey_(key)
  {}

  LabeledCategoricalData *LabeledCategoricalData::clone() const {
    return new LabeledCategoricalData(*this);
  }

  //======================================================================
  OrdinalData::OrdinalData(uint value, uint Nlevels)
      : CategoricalData(value, Nlevels) {}

  OrdinalData::OrdinalData(uint value, const Ptr<CatKeyBase> &key)
      : CategoricalData(value, key) {}

  OrdinalData::OrdinalData(const std::string &label, const Ptr<CatKey> &key)
      : CategoricalData(label, key) {}

  OrdinalData::OrdinalData(const OrdinalData &rhs)
      : Data(rhs), CategoricalData(rhs) {}

  OrdinalData *OrdinalData::clone() const { return new OrdinalData(*this); }

  bool OrdinalData::operator<(uint rhs) const { return value() < rhs; }
  bool OrdinalData::operator<=(uint rhs) const { return value() <= rhs; }
  bool OrdinalData::operator>(uint rhs) const { return value() > rhs; }
  bool OrdinalData::operator>=(uint rhs) const { return value() >= rhs; }

  bool OrdinalData::operator<(const std::string &rhs) const {
    uint v = key()->findstr(rhs);
    return value() < v;
  }
  bool OrdinalData::operator<=(const std::string &rhs) const {
    uint v = key()->findstr(rhs);
    return value() <= v;
  }
  bool OrdinalData::operator>(const std::string &rhs) const {
    uint v = key()->findstr(rhs);
    return value() > v;
  }
  bool OrdinalData::operator>=(const std::string &rhs) const {
    uint v = key()->findstr(rhs);
    return value() > v;
  }

  bool OrdinalData::operator<(const OrdinalData &rhs) const {
    if (!comparable(rhs)) {
      incompatible_category_error();
    }
    return value() < rhs.value();
  }
  bool OrdinalData::operator<=(const OrdinalData &rhs) const {
    if (!comparable(rhs)) {
      incompatible_category_error();
    }
    return value() <= rhs.value();
  }
  bool OrdinalData::operator>(const OrdinalData &rhs) const {
    if (!comparable(rhs)) {
      incompatible_category_error();
    }
    return value() > rhs.value();
  }
  bool OrdinalData::operator>=(const OrdinalData &rhs) const {
    if (!comparable(rhs)) {
      incompatible_category_error();
    }
    return value() >= rhs.value();
  }
  
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
      std::ostringstream err;
      err << values[step] << " was not found in level " << step
          << " of the taxonomy with taxonomy element ";
      for (int i = 0; i < values.size(); ++i) {
        err << value;
        if (i + 1 < values.size()) {
          err << "/";
        }
      }
      err << ".";

      err << "The available children are: \n";
      for (const auto &child : children_) {
        err << "   " << child->value() << "\n";
      }
      report_error(err.str());
    } else {
      const Ptr<TaxonomyNode> &node(*it);
      output.push_back(node->position());
      node->fill_position(values, output, less);
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
  
  //======================================================================

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

  TaxonomyIterator Taxonomy::begin() {
    std::vector<TaxonomyIterator::base_iterator> iterator_stack;
    if (!top_levels_.empty()) {
      iterator_stack.push_back(top_levels_.begin());
      top_levels_[0]->descend_left(iterator_stack);
    }
    return TaxonomyIterator(top_levels_[0]->descend_left(iterator_stack),
                            top_levels_.end());
    
  }

  TaxonomyIterator Taxonomy::end() {
    std::vector<TaxonomyIterator::base_iterator> iterator_stack;
    return TaxonomyIterator(iterator_stack, top_levels_.end());
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
  
  //======================================================================

  Ptr<Taxonomy> read_taxonomy(const std::vector<std::string> &values,
                              char sep) {
    std::string delim(1, sep);
    std::vector<std::vector<std::string>> unpacked;
    StringSplitter splitter(delim);
    for (const auto &el : values) {
      std::vector<std::string> split = splitter(el);
      unpacked.push_back(split);
    }
    return read_taxonomy(unpacked);
  }

  Ptr<Taxonomy> read_taxonomy(
      const std::vector<std::vector<std::string>> &values) {
    NEW(Taxonomy, tax)();
    
    for (const auto &entry : values) {
      tax->add(entry);
    }

    tax->finalize();
    return tax;
  }
  
  //======================================================================
  
  Ptr<CatKey> make_catkey(const std::vector<std::string> &sv) {
    std::vector<std::string> tmp(sv);
    std::sort(tmp.begin(), tmp.end());
    std::vector<std::string> labs;
    std::unique_copy(tmp.begin(), tmp.end(), back_inserter(labs));
    return new CatKey(labs);
  }

  std::vector<Ptr<CategoricalData> > create_categorical_data(
      const std::vector<std::string> &sv) {
    uint n = sv.size();
    Ptr<CatKey> labs = make_catkey(sv);
    std::vector<Ptr<CategoricalData> > ans(n);
    for (uint i = 0; i < n; ++i) {
      ans[i] = new CategoricalData(sv[i], labs);
    }
    return ans;
  }

  namespace {
    // Create vectors of pointers to CategoricalData from vectors of "integers"
    // including int, long, unsigned long, etc.
    template <class INT>
    std::vector<Ptr<CategoricalData>> create_integer_categorical_data(
        const std::vector<INT> &input) {
      INT max = *std::max_element(input.begin(), input.end());
      Ptr<CatKeyBase> labs = new FixedSizeIntCatKey(max + 1);
      std::vector<Ptr<CategoricalData>> ans;
      for (auto el : input) {
        ans.push_back(new CategoricalData(el, labs));
      }
      return ans;
    }
  }  // namespace

  std::vector<Ptr<CategoricalData> > create_categorical_data(
      const std::vector<uint> &iv) {
    return create_integer_categorical_data<uint>(iv);
  }
  std::vector<Ptr<CategoricalData> > create_categorical_data(
      const std::vector<int> &iv) {
    return create_integer_categorical_data<int>(iv);
  }

  std::vector<Ptr<OrdinalData>>
  create_ordinal_data(const std::vector<uint> &iv) {
    uint n = iv.size();
    uint Max = 0;
    for (uint i = 0; i < n; ++i) {
      Max = std::max(iv[i], Max);
    }
    Ptr<CatKeyBase> key(new FixedSizeIntCatKey(Max + 1));
    std::vector<Ptr<OrdinalData> > ans;
    ans.reserve(n);
    for (uint i = 0; i < n; ++i) {
      ans.push_back(new OrdinalData(iv[i], key));
    }
    return ans;
  }

}  // namespace BOOM
