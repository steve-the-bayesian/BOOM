#ifndef BOOM_MODELS_MULTILEVEL_CATEGORICAL_DATA_HPP_
#define BOOM_MODELS_MULTILEVEL_CATEGORICAL_DATA_HPP_
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

#include "uint.hpp"

#include "Models/DataTypes.hpp"

#include "cpputil/SortedVector.hpp"
#include "cpputil/RefCounted.hpp"

#include <list>
#include <string>
#include <vector>
#include <ostream>

//===========================================================================
// This file defines tools for dealing with multi-level categorical data.
// Ordinary categorical data describes categories from a category list.
// MultilevelCategoricalData describes data whose values are elements of a
// Taxonomy.
//
// A Taxonomy is a nested set of categories of the form "first/second/third".
// Each level of a Taxonomy can have an arbitrary number of child levels.  The
// Taxonomy.  A MultilevelCategoricalData can assume any value of the taxonomy,
// even an interior node with children.
// ===========================================================================

namespace BOOM {

  // A TaxonomyNode is a node in the directed graph of a Taxonomy.
  class TaxonomyNode;

  //---------------------------------------------------------------------------
  // TaxonomyNode's need to be less-than comparable so we can keep them in a
  // SortedVector.
  class TaxNodeLess {
   public:
    bool operator()(const Ptr<TaxonomyNode> &lhs,
                    const Ptr<TaxonomyNode> &rhs) const;
  };

  //---------------------------------------------------------------------------
  // A "less" functor for comparing TaxonomyNode Ptr's to string's.  Used to
  // quickly find a TaxonomyNode given its value.
  class TaxNodeStringLess {
   public:
    bool operator()(const Ptr<TaxonomyNode> &lhs,
                    const std::string &rhs) const;
  };

  //===========================================================================
  // An iterator class for iterating across a taxonomy.
  class TaxonomyIterator {
   public:
    typedef SortedVector<Ptr<TaxonomyNode>>::iterator base_iterator;

    // Args:
    //   stack: The begin value for the iterator.  The END value is indicated by
    //     an empty stack.
    TaxonomyIterator(const std::vector<base_iterator> &stack,
                     const base_iterator &top_level_end);

    bool operator==(const TaxonomyIterator &rhs) const {
      return iterator_stack_ == rhs.iterator_stack_;
    }

    bool operator!=(const TaxonomyIterator &rhs) const {
      return !(*this == rhs);
    }

    Ptr<TaxonomyNode> &operator*() { return *iterator_stack_.back(); }
    TaxonomyNode * operator->() { return iterator_stack_.back()->get(); }

    TaxonomyIterator &operator++();

    std::ostream &print_iterator_stack(std::ostream &out) const;

   private:
    std::vector<base_iterator> iterator_stack_;
    base_iterator top_level_end_;
    base_iterator current_end_;
  };

  //===========================================================================
  class TaxonomyConstIterator {
   public:
    typedef SortedVector<Ptr<TaxonomyNode>>::const_iterator base_iterator;

    // Args:
    //   stack: The begin value for the iterator.  The END value is indicated by
    //     an empty stack.
    TaxonomyConstIterator(const std::vector<base_iterator> &stack,
                          const base_iterator &top_level_end);

    bool operator==(const TaxonomyConstIterator &rhs) const {
      return iterator_stack_ == rhs.iterator_stack_;
    }

    bool operator!=(const TaxonomyConstIterator &rhs) const {
      return !(*this == rhs);
    }

    const Ptr<TaxonomyNode> &operator*() { return *iterator_stack_.back(); }
    const TaxonomyNode * operator->() { return iterator_stack_.back()->get(); }

    TaxonomyConstIterator &operator++();

    std::ostream &print_iterator_stack(std::ostream &out) const;

   private:
    std::vector<base_iterator> iterator_stack_;
    base_iterator top_level_end_;
    base_iterator current_end_;
  };

  //===========================================================================
  // One element of a Taxonomy.
  class TaxonomyNode : private RefCounted {
   public:
    typedef SortedVector<Ptr<TaxonomyNode>>::iterator base_iterator;
    typedef SortedVector<Ptr<TaxonomyNode>>::const_iterator base_const_iterator;

    TaxonomyNode(const std::string &value);

    // Create a TaxonomyNode with the given value.  Add it to children_.  Return
    // a raw pointer to the created node.
    TaxonomyNode * add_child(const std::string &value);

    // If one of the children has value equal to 'level' then return a raw
    // pointer to the child.  Otherwise return nullptr.
    TaxonomyNode *find_child(const std::string &level) const;

    TaxonomyNode *find_child(int level) const;

    // Find a descendant of this node.
    TaxonomyNode *find_node(std::list<std::string> &child_levels);
    const TaxonomyNode *find_node(std::list<std::string> &child_levels) const;

    // Set the parent of this node to the supplied node.
    void set_parent(TaxonomyNode *parent);

    bool operator==(const std::string &value) const;

    // Returns true iff value_ is the same for both nodes, both nodes have the
    // same number of children, and if recursive_equals evaluates to true for
    // each child.
    bool recursive_equals(const TaxonomyNode &rhs) const;

    // The index of this node's value in the parent node of the taxonomy.
    int position() const {return position_;}

    // Fill the next entry in the vector of integer positions/indexes describing
    // the numerical value
    void fill_position(const std::vector<std::string> &values,
                       std::vector<int> &output,
                       const TaxNodeStringLess &less) const;

    // The names of all the leaves appearing underneath this node.
    std::vector<std::string> leaf_names(char sep) const;

    // The names of all the nodes in the tree rooted by this node.
    std::vector<std::string> node_names(char sep) const;

    // This taxonomy node's full name, including the names of all
    // ancestors.
    std::string path_from_root() const;

    // Two TaxonomyNode's are equal if their values are equal, and if they have
    // the same parent.
    bool operator==(const TaxonomyNode &rhs) const;

    // The text describing this level of the taxonomy.
    const std::string &value() const {return value_;}

    // The total number of nodes, including this one, in the tree headed by this
    // node.
    Int tree_size() const;

    // The total number of leaves in the tree headed by this taxonomy.  If this
    // is a leaf then the return value is 1.
    Int number_of_leaves() const;

    Int number_of_children() const {
      return children_.size();
    }

    bool is_leaf() const {
      return children_.empty();
    }

    // Args:
    //   position:  This node's position in the hierarchy below its parent node.
    //
    // Effects:
    //   The node's children are all finalized as well.
    void finalize(int position);

    base_iterator begin() {
      return children_.begin();
    }

    base_const_iterator begin() const {
      return children_.begin();
    }

    base_iterator end() {
      return children_.end();
    }

    base_const_iterator end() const {
      return children_.end();
    }

    TaxonomyNode *child(int i) {
      return children_[i].get();
    }

    const TaxonomyNode *child(int i) const {
      return children_[i].get();
    }

    std::ostream &print(std::ostream &out) const {
      out << path_from_root();
      return out;
    }

   private:
    // The text describing this node in the taxonomy.
    std::string value_;

    // Position is set to -1 on construction, and then updated when finalize is
    // called.  This is the node's position in the hierarchy relative to its
    // parent.  It is the integer value assigned to a data point that hits this
    // node.
    int position_;

    // parent_ will be nullptr for nodes at the top of the hierarchy.
    TaxonomyNode *parent_;

    // The children of this node are stored in alphabetical order by their
    // value_ attribute.
    SortedVector<Ptr<TaxonomyNode>, TaxNodeLess> children_;

    // To be called when 'fill_position' encounters an error condition.  Throws
    // an exception an error message summarizing the information in 'values' and
    // 'step'.
    void fill_position_error(const std::vector<std::string> &values,
                             int step) const;

    friend void intrusive_ptr_add_ref(TaxonomyNode *node) {
      node->up_count();
    }

    friend void intrusive_ptr_release(TaxonomyNode *node) {
      node->down_count();
      if (node->ref_count() == 0) {
        delete node;
      }
    }
  };

  inline std::ostream &operator<<(std::ostream &out, const TaxonomyNode &node) {
    return node.print(out);
  }

  //===========================================================================
  // A hierarchical collection of taxonomy values, stored as a tree of
  // TaxonomyNode objects.  See comments at the top of the file for more.
  class Taxonomy : private RefCounted {
   public:
    // Args:
    //   levels: element [i] is a vector of levels for a single taxonomy
    //     element.  Element [i][j] is nested within element[i][j - 1].
    //     Example: ["shopping", "clothes", "shoes"].
    Taxonomy(const std::vector<std::vector<std::string>> &levels);

    // Args:
    //   levels: element [i] is a single taxonomy element of the form
    //     "shopping/clothes/shoes".
    //   sep: The field delimiter used to separate values in 'levels'.
    Taxonomy(const std::vector<std::string> &levels, char sep='/');

    bool operator==(const Taxonomy &rhs) const;
    bool operator!=(const Taxonomy &rhs) const {
      return !(rhs == *this);
    }

    // Args:
    //   levels:  The taxonomy levels identifying an observation.
    //
    // Returns:
    //   The numeric indices of each supplied level.
    //
    // Throws:
    //   If a taxonomy level is supplied, but that level is not present in the
    //   taxonomy then an exception is thrown.
    std::vector<int> index(const std::vector<std::string> &levels) const;

    // Args:
    //   values:  Numeric indices describing a taxonomy level.
    //
    // Returns:
    //   The name of the taxonomy level described by the indices.
    //
    // Throws:
    //   An exception is thrown if the values do not correspond to a valid
    //   taxonomy element.
    std::string name(const std::vector<int> &values) const;

    // Args:
    //   values: Integer indexes of a taxonomy level. For example, [0, 3, 2]
    //     means level 0 of taxonomy level 1, level 3 in the children of level
    //     0, and level 2 in the children of [0, 3].
    //
    // Throws:
    //   An exception is thrown if the level set does not exist within the
    //   taxonomy.  Otherwise nothing happens.
    void ensure_valid(const std::vector<int> &values) const;

    // The names of the leaf nodes in the taxonomy.
    std::vector<std::string> leaf_names(char sep='/') const;

    // The names of all nodes in the taxonomy, including interior nodes.
    std::vector<std::string> node_names(char sep='/') const;

    // The total number of nodes in the taxonomy, including interior nodes.
    Int tree_size() const;

    // The total number of leaves in the taxonomy.
    Int number_of_leaves() const;

    // The number of entries in the top level of the taxonomy.
    Int top_level_size() const {
      return top_levels_.size();
    }

    TaxonomyNode *top_level_node(int i) {
      return top_levels_[i].get();
    }

    const TaxonomyNode *top_level_node(int i) const {
      return top_levels_[i].get();
    }

    TaxonomyNode *node(const std::string &level, char sep='/');
    const TaxonomyNode *node(const std::string &level, char sep='/') const;

    TaxonomyNode *node(const std::vector<std::string> &levels);
    const TaxonomyNode *node(const std::vector<std::string> &levels) const;

    TaxonomyIterator begin();
    TaxonomyIterator end();

    TaxonomyConstIterator begin() const;
    TaxonomyConstIterator end() const;

    std::ostream &print(std::ostream &out) const;
    std::string to_string() const;

   private:
    // The first levels of the taxonomy tree.
    SortedVector<Ptr<TaxonomyNode>, TaxNodeLess> top_levels_;
    TaxNodeStringLess level_less_;

    // To be called during construction.
    void create(const std::vector<std::vector<std::string>> &levels);

    // Add an element to the taxonomy.  If the element is already present the
    // taxonomy remains unchanged.
    //
    // Args:
    //   element: A sequence of nested taxonomy levels.  For example
    //     ["shopping", "clothes", "shoes"].
    void add(const std::vector<std::string> &element);

    // To be called after the last taxonomy element has been added.
    //
    // Effects: Each TaxonomyNode is informed of its position relative to its
    // parent.
    void finalize();

    friend void intrusive_ptr_add_ref(Taxonomy *taxonomy) {
      taxonomy->up_count();
    }

    friend void intrusive_ptr_release(Taxonomy *taxonomy) {
      taxonomy->down_count();
      if (taxonomy->ref_count() == 0) {
        delete taxonomy;
      }
    }

  };

  // Read a taxonomy from a collection of strings of the form L1/L2/L3
  // describing different taxonomy levels separated by a character (by default
  // '/')
  Ptr<Taxonomy> read_taxonomy(const std::vector<std::string> &values,
                              char sep='/');

  // Read a taxonomy from a collection of strings of the form
  // [[L1, L2, L3],
  //  [L1, L2],
  //  [L1, L2, L3, L4]]
  Ptr<Taxonomy> read_taxonomy(
      const std::vector<std::vector<std::string>> &values);

  inline std::ostream &operator<<(std::ostream &out, const Taxonomy &tax) {
    return tax.print(out);
  }

  //===========================================================================
  // A categorical variable taking values from a taxonomy.  See comments at the
  // top of the file for more.
  class MultilevelCategoricalData : public Data {
   public:
    // An empty, invalid categorical data point.  Values should be assigned
    // later.
    //
    // Args:
    //   taxonomy:  The set of possible taxonomy values.
    MultilevelCategoricalData(const Ptr<Taxonomy> &taxonomy);

    // Args:
    //   taxonomy:  The set of possible taxonomy values.
    //   values:  The specific values for this data point.
    //
    // values = {2, 1, 7} means that this data point is an observation from
    // value 2 of the top taxonomy level, value 1 from the children of that
    // level, and value 7 from the children of level {2, 1}.
    MultilevelCategoricalData(const Ptr<Taxonomy> &taxonomy,
                              const std::vector<int> &values);

    // Args:
    //   taxonomy:  The set of possible taxonomy values.
    //   values:  The specific values for this data point.
    //
    // values = {"Animalia", "Chordata", "Mammalia"} means level "chordata"
    // within the top level "animalia" and level "mammalia" within the 
    MultilevelCategoricalData(const Ptr<Taxonomy> &taxonomy,
                              const std::vector<std::string> &values);
    
    MultilevelCategoricalData(const Ptr<Taxonomy> &taxonomy,
                              const std::string &value,
                              char sep = '/');

    MultilevelCategoricalData *clone() const override;
    std::ostream &display(std::ostream &out) const override;

    void set(const std::vector<int> &values);
    void set(const std::vector<std::string> &values);

    const std::vector<int> &levels() const {return values_;}
    std::string name() const {return taxonomy_->name(values_);}

    const Ptr<Taxonomy> &taxonomy() const {return taxonomy_;}

   private:
    Ptr<Taxonomy> taxonomy_;
    std::vector<int> values_;
  };
  
}  // namespace BOOM



#endif  //  BOOM_MODELS_MULTILEVEL_CATEGORICAL_DATA_HPP_
