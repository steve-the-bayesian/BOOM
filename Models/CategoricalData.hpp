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

#ifndef BOOM_CATEGORICAL_DATA_HPP
#define BOOM_CATEGORICAL_DATA_HPP

#include <set>
#include <vector>
#include "Models/DataTypes.hpp"
#include "cpputil/RefCounted.hpp"
#include "stats/FreqDist.hpp"

namespace BOOM {

  class CategoricalData;
  //======================================================================
  // A CatKeyBase manages the behavior of a CategoricalData object.
  //
  // a CatKey should be held by a smart pointer, as it will be shared
  // by many individual CategoricalData.  It contains a set of
  // observers, which must be dumb pointers.
  class CatKeyBase : private RefCounted {
   public:
    virtual CatKeyBase *clone() const = 0;

    // Establish a parent child relationship between the key and the
    // CategoricalData.  The argument adopts *this as its key, and the argument
    // is added to the set of observers.
    virtual void Register(CategoricalData *);

    // Removes the argument from the set of observers.
    void Remove(CategoricalData *);

    // If positive then the integer values that the CategoricalData can assume
    // have a fixed upper limit of max_levels() - 1.  If non-positive then there
    // is no upper limit on the value of each CategoricalData object.
    virtual int max_levels() const { return -1; }

    // Print the value of this key to the stream 'out'.
    virtual std::ostream &print(std::ostream &out) const = 0;

    // Print the label corresponding to 'value' on the stream 'out'.
    virtual std::ostream &print(uint value, std::ostream &out) const;

    // If true then the CategoricalData can be worked with as strings.
    virtual bool allows_strings() const { return false; }

    // Returns the numeric value associated with the string 's'.  The default
    // implementation of findstr throws an error.  It should be overwritten for
    // key types that allow for string values.
    virtual uint findstr(const std::string &s) const;

   protected:
    std::set<CategoricalData *> &observers() { return observers_; }

   private:
    friend void intrusive_ptr_add_ref(CatKeyBase *k) { k->up_count(); }
    friend void intrusive_ptr_release(CatKeyBase *k) {
      k->down_count();
      if (k->ref_count() == 0) {
        delete k;
      }
    }

    // The observers are the categorical data objects using *this as a key.
    // They are stored as ordinary pointers, not smart pointers, to avoid an
    // ownership loop.
    std::set<CategoricalData *> observers_;
  };

  //======================================================================
  // Numerical categorical data with no upper limit.
  class UnboundedIntCatKey : public CatKeyBase {
   public:
    UnboundedIntCatKey() = default;
    UnboundedIntCatKey *clone() const override;
    std::ostream &print(std::ostream &out) const override;
  };

  //======================================================================
  // Numerical categorical data with a fixed number of levels.
  class FixedSizeIntCatKey : public CatKeyBase {
   public:
    explicit FixedSizeIntCatKey(int max_levels) : max_levels_(max_levels) {}
    FixedSizeIntCatKey *clone() const override;
    int max_levels() const override { return max_levels_; }
    std::ostream &print(std::ostream &out) const override;

   private:
    int max_levels_;
  };
  //============================================================================
  // Categorical data based on labels.  This class keeps track of the mapping
  // between text labels and numerical values.
  class CatKey : public CatKeyBase {
   public:
    CatKey();
    explicit CatKey(int number_of_levels);
    explicit CatKey(const std::vector<std::string> &labels);
    CatKey(const CatKey &rhs) = default;

    CatKey *clone() const override;

    // Sets the 'grow_' flag.  If true then the max number of levels increases
    // by 1 when a new level is provided using RegisterWithLabel.  If false then
    // RegisterWithLabel will throw an error when a new label is seen.
    void allow_growth(bool allow = true);

    void Register(CategoricalData *dp) override;

    // Register a CategoricalData object that has a particular label.
    // Args:
    //   dp:  A pointer to the data object to be registered.
    //   label:  The label associated with dp.
    void RegisterWithLabel(CategoricalData *dp, const std::string &label);

    bool allows_strings() const override { return true; }
    int max_levels() const override { return labs_.size(); }

    const std::vector<std::string> &labels() const;
    const std::string &label(int value) const { return labs_[value]; }

    // Returns the position in labs_ containing the string 's'.  If the string
    // is found then 'found' is set to true, otherwise it is set to false.
    uint findstr_safe(const std::string &label, bool &found) const;

    // Returns the position in labs_ containing the string 's'.  If the string
    // is not found in labs_ an error is reported.
    uint findstr(const std::string &) const override;

    // Adds a label to the end of labs_.
    void add_label(const std::string &lab);

    // Change the order in which the categorical data appears.  This causes all
    // registered CategoricalData to change numerical values to comply with the
    // new ordering.
    void reorder(const std::vector<std::string> &new_ordering);

    // Change the levels of the variable to new_ordering.  This function may
    // just do the same thing as 'reorder'.
    void set_levels(const std::vector<std::string> &new_ordering);

    // Change the labels without changing the ordering or the values of the
    // underlying data.
    void relabel(const std::vector<std::string> &new_labels);

    // Print the value of this key to the stream 'out'.
    std::ostream &print(std::ostream &out) const override;

    // Print the level that corresponds to a particular value.
    std::ostream &print(uint value, std::ostream &out) const override;

   private:
    std::vector<std::string> labs_;
    bool grow_;
    std::vector<uint> map_levels(const std::vector<std::string> &sv) const;
  };

  inline std::ostream &operator<<(std::ostream &out, const CatKeyBase &k) {
    return k.print(out);
  }

  Ptr<CatKey> make_catkey(const std::vector<std::string> &);

  //----------------------------------------------------------------------
  // CategoricalData models discrete valued 'factor' data.  All CategoricalData
  // can be thought of as unsigned integers.  Some CategoricalData can also be
  // thought of as strings.
  //
  // The behavior of a CategoricalData is managed through a CatKeyBase, which
  // encodes two basic bits of information: whether the CategoricalData has a
  // maximum size, and whether it can be treated as a string in addition to
  // being treated as an integer.  If the CatKey does not allow string-based
  class CategoricalData : public DataTraits<uint> {
   public:
    // constructors, assingment, comparison...
    ~CategoricalData() override;
    CategoricalData(uint val, uint Nlevels);
    CategoricalData(uint val, const Ptr<CatKeyBase> &key);
    CategoricalData(uint val, CategoricalData &other);

    CategoricalData(const std::string &label, const Ptr<CatKey> &key);

    // Copying a CategoricalData will create a new CategoricalData that has the
    // same key as this one, and the same value.
    CategoricalData(const CategoricalData &rhs) = default;
    CategoricalData(CategoricalData &&rhs) = default;

    CategoricalData *clone() const override;

    // The const-ref in the signature is needed to match the signature of the
    // DataTraits parent class.
    void set(const uint &value, bool signal_observers = true) override;

    bool operator==(uint rhs) const;
    bool operator==(const CategoricalData &rhs) const;

    bool operator!=(uint rhs) const { return !(*this == rhs); }
    bool operator!=(const CategoricalData &rhs) const {
      return !(*this == rhs);
    }

    //  size querries...........
    uint nlevels() const;  //  'value()' can be 0..nelvels()-1

    // value querries.............
    const uint &value() const override;
    const Ptr<CatKeyBase> &key() const { return key_; }
    bool comparable(const CategoricalData &rhs) const;

    // input-output
    std::ostream &display(std::ostream &out) const override;

    void set_key(const Ptr<CatKeyBase> &key) { key_ = key; }

    void print_key(std::ostream &out) const;

   private:
    uint val_{};
    Ptr<CatKeyBase> key_;
  };

  //------------------------------------------------------------
  // LabeledCategoricalData is CategoricalData with labels assigned to the
  // categories.
  class LabeledCategoricalData
      : public CategoricalData {
   public:
    LabeledCategoricalData(uint value, const Ptr<CatKey> &key);
    LabeledCategoricalData(const std::string &value, const Ptr<CatKey> &key);
    LabeledCategoricalData *clone() const override;

    const std::string &label() const {return labels()[value()];}
    const std::vector<std::string> &labels() const {
      return catkey_->labels();
    }

    Ptr<CatKey> catkey() const {return catkey_;}

   private:
    Ptr<CatKey> catkey_;
  };

  //------------------------------------------------------------
  class OrdinalData : public CategoricalData {
   public:
    OrdinalData(uint value, uint Nlevels);
    OrdinalData(uint value, const Ptr<CatKeyBase> &key);
    OrdinalData(const std::string &label, const Ptr<CatKey> &key);

    OrdinalData(const OrdinalData &rhs);
    OrdinalData *clone() const override;

    bool operator<(const OrdinalData &rhs) const;
    bool operator<=(const OrdinalData &rhs) const;
    bool operator>(const OrdinalData &rhs) const;
    bool operator>=(const OrdinalData &rhs) const;

    bool operator<(uint rhs) const;
    bool operator<=(uint rhs) const;
    bool operator>(uint rhs) const;
    bool operator>=(uint rhs) const;

    bool operator<(const std::string &rhs) const;
    bool operator<=(const std::string &rhs) const;
    bool operator>(const std::string &rhs) const;
    bool operator>=(const std::string &rhs) const;
  };
  //======================================================================

  std::vector<Ptr<CategoricalData> > make_catdat_ptrs(
      const std::vector<std::string> &);
  std::vector<Ptr<CategoricalData> > make_catdat_ptrs(
      const std::vector<uint> &);

  std::vector<Ptr<OrdinalData> > make_ord_ptrs(const std::vector<uint> &);

}  // namespace BOOM

#endif  // BOOM_CATEGORICAL_DATA_HPP
