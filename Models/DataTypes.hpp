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

#ifndef BOOM_MODELS_DATA_TYPES_H
#define BOOM_MODELS_DATA_TYPES_H

#include <cmath>
#include <map>  // for STL's map container
#include <string>
#include <vector>
#include "uint.hpp"

#include "LinAlg/CorrelationMatrix.hpp"  // for VectorData
#include "LinAlg/Matrix.hpp"
#include "LinAlg/SpdMatrix.hpp"
#include "LinAlg/Vector.hpp"
#include "LinAlg/Selector.hpp"

#include <functional>
#include "cpputil/Ptr.hpp"
#include "cpputil/RefCounted.hpp"

namespace BOOM {

  class Data {  // abstract base class
   public:
    RefCounted rc_;
    void up_count() { rc_.up_count(); }
    void down_count() { rc_.down_count(); }
    unsigned int ref_count() { return rc_.ref_count(); }

    enum missing_status { observed = 0, completely_missing, partly_missing };

    Data() : missing_flag(observed) {}
    // When copying Data, the observers should not be copied.
    Data(const Data &rhs)
        : missing_flag(rhs.missing_flag),
          signals_() {}
    virtual Data *clone() const = 0;
    virtual ~Data() {signals_.clear();}
    virtual std::ostream &display(std::ostream &) const = 0;
    missing_status missing() const;
    void set_missing_status(missing_status m);
    void signal() {
      for (size_t i = 0; i < signals_.size(); ++i) {
        signals_[i]();
      }
    }
    // TODO: This implementation of the observer pattern is broken by
    // assignment.  When an object is created from an old object by assignment,
    // it should eliminate any observers it has placed on other objects.  This
    // will require the observing object to keep a collection of handles to the
    // observers and mark them as inactive.  The observed objects should check
    // that the observer is active when calling, and remove inactive observers
    // from the set of signals.  This fix will require making changes to all the
    // classes that use the current observer scheme.
    void add_observer(const std::function<void(void)> &f) {
      signals_.push_back(f);
    }

    // Remove all observers.
    void clear_observers() { signals_.clear(); }
    friend void intrusive_ptr_add_ref(Data *d);
    friend void intrusive_ptr_release(Data *d);

   private:
    missing_status missing_flag;
    std::vector<std::function<void(void)> > signals_;
  };
  //======================================================================
  std::ostream &operator<<(std::ostream &out, const Data &d);
  std::ostream &operator<<(std::ostream &out, const Ptr<Data> &dp);
  void print_data(const Data &d);

  //==========================================================================-
  // The DataTraits class enforces a uniform interface for set() and value().
  // It could probably be eliminated, but eliminating it is probably not worth
  // the effort.
  template <class DAT>
  class DataTraits : virtual public Data {
   public:
    DataTraits() = default;
    DataTraits(const DataTraits &rhs) : Data(rhs) {}
    using value_type = DAT;
    using Traits = DataTraits<DAT>;
    virtual void set(const value_type &, bool) = 0;
    virtual const value_type &value() const = 0;
  };
  //===========================================================================
  template <class T>
  class UnivData : public DataTraits<T> {  // univariate data
   public:
    using Traits = typename DataTraits<T>::Traits;
    // constructors
    UnivData() : value_() {}
    explicit UnivData(T y) : value_(y) {}
    UnivData(const UnivData &rhs)
        : Data(rhs), Traits(rhs), value_(rhs.value_) {}
    UnivData<T> *clone() const { return new UnivData<T>(*this); }

    const T &value() const { return value_; }
    virtual void set(const T &rhs, bool Signal = true) {
      value_ = rhs;
      if (Signal) {
        this->signal();
      }
    }
    std::ostream &display(std::ostream &out) const {
      out << value_;
      return out;
    }

   private:
    T value_;
  };

  //==========================================================================
  using IntData = UnivData<unsigned int>;
  using DoubleData = UnivData<double>;
  using BinaryData = UnivData<bool>;
  //==========================================================================
  class VectorData : public DataTraits<Vector> {
   public:
    explicit VectorData(uint n, double X = 0);
    explicit VectorData(const Vector &y);
    VectorData(const VectorData &rhs);
    VectorData *clone() const override;

    uint dim() const { return data_.size(); }
    std::ostream &display(std::ostream &out) const override;

    const Vector &value() const override { return data_; }
    void set(const Vector &rhs, bool signal_change = true) override;
    virtual void set_element(double value, int position, bool sig = true);

    // Set the contiguous subset of elements from start to start + subset.size()
    // - 1 with the elements of subset.
    virtual void set_subset(const Vector &subset, int start,
                            bool signal = true);

    double operator[](uint) const;
    double &operator[](uint);

   private:
    Vector data_;
  };

  //==========================================================================
  // A variant of VectorData that handles the partially missing case.
  class PartiallyObservedVectorData : public VectorData {
   public:
    // Args:
    //   y:  The numeric value of the data vector.
    //   obs:  Indicates which components of y are observed.
    explicit PartiallyObservedVectorData(const Vector &y,
                                         const Selector &obs = Selector());
    PartiallyObservedVectorData * clone() const override;
    void set(const Vector &value, bool signal_change = true) override;

    Selector &observation_status() { return obs_; }
    const Selector &observation_status() const { return obs_; }

   private:
    Selector obs_;
  };

  //===========================================================================
  class MatrixData : public DataTraits<Matrix> {
   public:
    MatrixData(int r, int c, double val = 0.0);
    explicit MatrixData(const Matrix &y);
    MatrixData(const MatrixData &rhs);
    MatrixData *clone() const override;

    uint nrow() const { return x.nrow(); }
    uint ncol() const { return x.ncol(); }

    std::ostream &display(std::ostream &out) const override;

    const Matrix &value() const override { return x; }
    void set(const Matrix &rhs, bool sig = true) override;
    virtual void set_element(double value, int row, int col, bool sig = true);

   private:
    Matrix x;
  };

}  // namespace BOOM

#endif  // BOOM_MODELS_DATA_TYPES_H
