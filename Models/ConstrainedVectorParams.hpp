// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2006 Steven L. Scott

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
#ifndef BOOM_CONSTRAINED_VECTOR_PARAMS
#define BOOM_CONSTRAINED_VECTOR_PARAMS
#include "Models/ParamTypes.hpp"
namespace BOOM {

  class VectorConstraint : private RefCounted {
   public:
    friend void intrusive_ptr_add_ref(VectorConstraint *d) { d->up_count(); }
    friend void intrusive_ptr_release(VectorConstraint *d) {
      d->down_count();
      if (d->ref_count() == 0) {
        delete d;
      }
    }

    ~VectorConstraint() override = default;

    // Return true iff the constraint is satisfied.
    virtual bool check(const Vector &v) const = 0;

    // Modify 'v' so that the constraint is true.  Return a reference to the
    // modified value.
    virtual Vector & impose(Vector &v) const = 0;

    // Return the constrained vector from a minimal information vector.
    // reduce() and expand() are inverse operations, so the form of the
    // expansion depends on the form of the reduction.
    virtual Vector expand(const Vector &small) const = 0;

    // Return a minimal information vector from constrained vector.
    virtual Vector reduce(const Vector &large) const = 0;

    // The number of elements in a vector that are made redundant by the
    // constraint.
    virtual int minimal_size_reduction() const = 0;
  };
  //---------------------------------------------------------------------------
  class NoConstraint : public VectorConstraint {
   public:
    bool check(const Vector &) const override { return true; }
    Vector &impose(Vector &v) const override {return v;}
    Vector expand(const Vector &v) const override { return v; }
    Vector reduce(const Vector &v) const override { return v; }
    int minimal_size_reduction() const override {return 0;}
  };

  //---------------------------------------------------------------------------
  // Constrain a particular element to be a particular value.  E.g. element 3
  // must be -1.
  class ElementConstraint : public VectorConstraint {
   public:
    explicit ElementConstraint(uint el = 0, double val = 0.0);
    bool check(const Vector &v) const override;
    Vector &impose(Vector &v) const override;
    Vector expand(const Vector &v) const override;
    Vector reduce(const Vector &v) const override;
    int minimal_size_reduction() const override {return 1;}

   private:
    uint element_;
    double value_;
  };

  //---------------------------------------------------------------------------
  // Constrain the elemets to sum to a particular value.  Impose the constraint
  // by subtracting a value to the final element.
  class SumConstraint : public VectorConstraint {
   public:
    explicit SumConstraint(double x);
    bool check(const Vector &v) const override;
    Vector &impose(Vector &v) const override;
    Vector expand(const Vector &v) const override;  // adds final element to
    Vector reduce(const Vector &v) const override;  // eliminates last element
    int minimal_size_reduction() const override {return 1;}

   private:
    double sum_;
  };

  //---------------------------------------------------------------------------
  // The vector elements must sum to a (nonzero) value. If they don't then the
  // vector elements will all be jointly scaled to satisfy the constraint.
  class ProportionalSumConstraint : public VectorConstraint {
   public:
    explicit ProportionalSumConstraint(double value)
        : sum_(value)
    {}

    bool check(const Vector &v) const override;
    Vector &impose(Vector &v) const override;
    Vector expand(const Vector &constrained) const override;
    Vector reduce(const Vector &full) const override;
    int minimal_size_reduction() const override {return 1;}

   private:
    double sum_;
  };

  //======================================================================
  class ConstrainedVectorParams : public VectorParams {
   public:
    // Args:
    //   v: Vector containing the initial values of the parameter.
    //   constraint:  A constraint on v that must be maintained.
    explicit ConstrainedVectorParams(
        const Vector &v,
        const Ptr<VectorConstraint> &constraint = nullptr);

    ConstrainedVectorParams(const ConstrainedVectorParams &rhs);
    ConstrainedVectorParams *clone() const override;

    uint size(bool minimal = true) const override;

    void set(const Vector &value, bool signal_change = true) override;

    Vector vectorize(bool minimal = true) const override;
    Vector::const_iterator unvectorize(Vector::const_iterator &v,
                                       bool minimal = true) override;
    using Params::unvectorize;

    bool check_constraint() const;

   private:
    Ptr<VectorConstraint> constraint_;
  };
  //------------------------------------------------------------

}  // namespace BOOM
#endif  // BOOM_CONSTRAINED_VECTOR_PARAMS
