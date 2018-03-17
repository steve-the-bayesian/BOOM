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

    virtual bool check(const Vector &v) const = 0;
    // returns true if constraint satisfied

    virtual void impose(Vector &v) const = 0;
    // forces constraint to hold

    virtual Vector expand(const Vector &small) const = 0;
    // returns constrained vector from minimal information vector

    virtual Vector reduce(const Vector &large) const = 0;
    // returns minimal information vector from constrained vector
  };
  //------------------------------------------------------------
  class NoConstraint : public VectorConstraint {
   public:
    bool check(const Vector &) const override { return true; }
    void impose(Vector &) const override {}
    Vector expand(const Vector &v) const override { return v; }
    Vector reduce(const Vector &v) const override { return v; }
  };
  //------------------------------------------------------------
  class ElementConstraint : public VectorConstraint {
   public:
    explicit ElementConstraint(uint el = 0, double val = 0.0);
    bool check(const Vector &v) const override;
    void impose(Vector &v) const override;
    Vector expand(const Vector &v) const override;
    Vector reduce(const Vector &v) const override;

   private:
    uint element_;
    double value_;
  };
  //------------------------------------------------------------
  class SumConstraint : public VectorConstraint {
   public:
    explicit SumConstraint(double x);
    bool check(const Vector &v) const override;
    void impose(Vector &v) const override;
    Vector expand(const Vector &v) const override;  // adds final element to
    Vector reduce(const Vector &v) const override;  // eliminates last element
   private:
    double sum_;
  };

  //======================================================================

  class ConstrainedVectorParams : public VectorParams {
   public:
    explicit ConstrainedVectorParams(uint p, double x = 0.0,
                                     const Ptr<VectorConstraint> &vc = nullptr);
    // copies v's data
    explicit ConstrainedVectorParams(const Vector &v,
                                     const Ptr<VectorConstraint> &vc = nullptr);
    // copies data
    ConstrainedVectorParams(const ConstrainedVectorParams &rhs);
    ConstrainedVectorParams *clone() const override;

    Vector vectorize(bool minimal = true) const override;
    Vector::const_iterator unvectorize(Vector::const_iterator &v,
                                       bool minimal = true) override;
    Vector::const_iterator unvectorize(const Vector &v,
                                       bool minimal = true) override;

    bool check_constraint() const;

   private:
    Ptr<VectorConstraint> c_;
  };
  //------------------------------------------------------------

}  // namespace BOOM
#endif  // BOOM_CONSTRAINED_VECTOR_PARAMS
