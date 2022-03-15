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
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
   USA
 */

#ifndef GLM_MODEL_H
#define GLM_MODEL_H
#include "uint.hpp"
#include "LinAlg/Selector.hpp"
#include "LinAlg/VectorView.hpp"
#include "Models/Glm/GlmCoefs.hpp"
#include "Models/ModelTypes.hpp"
#include "Models/ParamTypes.hpp"

namespace BOOM {

  // A base class for GlmData,
  class GlmBaseData : virtual public Data {
   public:
    explicit GlmBaseData(const Vector &x);
    explicit GlmBaseData(const Ptr<VectorData> &xp);
    GlmBaseData(const GlmBaseData &rhs);
    GlmBaseData *clone() const override = 0;

    // Dimension of the predictors.
    uint xdim() const;
    const Vector &x() const;

    // Args:
    //   X:  The new predictor values.
    //   allow_any: If true, then the new X can have different
    //     dimension from the old X.
    void set_x(const Vector &X, bool allow_any = false);

    // By default, GlmData are weighted with weight 1.0, but derived
    // classes can support different weights.
    virtual double weight() const { return 1.0; }

    Ptr<VectorData> Xptr() { return x_; }
    const Ptr<VectorData> Xptr() const { return x_; }

    // Replace the representation of x_ by a new pointer.  This is an unusual
    // operation.
    void set_Xptr(const Ptr<VectorData> &x) {
      x_ = x;
    }

   private:
    Ptr<VectorData> x_;
  };

  template <class DAT>
  class GlmData : public GlmBaseData {
   public:
    typedef typename DAT::value_type value_type;
    typedef const value_type const_value_type;

    // Args:
    //   y:  Response value
    //   X: Vector of predictors.  If an intercept is desired it must
    //     be explicitly added.
    GlmData(const value_type &y, const Vector &X);

    // Args:
    //   yp:  Ptr to response value.
    //   xp:  Ptr to predictor values.
    GlmData(const Ptr<DAT> &yp, Ptr<VectorData> xp);

    // Copies have value semantics.
    GlmData(const GlmData &);
    GlmData *clone() const override;

    // required virtual functions
    std::ostream &display(std::ostream &) const override;

    const value_type &y() const;
    void set_y(const value_type &Y);

    Ptr<DAT> Yptr() { return y_; }

    // Replace the representation of the response with a new object.  This is an
    // unusual operation.
    void set_Yptr(const Ptr<DAT> &y) {
      y_ = y;
    }

   private:
    // If an intercept is desired, it must be explicitly included.
    Ptr<DAT> y_;
  };

  //------------------------------------------------------------
  template <class D>
  class WeightedGlmData : public GlmData<D> {
   public:
    typedef GlmData<D> Base;
    typedef typename Base::value_type value_type;
    WeightedGlmData(const value_type &y, const Vector &x, double weight = 1.0)
        : Base(y, x), weight_(new DoubleData(weight)) {}

    WeightedGlmData(const Ptr<D> &yptr, const Ptr<VectorData> &xptr,
                    Ptr<DoubleData> wptr)
        : Base(yptr, xptr), weight_(wptr) {}

    WeightedGlmData(const WeightedGlmData &rhs)
        : Base(rhs), weight_(rhs.weight_->clone()) {}

    WeightedGlmData *clone() const override {
      return new WeightedGlmData(*this);
    }

    std::ostream &display(std::ostream &out) const override {
      out << *weight_ << " ";
      return (Base::display(out));
    }

    double weight() const override { return weight_->value(); }
    void set_weight(double W) { weight_->set(W); }
    Ptr<DoubleData> WeightPtr() { return weight_; }

   private:
    Ptr<DoubleData> weight_;
  };

  //============================================================
  class OrdinalData;

  typedef GlmData<DoubleData> RegressionData;
  typedef GlmData<VectorData> MvRegData;
  typedef GlmData<BinaryData> BinaryRegressionData;
  typedef GlmData<OrdinalData> OrdinalRegressionData;
  typedef WeightedGlmData<DoubleData> WeightedRegressionData;
  typedef WeightedGlmData<VectorData> WeightedMvRegData;

  // Base class for generalized linear models.  Regression models, Poisson
  // regression, logistic regression, etc.
  class GlmModel : virtual public PosteriorModeModel {
   public:
    GlmModel();
    GlmModel(const GlmModel &rhs);
    GlmModel *clone() const override = 0;

    // The coefficients relating the model to the predictor variables.  All
    // models have these.  Some models have other parameters too.
    virtual GlmCoefs &coef() = 0;
    virtual const GlmCoefs &coef() const = 0;
    virtual Ptr<GlmCoefs> coef_prm() = 0;
    virtual const Ptr<GlmCoefs> coef_prm() const = 0;

    uint xdim() const;
    // Utiliteis for variable selection.  These are syntactic sugar pass-through
    // calls to the underlying GlmCoefs owned by the model.
    void add_all();
    void drop_all();
    void drop_all_but_intercept();
    void add(uint p);
    void drop(uint p);
    void flip(uint p);
    const Selector &inc() const;
    bool inc(uint p) const;

    // The set of model coefficients for the subset of included variables.
    Vector included_coefficients() const;
    void set_included_coefficients(const Vector &b);

    // The full set of model coefficients, with 0's in positions corresponding
    // to excluded variables.
    virtual const Vector &Beta() const;

    // Setting the full vector of coefficients does not affect the 'inclusion
    // vector'.  It is up to the caller to ensure that excluded variables have
    // 0's in the right places.
    void set_Beta(const Vector &Beta);

    // The I'th coefficient from the full coefficient vector.
    double Beta(uint I) const;

    void set_sparse_coefficients(const Vector &nonzero_values,
                                 const std::vector<uint> &nonzero_positions);

    // Return the dot product of x and coefs().  The dimension of x must either
    // be the number of included predictor variables, or the number of available
    // predictor variables.
    virtual double predict(const Vector &x) const;
    virtual double predict(const VectorView &x) const;
    virtual double predict(const ConstVectorView &x) const;
  };

  //============================================================
  template <class D>
  GlmData<D>::GlmData(const value_type &Y, const Vector &X)
      : GlmBaseData(X), y_(new D(Y)) {}

  template <class D>
  GlmData<D>::GlmData(const Ptr<D> &Y, Ptr<VectorData> X)
      : GlmBaseData(X), y_(Y) {}

  template <class D>
  GlmData<D>::GlmData(const GlmData &rhs)
      : GlmBaseData(rhs), y_(rhs.y_->clone()) {}

  template <class D>
  GlmData<D> *GlmData<D>::clone() const {
    return new GlmData(*this);
  }

  template <class D>
  std::ostream &GlmData<D>::display(std::ostream &out) const {
    y_->display(out);
    out << " ";
    Xptr()->display(out);
    return out;
  }

  template <class D>
  const typename D::value_type &GlmData<D>::y() const {
    return y_->value();
  }

  template <class D>
  void GlmData<D>::set_y(const value_type &Y) {
    y_->set(Y);
  }

}  // namespace BOOM

#endif  // GLM_MODEL_H
