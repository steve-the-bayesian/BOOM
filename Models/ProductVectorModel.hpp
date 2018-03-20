// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2016 Steven L. Scott

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
#ifndef BOOM_PRODUCT_VECTOR_MODEL_HPP_
#define BOOM_PRODUCT_VECTOR_MODEL_HPP_

#include "Models/DoubleModel.hpp"
#include "Models/Policies/CompositeParamPolicy.hpp"
#include "Models/Policies/NullDataPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/VectorModel.hpp"

namespace BOOM {

  // Create a model for a Vector y through independent scalar marginal
  // models for each y[i].
  //
  // This is not a model for learning about parameters.  If parameter
  // learning is needed then separate pointers to the component models
  // should be held and used to add data and posterior samplers.
  class ProductVectorModel : virtual public VectorModel,
                             public CompositeParamPolicy,
                             public NullDataPolicy,
                             public PriorPolicy {
   public:
    // To build the model incrementally, use the default constructor
    // and add models with add_model().
    ProductVectorModel() {}

    // To build the model all at once pass in a vector of pointers to
    // DoubleModel.
    explicit ProductVectorModel(const std::vector<Ptr<DoubleModel>> &marginals);
    ProductVectorModel(const ProductVectorModel &rhs);
    ProductVectorModel(ProductVectorModel &&rhs) = default;
    ProductVectorModel &operator=(const ProductVectorModel &rhs);
    ProductVectorModel &operator=(ProductVectorModel &&rhs) = default;
    ProductVectorModel *clone() const override;

    // The dimension of the Vector being modeled.
    int dimension() const { return marginal_distributions_.size(); }

    double logp(const Vector &y) const override;
    Vector sim(RNG &rng = GlobalRng::rng) const override;

    virtual void add_model(const Ptr<DoubleModel> &model);
    virtual void clear_models();

   private:
    std::vector<Ptr<DoubleModel>> marginal_distributions_;

    // A non-virtual implementation of add_model that can be called
    // from constructors.
    void non_virtual_add_model(const Ptr<DoubleModel> &model);
  };

  //======================================================================
  // A ProductVectorModel that knows about its mean and variance.
  class ProductLocationScaleVectorModel : public ProductVectorModel,
                                          public LocationScaleVectorModel {
   public:
    ProductLocationScaleVectorModel();
    explicit ProductLocationScaleVectorModel(
        const std::vector<Ptr<LocationScaleDoubleModel>> &marginals);
    ProductLocationScaleVectorModel(const ProductLocationScaleVectorModel &rhs);
    ProductLocationScaleVectorModel(ProductLocationScaleVectorModel &&rhs) =
        default;
    ProductLocationScaleVectorModel &operator=(
        const ProductLocationScaleVectorModel &rhs);
    ProductLocationScaleVectorModel &operator=(
        ProductLocationScaleVectorModel &&rhs) = default;
    ProductLocationScaleVectorModel *clone() const override;

    const Vector &mu() const override {
      refresh_moments();
      return mu_;
    }
    const SpdMatrix &Sigma() const override {
      refresh_moments();
      return Sigma_;
    }
    const SpdMatrix &siginv() const override {
      refresh_moments();
      return siginv_;
    }
    double ldsi() const override {
      refresh_moments();
      return ldsi_;
    }

    void add_model(const Ptr<DoubleModel> &model) override;
    void add_location_scale_model(const Ptr<LocationScaleDoubleModel> &model);
    void clear_models() override;

   private:
    // An observer that should be passed to the parameters of the
    // marginal distributions.  When they change the
    // moments_are_current_ flag gets flipped to false.
    void observe_parameter_changes() { moments_are_current_ = false; }

    // Moments are logically const.  refresh_moments has to be marked
    // const so it can be called from the moment accessor functions
    // that are part of the class's virtual signature.
    void refresh_moments() const;

    std::vector<Ptr<LocationScaleDoubleModel>> ls_marginal_distributions_;
    mutable bool moments_are_current_;
    mutable Vector mu_;
    mutable SpdMatrix Sigma_;
    mutable SpdMatrix siginv_;
    mutable double ldsi_;
  };

}  // namespace BOOM

#endif  //  BOOM_PRODUCT_VECTOR_MODEL_HPP_
