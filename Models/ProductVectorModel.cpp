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

#include "Models/ProductVectorModel.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  namespace {
    typedef ProductVectorModel PVM;
    typedef ProductLocationScaleVectorModel PLSVM;
  }  // namespace

  PVM::ProductVectorModel(const std::vector<Ptr<DoubleModel>> &marginals) {
    for (int i = 0; i < marginals.size(); ++i) {
      non_virtual_add_model(marginals[i]);
    }
  }

  PVM::ProductVectorModel(const ProductVectorModel &rhs)
      : VectorModel(rhs), CompositeParamPolicy(rhs) {
    for (int i = 0; i < rhs.marginal_distributions_.size(); ++i) {
      non_virtual_add_model(rhs.marginal_distributions_[i]->clone());
    }
  }

  ProductVectorModel &PVM::operator=(const ProductVectorModel &rhs) {
    if (&rhs == this) {
      return *this;
    }
    clear_models();
    for (int i = 0; i < rhs.marginal_distributions_.size(); ++i) {
      add_model(rhs.marginal_distributions_[i]->clone());
    }
    return *this;
  }

  ProductVectorModel *PVM::clone() const {
    return new ProductVectorModel(*this);
  }

  void PVM::add_model(const Ptr<DoubleModel> &m) { non_virtual_add_model(m); }

  void PVM::non_virtual_add_model(const Ptr<DoubleModel> &m) {
    marginal_distributions_.push_back(m);
    CompositeParamPolicy::add_model(m);
  }

  void PVM::clear_models() {
    marginal_distributions_.clear();
    CompositeParamPolicy::clear();
  }

  double PVM::logp(const Vector &y) const {
    double ans = 0;
    if (y.size() != marginal_distributions_.size()) {
      report_error("Wrong size argument.");
    }
    for (int i = 0; i < y.size(); ++i) {
      ans += marginal_distributions_[i]->logp(y[i]);
    }
    return ans;
  }

  Vector PVM::sim(RNG &rng) const {
    Vector ans(marginal_distributions_.size());
    for (int i = 0; i < ans.size(); ++i) {
      ans[i] = marginal_distributions_[i]->sim(rng);
    }
    return ans;
  }

  //======================================================================
  PLSVM::ProductLocationScaleVectorModel() : moments_are_current_(false) {}

  PLSVM::ProductLocationScaleVectorModel(
      const std::vector<Ptr<LocationScaleDoubleModel>> &marginals)
      : ProductVectorModel(), moments_are_current_(false) {
    for (int i = 0; i < marginals.size(); ++i) {
      add_location_scale_model(marginals[i]);
    }
    refresh_moments();
  }

  PLSVM::ProductLocationScaleVectorModel(const PLSVM &rhs)
      : ProductVectorModel(), moments_are_current_(false) {
    for (int i = 0; i < rhs.ls_marginal_distributions_.size(); ++i) {
      Ptr<LocationScaleDoubleModel> model =
          rhs.ls_marginal_distributions_[i]->clone();
      add_location_scale_model(model);
    }
    refresh_moments();
  }

  PLSVM *PLSVM::clone() const { return new PLSVM(*this); }

  PLSVM &PLSVM::operator=(const PLSVM &rhs) {
    if (&rhs == this) {
      return *this;
    }
    clear_models();
    for (int i = 0; i < rhs.dimension(); ++i) {
      add_location_scale_model(rhs.ls_marginal_distributions_[i]->clone());
    }
    return *this;
  }

  void PLSVM::add_model(const Ptr<DoubleModel> &model) {
    Ptr<LocationScaleDoubleModel> model_with_correct_type(
        model.dcast<LocationScaleDoubleModel>());
    if (!model_with_correct_type) {
      report_error(
          "Argument to ProductLocationScaleVectorModel::add_model "
          "must inherit from LocationScaleDoubleModel.");
    }
    add_location_scale_model(model_with_correct_type);
  }

  void PLSVM::add_location_scale_model(
      const Ptr<LocationScaleDoubleModel> &model) {
    ls_marginal_distributions_.push_back(model);
    moments_are_current_ = false;
    std::vector<Ptr<Params>> parameter_vector = model->parameter_vector();
    std::function<void(void)> observer = [this]() {
      this->observe_parameter_changes();
    };
    for (int i = 0; i < parameter_vector.size(); ++i) {
      parameter_vector[i]->add_observer(this, observer);
    }
    PVM::add_model(model);
  }

  void PLSVM::clear_models() {
    moments_are_current_ = false;
    ls_marginal_distributions_.clear();
    PVM::clear_models();
  }

  void PLSVM::refresh_moments() const {
    if (moments_are_current_) return;
    int dimension = ls_marginal_distributions_.size();
    if (mu_.size() != dimension) {
      mu_.resize(dimension);
      Sigma_.resize(dimension);
      siginv_.resize(dimension);
      Sigma_ = 0;
      siginv_ = 0;
    }
    ldsi_ = 0;
    for (int i = 0; i < dimension; ++i) {
      mu_[i] = ls_marginal_distributions_[i]->mean();
      Sigma_(i, i) = ls_marginal_distributions_[i]->variance();
      siginv_(i, i) = 1.0 / Sigma_(i, i);
      ldsi_ += log(siginv_(i, i));
    }
    moments_are_current_ = true;
  }

}  // namespace BOOM
