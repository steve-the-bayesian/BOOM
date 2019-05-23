/*
  Copyright (C) 2005-2019 Steven L. Scott

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

#include "Models/ExponentialIncrementModel.hpp"
#include "distributions.hpp"

namespace BOOM {

  namespace {
    using EIM = ExponentialIncrementModel;
  }
  
  EIM::ExponentialIncrementModel(const Vector &increment_rates)
  {
    for (int i = 0; i < increment_rates.size(); ++i) {
      add_increment_model(new ExponentialModel(increment_rates[i]));
    }
  }

  
  EIM::ExponentialIncrementModel(
      const std::vector<Ptr<ExponentialModel>> &increment_models)
  {
    for (const auto &m : increment_models) {
      add_increment_model(m);
    }
  }

  EIM::ExponentialIncrementModel(const EIM &rhs)
      : Model(rhs),
        VectorModel(rhs),
        CompositeParamPolicy(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs)
  {
    for (int i = 0; i < rhs.models_.size(); ++i) {
      add_increment_model(rhs.models_[i]->clone());
    }
  }

  EIM & EIM::operator=(const EIM &rhs) {
    if (&rhs != this) {
      Model::operator=(rhs);
      ParamPolicy::operator=(rhs);
      DataPolicy::operator=(rhs);
      PriorPolicy::operator=(rhs);
      ParamPolicy::clear();
      models_.clear();
      for (const auto &m : rhs.models_) {
        add_increment_model(m->clone());
      }
    }
    return *this;
  }

  EIM & EIM::operator=(EIM &&rhs) {
    if (&rhs != this) {
      Model::operator=(rhs);
      VectorModel::operator=(rhs);
      ParamPolicy::operator=(rhs);
      DataPolicy::operator=(rhs);
      PriorPolicy::operator=(rhs);
      models_ = std::move(rhs.models_);
    }
    return *this;
  }

  EIM * EIM::clone() const {return new EIM(*this);}
  
  void EIM::add_increment_model(const Ptr<ExponentialModel> &m) {
    models_.push_back(m);
    ParamPolicy::add_model(m);
  }
  

  double EIM::logp(const Vector &x) const {
    if (x.size() != models_.size()) {
      return negative_infinity();
    }
    double ans = 0;
    for (int i = 0; i < models_.size(); ++i) {
      double dx = (i == 0) ? x[0] : x[i] - x[i - 1];
      ans += models_[i]->logp(dx);
    }
    return ans;
  }

  Vector EIM::sim(RNG &rng) const {
    Vector ans;
    ans.reserve(models_.size());
    for (int i = 0; i < models_.size(); ++i) {
      double increment = models_[i]->sim(rng);
      ans.push_back(i == 0 ? increment : ans.back() + increment);
    }
    return ans;
  }
  
}
