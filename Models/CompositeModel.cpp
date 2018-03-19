// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2007 Steven L. Scott

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

#include "Models/CompositeModel.hpp"

namespace BOOM {

  using CM = BOOM::CompositeModel;

  CM::CompositeModel() = default;

  void CM::setup() { ParamPolicy::set_models(m_.begin(), m_.end()); }

  CM::CompositeModel(const CM &rhs)
      : Model(rhs), ParamPolicy(rhs), DataPolicy(rhs), PriorPolicy(rhs) {
    uint S = rhs.m_.size();
    for (uint s = 0; s < S; ++s) {
      m_.emplace_back(rhs.m_[s]->clone());
    }
    setup();
  }

  CM *CM::clone() const { return new CM(*this); }

  void CM::add_model(const Ptr<MixtureComponent> &new_model) {
    m_.push_back(new_model);
    ParamPolicy::add_model(new_model);
  }

  void CM::add_data(const Ptr<CompositeData> &dp) {
    DataPolicy::add_data(dp);
    uint n = dp->dim();
    assert(n == m_.size());
    for (uint i = 0; i < n; ++i) {
      m_[i]->add_data(dp->get_ptr(i));
    }
  }

  void CM::add_data(const Ptr<Data> &dp) {
    Ptr<CompositeData> d = DAT(dp);
    add_data(d);
  }

  void CM::clear_data() {
    int n = m_.size();
    for (int i = 0; i < n; ++i) {
      m_[i]->clear_data();
    }
    DataPolicy::clear_data();
  }

  double CM::pdf(const Ptr<Data> &dp, bool logscale) const {
    return pdf(*DAT(dp), logscale);
  }

  double CM::pdf(const Data *dp, bool logscale) const {
    return pdf(*DAT(dp), logscale);
  }

  double CM::pdf(const CompositeData &dp, bool logscale) const {
    uint n = dp.dim();
    assert(n == m_.size());
    double ans = 0;
    for (uint i = 0; i < n; ++i) {
      if (dp.get(i)->missing() == 0u) {
        ans += m_[i]->pdf(dp.get(i), true);
      }
    }
    return logscale ? ans : exp(ans);
  }

  std::vector<Ptr<MixtureComponent> > &CM::components() { return m_; }
  const std::vector<Ptr<MixtureComponent> > &CM::components() const {
    return m_;
  }

}  // namespace BOOM
