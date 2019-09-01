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

#include "Models/Policies/MixtureDataPolicy.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {

  MixtureDataPolicy::MixtureDataPolicy(int number_of_latent_levels)
  {
    if (number_of_latent_levels > 0) {
      pkey_ = new FixedSizeIntCatKey(number_of_latent_levels);
    } else {
      pkey_ = new UnboundedIntCatKey;
    }
  }

  MixtureDataPolicy::MixtureDataPolicy(const MixtureDataPolicy &rhs)
      : Model(rhs),
        DataTraits(rhs),
        dat_(rhs.dat_),
        latent_(std::vector<Ptr<CategoricalData>>(rhs.latent_)),
        pkey_(rhs.pkey_) {
    // copy pointer elements for observed data
    // create new data storage for latent data
    uint n = latent_.size();
    for (uint i = 0; i < n; ++i) {
      latent_[i] = latent_[i]->clone();
    }
  }

  MixtureDataPolicy &MixtureDataPolicy::operator=(
      const MixtureDataPolicy &rhs) {
    if (&rhs != this) {
      dat_ = rhs.dat_;
      latent_ = rhs.latent_;
    }
    return *this;
  }

  void MixtureDataPolicy::clear_data() {
    dat().clear();
    latent_data().clear();
    known_data_source_.clear();
  }

  //------------------------------------------------------------
  void MixtureDataPolicy::set_data(const DatasetType &d) {
    clear_data();
    for (uint i = 0; i < d.size(); ++i) add_data(d[i]);
  }

  std::vector<Ptr<CategoricalData>> &MixtureDataPolicy::latent_data() {
    return latent_;
  }

  const std::vector<Ptr<CategoricalData>> &MixtureDataPolicy::latent_data()
      const {
    return latent_;
  }

  void MixtureDataPolicy::add_data(const Ptr<DataType> &d) {
    dat().push_back(d);
    int max_levels = pkey_->max_levels();
    if (max_levels > 0) {
      uint h = random_int(0, max_levels - 1);
      NEW(CategoricalData, pcat)(h, pkey_);
      latent_data().push_back(pcat);
    } else {
      NEW(CategoricalData, pcat)(uint(0), pkey_);
      latent_data().push_back(pcat);
    }
    if (!known_data_source_.empty()) {
      known_data_source_.push_back(-1);
    }
  }

  void MixtureDataPolicy::add_data_with_known_source(
      const Ptr<DataType> &data_point, int source) {
    if (known_data_source_.empty()) {
      known_data_source_.assign(dat().size(), -1);
    }
    add_data(data_point);
    known_data_source_.push_back(source);
  }

  int MixtureDataPolicy::which_mixture_component(int observation_number) const {
    if (known_data_source_.empty()) return -1;
    return known_data_source_[observation_number];
  }

  void MixtureDataPolicy::set_data_source(
      const std::vector<int> &known_data_source) {
    if (dat().size() != known_data_source.size()) {
      ostringstream err;
      err << "Error in MixtureDataPolicy::set_data_source.  "
          << "The size of known_data_source (" << known_data_source.size()
          << ") does not match that of the data (" << dat().size() << ").";
      report_error(err.str());
    }
    known_data_source_ = known_data_source;
  }

  void MixtureDataPolicy::combine_data(const Model &other, bool) {
    const MixtureDataPolicy &m(dynamic_cast<const MixtureDataPolicy &>(other));
    const std::vector<Ptr<Data>> &d(m.dat_);
    dat_.reserve(dat_.size() + d.size());
    dat_.insert(dat_.end(), d.begin(), d.end());

    const std::vector<Ptr<CategoricalData>> &mis(m.latent_);
    latent_.reserve(latent_.size() + mis.size());
    latent_.insert(latent_.end(), mis.begin(), mis.end());

    if (known_data_source_.empty()) {
      if (!m.known_data_source_.empty()) {
        known_data_source_.assign(dat().size(), -1);
        std::copy(m.known_data_source_.begin(), m.known_data_source_.end(),
                  std::back_inserter(known_data_source_));
      }
    } else {
      bool other_empty = m.known_data_source_.empty();
      known_data_source_.reserve(dat().size() + m.dat().size());
      for (int i = 0; i < m.dat().size(); ++i) {
        known_data_source_.push_back(other_empty ? -1
                                                 : m.known_data_source_[i]);
      }
    }
  }
}  // namespace BOOM
