// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2015 Steven L. Scott

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

#include "Models/Hierarchical/HierarchicalDirichletModel.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  namespace {
    typedef HierarchicalDirichletModel HDM;
    typedef HierarchicalDirichletData HDD;
  }  // namespace

  HDD::HierarchicalDirichletData(uint p) : counts_(p) {}

  HDD::HierarchicalDirichletData(const MultinomialSuf &suf) : counts_(suf) {}

  HDD *HDD::clone() const { return new HDD(*this); }

  std::ostream &HDD::display(std::ostream &out) const { return out << counts_; }

  HDM::HierarchicalDirichletModel(double sample_size, const Vector &mean)
      : HierarchicalBase(new DirichletModel(sample_size * mean)) {
    double mean_sum = sum(mean);
    double mean_min = min(mean);
    if (mean_min < 0) {
      report_error("All elements of must be non-negative.");
    }
    if (fabs(mean_sum - 1.0) > .000001) {
      report_error("Elements of mean must sum to 1.");
    }
    if (sample_size <= 0.0) {
      report_error("sample_size must be positive.");
    }
  }

  HDM::HierarchicalDirichletModel(const Ptr<DirichletModel> &prior)
      : HierarchicalBase(prior) {}

  HDM *HDM::clone() const { return new HDM(*this); }

  void HDM::add_data(const Ptr<Data> &dp) {
    Ptr<HierarchicalDirichletData> data_point =
        dp.dcast<HierarchicalDirichletData>();
    NEW(MultinomialModel, data_model)(data_point->suf());
    add_data_level_model(data_model);
  }

}  // namespace BOOM
