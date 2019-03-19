// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2013 Steven L. Scott

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

#include "Models/Mixtures/MvnMetaAnalysisDPMPriorModel.hpp"
#include <utility>
#include "Models/MvnBase.hpp"

namespace BOOM {

  namespace {
    typedef MvnMetaAnalysisDPMPriorData mDPMData;
    typedef MvnMetaAnalysisDPMPriorModel mDPMPrior;
  }  // namespace

  mDPMData::MvnMetaAnalysisDPMPriorData(const Vector& observation,
                                        const SpdMatrix& observation_variance)
      : observation_(new VectorData(observation)),
        observation_variance_(observation_variance) {}

  mDPMData* mDPMData::clone() const { return new mDPMData(*this); }

  std::ostream& mDPMData::display(std::ostream& out) const {
    out << observation() << " " << observation_variance().vectorize();
    return out;
  }

  mDPMPrior::MvnMetaAnalysisDPMPriorModel(int dim, double alpha)
      : HierarchicalBase(new DirichletProcessMvnModel(dim, alpha)) {}

  mDPMPrior::MvnMetaAnalysisDPMPriorModel(
      const Ptr<DirichletProcessMvnModel>& prior)
      : HierarchicalBase(prior) {}

  mDPMPrior* mDPMPrior::clone() const { return new mDPMPrior(*this); }

  void mDPMPrior::add_data(const Ptr<Data>& dp) {
    Ptr<mDPMData> data_point = dp.dcast<mDPMData>();
    const Vector& value(data_point->observation());
    NEW(MvnModel, model)(value, data_point->observation_variance());
    model->suf()->update_raw(value);
    add_data_level_model(model);
  }

  std::vector<Vector> mDPMPrior::group_means() const {
    std::vector<Vector> out;
    out.reserve(number_of_groups());
    for (int i = 0; i < number_of_groups(); ++i) {
      out.push_back(data_model(i)->mu());
    }
    return out;
  }

  int mDPMPrior::number_of_clusters() const {
    return prior_model()->number_of_clusters();
  }

}  // namespace BOOM
