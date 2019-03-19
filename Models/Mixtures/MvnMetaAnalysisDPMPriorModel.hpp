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

#ifndef BOOM_MVN_METAANALYSIS_MVN_DPM_HPP_
#define BOOM_MVN_METAANALYSIS_MVN_DPM_HPP_

#include "Models/Hierarchical/HierarchicalModel.hpp"
#include "Models/Mixtures/DirichletProcessMvnModel.hpp"
#include "Models/MvnModel.hpp"

namespace BOOM {

  // Meta-analysis style data with group (study) mean 'observation' and
  // known measurement error covariance matrix 'observation_variance'
  class MvnMetaAnalysisDPMPriorData : public Data {
   public:
    MvnMetaAnalysisDPMPriorData(const Vector &y,
                                const SpdMatrix &observation_variance);
    MvnMetaAnalysisDPMPriorData *clone() const override;
    std::ostream &display(std::ostream &out) const override;
    Vector observation() const { return observation_->value(); }
    SpdMatrix observation_variance() const { return observation_variance_; }

   private:
    Ptr<VectorData> observation_;
    SpdMatrix observation_variance_;
  };

  // Meta-analysis model with DPM of Gaussians prior distribution
  // Observation error covariance matrix V known.
  //
  //      y[i] | theta[i], V[i] ~ N(theta[i], V[i])
  //      theta[i] ~ N(mu, S)
  //      mu, S ~ DP(alpha*G0)
  //      G0 = Normal-Inverse-Wishart
  class MvnMetaAnalysisDPMPriorModel
      : public HierarchicalModelBase<MvnModel, DirichletProcessMvnModel> {
   public:
    explicit MvnMetaAnalysisDPMPriorModel(int dim, double alpha = 1.0);
    explicit MvnMetaAnalysisDPMPriorModel(
        const Ptr<DirichletProcessMvnModel> &prior_model);
    MvnMetaAnalysisDPMPriorModel *clone() const override;

    // Creates a new data_level_model with data assigned.
    void add_data(const Ptr<Data> &) override;

    std::vector<Vector> group_means() const;
    int number_of_clusters() const;
    int dim() const;
  };
}  // namespace BOOM

#endif  //  BOOM_MVN_METAANALYSIS_MVN_DPM_HPP_
