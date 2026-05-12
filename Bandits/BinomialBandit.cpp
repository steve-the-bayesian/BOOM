/*
  Copyright (C) 2005-2026 Steven L. Scott

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

#include "Bandits/BinomialBandit.hpp"
#include "Models/ParamTypes.hpp"
#include "Models/DataTypes.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  BinomialBandit::BinomialBandit(const std::vector<Ptr<BinomialModel>> &models) 
      : models_(models)
  {
    if (models_.empty()) {
      report_error("Vector of models was empty.");
    } else if (models_.size() == 1) {
      report_error("Vector of models only had a single element.");
    }

    for (int i = 0; i < models_.size(); ++i) {
      if (!models_[i]) {
        std::ostringstream msg;
        msg << "Element " << i << " of models vector is (nullptr).";
        report_error(msg.str());
      }
    }
  }

  double BinomialBandit::Value(int arm,
                               const Params *model_params,
                               const Data *user_data,
                               const RNG *rng) const {
    if (model_params) {
      const VectorParams *arm_probs(
          dynamic_cast<const VectorParams *>(model_params));
      return (*arm_probs)[arm];
    } else {
      return models_[arm]->prob();
    }
  }
  
  void BinomialBandit::ObserveData(int arm, int numSuccess, int numTrials) {
    models_[arm]->suf()->batch_update(numTrials, numSuccess);
  }
  
  void BinomialBandit::UpdatePosterior() {
    for (auto &model : models_) {
      model->sample_posterior();
    }
  }
  
}  // namespace BOOM
