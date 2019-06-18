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
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA
*/
#ifndef BOOM_IRT_MULTISUBSCALE_LOGIT_CUTPOINT_MODEL_HPP
#define BOOM_IRT_MULTISUBSCALE_LOGIT_CUTPOINT_MODEL_HPP
#include "Models/Glm/OrdinalCutpointModel.hpp"

#include "Models/IRT/IRT.hpp"
#include "Models/IRT/Item.hpp"
#include "Models/IRT/ItemDataPolicy.hpp"

namespace BOOM {
  namespace IRT {
    class MultisubscaleLogitCutpointModel
        : public OrdinalCutpointModel,
          public ItemDataPolicy<OrdinalRegressionData> {
     public:
      // don't forget to call model_selection(false)
      typedef MultisubscaleLogitCutpointModel MLCM;
      typedef OrdinalCutpointModel base;

      MultisubscaleLogitCutpointModel(const Selector &subs, uint Maxscore);
      MultisubscaleLogitCutpointModel(const MLCM &rhs);
      MLCM *clone() const override;

      std::ostream &display_item_params(std::ostream &, bool decorate = true) const;

      // set to observed proportions assuming theta=0
      void initialize_params() override;

      double response_prob(Response r, const Vector &Theta,
                           bool logscale) const;
      double response_prob(uint r, const Vector &Theta, bool logscale) const;

      DataPolicy::DatasetType &dat() { return DataPolicy::dat(); }
      const DataPolicy::DatasetType &dat() const { return DataPolicy::dat(); }
    };

    Ptr<MultisubscaleLogitCutpointModel> random_mlcm(RNG &rng,
                                                     const Selector &subs,
                                                     uint Maxscore);

  }  // namespace IRT
}  // namespace BOOM
#endif  // BOOM_IRT_MULTISUBSCALE_LOGIT_CUTPOINT_MODEL_HPP
