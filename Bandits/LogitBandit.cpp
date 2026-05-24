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

#include "Bandits/LogitBandit.hpp"

namespace BOOM {

  double LogitBandit::value(int arm, const MixedMultivariateData &context) const {
    Vector predictor_vector = encoder_->encode_row(arm, context);
    return model_->predict(predictor_vector);
  }

  void LogitBandit::observe_data(int arm,
                                 int num_successes,
                                 int num_trials,
                                 const MixedMultivariateData &context) {
    Vector predictor_vector = encoder_->encode_row(arm, context);
    NEW(BinomialRegressionData, data_point)(
        num_successes, num_trials, predictor_vector);
    model_->add_data(data_point);
  }

  void LogitBandit::update_posterior(int ndraws) {
    coefficient_draws_.resize(ndraws, model_->xdim());
    for (int i = 0; i < ndraws; ++i) {
      model_->sample_posterior();
      coefficient_draws_.row(i) = model_->Beta();
    }
  }

  // Vector OptimalArmProbabilities(const MixedMultivariateData &context) const {
  //   // TODO
  // }
  
}  // namespace BOOM
