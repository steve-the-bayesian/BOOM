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
#include <Models/StateSpace/StateModels/TrigStateModel.hpp>
#include <distributions.hpp>
#include <cmath>
#include <cpputil/Constants.hpp>

namespace BOOM {

  TrigStateModel::TrigStateModel(double period, const Vector &frequencies)
      : IndependentMvnModel(2 * frequencies.size()),
        period_(period),
        frequencies_(frequencies),
        state_transition_matrix_(new IdentityMatrix(state_dimension())),
        state_variance_matrix_(
            new DiagonalMatrixBlockVectorParamView(
                Sigsq_prm()))
  {
    if (frequencies_.empty()) {
      report_error("At least one frequency needed to "
                   "initialize TrigStateModel.");
    }
    for (int i = 0; i < frequencies_.size(); ++i) {
      frequencies_[i] = 2 * Constants::pi * frequencies_[i] / period_;
    }
    set_mu(Vector(state_dimension(), 0));
  }

  TrigStateModel * TrigStateModel::clone() const {
    return new TrigStateModel(*this);
  }

  void TrigStateModel::observe_state(const ConstVectorView then,
                                     const ConstVectorView now,
                                     int time_now) {
    suf()->update_raw(now - then);
  }

  void TrigStateModel::update_complete_data_sufficient_statistics(
      int t,
      const ConstVectorView &state_error_mean,
      const ConstSubMatrix &state_error_variance) {
    suf()->update_expected_value(
        1.0,
        state_error_mean,
        state_error_variance.diag() + pow(state_error_mean, 2));
  }

  void TrigStateModel::simulate_state_error(RNG &rng, VectorView eta, int t) const {
    eta = sim(rng);
  }

  SparseVector TrigStateModel::observation_matrix(int t) const {
    Vector trig(state_dimension());
    for (int i = 0; i < frequencies_.size(); ++i) {
      trig[2 * i] = cos(frequencies_[i] * t);
      trig[2 * i + 1] = sin(frequencies_[i] * t);
    }
    return SparseVector(trig);
  }

  Vector TrigStateModel::initial_state_mean() const {
    return initial_state_mean_;
  }

  void TrigStateModel::set_initial_state_mean(const Vector &mean) {
    if (mean.size() != state_dimension()) {
      report_error("Initial state mean is the wrong size for TrigStateModel.");
    }
    initial_state_mean_ = mean;
  }

  SpdMatrix TrigStateModel::initial_state_variance() const {
    return initial_state_variance_;
  }

  void TrigStateModel::set_initial_state_variance(
      const SpdMatrix &variance) {
    if (nrow(variance) != state_dimension()) {
      report_error("initial_state_variance is the wrong size "
                   "in TrigStateModel.");
    }
    initial_state_variance_ = variance;
  }

}
