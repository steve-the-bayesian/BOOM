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
#include "Models/StateSpace/StateModels/TrigStateModel.hpp"
#include <cmath>
#include "cpputil/Constants.hpp"
#include "distributions.hpp"

namespace BOOM {
  using std::cout;
  using std::endl;

  TrigStateModel::TrigStateModel(double period, const Vector &frequencies)
      : IndependentMvnModel(2 * frequencies.size()),
        period_(period),
        frequencies_(frequencies),
        state_transition_matrix_(new IdentityMatrix(state_dimension())),
        state_variance_matrix_(
            new DiagonalMatrixBlockVectorParamView(Sigsq_prm())) {
    if (frequencies_.empty()) {
      report_error(
          "At least one frequency needed to "
          "initialize TrigStateModel.");
    }
    for (int i = 0; i < frequencies_.size(); ++i) {
      frequencies_[i] = 2 * Constants::pi * frequencies_[i] / period_;
    }
    set_mu(Vector(state_dimension(), 0));
  }

  TrigStateModel::TrigStateModel(const TrigStateModel &rhs)
      : Model(rhs),
        StateModel(rhs),
        IndependentMvnModel(rhs),
        period_(rhs.period_),
        frequencies_(rhs.frequencies_),
        state_transition_matrix_(new IdentityMatrix(state_dimension())),
        state_variance_matrix_(
            new DiagonalMatrixBlockVectorParamView(Sigsq_prm())),
        initial_state_mean_(rhs.initial_state_mean_),
        initial_state_variance_(rhs.initial_state_variance_) {}

  TrigStateModel *TrigStateModel::clone() const {
    return new TrigStateModel(*this);
  }

  void TrigStateModel::observe_state(const ConstVectorView &then,
                                     const ConstVectorView &now, int time_now,
                                     ScalarStateSpaceModelBase *) {
    suf()->update_raw(now - then);
  }

  void TrigStateModel::update_complete_data_sufficient_statistics(
      int t, const ConstVectorView &state_error_mean,
      const ConstSubMatrix &state_error_variance) {
    suf()->update_expected_value(
        1.0, state_error_mean,
        state_error_variance.diag() + pow(state_error_mean, 2));
  }

  void TrigStateModel::simulate_state_error(RNG &rng, VectorView eta,
                                            int t) const {
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

  void TrigStateModel::set_initial_state_variance(const SpdMatrix &variance) {
    if (nrow(variance) != state_dimension()) {
      report_error(
          "initial_state_variance is the wrong size "
          "in TrigStateModel.");
    }
    initial_state_variance_ = variance;
  }

  //===========================================================================
  QuasiTrigStateModel::QuasiTrigStateModel(double period,
                                           const Vector &frequencies)
      : period_(period),
        frequencies_(frequencies),
        error_distribution_(new ZeroMeanGaussianModel),
        cosines_(frequencies.size()),
        sines_(frequencies.size()),
        state_error_variance_(new ConstantMatrixParamView(
            2 * frequencies_.size(),
            error_distribution_->Sigsq_prm())),
        state_error_expander_(new IdentityMatrix(2 * frequencies_.size())),
        observation_matrix_(2 * frequencies_.size()),
        initial_state_mean_(2 * frequencies_.size(), 0.0),
        initial_state_variance_(2 * frequencies_.size(), 1.0)
  {
    ParamPolicy::add_model(error_distribution_);
    for (int i = 0; i < 2 * frequencies_.size(); i += 2) {
      observation_matrix_[i] = 1.0;
    }
    for (int i = 0; i < frequencies_.size(); ++i) {
      double freq = 2 * Constants::pi * frequencies_[i] / period_;
      cosines_[i] = cos(freq);
      sines_[i] = sin(freq);
    }
  }

  QuasiTrigStateModel::QuasiTrigStateModel(const QuasiTrigStateModel &rhs)
      : StateModel(rhs),
        period_(rhs.period_),
        frequencies_(rhs.frequencies_),
        error_distribution_(rhs.error_distribution_->clone()),
        cosines_(rhs.cosines_),
        sines_(rhs.sines_),
        state_error_variance_(new ConstantMatrixParamView(
            2 * frequencies_.size(),
            error_distribution_->Sigsq_prm())),
        state_error_expander_(rhs.state_error_expander_->clone()),
        observation_matrix_(rhs.observation_matrix_),
        initial_state_mean_(rhs.initial_state_mean_),
        initial_state_variance_(rhs.initial_state_variance_) {
    ParamPolicy::add_model(error_distribution_);
  }

  QuasiTrigStateModel & QuasiTrigStateModel::operator=(
      const QuasiTrigStateModel &rhs) {
    if (&rhs != this) {
      StateModel::operator=(rhs);
      period_ = rhs.period_;
      frequencies_ = rhs.frequencies_;
      error_distribution_ = rhs.error_distribution_->clone();
      cosines_ = rhs.cosines_;
      sines_ = rhs.sines_;
      state_error_variance_.reset(new ConstantMatrixParamView(
          2 * frequencies_.size(),
          error_distribution_->Sigsq_prm()));
      state_error_expander_ = rhs.state_error_expander_->clone();
      observation_matrix_ = rhs.observation_matrix_;
      initial_state_mean_ = rhs.initial_state_mean_;
      initial_state_variance_ = rhs.initial_state_variance_;
      ParamPolicy::clear();
      ParamPolicy::add_model(error_distribution_);
    }
    return *this;
  }

  void QuasiTrigStateModel::observe_state(const ConstVectorView &then,
                                          const ConstVectorView &now,
                                          int time_now,
                                          ScalarStateSpaceModelBase *model) {
    Matrix rotation(2, 2);
    Vector rotated(2);
    if (time_now <= 0) {
      report_error("observe_state called with time_now = 0.");
    }
    for (int i = 0; i < frequencies_.size(); ++i) {
      rotation(0, 0) = cosines_[i];
      rotation(0, 1) = sines_[i];
      rotation(1, 0) = -sines_[i];
      rotation(1, 1) = cosines_[i];
      rotated = rotation * then;
      error_distribution_->suf()->update_raw(now[0] - rotated[0]);
      error_distribution_->suf()->update_raw(now[1] - rotated[1]);
    }
  }

  void QuasiTrigStateModel::update_complete_data_sufficient_statistics(
      int t,
      const ConstVectorView &state_error_mean,
      const ConstSubMatrix &state_error_variance)  {
    ////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////
    // TODO
    ////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////
  }

  void QuasiTrigStateModel::increment_expected_gradient(
      VectorView gradient,
      int t,
      const ConstVectorView &state_error_mean,
      const ConstSubMatrix &state_error_variance) {
    ////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////
    // TODO
    ////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////
  }

  void QuasiTrigStateModel::simulate_state_error(
      RNG &rng, VectorView eta, int t) const {
    double sigma = error_distribution_->sigma();
    for (int i = 0; i < eta.size(); ++i) {
      eta[i] = rnorm_mt(rng, 0, sigma);
    }
  }

  Ptr<SparseMatrixBlock>
  QuasiTrigStateModel::state_transition_matrix(int t) const {
    NEW(BlockDiagonalMatrixBlock, transition)();
    Matrix rotation(2, 2);
    for (int i = 0; i < frequencies_.size(); ++i) {
      rotation(0, 0) = cosines_[i];
      rotation(0, 1) = sines_[i];
      rotation(1, 0) = -sines_[i];
      rotation(1, 1) = cosines_[i];
      transition->add_block(new DenseMatrix(rotation))
    }
    return transition;
  }

}  // namespace BOOM
