// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2011 Steven L. Scott

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

#include "Models/StateSpace/StateModels/SemilocalLinearTrend.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {
  typedef SemilocalLinearTrendMatrix LMAT;
  typedef SemilocalLinearTrendStateModel SLLT;

  LMAT::SemilocalLinearTrendMatrix(const Ptr<UnivParams> &phi) : phi_(phi) {}

  LMAT::SemilocalLinearTrendMatrix(const LMAT &rhs)
      : SparseMatrixBlock(rhs), phi_(rhs.phi_) {}

  LMAT *LMAT::clone() const { return new LMAT(*this); }

  void LMAT::multiply(VectorView lhs, const ConstVectorView &rhs) const {
    if (lhs.size() != 3) {
      report_error("lhs is the wrong size in LMAT::multiply");
    }
    if (rhs.size() != 3) {
      report_error("rhs is the wrong size in LMAT::multiply");
    }
    double phi = phi_->value();
    lhs[0] = rhs[0] + rhs[1];
    lhs[1] = phi * rhs[1] + (1 - phi) * rhs[2];
    lhs[2] = rhs[2];
  }

  void LMAT::multiply_and_add(VectorView lhs,
                              const ConstVectorView &rhs) const {
    if (lhs.size() != 3) {
      report_error("lhs is the wrong size in LMAT::multiply");
    }
    if (rhs.size() != 3) {
      report_error("rhs is the wrong size in LMAT::multiply");
    }
    double phi = phi_->value();
    lhs[0] += rhs[0] + rhs[1];
    lhs[1] += phi * rhs[1] + (1 - phi) * rhs[2];
    lhs[2] += rhs[2];
  }

  void LMAT::Tmult(VectorView lhs, const ConstVectorView &rhs) const {
    if (lhs.size() != 3) {
      report_error("lhs is the wrong size in LMAT::Tmult");
    }
    if (rhs.size() != 3) {
      report_error("rhs is the wrong size in LMAT::Tmult");
    }
    lhs[0] = rhs[0];
    double phi = phi_->value();
    lhs[1] = rhs[0] + phi * rhs[1];
    lhs[2] = (1 - phi) * rhs[1] + rhs[2];
  }

  void LMAT::multiply_inplace(VectorView x) const {
    x[0] += x[1];
    double phi = phi_->value();
    x[1] = phi * x[1] + (1 - phi) * x[2];
  }

  SpdMatrix LMAT::inner() const {
    SpdMatrix ans(3, 1.0);
    ans(0, 1) = 1.0;
    ans(1, 0) = 1.0;
    double phi = phi_->value();
    ans(1, 1) += phi * phi;
    ans(1, 2) = ans(2, 1) = phi * (1 - phi);
    ans(2, 2) += square(1 - phi);
    return ans;
  }

  SpdMatrix LMAT::inner(const ConstVectorView &weights) const {
    SpdMatrix ans(3, 0.0);
    ans(0, 0) = ans(0, 1) = ans(1, 0) = weights[0];
    double phi = phi_->value();
    ans(1, 1) = weights[0] + weights[1] * square(phi);
    ans(1, 2) = ans(2, 1) = weights[1] * phi * (1 - phi);
    ans(2, 2) = weights[2] + weights[1] * square(1 - phi);
    return ans;
  }
  
  void LMAT::add_to_block(SubMatrix block) const {
    if (block.nrow() != 3 || block.ncol() != 3) {
      report_error("block is the wrong size in LMAT::add_to_block");
    }
    double phi = phi_->value();
    block(0, 0) += 1;
    block(0, 1) += 1;
    block(1, 1) += phi;
    block(1, 2) += 1 - phi;
    block(2, 2) += 1;
  }

  Matrix LMAT::dense() const {
    Matrix ans(3, 3, 0.0);
    ans(0, 0) = 1.0;
    ans(0, 1) = 1.0;
    double phi = phi_->value();
    ans(1, 1) = phi;
    ans(1, 2) = 1 - phi;
    ans(2, 2) = 1.0;
    return ans;
  }

  SLLT::SemilocalLinearTrendStateModel(const Ptr<ZeroMeanGaussianModel> &level,
                                       const Ptr<NonzeroMeanAr1Model> &slope)
      : level_(level),
        slope_(slope),
        observation_matrix_(3),
        state_transition_matrix_(new LMAT(slope_->Phi_prm())),
        state_variance_matrix_(new UpperLeftDiagonalMatrix(get_variances(), 3)),
        state_error_expander_(new ZeroPaddedIdentityMatrix(3, 2)),
        state_error_variance_(new UpperLeftDiagonalMatrix(get_variances(), 2)),
        initial_level_mean_(0.0),
        initial_slope_mean_(0.0),
        initial_state_variance_(3, 1.0) {
    observation_matrix_[0] = 1;
    ParamPolicy::add_model(level_);
    ParamPolicy::add_model(slope_);

    // The mean of the slope is a known model parameter.
    initial_state_variance_(2, 2) = 0;
  }

  SLLT::SemilocalLinearTrendStateModel(const SLLT &rhs)
      : Model(rhs),
        StateModel(rhs),
        ParamPolicy(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs),
        level_(rhs.level_->clone()),
        slope_(rhs.slope_->clone()),
        observation_matrix_(rhs.observation_matrix_),
        state_transition_matrix_(new LMAT(slope_->Phi_prm())),
        state_variance_matrix_(new UpperLeftDiagonalMatrix(get_variances(), 3)),
        state_error_expander_(new ZeroPaddedIdentityMatrix(3, 2)),
        state_error_variance_(new UpperLeftDiagonalMatrix(get_variances(), 2)),
        initial_level_mean_(rhs.initial_level_mean_),
        initial_slope_mean_(rhs.initial_slope_mean_),
        initial_state_variance_(rhs.initial_state_variance_) {
    ParamPolicy::add_model(level_);
    ParamPolicy::add_model(slope_);
  }

  SLLT *SLLT::clone() const { return new SLLT(*this); }

  void SLLT::clear_data() {
    level_->clear_data();
    slope_->clear_data();
  }

  // State is (level, slope, slope_mean) The level model expects the
  // error term in level.  The slope model expects the current value
  // of the slope.
  void SLLT::observe_state(const ConstVectorView &then,
                           const ConstVectorView &now, int time_now) {
    double change_in_level = now[0] - then[0] - then[1];
    level_->suf()->update_raw(change_in_level);

    double current_slope = now[1];
    slope_->suf()->update_raw(current_slope);
  }

  void SLLT::observe_initial_state(const ConstVectorView &state) {
    slope_->suf()->update_raw(state[1]);
  }

  void SLLT::update_complete_data_sufficient_statistics(
      int t, const ConstVectorView &, const ConstSubMatrix &) {
    report_error(
        "SemilocalLinearTrendStateModel cannot "
        "be part of an EM algorithm.");
  }

  void SLLT::simulate_state_error(RNG &rng, VectorView eta, int t) const {
    assert(eta.size() == state_dimension());
    eta[0] = rnorm_mt(rng, 0, level_->sigma());
    eta[1] = rnorm_mt(rng, 0, slope_->sigma());
    eta[2] = 0;
  }

  Ptr<SparseMatrixBlock> SLLT::state_transition_matrix(int) const {
    return state_transition_matrix_;
  }

  Ptr<SparseMatrixBlock> SLLT::state_variance_matrix(int) const {
    return state_variance_matrix_;
  }

  Ptr<SparseMatrixBlock> SLLT::state_error_expander(int) const {
    return state_error_expander_;
  }

  Ptr<SparseMatrixBlock> SLLT::state_error_variance(int) const {
    return state_error_variance_;
  }

  SparseVector SLLT::observation_matrix(int) const {
    return observation_matrix_;
  }

  Vector SLLT::initial_state_mean() const {
    Vector ans(3);
    ans[0] = initial_level_mean_;
    ans[1] = initial_slope_mean_;
    ans[2] = slope_->mu();
    return ans;
  }

  SpdMatrix SLLT::initial_state_variance() const {
    return initial_state_variance_;
  }
  void SLLT::set_initial_level_mean(double level_mean) {
    initial_level_mean_ = level_mean;
  }
  void SLLT::set_initial_level_sd(double level_sd) {
    initial_state_variance_(0, 0) = pow(level_sd, 2);
  }
  void SLLT::set_initial_slope_mean(double slope_mean) {
    initial_slope_mean_ = slope_mean;
  }
  void SLLT::set_initial_slope_sd(double sd) {
    initial_state_variance_(1, 1) = pow(sd, 2);
  }

  void SLLT::check_dim(const ConstVectorView &v) const {
    if (v.size() != 3) {
      ostringstream err;
      err << "improper dimesion of ConstVectorView v = :" << v << endl
          << "in SemilocalLinearTrendStateModel.  "
          << "Should be of dimension 3" << endl;
      report_error(err.str());
    }
  }

  std::vector<Ptr<UnivParams> > SLLT::get_variances() {
    std::vector<Ptr<UnivParams> > ans(2);
    ans[0] = level_->Sigsq_prm();
    ans[1] = slope_->Sigsq_prm();
    return ans;
  }

  void SLLT::simulate_initial_state(RNG &rng, VectorView state) const {
    check_dim(state);
    state[0] =
        rnorm_mt(rng, initial_level_mean_, sqrt(initial_state_variance_(0, 0)));
    state[1] =
        rnorm_mt(rng, initial_slope_mean_, sqrt(initial_state_variance_(1, 1)));
    state[2] = slope_->mu();
  }

}  // namespace BOOM
