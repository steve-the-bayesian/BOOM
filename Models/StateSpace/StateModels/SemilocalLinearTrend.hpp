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

#ifndef BOOM_SEMILOCAL_LINEAR_TREND_STATE_MODEL_HPP_
#define BOOM_SEMILOCAL_LINEAR_TREND_STATE_MODEL_HPP_
#include "Models/Policies/CompositeParamPolicy.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/StateSpace/Filters/SparseMatrix.hpp"
#include "Models/StateSpace/Filters/SparseVector.hpp"
#include "Models/StateSpace/StateModels/StateModel.hpp"
#include "Models/TimeSeries/NonzeroMeanAr1Model.hpp"
#include "Models/ZeroMeanGaussianModel.hpp"
namespace BOOM {

  // The state transition matrix for the
  // SemilocalLinearTrendMatrix is
  //  1   1   0
  //  0  phi (1-phi)
  //  0   0   1
  //
  // This class is tested in the test suite for the other sparse matrices.
  class SemilocalLinearTrendMatrix : public SparseMatrixBlock {
   public:
    explicit SemilocalLinearTrendMatrix(const Ptr<UnivParams> &phi);

    // Can safely copy with pointer semantics, becasue nothing in this
    // class can change the value of the pointer.
    SemilocalLinearTrendMatrix(const SemilocalLinearTrendMatrix &rhs);
    SemilocalLinearTrendMatrix *clone() const override;
    int nrow() const override { return 3; }
    int ncol() const override { return 3; }
    void multiply(VectorView lhs, const ConstVectorView &rhs) const override;
    void multiply_and_add(VectorView lhs,
                          const ConstVectorView &rhs) const override;
    void Tmult(VectorView lhs, const ConstVectorView &rhs) const override;
    void multiply_inplace(VectorView x) const override;
    SpdMatrix inner() const override;
    SpdMatrix inner(const ConstVectorView &weights) const override;
    void add_to_block(SubMatrix block) const override;
    Matrix dense() const override;

   private:
    Ptr<UnivParams> phi_;
  };

  // The state equations are:
  //  mu[t+1] = mu[t] + delta[t] + u[t]
  //  delta[t+1] = D + phi * (delta[t]-D) + v[t]
  // To put this model in state space form requires a 3-dimensional state:
  //     alpha[t] = (mu[t], delta[t], D)
  // Here D is the time-invariant mean parameter of an AR1 model for
  // which phi is the AR1 coefficient.
  //
  // The error expander is
  //   | 1 0 |
  //   | 0 1 |
  //   | 0 0 |
  class SemilocalLinearTrendStateModel : virtual public StateModel,
                                         public CompositeParamPolicy,
                                         public IID_DataPolicy<VectorData>,
                                         public PriorPolicy {
   public:
    SemilocalLinearTrendStateModel(const Ptr<ZeroMeanGaussianModel> &level,
                                   const Ptr<NonzeroMeanAr1Model> &slope);
    SemilocalLinearTrendStateModel(const SemilocalLinearTrendStateModel &rhs);
    SemilocalLinearTrendStateModel *clone() const override;

    void clear_data() override;
    void observe_state(const ConstVectorView &then, const ConstVectorView &now,
                       int time_now) override;

    void observe_initial_state(const ConstVectorView &state) override;
    uint state_dimension() const override { return 3; }
    uint state_error_dimension() const override { return 2; }

    // This model throws, because the AR1 state model for the slope
    // cannot be part of an EM algorithm.
    void update_complete_data_sufficient_statistics(
        int t, const ConstVectorView &state_error_mean,
        const ConstSubMatrix &state_error_variance) override;

    void simulate_state_error(RNG &rng, VectorView eta, int t) const override;
    Ptr<SparseMatrixBlock> state_transition_matrix(int t) const override;
    Ptr<SparseMatrixBlock> state_variance_matrix(int t) const override;
    Ptr<SparseMatrixBlock> state_error_expander(int t) const override;
    Ptr<SparseMatrixBlock> state_error_variance(int t) const override;

    SparseVector observation_matrix(int t) const override;

    Vector initial_state_mean() const override;
    SpdMatrix initial_state_variance() const override;

    void set_initial_level_mean(double level_mean);
    void set_initial_level_sd(double level_sd);
    void set_initial_slope_mean(double slope_mean);
    void set_initial_slope_sd(double slope_sd);

    void simulate_initial_state(RNG &rng, VectorView state) const override;

    double level_sd() const {return level_->sigma();}
    double slope_sd() const {return slope_->sigma();}
    double slope_mean() const {return slope_->mu();}
    double slope_ar_coefficient() const {return slope_->phi();}

    void set_level_sd(double sd) {level_->set_sigsq(sd * sd);}
    void set_slope_sd(double sd) {slope_->set_sigsq(sd * sd);}
    void set_slope_mean(double mean) {slope_->set_mu(mean);}
    void set_slope_ar_coefficient(double ar) {slope_->set_phi(ar);}

   private:
    void check_dim(const ConstVectorView &) const;
    std::vector<Ptr<UnivParams> > get_variances();
    Ptr<ZeroMeanGaussianModel> level_;
    Ptr<NonzeroMeanAr1Model> slope_;

    SparseVector observation_matrix_;
    Ptr<SemilocalLinearTrendMatrix> state_transition_matrix_;
    Ptr<UpperLeftDiagonalMatrix> state_variance_matrix_;
    Ptr<ZeroPaddedIdentityMatrix> state_error_expander_;
    Ptr<UpperLeftDiagonalMatrix> state_error_variance_;
    double initial_level_mean_;
    double initial_slope_mean_;
    SpdMatrix initial_state_variance_;
  };

}  // namespace BOOM
#endif  // BOOM_SEMILOCAL_LINEAR_TREND_STATE_MODEL_HPP_
