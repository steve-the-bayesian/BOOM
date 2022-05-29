#ifndef BOOM_GENERAL_SEASONAL_STATE_MODEL_HPP_
#define BOOM_GENERAL_SEASONAL_STATE_MODEL_HPP_
/*
  Copyright (C) 2018 Steven L. Scott

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

#include "Models/StateSpace/StateModels/StateModel.hpp"
#include "Models/StateSpace/StateModels/LocalLinearTrend.hpp"

#include "Models/Policies/CompositeParamPolicy.hpp"
#include "Models/Policies/NullDataPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"

namespace BOOM {
  // A seasonal state model created by assuming a local linear trend model for
  // each season in the cycle.  The model state evolves each time point, but ony
  // one model at a time contributes directly to the observation equation.  The
  // level components of the model are adjusted with each time step, so that the
  // levels always sum to zero.
  //
  // This model is intended to model things like a sinusoid with a growing
  // amplitude.
  //
  // State:
  //   The state vector is an alternating sequence of level and slope
  //   components: phi[0, t], psi[0, t], ..., phi[S-1, t], psi[S-1, t], where
  //   phi[s,t] is the level of a local linear trend for season s at time t, and
  //   psi[s,t] is the corresponding local slope.
  //
  //   The unadjusted levels and slopes evolve according to independent local
  //   linear trend models.  Thus
  //       phi0[s, t+1] = phi[s, t] + psi[s, t] + eta[s, t, 0], and
  //       psi0[s, t+1] = psi[s, t]             + eta[s, t, 1].
  //
  //   phi0, psi0 -> phi, psi by way of a matrix multiplication that subtracts
  //   the means of the phi's while leaving the psi's unchanged.
  //
  // Model matrices:
  //   The observation matrix Z[t] is all 0's except for a 1 in the spot
  //     corresponding to the active level element.  If s == t mod S then Z[t]
  //     has a 1 in spot 2 * s.
  //
  //   The transition matrix T[t] is M * T0[t], where T0[t] is block diagonal
  //     with 2x2 blocks corresponding to the local linear trend transition
  //     matrix.  M is a matrix that subtracts the mean phi_bar from each
  //     phi[s,t].
  //
  //   The unadjusted model is full rank, so the unadjusted error expander
  //     matrix is I.  Subtracting the mean each time multiplies by M (see
  //     above), so R[t] = M.
  //
  //   The error variance matrix is a block diagonal SpdMatrix with 2x2 Spd
  //   blocks corresponding to the error variances of the underlying trend
  //   models.

  class GeneralSeasonalLLT
      : virtual public StateModel,
        public CompositeParamPolicy,
        public NullDataPolicy,
        public PriorPolicy
  {
   public:
    // Args:
    //   nseasons:  The number of seasons in a full cycle.
    //   season_duration: The number of time periods each season lasts.  This
    //     argument is currently ignored.
    explicit GeneralSeasonalLLT(int nseasons, int season_duration = 1);
    GeneralSeasonalLLT(const GeneralSeasonalLLT &rhs);
    GeneralSeasonalLLT & operator=(const GeneralSeasonalLLT &rhs);
    GeneralSeasonalLLT(GeneralSeasonalLLT &&rhs);
    GeneralSeasonalLLT & operator=(GeneralSeasonalLLT &&rhs);
    GeneralSeasonalLLT *clone() const override;

    void observe_time_dimension(int max_time) override {}
    void observe_state(const ConstVectorView &then,
                       const ConstVectorView &now,
                       int time_now) override;

    void observe_initial_state(const ConstVectorView &state) override;
    uint state_dimension() const override {return 2 * nseasons_;}
    uint state_error_dimension() const override {return 2 * nseasons_;}

    void update_complete_data_sufficient_statistics(
        int t,
        const ConstVectorView &state_error_mean,
        const ConstSubMatrix &state_error_variance) override;

    void increment_expected_gradient(
        VectorView gradient,
        int t,
        const ConstVectorView &state_error_mean,
        const ConstSubMatrix &state_error_variance) override;

    // Simulates the state eror at time t, for moving to time t+1.
    // Args:
    //   rng:  The random number generator to use for the simulation.
    //   eta: A view into the error term to be simulated.  ***NOTE*** eta.size()
    //     matches state_dimension(), not state_error_dimension().  If the error
    //     distribution is not full rank then some components of eta will be
    //     deterministic functions of others (most likely just zero).
    //   t: The time index of the error.  The convention is that state[t+1] =
    //     T[t] * state[t] + error[t], so errors at time t are part of the state
    //     at time t+1.
    void simulate_state_error(RNG &rng, VectorView eta, int t) const override;

    Ptr<SparseMatrixBlock> state_transition_matrix(int t) const override {
      return state_transition_matrix_;
    }

    Ptr<SparseMatrixBlock> state_variance_matrix(int t) const override {
      update_state_variance_matrix();
      return state_variance_matrix_;
    }

    Ptr<SparseMatrixBlock> state_error_expander(int t) const override {
      return state_error_expander_;
    }

    Ptr<SparseMatrixBlock> state_error_variance(int t) const override {
      return state_error_variance_;
    }

    SparseVector observation_matrix(int t) const override;

    void set_initial_state_mean(const Vector &initial_state_mean);
    Vector initial_state_mean() const override;

    void set_initial_state_variance(const SpdMatrix &initial_state_variance);
    SpdMatrix initial_state_variance() const override;

    int nseasons() const { return nseasons_; }
    int season_duration() const { return season_duration_; }

    LocalLinearTrendStateModel *subordinate_model(int i) {
      return subordinate_models_[i].get();
    }

   private:
    int nseasons_;
    int season_duration_;

    Ptr<ProductSparseMatrixBlock> state_transition_matrix_;
    Ptr<BlockDiagonalMatrixBlock> state_error_variance_;
    mutable Ptr<DenseMatrix> state_variance_matrix_;
    mutable bool state_variance_matrix_current_;

    // This is the matrix M in the comments.
    Ptr<SubsetEffectConstraintMatrix> state_error_expander_;

    std::vector<Ptr<LocalLinearTrendStateModel>> subordinate_models_;

    Vector initial_state_mean_;
    SpdMatrix initial_state_variance_;

    // Constructor utilities.
    void build_subordinate_models();
    void build_state_matrices();

    // Efficiency utilities.
    void update_state_variance_matrix() const;
  };

}  // namespace BOOM

#endif  // BOOM_GENERAL_SEASONAL_STATE_MODEL_HPP_
