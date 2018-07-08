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

#ifndef BOOM_STATE_SPACE_TRIG_MODEL_HPP_
#define BOOM_STATE_SPACE_TRIG_MODEL_HPP_

#include "Models/IndependentMvnModel.hpp"
#include "Models/StateSpace/Filters/SparseMatrix.hpp"
#include "Models/StateSpace/StateModels/StateModel.hpp"
#include "Models/Policies/CompositeParamPolicy.hpp"
#include "Models/Policies/NullDataPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/ZeroMeanGaussianModel.hpp"

namespace BOOM {

  // A stable version of the trig state model.  Each frequency 
  // lambda_j = 2 * pi * j / S, where S is the period (number of time points in
  // a full cycle) is associated with two time-varying random components:
  // gamma[j, t], and gamma^*[j, t].  They evolve through time as
  //
  // gamma[j, t + 1] = \gamma[j, t] * cos(lambda_j)
  //                   + \gamma^*[j, t] * sin(\lambda_j) + error_0
  // gamma^[j, t + 1] = \gamma^*[j, t] * cos(\lambda_j)
  //                   - \gamma[j, t] * sin(lambda_j) + error_1
  //
  // where error_0 and error_1 are independent with the same variance.  This is
  // the real-valued version of a harmonic function: gamma * exp(i * theta).
  // The transition matrix multiplies the function by exp(i * lambda), so that
  // after 't' steps the harmonic's value is gamma * exp(i * lambda * t).  The
  // state allows gamma to drift over time in a random walk.
  //
  // The state is (gamma_jt, gamma^*_jt)', for j = 1, ... number of frequencies,
  // and the contribution of the state to the mean of y[t] is the sum (over j)
  // of the gamma_jt entries (ignoring the gamma^*_jt entries.
  //
  // The state transition matrix is a block diagonal matrix, where block 'j' is
  //
  //    cos(lambda_j)   sin(lambda_j)
  //   -sin(lambda_j)   cos(lambda_j)
  //
  // The error variance matrix is sigma^2 * I.  There is a common sigma^2
  // parameter shared by all frequencies.
  //
  // The model is full rank, so the state error expander matrix R_t is the
  // identity.
  //
  // The observation_matrix is (1, 0, 1, 0, ...), where the 1's pick out the
  // 'real' part of the state contributions.
  class TrigStateModel
      : virtual public StateModel,
        public CompositeParamPolicy,
        public NullDataPolicy,
        public PriorPolicy 
  {
   public:
    // Args:
    //   period: The number of time steps (need not be an integer) that it takes
    //     for the longest cycle to repeat.
    //   frequencies: A vector giving the number of times each sinusoid repeats
    //     in a period.  One sine and one cosine will be added to the model for
    //     each entry in frequencies.  
    //
    // A typical value of frequencies is {1, 2, 3, ...}.  The number of
    // frequencies should not exceed half the period, and the largest entry in
    // frequencies should not exceed half the period.
    TrigStateModel(double period, const Vector &frequencies);

    ~TrigStateModel() {}
    TrigStateModel(const TrigStateModel &rhs);
    TrigStateModel & operator=(const TrigStateModel &rhs);
    TrigStateModel(TrigStateModel &&rhs) = default;
    TrigStateModel & operator=(TrigStateModel &&rhs) = default;

    TrigStateModel *clone() const override {
      return new TrigStateModel(*this);
    }

    // Member functions inherited from Model that would normally be supplied by
    // a data policy.  These are deferred to the error distribution.
    void clear_data() override {
      error_distribution_->clear_data();
    }
    void add_data(const Ptr<Data> &dp) override {
      error_distribution_->add_data(dp);
    }
    void combine_data(const Model &other_model, bool just_suf = true) override {
      error_distribution_->combine_data(other_model, just_suf);
    }

    // This is the model that needs a posterior sampler.
    ZeroMeanGaussianModel *error_distribution() {
      return error_distribution_.get();
    }

    // Overrides from StateModel.  Please see documentation in the base class.
    void observe_state(const ConstVectorView &then,
                       const ConstVectorView &now,
                       int time_now) override;

    uint state_dimension() const override {
      return 2 * frequencies_.size();
    }

    uint state_error_dimension() const override {
      return state_dimension();
    }

    void update_complete_data_sufficient_statistics(
        int t, const ConstVectorView &state_error_mean,
        const ConstSubMatrix &state_error_variance) override;

    void increment_expected_gradient(
        VectorView gradient,
        int t,
        const ConstVectorView &state_error_mean,
        const ConstSubMatrix &state_error_variance) override;

    void simulate_state_error(RNG &rng, VectorView eta, int t) const override;

    Ptr<SparseMatrixBlock> state_transition_matrix(int t) const override;

    Ptr<SparseMatrixBlock> state_variance_matrix(int t) const override {
      return state_error_variance_;
    }

    Ptr<SparseMatrixBlock> state_error_expander(int t) const override {
      return state_error_expander_;
    }
    
    Ptr<SparseMatrixBlock> state_error_variance(int t) const override {
      return state_error_variance_;
    }
      
    SparseVector observation_matrix(int t) const override {
      return observation_matrix_;
    }
    
    Vector initial_state_mean() const override {
      return initial_state_mean_;
    }
    void set_initial_state_mean(const ConstVectorView &mean);

    SpdMatrix initial_state_variance() const override {
      return initial_state_variance_;
    }
    void set_initial_state_variance(const SpdMatrix &variance);
    
   private:
    // The number of time steps in the longest full cycle.
    double period_;

    // A vector of numbers (typically, but not necessarily, integers) indicating
    // how many complete cycles each sinusoid will go through in one period.
    Vector frequencies_;
    
    // The model describing innovations in the harmonic coefficients over time.
    Ptr<ZeroMeanGaussianModel> error_distribution_;

    Ptr<BlockDiagonalMatrixBlock> state_transition_matrix_;
    Ptr<ConstantMatrixParamView> state_error_variance_;
    Ptr<IdentityMatrix> state_error_expander_;
    SparseVector observation_matrix_;

    Vector initial_state_mean_;
    SpdMatrix initial_state_variance_;
  };

  //===========================================================================
  // A state model with trigonometric components (one sine and one cosine at
  // each frequency) that cycle 1, 2, 3, ..., number_of_frequencies times per
  // period.
  //
  // NOTE: This model is inferior to the 'TrigStateModel' based on harmonics.
  // The main reason is that beta * x has variance increasing in |x|, so by
  // putting the sines and cosines in a regression setting, this model
  // introduces an unwanted form of heteroskedasticity.
  //
  // The state of this model is a set of 2 * number_of_frequencies
  // coefficients that move according to a random walk.  The model
  // matrices are
  //   Z[t]     = sines and cosines evaluated at time t.
  //   alpha[t] = coefficients at time t.
  //   T[t]     = Identity matrix.
  //   Q[t]     = diagonal variance matrix for the changes in the
  //              coefficients.
  class TrigRegressionStateModel : virtual public StateModel, public IndependentMvnModel {
   public:
    // Args:
    //   period: The number of time steps (need not be an integer) that it takes
    //     for the longest cycle to repeat.
    //   frequencies: A vector giving the number of times each sinusoid repeats
    //     in a period.  One sine and one cosine will be added to the model for
    //     each entry in frequencies.
    TrigRegressionStateModel(double period, const Vector &frequencies);
    TrigRegressionStateModel(const TrigRegressionStateModel &rhs);
    TrigRegressionStateModel(TrigRegressionStateModel &&rhs) = default;
    TrigRegressionStateModel *clone() const override;

    void observe_state(const ConstVectorView &then, const ConstVectorView &now,
                       int time_now) override;

    uint state_dimension() const override { return 2 * frequencies_.size(); }
    uint state_error_dimension() const override { return state_dimension(); }

    void update_complete_data_sufficient_statistics(
        int t, const ConstVectorView &state_error_mean,
        const ConstSubMatrix &state_error_variance) override;

    void simulate_state_error(RNG &rng, VectorView eta, int t) const override;

    Ptr<SparseMatrixBlock> state_transition_matrix(int t) const override {
      return state_transition_matrix_;
    }

    Ptr<SparseMatrixBlock> state_variance_matrix(int t) const override {
      return state_variance_matrix_;
    }

    Ptr<SparseMatrixBlock> state_error_expander(int t) const override {
      return state_transition_matrix(t);
    }

    Ptr<SparseMatrixBlock> state_error_variance(int t) const override {
      return state_variance_matrix(t);
    }

    SparseVector observation_matrix(int t) const override;

    Vector initial_state_mean() const override;
    void set_initial_state_mean(const Vector &v);

    SpdMatrix initial_state_variance() const override;
    void set_initial_state_variance(const SpdMatrix &V);

   private:
    double period_;
    Vector frequencies_;
    Ptr<IdentityMatrix> state_transition_matrix_;
    Ptr<DiagonalMatrixBlockVectorParamView> state_variance_matrix_;

    Vector initial_state_mean_;
    SpdMatrix initial_state_variance_;
  };
  
}  // namespace BOOM

#endif  //  BOOM_STATE_SPACE_TRIG_MODEL_HPP_
