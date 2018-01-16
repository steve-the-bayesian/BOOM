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

#include <Models/StateSpace/StateModels/StateModel.hpp>
#include <Models/StateSpace/Filters/SparseMatrix.hpp>
#include <Models/IndependentMvnModel.hpp>

namespace BOOM {

  // A state model with trigonometric components (one sine and one
  // cosine at each frequency) that cycle 1, 2, 3, ...,
  // number_of_frequencies times per period.
  //
  // The state of this model is a set of 2 * number_of_frequencies
  // coefficients that move according to a random walk.  The model
  // matrices are
  //   Z[t]     = sines and cosines evaluated at time t.
  //   alpha[t] = coefficients at time t.
  //   T[t]     = Identity matrix.
  //   Q[t]     = diagonal variance matrix for the changes in the
  //              coefficients.
  class TrigStateModel
      : public StateModel,
        public IndependentMvnModel {
   public:
    // Args:
    //   period: The number of time steps (need not be an integer)
    //     that it takes for the longest cycle to repeat.
    //   frequencies: A vector giving the number of times each
    //     sinusoid repeats in a period.  One sine and one cosine will
    //     be added to the model for each entry in frequencies.
    TrigStateModel(double period, const Vector &frequencies);
    TrigStateModel * clone() const override;

    void observe_state(const ConstVectorView then, const ConstVectorView now,
                       int time_now) override;

    uint state_dimension() const override {return 2 * frequencies_.size();}
    uint state_error_dimension() const override {return state_dimension();}

    void update_complete_data_sufficient_statistics(
        int t,
        const ConstVectorView &state_error_mean,
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

    SparseVector observation_matrix(int t)const override;

    Vector initial_state_mean()const override;
    void set_initial_state_mean(const Vector &v);

    SpdMatrix initial_state_variance()const override;
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
