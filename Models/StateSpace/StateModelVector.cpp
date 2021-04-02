/*
  Copyright (C) 2005-2019 Steven L. Scott

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

#include "Models/StateSpace/StateModelVector.hpp"

namespace BOOM {
  namespace StateSpaceUtils {

    void StateModelVectorBase::add_state_model(
        Ptr<StateModelBase> state_model) {
        state_model->set_index(state_models_.size());
        state_models_.push_back(state_model);
        state_dimension_ += state_model->state_dimension();
        int next_position = state_positions_.back()
            + state_model->state_dimension();
        state_positions_.push_back(next_position);

        state_error_dimension_ += state_model->state_error_dimension();
        next_position = state_error_positions_.back()
            + state_model->state_error_dimension();
        state_error_positions_.push_back(next_position);
    }

    void StateModelVectorBase::clear_state_models() {
      state_dimension_ = 0;
      state_error_dimension_ = 0;
      state_positions_.clear();
      state_positions_.push_back(0);
      state_error_positions_.clear();
      state_error_positions_.push_back(0);

      state_transition_matrix_->clear();
      state_variance_matrix_->clear();
      state_error_expander_->clear();
      state_error_variance_->clear();
    }

    void StateModelVectorBase::clear_data() {
      for (int s = 0; s < state_models_.size(); ++s) {
        state_models_[s]->clear_data();
      }
    }

    VectorView StateModelVectorBase::state_component(
        Vector &state, int s) const {
      int start = state_positions_[s];
      int size = state_models_[s]->state_dimension();
      return VectorView(state, start, size);
    }

    VectorView StateModelVectorBase::state_component(
        VectorView &state, int s) const {
      int start = state_positions_[s];
      int size = state_models_[s]->state_dimension();
      return VectorView(state, start, size);
    }

    ConstVectorView StateModelVectorBase::state_component(
        const ConstVectorView &state,
        int s) const {
      int start = state_positions_[s];
      int size = state_models_[s]->state_dimension();
      return ConstVectorView(state, start, size);
    }

    ConstVectorView StateModelVectorBase::const_state_error_component(
        const Vector &full_state_error,
        int state_model_number) const {
      int start = state_error_positions_[state_model_number];
      int size = state_models_[state_model_number]->state_error_dimension();
      return ConstVectorView(full_state_error, start, size);
    }
    VectorView StateModelVectorBase::state_error_component(
        Vector &full_state_error, int state_model_number) const {
      int start = state_error_positions_[state_model_number];
      int size = state_models_[state_model_number]->state_error_dimension();
      return VectorView(full_state_error, start, size);
    }

    ConstSubMatrix StateModelVectorBase::state_error_variance_component(
        const SpdMatrix &full_error_variance, int state) const {
      int start = state_error_positions_[state];
      int size = state_models_[state]->state_error_dimension();
      return ConstSubMatrix(full_error_variance, start, start + size - 1, start,
                            start + size - 1);
    }

    ConstSubMatrix StateModelVectorBase::full_state_subcomponent(
        const Matrix &state, int state_model_index) const {
      int start = state_positions_[state_model_index];
      int size = state_models_[state_model_index]->state_dimension();
      return ConstSubMatrix(state, start, start + size - 1, 0,
                            state.ncol() - 1);
    }
    SubMatrix StateModelVectorBase::mutable_full_state_subcomponent(
        Matrix &state, int state_model_index) const {
      int start = state_positions_[state_model_index];
      int size = state_models_[state_model_index]->state_dimension();
      return SubMatrix(state, start, start + size - 1, 0,
                       state.ncol() - 1);
    }

    const SparseKalmanMatrix *StateModelVectorBase::state_transition_matrix(
        int t) const {
      // Size comparisons should be made with respect to state_dimension(), not
      // state_dimension() which is virtual.
      if (state_transition_matrix_->nrow() != state_dimension() ||
          state_transition_matrix_->ncol() != state_dimension()) {
        state_transition_matrix_->clear();
        for (int s = 0; s < state_models_.size(); ++s) {
          state_transition_matrix_->add_block(
              state_models_[s]->state_transition_matrix(t));
        }
      } else {
        // If we're in this block, then the matrix must have been created already,
        // and we just need to update the blocks.
        for (int s = 0; s < state_models_.size(); ++s) {
          state_transition_matrix_->replace_block(
              s, state_models_[s]->state_transition_matrix(t));
        }
      }
      return state_transition_matrix_.get();
    }

    const SparseKalmanMatrix *StateModelVectorBase::state_variance_matrix(
        int t) const {
      state_variance_matrix_->clear();
      for (int s = 0; s < state_models_.size(); ++s) {
        state_variance_matrix_->add_block(
            state_models_[s]->state_variance_matrix(t));
      }
      return state_variance_matrix_.get();
    }

    const ErrorExpanderMatrix *StateModelVectorBase::state_error_expander(
        int t) const {
      state_error_expander_->clear();
      for (int s = 0; s < state_models_.size(); ++s) {
        state_error_expander_->add_block(state_models_[s]->state_error_expander(t));
      }
      return state_error_expander_.get();
    }

    const SparseKalmanMatrix *StateModelVectorBase::state_error_variance(
        int t) const {
      state_error_variance_->clear();
      for (int s = 0; s < state_models_.size(); ++s) {
        state_error_variance_->add_block(
            state_models_[s]->state_error_variance(t));
      }
      return state_error_variance_.get();
    }

  }  // namespace StateSpaceUtils
}  // namespace
