#ifndef BOOM_STATE_SPACE_STATE_MODEL_VECTOR_HPP_
#define BOOM_STATE_SPACE_STATE_MODEL_VECTOR_HPP_

/*
  Copyright (C) 2019 Steven L. Scott

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

#include <vector>
#include "cpputil/Ptr.hpp"
#include "LinAlg/Vector.hpp"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/VectorView.hpp"
#include "LinAlg/SubMatrix.hpp"

namespace BOOM {
  namespace StateSpaceUtils {

    // Much of the work done by state space models involves dividing work among
    // the state models.  A StateModelVector keeps track of the state models
    // stored by a state space model, and handles things like keeping track of the
    // state dimension and state error dimension, and finding the subvector of a
    // full state vector associated with a particular state model.

    template <class STATE_MODEL>
    class StateModelVector {
     public:

      StateModelVector()
          : state_dimension_(0),
            state_error_dimension_(0),
            state_positions_(1, 0),
            state_error_positions_(1, 0)
      {}
    
      void add_state(Ptr<STATE_MODEL> state_model) {
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

      void clear() {
        state_models_.clear();
        state_dimension_ = 0;
        state_error_dimension_ = 0;
        state_positions_.clear();
        state_positions_.push_back(0);
        state_error_positions_.clear();
        state_error_positions_.push_back(0);
      }
    
      Ptr<STATE_MODEL> operator[](int s) {return state_models_[s];}
      const Ptr<STATE_MODEL> operator[](int s) const {return state_models_[s];}

      int state_dimension() const {
        return state_dimension_;
      }

      int size() const {
        return state_models_.size();
      }

      // Takes the full state vector as input, and returns the component of the
      // state vector belonging to state model s.
      //
      // Args:
      //   state:  The full state vector.
      //   s:  The index of the state model whose state component is desired.
      //
      // Returns:
      //   The subset of the 'state' argument corresponding to state model 's'.
      VectorView state_component(Vector &state, int s) const {
        int start = state_positions_[s];
        int size = state_models_[s]->state_dimension();
        return VectorView(state, start, size);
      }
      VectorView state_component(VectorView &state, int s) const {
        int start = state_positions_[s];
        int size = state_models_[s]->state_dimension();
        return VectorView(state, start, size);
      }
      ConstVectorView state_component(const ConstVectorView &state,
                                      int s) const {
        int start = state_positions_[s];
        int size = state_models_[s]->state_dimension();
        return ConstVectorView(state, start, size);
      }

      // Return the component of the full state error vector corresponding to a
      // given state model.
      //
      // Args:
      //   full_state_error: The error for the full state vector (i.e. all state
      //     models).
      //   state_model_number:  The index of the desired state model.
      //
      // Returns:
      //   The error vector for just the specified state model.
      ConstVectorView const_state_error_component(const Vector &full_state_error,
                                                  int state_model_number) const {
        int start = state_error_positions_[state_model_number];
        int size = state_models_[state_model_number]->state_error_dimension();
        return ConstVectorView(full_state_error, start, size);
      }
      VectorView state_error_component(Vector &full_state_error,
                                       int state_model_number) const {
        int start = state_error_positions_[state_model_number];
        int size = state_models_[state_model_number]->state_error_dimension();
        return VectorView(full_state_error, start, size);
      }

      // Returns the subcomponent of the (block diagonal) error variance matrix
      // corresponding to a specific state model.
      //
      // Args:
      //   full_error_variance:  The full state error variance matrix.
      //   state: The index of the state model defining the desired sub-component.
      ConstSubMatrix state_error_variance_component(
          const SpdMatrix &full_error_variance, int state) const {
        int start = state_error_positions_[state];
        int size = state_models_[state]->state_error_dimension();
        return ConstSubMatrix(full_error_variance, start, start + size - 1, start,
                              start + size - 1);
      }

      //----------------------------------------------------------------------
      // Returns the complete state vector (across time, so the return value is a
      // matrix) for a specified state component.
      //
      // Args:
      //   state: The state matrix (rows are state variables, columns are time) to
      //     subset.
      //   state_model_index:  The index of the desired state model.
      //
      // Returns:
      //   A matrix giving the imputed value of the state vector for the specified
      //   state model.  The matrix has S rows and T columns, where S is the
      //   dimension of the state vector for the specified state model, and T is
      //   the number of time points.
      ConstSubMatrix full_state_subcomponent(
          const Matrix &state, int state_model_index) const {
        int start = state_positions_[state_model_index];
        int size = state_models_[state_model_index]->state_dimension();
        return ConstSubMatrix(state, start, start + size - 1, 0,
                              state.ncol() - 1);
      }
      SubMatrix mutable_full_state_subcomponent(
          Matrix &state, int state_model_index) {
        int start = state_positions_[state_model_index];
        int size = state_models_[state_model_index]->state_dimension();
        return SubMatrix(state, start, start + size - 1, 0,
                         state.ncol() - 1);
      }
    
     private:
      std::vector<Ptr<STATE_MODEL>> state_models_;

      // Dimension of the latent state vector.  Constructors set state_dimension
      // to zero.  It is incremented during calls to add_state.
      int state_dimension_;

      // At construction time state_error_dimension_ is set to zero.  It is
      // incremented during calls to add_state.  It gives the dimension of the
      // state innovation vector (from the transition equation), which can be of
      // lower dimension than the state itself.
      int state_error_dimension_;

      // state_positions_[s] is the index in the state vector where the state for
      // state_models_[s] begins.  There will be one more entry in this vector
      // than the number of state models.  The last entry can be ignored.
      std::vector<int> state_positions_;

      // state_error_positions_[s] is the index in the vector of state errors
      // where the error for state_models_[s] begins.  This vector should have the
      // same number of elements as state_positions_, but the entries can be
      // different because state errors can be lower dimensional than the states
      // themselves.
      std::vector<int> state_error_positions_;
    };
    
  }  // namespace StateSpaceUtils  
}  // namespace BOOM

#endif  // BOOM_STATE_SPACE_STATE_MODEL_VECTOR_HPP_
