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

#include "Models/StateSpace/Filters/SparseMatrix.hpp"
#include "Models/StateSpace/StateModels/StateModel.hpp"

namespace BOOM {
  namespace StateSpaceUtils {

    // A StateModelVector manages state models for a state space model.  It
    // stores the models, keeps track of where the state for each model begins
    // and ends, and handles tasks like partitioning a state vector into
    // model-specific bits.  A StateModelVector knows the type of the state
    // model it holds.  This base class captures the parts of the interface of a
    // StateModelVector that don't depend on the type of state model being
    // stored.
    class StateModelVectorBase {
     public:
      StateModelVectorBase()
          : state_transition_matrix_(new BlockDiagonalMatrix),
            state_variance_matrix_(new BlockDiagonalMatrix),
            state_error_expander_(new ErrorExpanderMatrix),
            state_error_variance_(new BlockDiagonalMatrix)
      {
        clear_state_models();
      }

      StateModelVectorBase(const StateModelVectorBase &rhs)
      {
        clear_state_models();
        for (int m = 0; m < rhs.state_models_.size(); ++m) {
          add_state_model(rhs.state_models_[m]);
        }
      }

      StateModelVectorBase(StateModelVectorBase &&rhs):
          state_models_(std::move(rhs.state_models_)),
          state_dimension_(rhs.state_dimension_),
          state_error_dimension_(rhs.state_error_dimension_),
          state_positions_(std::move(rhs.state_positions_)),
          state_error_positions_(std::move(rhs.state_error_positions_)),
          state_transition_matrix_(std::move(rhs.state_transition_matrix_)),
          state_variance_matrix_(std::move(rhs.state_variance_matrix_)),
          state_error_expander_(std::move(rhs.state_error_expander_)),
          state_error_variance_(std::move(rhs.state_error_variance_))
      {}

      StateModelVectorBase & operator=(const StateModelVectorBase &rhs) {
        if (&rhs != this) {
          clear_state_models();
          for (int m = 0; m < rhs.state_models_.size(); ++m) {
            add_state_model(rhs.state_models_[m]);
          }
        }
        return *this;
      }

      StateModelVectorBase &operator=(StateModelVectorBase &&rhs) {
        if (&rhs != this) {
          clear_state_models();
          state_models_ = std::move(rhs.state_models_);
          state_dimension_ = rhs.state_dimension_;
          state_error_dimension_ = rhs.state_error_dimension_;
          state_positions_ = std::move(rhs.state_positions_);
          state_error_positions_ = std::move(rhs.state_error_positions_);
          state_transition_matrix_ = std::move(rhs.state_transition_matrix_);
          state_variance_matrix_ = std::move(rhs.state_variance_matrix_);
          state_error_expander_ = std::move(rhs.state_error_expander_);
          state_error_variance_ = std::move(rhs.state_error_variance_);
        }
        return *this;
      }

      virtual ~StateModelVectorBase() {}

      // The dimension of the state vector associated with the stored models.
      int state_dimension() const { return state_dimension_; }

      // The number of state models stored by this object.
      int size() const { return state_models_.size(); }

      // Clear the vector of models and restore the state of the object to that
      // produced by the default constructor.
      virtual void clear() = 0;

      // Clear the data from the stored models.
      void clear_data();

      // Access to individual state models.
      StateModelBase *state_model(int s);
      const StateModelBase *state_model(int s) const;

      //----------------------------------------------------------------------
      // The subset of the full state vector belonging to state model s.
      //
      // Args:
      //   state:  The full state vector.
      //   s:  The index of the state model whose state component is desired.
      //
      // Returns:
      //   The subset of the 'state' argument corresponding to state model 's'.
      VectorView state_component(Vector &state, int s) const;
      VectorView state_component(VectorView &state, int s) const;
      ConstVectorView state_component(
          const ConstVectorView &state, int s) const;

      //----------------------------------------------------------------------
      // The subset of the vector of state errors (or innovations) corresponding
      // to state model s.
      //
      // Args:
      //   full_state_error: The error for the full state vector (i.e. all state
      //     models).
      //   state_model_number:  The index of the desired state model.
      //
      // Returns:
      //   The subset of 'full_state_error' for the specified state model.
      ConstVectorView const_state_error_component(
          const Vector &full_state_error, int state_model_number) const;
      VectorView state_error_component(
          Vector &full_state_error, int state_model_number) const;

      //----------------------------------------------------------------------
      // The subcomponent of the block-diagonal error variance matrix
      // corresponding to a specific state model.
      //
      // Args:
      //   full_error_variance:  The full state error variance matrix.
      //   state_model_index: The index of the desired state model.
      //
      // Returns:
      //   The diagonal block of full_error_variance corresponding to
      //   state_model_index.
      ConstSubMatrix state_error_variance_component(
          const SpdMatrix &full_error_variance, int state_model_index) const;

      //----------------------------------------------------------------------
      // The complete state vector (across time, so the return value is a
      // matrix) for a specified state component.
      //
      // Args:
      //   state: The state matrix (rows are state variables, columns are time) to
      //     subset.
      //   state_model_index:  The index of the desired state model.
      //
      // Returns:
      //   A matrix containing the imputed value of the state vector for the
      //   specified state model.  The matrix has S rows and T columns, where S
      //   is the dimension of the state vector for the specified state model,
      //   and T is the number of time points.
      ConstSubMatrix full_state_subcomponent(
          const Matrix &state, int state_model_index) const;
      SubMatrix mutable_full_state_subcomponent(
          Matrix &state, int state_model_index) const;

      // Structural matrices for Kalman filtering.
      const SparseKalmanMatrix *state_transition_matrix(int t) const;
      const SparseKalmanMatrix *state_variance_matrix(int t) const;
      const ErrorExpanderMatrix *state_error_expander(int t) const;
      const SparseKalmanMatrix *state_error_variance(int t) const;

     protected:
      // Child classes should call this method when implementing add_state.
      void add_state_model(Ptr<StateModelBase> state_model);

      // Child classes should call this method when implementing clear().
      // Clears the vector of state model pointers, and resets all metadata
      // accordingly.
      void clear_state_models();

     private:
      std::vector<Ptr<StateModelBase>> state_models_;

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

      // Model matrices for Kalman filtering.
      mutable std::unique_ptr<BlockDiagonalMatrix> state_transition_matrix_;
      mutable std::unique_ptr<BlockDiagonalMatrix> state_variance_matrix_;
      mutable std::unique_ptr<ErrorExpanderMatrix> state_error_expander_;
      mutable std::unique_ptr<BlockDiagonalMatrix> state_error_variance_;
    };

    // Concrete StateModelVector objects are parameterized by the type of the
    // state model they store.
    template <class STATE_MODEL>
    class StateModelVector : public StateModelVectorBase{
     public:
      void add_state(Ptr<STATE_MODEL> state_model) {
        add_state_model(state_model);
        state_models_.push_back(state_model);
      }

      // Clear the vector of models and restore the state of the object to that
      // produced by the default constructor.
      void clear() override {
        state_models_.clear();
        clear_state_models();
      }

      Ptr<STATE_MODEL> operator[](int s) {return state_models_[s];}
      const Ptr<STATE_MODEL> operator[](int s) const {return state_models_[s];}

     private:
      std::vector<Ptr<STATE_MODEL>> state_models_;
    };

  }  // namespace StateSpaceUtils
}  // namespace BOOM

#endif  // BOOM_STATE_SPACE_STATE_MODEL_VECTOR_HPP_
