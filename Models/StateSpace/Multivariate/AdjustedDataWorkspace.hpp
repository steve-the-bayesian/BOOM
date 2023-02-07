#ifndef BOOM_STATE_SPACE_MULTIVARIATE_ADJUSTED_DATA_WORKSPACE_HPP_
#define BOOM_STATE_SPACE_MULTIVARIATE_ADJUSTED_DATA_WORKSPACE_HPP_

/*
  Copyright (C) 2005-2022 Steven L. Scott

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

namespace BOOM {

  namespace StateSpaceUtilities {

    class AdjustedDataWorkspace {
     public:

      AdjustedDataWorkspace()
          : adjusted_data_workspace_(),
            workspace_current_(false),
            workspace_time_index_(-1),
            workspace_status_(WorkspaceStatus::UNSET)
      {}

      template <class DATA_POLICY, class STATE_MANAGER, class OBSERVATION_MODEL>
      void isolate_shared_state(int time,
                                const DATA_POLICY &data_policy,
                                const STATE_MANAGER &state_manager,
                                const OBSERVATION_MODEL *observation_model) {
        if (workspace_current_
            && workspace_time_index_ == time
            && workspace_status_ == ISOLATE_SHARED_STATE) {
          return;
        }
        const Selector &observed(data_policy.observed(time));
        adjusted_data_workspace_.resize(observed.nvars());
        for (int series = 0; series < data_policy.nseries(); ++series) {
          if (observed[series]) {
            int s = observed.dense_index(series);
            Ptr<typename DATA_POLICY::DataType> data_point =
                data_policy.data_point(series, time);
            adjusted_data_workspace_[s] = data_point->y()
                - state_manager.series_specific_state_contribution(
                    series, time);

            double regression_contribution = observation_model->model(
                series)->predict(data_point->x());
            adjusted_data_workspace_[s] -= regression_contribution;
          }
        }
        workspace_current_ = true;
        workspace_time_index_ = time;
        workspace_status_ = ISOLATE_SHARED_STATE;
      }

      template <class DATA_POLICY, class STATE_MANAGER, class OBSERVATION_MODEL>
      void isolate_series_specific_state(
          int time,
          const DATA_POLICY &data_policy,
          const STATE_MANAGER &state_manager,
          const OBSERVATION_MODEL *observation_model,
          const SparseKalmanMatrix &observation_coefficients,
          const Matrix &shared_state) {
        if (workspace_status_ == ISOLATE_SERIES_SPECIFIC_STATE
            && workspace_time_index_ == time
            && workspace_current_) {
          return;
        }
        const Selector &observed(data_policy.observed(time));
        adjusted_data_workspace_.resize(observed.nvars());
        Vector shared_state_contribution =
            observation_coefficients * shared_state.col(time);

        for (int s = 0; s < observed.nvars(); ++s) {
          int series = observed.sparse_index(s);
          const typename DATA_POLICY::DataType *data_point =
              data_policy.data_point(series, time).get();
          const Vector &predictors(data_point->x());
          adjusted_data_workspace_[s] = data_point->y()
              - shared_state_contribution[s]
              - observation_model->model(series)->predict(predictors);
        }
        workspace_current_ = true;
        workspace_time_index_ = time;
        workspace_status_ = ISOLATE_SERIES_SPECIFIC_STATE;
      }

      template <class DATA_POLICY, class STATE_MANAGER, class OBSERVATION_MODEL>
      void isolate_state(int time,
                         const DATA_POLICY &data_policy,
                         const STATE_MANAGER &state_manager,
                         const OBSERVATION_MODEL *observation_model,
                         const SparseKalmanMatrix &observation_coefficients,
                         const Matrix &shared_state) {
        if (workspace_status_ == ISOLATE_SHARED_STATE) {
          isolate_shared_state(
              time, data_policy, state_manager, observation_model);
        } else if (workspace_status_ == ISOLATE_SERIES_SPECIFIC_STATE) {
          isolate_series_specific_state(
              time, data_policy, state_manager, observation_model,
              observation_coefficients, shared_state);
        } else {
          report_error("The workspace_status_ flag must be set before "
                       "calling isolate_state.");
        }
      }

      const Vector & adjusted_data_workspace() const {
        return adjusted_data_workspace_;
      }

      void set_observers(std::vector<Ptr<Params>> &params) {
        for (auto &el : params) {
          el->add_observer(
              this,
              [this]() {this->workspace_current_ = false;});
        }
      }

      void isolate_shared_state() {
        workspace_status_ = ISOLATE_SHARED_STATE;
      }

      void isolate_series_specific_state() {
        workspace_status_ = ISOLATE_SERIES_SPECIFIC_STATE;
      }

      void unset() {
        workspace_status_ = UNSET;
      }

     private:
      // A workspace where observed data can be modified by subtracting off
      // components on which we wish to condition.
      //
      // The point of having this workspace, as opposed to simply making the
      // adjustments on demand, is that the adjustments are best performed on a
      // time-by-time basis, but the model also supports a (series, time)
      // interface.  To make the latter more efficient we make adjustments on the
      // time basis, store the results, and then look up the (series, time)
      // answer.
      Vector adjusted_data_workspace_;

      // Metadata about the adjusted_data_workspace_.
      //
      // A flag indicating that the workspace holds current values.  This is set
      // to false whenever new parameters are assigned or new state is drawn.
      bool workspace_current_;

      // The time index that the workspace currently describes.
      int workspace_time_index_;

      // The type of information currently stored in the workspace.
      enum WorkspaceStatus {
        UNSET,
        ISOLATE_SHARED_STATE,
        ISOLATE_SERIES_SPECIFIC_STATE,
        ISOLATE_REGRESSION_EFFECTS
      };
      WorkspaceStatus workspace_status_;
    };

  } // namespace StateSpaceUtilities
}  // namespace BOOM

#endif  //  BOOM_STATE_SPACE_MULTIVARIATE_ADJUSTED_DATA_WORKSPACE_HPP_
