#ifndef BOOM_STATE_SPACE_MULTIVARIATE_STATE_MODEL_MANAGER_HPP_
#define BOOM_STATE_SPACE_MULTIVARIATE_STATE_MODEL_MANAGER_HPP_
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

#include "Models/StateSpace/StateModelVector.hpp"

namespace BOOM {
  namespace StateSpaceUtils {

    // A MultivariateStateModelManager manages the SharedStateModels and the
    // ProxyModels for a multivariate state space regression.  There are several
    // such regression models with different types of observation models and
    // different types of raw data.  This class handles the common tasks of
    // trading between shared state and series specific state.
    //
    // Args:
    //   PROXY:  A specific instance of a ProxyScalarStateSpaceModel.
    template <class PROXY>
    class SharedStateModelManager {
     public:

      void add_series_specific_state(const Ptr<StateModel> &state_model,
                                     int series) {
        proxy_models_[series]->add_state(state_model);
      }

      void add_shared_state(const Ptr<SharedStateModel> &state_model) {
        shared_state_models_.add_state(state_model);
      }

      bool has_series_specific_state() const {
        for (int i = 0; i < proxy_models_.size(); ++i) {
          if (proxy_models_[i]->state_dimension() > 0) {
            return true;
          }
        }
        return false;
      }

      PROXY *series_specific_model(int index) {
        return proxy_models_[index].get();
      }

      const PROXY *series_specific_model(int index) const {
        return proxy_models_[index].get();
      }

      // The series-specific dimension of
      int series_state_dimension(int series) const {
        return proxy_models_.empty() ? 0 :
            proxy_models_[series]->state_dimension();
      }

      void observe_time_dimension(int t) {
        for (int s = 0; s < number_of_shared_state_models(); ++s) {
          shared_state_model(s)->observe_time_dimension(t);
        }
        for (int m = 0; m < proxy_models_.size(); ++m) {
          if (!!proxy_models_[m]) {
            proxy_models_[m]->observe_time_dimension(t);
          }
        }
      }

      StateSpaceUtils::StateModelVector<SharedStateModel>
      &shared_state_models() { return shared_state_models_; }

      const StateSpaceUtils::StateModelVector<SharedStateModel>
      &shared_state_models() const { return shared_state_models_; }

      int shared_state_dimension() const {
        return shared_state_models_.state_dimension();
      }

      int number_of_shared_state_models() const {
        return shared_state_models_.size();
      }

      SharedStateModel *shared_state_model(int s) {
        if (s < 0 || s >= shared_state_models_.size()) {
          return nullptr;
        } else {
          return shared_state_models_[s].get();
        }
      }

      const SharedStateModel *shared_state_model(int s) const {
        if (s < 0 || s >= shared_state_models_.size()) {
          return nullptr;
        } else {
          return shared_state_models_[s].get();
        }
      }

      template <class HOST>
      void initialize_proxy_models(HOST *host) {
        proxy_models_.clear();
        proxy_models_.reserve(host->nseries());
        for (int i = 0; i < host->nseries(); ++i) {
          proxy_models_.push_back(new PROXY(host, i));
        }
      }

     private:
      StateModelVector<SharedStateModel> shared_state_models_;
      std::vector<Ptr<PROXY>> proxy_models_;

    };

  }  // namespace StateSpaceUtils
}  // namespace BOOM`

#endif  //  BOOM_STATE_SPACE_MULTIVARIATE_STATE_MODEL_MANAGER_HPP_
