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

#ifndef BOOM_STATE_SPACE_SUFSTAT_MANAGER_HPP_
#define BOOM_STATE_SPACE_SUFSTAT_MANAGER_HPP_

#include <memory>

namespace BOOM {
  namespace StateSpace {

    // A concrete instance of a SufstatManager will typically hold a
    // pointer to a PosteriorSampler that manages the complete data
    // sufficient statistics for a model.
    class SufstatManagerBase {
     public:
      virtual ~SufstatManagerBase() {}

      // Signal that the complete data sufficient statistics should be
      // cleared.
      virtual void clear_complete_data_sufficient_statistics() = 0;

      // Signal that the latent data for observation t has changed,
      // and the observation is ready to be added to the complete data
      // sufficient statistics.
      virtual void update_complete_data_sufficient_statistics(int t) = 0;
    };

    // A container class to a concrete instance of a
    // SufstatManagerBase.
    class SufstatManager {
     public:
      // Assumes ownership of the passed pointer.
      explicit SufstatManager(SufstatManagerBase *impl) : impl_(impl) {}

      void clear_complete_data_sufficient_statistics() {
        impl_->clear_complete_data_sufficient_statistics();
      }

      void update_complete_data_sufficient_statistics(int t) {
        impl_->update_complete_data_sufficient_statistics(t);
      }

     private:
      std::shared_ptr<SufstatManagerBase> impl_;
    };

  }  // namespace StateSpace
}  // namespace BOOM

#endif  //  BOOM_STATE_SPACE_SUFSTAT_MANAGER_HPP_
