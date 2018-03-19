// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2008 Steven L. Scott

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

#ifndef BOOM_HMM_DATA_IMPUTER_HPP
#define BOOM_HMM_DATA_IMPUTER_HPP

#include "Models/HMM/HMM2.hpp"
#include "Models/HMM/HmmFilter.hpp"
#include "Models/MarkovModel.hpp"
#include "Models/ModelTypes.hpp"
#include "Models/TimeSeries/TimeSeries.hpp"
#include "distributions/rng.hpp"

namespace BOOM {

  class HmmDataImputer : private RefCounted {
   public:
    HmmDataImputer(HiddenMarkovModel *hmm, uint id, uint nworkers);
    Ptr<MarkovModel> mark();
    Ptr<MixtureComponent> models(uint s);
    double loglike() const;

    void setup(HiddenMarkovModel *);
    void clear_client_data();
    void impute_data();

    friend void intrusive_ptr_add_ref(HmmDataImputer *d) { d->up_count(); }
    friend void intrusive_ptr_release(HmmDataImputer *d) {
      d->down_count();
      if (d->ref_count() == 0) delete d;
    }

   private:
    uint id_;
    uint nworkers_;
    Ptr<MarkovModel> mark_;
    std::vector<Ptr<MixtureComponent>> mix_;
    Ptr<HmmFilter> filter_;
    double loglike_;
    std::vector<TimeSeries<Data> *> dat_;

    RNG eng;
  };

}  // namespace BOOM

#endif  // BOOM_HMM_DATA_IMPUTER_HPP
