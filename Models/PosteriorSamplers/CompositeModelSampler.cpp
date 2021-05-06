// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2010 Steven L. Scott

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
#include "Models/PosteriorSamplers/CompositeModelSampler.hpp"

namespace BOOM {
  typedef CompositeModel CM;
  typedef CompositeModelSampler CMS;

  CMS::CompositeModelSampler(CM *model, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng), m_(model) {}

  CMS *CMS::clone_to_new_host(Model *new_host) const {
    return new CMS(dynamic_cast<CompositeModel *>(new_host),
                   rng());
  }

  double CMS::logpri() const {
    const std::vector<Ptr<MixtureComponent> > &components(m_->components());
    double ans = 0;
    for (int i = 0; i < components.size(); ++i) {
      ans += components[i]->logpri();
    }
    return ans;
  }

  void CMS::draw() {
    std::vector<Ptr<MixtureComponent> > &components(m_->components());
    for (int i = 0; i < components.size(); ++i) {
      components[i]->sample_posterior();
    }
  }
}  // namespace BOOM
