#ifndef BOOM_STATE_SPACE_STATE_MODEL_POSTERIOR_SAMPLER_HPP_
#define BOOM_STATE_SPACE_STATE_MODEL_POSTERIOR_SAMPLER_HPP_
/*
  Copyright (C) 2018 Steven L. Scott

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

namespace {

  // Some state models require constraints on their parameters (or on the state)
  // in order to maintain identifiability.  A StateModelPosteriorSampler is just
  // a PosteriorSampler that offers an impose_identifiability_constriaint()
  // member.

  class StateModelPosteriorSampler : public PosteriorSampler {
   public:
    // The default is a no-op.
    // Args:
    //   state: The portion of the model state associated with this state model.
    //
    // Effects:
    //   The state is modified so that it satisfies the constraint.  Model
    //   parameters for the associated state model may also be modified.
    virtual void impose_identifiability_constriaint(MatrixView &state) {}
  };

}  // namespace BOOM


#endif //  BOOM_STATE_SPACE_STATE_MODEL_POSTERIOR_SAMPLER_HPP_



