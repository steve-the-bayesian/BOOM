#ifndef BOOM_GLM_CORRELATION_MAP_HPP_
#define BOOM_GLM_CORRELATION_MAP_HPP_
/*
  Copyright (C) 2005-2018 Steven L. Scott

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

#include "Models/Glm/RegressionModel.hpp"
#include <map>
#include "LinAlg/Vector.hpp"
#include "LinAlg/Selector.hpp"

namespace BOOM {

  // A class to keep track of all the 'large' correlations between predictor
  // variables in a regresion model.  A coefficient is large if its absolute
  // value exceeds a threshold defined in the constructor.
  class CorrelationMap {
   public:
    // Build an empty CorrelationMap to be filled in later.
    // Args:
    //   threshold: The minimal absolute value required for a correlation to be
    //     included in the map.
    explicit CorrelationMap(double threshold = .8);

    // Build a CorrelationMap to be filled by the sufficient statistics from a
    // regression model.
    //
    // Args:
    //   threshold: The minimal absolute value required for a correlation to be
    //     included in the map.
    //   suf: The sufficient statistics from a regression model.  Correlations
    //     are obtained from the X'X matrix in the sufficient statistics object.
    CorrelationMap(double threshold, const RegSuf &suf);

    // Recompute the set of large correlations.
    void fill(const RegSuf &suf);
    bool filled() const {return filled_;}

    // Resets the threshold to a new value and clears the map, which must be
    // filled again using fill().
    void set_threshold(double threshold);
    double threshold() const {return threshold_;}

    // Args:
    //   rng: The random number generator to use for the proposal.
    //   included: A set of indicators describing which variables are currently
    //     in the model.
    //   index:  The index of one of the included variables.
    //   proposal_weight: This is an output-only parameter.  It will be filled
    //     with the probability of the returned value.
    //
    // Returns:
    //   The index of a swap candidate to be paired with 'index'.  If no
    //   variables are strongly correlated to 'index' then -1 is returned and
    //   negative infinity is assigned to the proposal weight.
    int propose_swap(RNG &rng, const Selector &included, int index,
                     double *proposal_weight) const;

    // Compute the probability of proposing 'candidate' as a swap partner to
    // 'index' if the current model is given by 'included'.
    double proposal_weight(const Selector &included, int index,
                           int candidate) const;

   private:
    double threshold_;
    bool filled_;

    // The first entry in the map indexes a column in a predictor matrix.  The
    // second entry is a table listing the indices of all the columns whose
    // absolute correlation with the first column exceed a threshold.
    std::map<int, std::pair<std::vector<int>, Vector>> correlations_;
  };
  
}  // namespace BOOM

#endif  // BOOM_GLM_CORRELATION_MAP_HPP_
