// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2007 Steven L. Scott

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

#ifndef BOOM_IQ_AGENT_HPP
#define BOOM_IQ_AGENT_HPP
// Implementation of the incremental quantile estimator from Chambers
// et. al. in Stat Science 2006, pp 463-475.
//

#include <vector>
#include "LinAlg/Vector.hpp"
#include "stats/ECDF.hpp"
#include "uint.hpp"

namespace BOOM {

  struct IqAgentState {
    uint max_buffer_size;
    uint nobs;
    Vector data_buffer;
    Vector probs;
    Vector quantiles;
    Vector ecdf_sorted_data;
    Vector fplus;
    Vector fminus;
  };

  // An approximate empirical distribution of continuous numeric data.
  class IQagent {
   public:
    // Args:
    //   bufsize: The number of data points to store before triggering a CDF
    //     refresh.
    explicit IQagent(uint BufSize = 20);

    // Args:
    //   probs:  The focal quantiles of the empirical distribution.
    //   bufsize: The number of data points to store before triggering a CDF
    //     refresh.
    explicit IQagent(const Vector& probs, uint BufSize = 20);

    // Args:
    //   state:  The serialized state of a previously fit IQagent.
    explicit IQagent(const IqAgentState &state);

    // Add a data point to the empirical distribution.
    void add(double x);

    // Add a collection of data points to the empirical distribution.
    void add(const Vector &x);

    // Return the approximate quantile associated with the given probability.
    double quantile(double prob) const;

    // Return the approximate fraction of data less than the supplied x value.
    double cdf(double x) const;

    // Incorporate any buffered data into the approximate CDF, emptying the
    // buffer in the process.
    void update_cdf();

    IqAgentState save_state() const;
    void restore_from_state(const IqAgentState &state);

   private:
    void set_default_probs();

    double Fq(double x) const;
    double F(double x, bool plus) const;
    double find_xplus(double p) const;
    double find_xminus(double p) const;

    uint max_buffer_size_, nobs_;
    Vector data_buffer_;
    Vector probs_;
    Vector quantiles_;

    ECDF ecdf_;
    Vector Fplus_;
    Vector Fminus_;
  };

}  // namespace BOOM

#endif  // BOOM_IQ_AGENT_HPP
