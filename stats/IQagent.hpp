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
#include <stats/ECDF.hpp>
#include <uint.hpp>

namespace BOOM{
  class IQagent{
    typedef std::vector<double> VEC;
  public:
    IQagent(uint BufSize=20);
    IQagent(const VEC & probs, uint BufSize=20);
    void add(double x);
    double quantile(double prob)const;
    double cdf(double x)const;
    void update_cdf();
  private:
    void flush();
    void set_default_probs();

    double Fq(double x)const;
    double F(double x, bool plus)const;
    double find_xplus(double p)const;
    double find_xminus(double p)const;

    uint max_buffer_size_, nobs_;
    VEC data_buffer_;
    VEC probs_;
    VEC quantiles_;

    ECDF ecdf_;
    VEC Fplus_;
    VEC Fminus_;

  };

}
#endif// BOOM_IQ_AGENT_HPP
