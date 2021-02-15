/*
  Copyright (C) 2005 Steven L. Scott

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

/*
 *  Mathlib : A C Library of Special Functions
 *  Copyright (C) 2003        The R Foundation
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2, or (at your option)
 *  any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  A copy of the GNU General Public License is available via WWW at
 *  http://www.gnu.org/copyleft/gpl.html.  You can also obtain it by
 *  writing to the Free Software Foundation, Inc., 59 Temple Place,
 *  Suite 330, Boston, MA  02111-1307  USA.
 *
 *
 *  SYNOPSIS
 *
 *      #include "Bmath.hpp"
 *      void rmultinom(int n, double* prob, int K, int* rN);
 *
 *  DESCRIPTION
 *
 *      Random Vector from the multinomial distribution.
 *             ~~~~~~
 *  NOTE
 *      Because we generate random _vectors_ this doesn't fit easily
 *      into the do_random[1-4](.) framework setup in ../main/random.c
 *      as that is used only for the univariate random generators.
 *      Multivariate distributions typically have too complex parameter spaces
 *      to be treated uniformly.
 *      => Hence also can have  int arguments.
 */

#include "Bmath/Bmath.hpp"
#include "cpputil/report_error.hpp"

#include <vector>
#include <stdexcept>
#include <sstream>
#include <cstdlib>

namespace Rmath{
  std::vector<int> rmultinom_mt(RNG &rng, int n, const std::vector<double> &prob){
    std::vector<int> result;
    rmultinom_mt(rng, n, prob, result);
    return result;
  }

  void rmultinom(int n, const std::vector<double> &prob, std::vector<int> &result){
    rmultinom_mt(BOOM::GlobalRng::rng, n, prob, result);
  }

  std::vector<int> rmultinom(int n, const std::vector<double> &prob){
    std::vector<int> result;
    rmultinom_mt(BOOM::GlobalRng::rng, n, prob, result);
    return result;
  }

  void rmultinom_mt(BOOM::RNG & rng,
                    int n,
                    const std::vector<double> & prob,
                    std::vector<int> &rN){
    /* `Return' vector  rN[1:K] {K := length(prob)}
     *  where rN[j] ~ Bin(n, prob[j]) ,  sum_j rN[j] == n,  sum_j prob[j] == 1,
     */

    int K = prob.size();
    if(rN.size()!=K) rN.resize(K);
    if(K < 1){
      BOOM::report_error("empty argument 'prob' in rmultinom_mt");
    }

    int k;
    double pp, p_tot = 0.;

    /* Note: prob[K] is only used here for checking  sum_k prob[k] = 1 ;
     *       Could make loop one shorter and drop that check !
     */
    for(k = 0; k < K; k++) {
      pp = prob[k];
      if (!std::isfinite(pp) || pp < 0. || pp > 1.){
        std::ostringstream err;
        err << "rmultinom:  element " << k
            << " (counting from 0) of 'prob' is illegal."
            << std::endl << "prob =";
        for(int m = 0; m < K; ++m){
          err << " " << prob[m];
        }
        err << std::endl;
        BOOM::report_error(err.str());
      }
      p_tot += pp;
      rN[k] = 0;
    }
    if(fabs(p_tot - 1.) > 1e-7){
      std::ostringstream err;
      err << "rmultinom: probability sum should be 1, but is " << p_tot << std::endl;
      BOOM::report_error(err.str());
    }
    if (n == 0) return;
    if (K == 1 && p_tot == 0.) return;/* trivial border case: do as rbinom */

    /* Generate the first K-1 obs. via binomials */

    for(k = 0; k < K-1; k++) { /* (p_tot, n) are for "remaining binomial" */
      pp = prob[k] / p_tot;
      rN[k] = rbinom_mt(rng, n,  pp);
      n -= rN[k];
      if(n <= 0) /* we have all*/ return;
      p_tot -= prob[k]; /* i.e. = sum(prob[(k+1):K]) */
    }
    rN[K-1] = n;
  }
}  // namespace Rmath
