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
 *  R : A Computer Language for Statistical Data Analysis
 *  Copyright (C) 1995, 1996  Robert Gentleman and Ross Ihaka
 *  Copyright (C) 2000 The R Development Core Team
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA.
 */

/* Reference:
 * R. C. H. Cheng (1978).
 * Generating beta variates with nonintegral shape parameters.
 * Communications of the ACM 21, 317-322.
 * (Algorithms BB and BC)
 */

#include "nmath.hpp"
#include "distributions/rng.hpp"
namespace Rmath{

constexpr double expmax = (std::numeric_limits<double>::max_exponent * M_LN2);

  double rbeta_mt(BOOM::RNG & rng, double aa, double bb);

  double rbeta(double a, double b){
    return rbeta_mt(BOOM::GlobalRng::rng, a,b);
  }

  double rbeta_mt(BOOM::RNG & rng, double aa, double bb){
    double a, b, alpha;
    double r, s, t, u1, u2, v, w, y, z;
    double beta, gamma, delta, k1, k2;

    if (aa <= 0. || bb <= 0. || (!R_FINITE(aa) && !R_FINITE(bb))) {
      std::ostringstream err;
      err << "Illegal parameter values a = " << aa
          << " and b = " << bb << " in call to rbeta.";
      report_error(err.str());
    }

    if (!R_FINITE(aa))
      return 1.0;

    if (!R_FINITE(bb))
      return 0.0;

    /* Test if we need new "initializing" */

    a = std::min(aa, bb);
    b = std::max(aa, bb); /* a <= b */
    alpha = a + b;

#define v_w_from__u1_bet(AA)                    \
    v = beta * log(u1 / (1.0 - u1));            \
    if (v <= expmax)                            \
      w = AA * exp(v);                          \
    else                                        \
      w = std::numeric_limits<double>::max()


    if (a <= 1.0) {     /* --- Algorithm BC --- */

      /* changed notation, now also a <= b (was reversed) */

      beta = 1.0 / a;
      delta = 1.0 + b - a;
      k1 = delta * (0.0138889 + 0.0416667 * a) / (b * beta - 0.777778);
      k2 = 0.25 + (0.5 + 0.25 / delta) * a;
        //      }
      /* FIXME: "do { } while()", but not trivially because of "continue"s:*/
      for(;;) {
        u1 = rng();
        u2 = rng();
        if (u1 < 0.5) {
          y = u1 * u2;
          z = u1 * y;
          if (0.25 * u2 + z - y >= k1)
            continue;
        } else {
          z = u1 * u1 * u2;
          if (z <= 0.25) {
            v_w_from__u1_bet(b);
            break;
          }
          if (z >= k2)
            continue;
        }

        v_w_from__u1_bet(b);

        if (alpha * (log(alpha / (a + w)) + v) - 1.3862944 >= log(z))
          break;
      }
      double ans = (aa == a) ? a / (a + w) : w / (a + w);
      if (std::isnan(ans)) {
        const double zero = std::numeric_limits<double>::epsilon();
        const double one = 1.0 - zero;
        if (aa == a) {
          // return a / (a + w), but be careful because a+w is infinite.
          return std::isfinite(a) ? zero : one;
        } else {
          // return w / (a + w) but be careful because a+w is infinite.
          return std::isfinite(w) ? zero : one;
        }
      } else return ans;

    }
    else {              /* Algorithm BB */
      beta = sqrt((alpha - 2.0) / (2.0 * a * b - alpha));
      gamma = a + 1.0 / beta;
      do {
        u1 = rng();
        u2 = rng();
        v_w_from__u1_bet(a);
        z = u1 * u1 * u2;
        r = gamma * v - 1.3862944;
        s = a + r - w;
        if (s + 2.609438 >= 5.0 * z) {
          break;
        }
        t = log(z);
        if (s > t) {
          break;
        }
      }
      while (r + alpha * log(alpha / (b + w)) < t);
      double ans = (aa != a) ? b / (b + w) : w / (b + w);
      if (std::isnan(ans)) {
        const double zero = std::numeric_limits<double>::epsilon();
        const double one = 1.0 - zero;
        if (aa != a) {
          // return b / b + w, but be careful because b + w is infinite.
          return std::isfinite(b) ? zero : one;
        } else {
          // return w / (b + w), but be careful because b + w is infinite.
          return std::isfinite(w) ? zero : one;
        }
      }
      return ans;
    }
  }
}
