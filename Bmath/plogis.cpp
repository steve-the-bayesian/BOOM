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
 *  Copyright (C) 1995, 1996    Robert Gentleman and Ross Ihaka
 *  Copyright (C) 2000          The R Development Core Team
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
#include "nmath.hpp"
#include "dpq.hpp"

namespace Rmath {

  double plogis(double x, double location, double scale,
                int lower_tail, int log_p)
  {
#ifdef IEEE_754
    if (ISNAN(x) || ISNAN(location) || ISNAN(scale))
      return x + location + scale;
#endif
    if (scale <= 0.0)   ML_ERR_return_NAN;

    x = (x - location) / scale;
    if (ISNAN(x))       ML_ERR_return_NAN;
    if(!R_FINITE(x)) {
      if (x > 0) {
        return R_DT_1;
      } else{
        /* x <= 0 */
        return R_DT_0;
      }
    }
    x = exp(lower_tail ? -x : x);
    return (log_p ? -log1p(x) : 1 / (1 + x));
  }

}  // namespace Rmath
