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
 *  Copyright (C) 1998 Ross Ihaka
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
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 *  SYNOPSIS
 *
 *    #include "Bmath.hpp"
 *    double choose(double n, double k);
 *    double lchoose(double n, double k);
 *    (and private)
 *    double fastchoose(double n, double k);
 *    double lfastchoose(double n, double k);
 *
 *  DESCRIPTION
 *
 *    Binomial coefficients.
 */
#include "nmath.hpp"
namespace Rmath{

double lfastchoose(double n, double k)
{
        return lgammafn(n + 1.0) - lgammafn(k + 1.0) - lgammafn(n - k + 1.0);
}

double fastchoose(double n, double k)
{
        return exp(lfastchoose(n, k));
}

double lchoose(double n, double k)
{
        n = FLOOR(n + 0.5);
        k = FLOOR(k + 0.5);
#ifdef IEEE_754
        /* NaNs propagated correctly */
        if(ISNAN(n) || ISNAN(k)) return n + k;
#endif
        if (k < 0 || n < k) ML_ERR_return_NAN;

        return lfastchoose(n, k);
}

double choose(double n, double k)
{
        n = FLOOR(n + 0.5);
        k = FLOOR(k + 0.5);
#ifdef IEEE_754
        /* NaNs propagated correctly */
        if(ISNAN(n) || ISNAN(k)) return n + k;
#endif
        if (k < 0 || n < k) ML_ERR_return_NAN;

        return FLOOR(exp(lfastchoose(n, k)) + 0.5);
}
}

