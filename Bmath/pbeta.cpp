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
 *
 *  SYNOPSIS
 *
 * #include "Bmath.hpp"
 *
 * double pbeta_raw(double x, double pin, double qin, int lower_tail)
 * double pbeta    (double x, double pin, double qin, int lower_tail, int log_p)
 *
 *  DESCRIPTION
 *
 *      Returns distribution function of the beta distribution.
 *      ( = The incomplete beta ratio I_x(p,q) ).
 *
 *  NOTES
 *
 *      This routine is a translation into C of a Fortran subroutine
 *      by W. Fullerton of Los Alamos Scientific Laboratory.
 *
 *  REFERENCE
 *
 *      Bosten and Battiste (1974).
 *      Remark on Algorithm 179, CACM 17, p153, (1974).
 */

#include "nmath.hpp"
#include "dpq.hpp"

namespace Rmath{

/* This is called from  qbeta(.) in a root-finding loop --- be FAST! */

double pbeta_raw(double x, double pin, double qin, int lower_tail, int log_p)
{
    double x1 = 0.5 - x + 0.5, w, wc;
    int ierr;
    bratio(pin, qin, x, x1, &w, &wc, &ierr, log_p); /* -> ./toms708.c */
    /* ierr = 8 is about inaccuracy in extreme cases */
    if(ierr && (ierr != 8 || log_p) ) {
      std::ostringstream err;
      err << "pbeta_raw() -> bratio() gave error code " << ierr << ".";
      report_error(err.str());
    }
    return lower_tail ? w : wc;
} /* pbeta_raw() */

double pbeta(double x, double pin, double qin, int lower_tail, int log_p)
{
  if (isnan(x) || isnan(pin) || isnan(qin)) return x + pin + qin;

  if (pin <= 0 || qin <= 0) {
    report_error("arguments to pbeta/qbeta must be > 0");
  }

  if (x <= 0)
    return R_DT_0;
  if (x >= 1)
    return R_DT_1;
  return pbeta_raw(x, pin, qin, lower_tail, log_p);
}
}
