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
 *  Copyright (C) 1998-2000  The R Development Core Team
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
 */

/* Private header file for use during compilation of Mathlib */
#ifndef MATHLIB_PRIVATE_H
#define MATHLIB_PRIVATE_H

#define MATHLIB_STANDALONE

#include <cmath>
#include <cfloat>
#include <cerrno>
#include <cstdio>

#include <limits>
#include <climits>
#include <stdexcept>
#include "Bmath/Bmath.hpp"
#include "cpputil/math_utils.hpp"

// TODO(stevescott): Once all CRAN platforms support the thread_local
// keyword remove this macro.
#if defined(NO_BOOST_THREADS)
#define PLATFORM_THREAD_LOCAL
#else
#define PLATFORM_THREAD_LOCAL thread_local
#endif

namespace Rmath{
  using std::isnan;
  typedef bool Rboolean;

  void mathlib_error(const std::string &s);
  void mathlib_error(const std::string &s, int d);
  void mathlib_error(const std::string &s, double d);

  const double ML_POSINF = std::numeric_limits<double>::infinity();
  const double ML_NEGINF = -1*std::numeric_limits<double>::infinity();

  inline bool ISNAN(double x){return std::isnan(x);}
  int R_IsNaNorNA(double);
  inline bool R_finite(double x){
    if(std::isnan(x)) return false;
    if(x==ML_POSINF || x== ML_NEGINF) return false;
    return true;
  }
  inline bool R_FINITE(double x){return R_finite(x);}

#ifdef IEEE_754
#define ML_ERROR(x)     /* nothing */
#define ML_UNDERFLOW    (DBL_MIN * DBL_MIN)
#define ML_VALID(x)     (!ISNAN(x))
#else/*--- NO IEEE: No +/-Inf, NAN,... ---*/
  void ml_error(int n);
#define ML_ERROR(x)     ml_error(x)
#define ML_UNDERFLOW    0
#define ML_VALID(x)     (errno == 0)
#endif

  const int ME_NONE       = 0;
  const int ME_DOMAIN     = 1;
  const int ME_RANGE      = 2;
  const int ME_NOCONV     = 4;
  const int ME_PRECISION  = 8;
  const int ME_UNDERFLOW  =16;

#define ML_ERR_return_NAN { ML_ERROR(ME_DOMAIN); return std::numeric_limits<double>::quiet_NaN(); }

  /* Wilcoxon Rank Sum Distribution */
  const int WILCOX_MAX= 50;

  /* Wilcoxon Signed Rank Distribution */

  const int SIGNRANK_MAX= 50;

  /* internal R functions */
  /* Chebyshev Series */
  int   chebyshev_init(double*, int, double);
  double        chebyshev_eval(double, const double *, const int);

  /* Gamma and Related Functions */

  void  gammalims(double*, double*);
  double        lgammacor(double); /* log(gamma) correction */
  double  stirlerr(double);  /* Stirling expansion "error" */

  double        fastchoose(double, double);
  double        lfastchoose(double, double);

  double  bd0(double, double);

  /* Consider adding these two to the API (Bmath.hpp): */
  double        dbinom_raw(double, double, double, double, int);
  double        dpois_raw (double, double, int);
  double        pnchisq_raw(double, double, double, double, int);

  int   i1mach(int);

  inline long FLOOR(double x){
    return static_cast<long>(std::floor(x));}

  void          bratio(double a, double b, double x, double y,
                       double *w, double *w1, int *ierr, int log_p);
}

#endif /* MATHLIB_PRIVATE_H */
