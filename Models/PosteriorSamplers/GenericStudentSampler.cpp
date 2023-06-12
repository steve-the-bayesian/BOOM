/*
  Copyright (C) 2005-2022 Steven L. Scott

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

#include "Models/PosteriorSamplers/GenericStudentSampler.hpp"
#include "distributions.hpp"

namespace BOOM {

  using ZMSLL = ZeroMeanStudentLogLikelihood;

  double ZeroMeanStudentLogLikelihood::operator()(
      const Vector &x, Vector *g, Matrix *h, bool reset_derivatives) const {

    double ans = 0;
    double sigma = x[0];
    double nu = x[1];
    if (reset_derivatives) {
      if (g) {
        *g = 0.0;
        if (h) {
          *h = 0.0;
        }
      }
    }

    long n = residuals_.size();
    double halfn = n / 2.0;
    double half_nu_plus1 = (nu + 1.0) / 2.0;
    double sigsq = sigma * sigma;
    if (g) {
      // Set the terms for the gradient (and possibly hessian) that don't depend
      // on the individual data values.
      (*g)[0] -= n / sigma;
      (*g)[1] += halfn * digamma(half_nu_plus1) - halfn * digamma(nu / 2.0) - halfn / nu;

      if (h) {
        (*h)(0, 0) += n / sigsq;
        (*h)(1, 1) +=  .5 * halfn * trigamma(half_nu_plus1)
            - .5 * halfn*trigamma(nu / 2.0)
            + halfn / square(nu);
      }
    }

    double g0_coefficient = (nu + 1) / (sigma);
    for (long i = 0; i < n; ++i) {
      ans += dstudent(residuals_[i], 0, sigma, nu, true);

      if (g) {
        // Now add in the portion of the gradient (and perhaps hessian) that
        // depends on the individual data values.
        double rsq = residuals_[i] * residuals_[i];
        double sumsq = nu * sigsq + rsq;
        double ratio = rsq / sumsq;

        (*g)[0] += g0_coefficient * ratio;
        (*g)[1] += (half_nu_plus1 / nu) * ratio;
        (*g)[1] -= .5 * log1p(rsq / (sigsq * nu));

        if (h) {
          (*h)(0, 0) -= (nu + 1) * ratio / sigsq;
          (*h)(0, 0) -= 2 * nu * (nu + 1) * ratio / sumsq;

          (*h)(0, 1) += ratio / sigma;
          (*h)(0, 1) -= (nu + 1) * ratio * sigsq / sumsq;

          (*h)(1, 1) -= .5 * ratio / square(nu);
          (*h)(1, 1) -= sigsq * (nu + 1) * ratio / (sumsq * 2 * nu);
          (*h)(1, 1) += .5 * ratio / nu;
        }
      }
    }

    if (h) {
      (*h)(1, 0) = (*h)(0, 1);
    }
    return ans;
  }

  /*
    LaTeX describing the derivatives.  Might need to be double checked.

\documentclass{article}
\usepackage{amsmath}

\begin{document}

\begin{equation*}
  \ell = n \log \Gamma\left(
    \frac{\nu + 1}{2}\right)
  - n \log \Gamma\left(\frac{\nu}{2} \right)
  - \frac{n}{2}\log \nu
  - n\log \sigma
  - \frac{\nu + 1}{2} \sum_i \log\left( 1 + \frac{x_i^2}{\sigma^2 \nu}\right)
\end{equation*}

\begin{equation*}
  \frac{\partial \ell}{\partial \sigma} =
  - \frac{n}{\sigma}
  + \frac{\nu + 1}{\sigma} \sum_i \frac{x_i^2}{(\sigma^2 \nu + x_i^2)}
\end{equation*}

\begin{equation*}
  \frac{\partial \ell}{\partial \nu} =
  \frac{n}{2} \Psi\left(\frac{\nu + 1}{2}\right)
  - \frac{n}{2} \Psi\left(\frac{\nu}{2} \right)
  - \frac{n}{2\nu}
  + \frac{\nu + 1}{2\nu} \sum_i \frac{x_i^2}{\nu\sigma^2 + x_i^2}
  - \frac{1}{2}\sum_i\log(1 + \frac{x_i^2}{\nu \sigma^2})
\end{equation*}

\begin{equation*}
  \frac{\partial^2 \ell}{\partial \sigma^2} =
  \frac{n}{\sigma^2} -2 \nu(\nu + 1) \sum_i \frac{x_i^2}{(\nu \sigma^2 + x_i^2)^2}
  - \frac{\nu + 1}{\sigma^2} \sum_i \frac{x_i^2}{\nu \sigma^2 + x_i^2}
\end{equation*}

\begin{equation*}
  \frac{\partial^2 \ell}{\partial \sigma \partial \nu} =
   -\frac{1}{2} \sum_i \frac{x_i^2}{\sigma^2 + x_i^2}
\end{equation*}

\begin{equation*}
  \frac{\partial^2\ell}{\partial \nu^2} =
  \frac{n}{4} \Psi'\left(\frac{\nu + 1}{2}\right)
  -\frac{n}{4} \Psi'\left(\frac{\nu}{2}\right)
  + \frac{n}{2\nu^2}
  - \frac{1}{2\nu^2} \sum_i \frac{x_i^2}{\nu\sigma^2 + x_i^2}
  - \frac{\sigma^2(\nu + 1)}{2\nu} \sum_i \frac{x_i^2}{(\nu\sigma^2 + x_i^2)^2}
  + \frac{1}{2\nu} \sum_i \frac{x_i^2}{\nu\sigma^2 + x_i^2}
\end{equation*}

\end{document}

   */


}  // namespace BOOM
