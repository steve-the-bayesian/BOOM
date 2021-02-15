// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2012 Steven L. Scott

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

#ifndef BOOM_POLYNOMIAL_HPP_
#define BOOM_POLYNOMIAL_HPP_

#include <complex>
#include "LinAlg/Vector.hpp"

namespace BOOM {

  class Polynomial {
   public:
    typedef std::complex<double> Complex;
    // If ascending is true then the polynomial is
    //
    // coef[0] + coef[1]*x + ... + coef[n]*x^n
    //
    // Otherwise it is
    //
    // coef[0]*x^n + coef[1]*x^{n-1} + ... + coef[n-1]*x + coef[n]
    explicit Polynomial(const Vector &coef, bool ascending = true);

    int degree() const;
    double operator()(double x) const;
    Complex operator()(Complex z) const;

    std::vector<Complex> roots();
    std::vector<double> real_roots();
    std::ostream &print(std::ostream &out) const;

    const Vector &coefficients() const { return coefficients_; }

    bool operator==(const Polynomial &p2) const {
      return coefficients_ == p2.coefficients_;
    }
    bool operator!=(const Polynomial &p2) const {
      return coefficients_ != p2.coefficients_;
    }

   private:
    // Fill roots_real_ and roots_imag_ with the real and complex parts of the
    // roots.
    void find_roots();

    // coefficients_[0] is the constant term.  coefficients_.back()
    // must not be zero.
    Vector coefficients_;

    Vector roots_real_;
    Vector roots_imag_;
  };

  inline std::ostream &operator<<(std::ostream &out, const Polynomial &p) {
    return p.print(out);
  }

  Polynomial operator+(const Polynomial &p1, const Polynomial &p2);
  Polynomial operator-(const Polynomial &p1, const Polynomial &p2);
  Polynomial operator*(const Polynomial &p1, const Polynomial &p2);

}  // namespace BOOM

#endif  //  BOOM_POLYNOMIAL_HPP_
