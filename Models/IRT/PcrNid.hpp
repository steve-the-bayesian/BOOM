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
#ifndef BOOM_PCR_NID_HPP
#define BOOM_PCR_NID_HPP

#include "Models/IRT/PartialCreditModel.hpp"

namespace BOOM {
  namespace IRT {

    class PcrNid : public PartialCreditModel {
      /*------------------------------------------------------------
        An item with maxscore()==M yields log score probabilities = C
        + X*beta where C is a normalizing constant X[0..M, 0..M] is an
        (M+1)x(M+2) matrix and beta[0..M+1] is an M+2 vector as follows
        (for M==4)

        X:                             beta:
        1  0  0  0  0  1*theta         a*(d0-b)
        0  1  0  0  0  2*theta         a*(d0+d1-2b)
        0  0  1  0  0  3*theta         a*(d0+d1+d2-3b)
        0  0  0  1  0  4*theta         a*(d0+d1+d2+d3-4b)
        0  0  0  0  1  5*theta         a*(-5b)        // sum of d's is 0
                                       a

        ------------------------------------------------------------*/

     public:
      PcrNid(const string &Id, uint Mscore, uint which_sub, uint Nscales,
             const string &Name = "");
      PcrNid(const string &Id, uint Mscore, uint which_sub, uint Nscales,
             double a, double b, const Vector &d, const string &Name = "");
      PcrNid(const PcrNid &rhs);
      PcrNid *clone() const override;

      virtual const Vector &d() const;
      virtual double d(uint m) const;
      virtual void set_d(const Vector &D);
      virtual bool is_d0_fixed() const { return false; }

      virtual const Matrix &X(const Vector &Theta) const;
      virtual const Matrix &X(double theta) const;

     private:
      virtual void fill_beta(bool first_time = false) const;
      virtual void fill_abd() const;
      // to be called during construction:
      void setup_X();
      void setup_beta();
    };
  }  // namespace IRT
}  // namespace BOOM
#endif  // BOOM_PCR_NID_HPP
