/*
  Copyright (C) 2005-2010 Steven L. Scott

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
#ifndef BOOM_CHI_SQUARE_TEST_HPP_
#define BOOM_CHI_SQUARE_TEST_HPP_

#include <iostream>
#include <LinAlg/Vector.hpp>
#include <LinAlg/Matrix.hpp>


namespace BOOM{
class OneWayChiSquareTest{
 public:
  // for testing counts vs. a known distribution
  OneWayChiSquareTest(const Vector & observed, const Vector & distribution);
  double p_value()const;
  double degrees_of_freedom()const;
  double chi_square()const;

  // Returns true if all the assumptions of the test have been met.
  // Returns false if any expected cell counts are less than 5.0.
  bool is_valid()const;
  std::ostream & print(std::ostream & out)const;
 private:
  Vector observed_;
  Vector expected_;
  double chi_square_;
  double df_;
  double p_value_;
};

inline std::ostream & operator<<(std::ostream &out,
                                 const OneWayChiSquareTest &test){
  return test.print(out);
}

class TwoWayChiSquareTest{
 public:
  TwoWayChiSquareTest(const Matrix & observed_cell_counts);
  double p_value()const;
  double degrees_of_freedom()const;
  double chi_square()const;

  // Returns true if all the assumptions of the test have been met.
  // Returns false if any expected cell counts are less than 5.0.
  bool is_valid()const;
  std::ostream & print(std::ostream & out)const;
 private:
  double chi_square_;
  double df_;
  double p_value_;
  bool assumptions_are_met_;
};

inline std::ostream & operator<<(std::ostream &out,
                                 const TwoWayChiSquareTest &test){
  return test.print(out);
}

}
#endif // BOOM_CHI_SQUARE_TEST_HPP_
