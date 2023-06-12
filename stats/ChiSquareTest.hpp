// Copyright 2018 Google LLC. All Rights Reserved.
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

#include "LinAlg/Matrix.hpp"
#include "LinAlg/Vector.hpp"

namespace BOOM {
  class FrequencyDistribution;

  // For testing counts vs. a known distribution
  class OneWayChiSquareTest {
   public:
    // Args:
    //   observed:  The observed cell counts in each cell of the frequency distribution.
    //   distribution:  The discrete probability distribution being tested against.
    //   collapse: The minimium expected cell count.  Cells with less than this
    //     minimum count are collapsed into neighboring cells.  A negative value
    //     indicates no cell collapsing is desired.
    OneWayChiSquareTest(const Vector &observed, const Vector &distribution, double collapse = -1.0);

    // Args:
    //   freq:  The empirical frequency distribution of observed events.
    //   distribution:  The discrete probability distribution being tested against.
    //   collapse: The minimium expected cell count.  Cells with less than this
    //     minimum count are collapsed into neighboring cells.  A negative value
    //     indicates no cell collapsing is desired.
    OneWayChiSquareTest(const FrequencyDistribution &freq,
                        const Vector &distribution,
                        double collapse = -1.0);

    double p_value() const;
    double degrees_of_freedom() const;
    double chi_square() const;

    // Returns true if all the assumptions of the test have been met.
    // Returns false if any expected cell counts are less than 5.0.
    bool is_valid() const;
    std::ostream &print(std::ostream &out) const;

   private:
    // Collapse all cells with expected counts less than min_count into
    // neighboring cells.
    //
    // Effect:
    //   observed_ and expected_ are replaced with collapsed versions of themselves.
    void collapse_cells(double min_count);

    Vector observed_;
    Vector expected_;
    double chi_square_;
    double df_;
    double p_value_;
  };

  inline std::ostream &operator<<(std::ostream &out,
                                  const OneWayChiSquareTest &test) {
    return test.print(out);
  }

  class TwoWayChiSquareTest {
   public:
    explicit TwoWayChiSquareTest(const Matrix &observed_cell_counts);
    double p_value() const;
    double degrees_of_freedom() const;
    double chi_square() const;

    // Returns true if all the assumptions of the test have been met.
    // Returns false if any expected cell counts are less than 5.0.
    bool is_valid() const;
    std::ostream &print(std::ostream &out) const;

   private:
    double chi_square_;
    double df_;
    double p_value_;
    bool assumptions_are_met_;
  };

  inline std::ostream &operator<<(std::ostream &out,
                                  const TwoWayChiSquareTest &test) {
    return test.print(out);
  }

}  // namespace BOOM
#endif  // BOOM_CHI_SQUARE_TEST_HPP_
