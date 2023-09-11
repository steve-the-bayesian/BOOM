#ifndef BOOM_MODELS_POSITIVE_SEMIDEFINITE_DATA_HPP_
#define BOOM_MODELS_POSITIVE_SEMIDEFINITE_DATA_HPP_

/*
  Copyright (C) 2005-2023 Steven L. Scott

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

#include <functional>
#include "Models/DataTypes.hpp"
#include "Models/ParamTypes.hpp"
#include "LinAlg/SpdMatrix.hpp"
#include "LinAlg/Cholesky.hpp"
#include "LinAlg/SVD.hpp"

namespace BOOM {
  // A non-negative definite (positive semidefinite) matrix, and its lower
  // triangular square root.
  //
  // virtual public inheritance is needed because PositiveSemidefiniteParams
  // inherit from this class but also from Params, which inherit from Data.
  class PositiveSemidefiniteData : virtual public Data {
   public:
    explicit PositiveSemidefiniteData(const SpdMatrix &S);
    PositiveSemidefiniteData(const PositiveSemidefiniteData &rhs);
    PositiveSemidefiniteData& operator=(const PositiveSemidefiniteData &rhs);
    PositiveSemidefiniteData(PositiveSemidefiniteData &&rhs);
    PositiveSemidefiniteData &operator=(PositiveSemidefiniteData &&rhs);

    PositiveSemidefiniteData *clone() const override;

    // The number of elements in the matrix.
    // Args:
    //   minimal: If true then only the elements in the diagonal and the upper
    //     triangle are counted.  Otherwise all elements (including elements
    //     duplicated by symmetry) are counted.
    virtual uint size(bool minimal = true) const;

    int nrow() const {return value_.nrow();}
    int ncol() const {return value_.ncol();}

    // The number of rows in the matrix (which is the same as the number of
    // columns).
    uint dim() const {return value_.nrow();}

    // Show the variance matrix.
    std::ostream &display(std::ostream &out) const override;

    const SpdMatrix &value() const {return value_;}
    void set(const SpdMatrix &value, bool signal = true);

    // A matrix A such that A * A' = value()
    const Matrix &root() const {return root_;}

    const SpdMatrix &generalized_inverse() const {return generalized_inverse_;}

    // The sum of the logs of the reciprocals of the (absolute) nonzero
    // eigenvalues in the original matrix.  If the original matrix is positive
    // definite this is the log determinant of the inverse matrix.
    double generalized_ldsi() const { return ldsi_; }

   private:
    // Update the values of root_ and generalized_inverse_
    void update();

    SpdMatrix value_;
    Matrix root_;
    SpdMatrix generalized_inverse_;

    // The sum of the logs of the absolute reciprocals of the nonzero
    // eigenvalues.
    double ldsi_;
  };

  class PositiveSemidefiniteParams :
      public PositiveSemidefiniteData,
      virtual public Params {
   public:
    explicit PositiveSemidefiniteParams(const SpdMatrix &S);
    PositiveSemidefiniteParams(const PositiveSemidefiniteParams &rhs);
    PositiveSemidefiniteParams(PositiveSemidefiniteParams &&rhs);
    PositiveSemidefiniteParams &operator=(const PositiveSemidefiniteParams &rhs);
    PositiveSemidefiniteParams &operator=(PositiveSemidefiniteParams &&rhs);
    PositiveSemidefiniteParams *clone() const override;

    uint size(bool minimal = true) const override;

    Vector vectorize(bool minimal = true) const override;
    Vector::const_iterator unvectorize(
        Vector::const_iterator &v, bool minimal = true) override;
    using Params::unvectorize;
  };

}  // namespace BOOM
#endif  // BOOM_MODELS_POSITIVE_SEMIDEFINITE_DATA_HPP_
