/*
  Copyright (C) 2005-2019 Steven L. Scott

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

#include "LinAlg/LU.hpp"
#include "LinAlg/EigenMap.hpp"
#include "Eigen/LU"

#include "cpputil/report_error.hpp"

namespace BOOM {
  namespace LuImpl {

    class LU_impl_ {
     public:
      explicit LU_impl_(const Matrix &foo) : dcmp_(EigenMap(foo)) {
        permutation_sign_ = dcmp_.permutationP().determinant() *
            dcmp_.permutationQ().determinant();
      }

      LU_impl_ * clone() const { return new LU_impl_(*this); }

      Vector solve(const ConstVectorView &rhs) const {
        if (rhs.size() != ncol()) {
          std::ostringstream err;
          err << "The decomposed matrix has " << ncol() << " columns, but the "
              << "right hand side is of length " << rhs.size();
        }

        Vector ans(nrow());
        EigenMap(ans) = dcmp_.solve(EigenMap(rhs));
        return ans;
      }

      Matrix solve(const Matrix &rhs) const {
        if (rhs.nrow() != this->ncol()) {
          std::ostringstream err;
          err << "The decomposed matrix has " << ncol() << " columns, but the "
              << "right hand side has " << rhs.nrow() << " rows." << std::endl;
          report_error(err.str());
        }
        Matrix ans(nrow(), rhs.ncol());
        EigenMap(ans) = dcmp_.solve(EigenMap(rhs));
        return ans;
      }

      int nrow() const { return dcmp_.matrixLU().rows();}
      int ncol() const { return dcmp_.matrixLU().cols();}

      Matrix original_matrix() const {
        Matrix ans(nrow(), ncol());
        EigenMap(ans) = dcmp_.reconstructedMatrix();
        return ans;
      }

      double det() const {
        return dcmp_.determinant();
      }

      double logdet() const {
        const Eigen::MatrixXd &lu(dcmp_.matrixLU());
        int dim = lu.rows();
        // Keep track of the number of negative signs in the diagonal elements
        // of LU.
        int numneg = (permutation_sign_ == -1);
        double ans = 0;
        for (int i = 0; i < dim; ++i) {
          double x = lu(i, i);
          if (x < 0) {
            ++numneg;
            ans += std::log(-x);
          } else {
            ans += std::log(x);
          }
        }
        return numneg %2 == 0 ? ans : negative_infinity();
      }


     private:
      Eigen::FullPivLU<Eigen::MatrixXd> dcmp_;
      int permutation_sign_;

    };

  } // namespace LuImpl

  LU::LU() : impl_(nullptr) {}

  LU::~LU() {}

  LU::LU(const LU &rhs) : impl_(rhs.impl_->clone()) {}

  LU::LU(LU &&rhs) : impl_(std::move(rhs.impl_)) {
    rhs.impl_.reset(nullptr);
  }

  LU & LU::operator=(const LU &rhs) {
    if (&rhs != this) {
      impl_.reset(rhs.impl_->clone());
    }
    return *this;
  }

  LU & LU::operator=(LU &&rhs) {
    if (&rhs != this) {
      impl_ = std::move(rhs.impl_);
      rhs.impl_.reset(nullptr);
    }
    return *this;
  }

  LU::LU(const Matrix &square_matrix) {
    decompose(square_matrix);
  }

  void LU::decompose(const Matrix &square_matrix) {
    if (square_matrix.nrow() != square_matrix.ncol()) {
      report_error("LU requires a square matrix.");
    }
    impl_.reset(new LuImpl::LU_impl_(square_matrix));
  }

  Matrix LU::original_matrix() const {
    if (!impl_) {
      report_error("No matrix was ever decomposed.");
    }
    return impl_->original_matrix();
  }

  Vector LU::solve(const ConstVectorView &rhs) const {
    if (!impl_) {
      report_error("Decmpose a matrix before calling LU::solve.");
    }
    return impl_->solve(rhs);
  }

  Matrix LU::solve(const Matrix &rhs) const {
    if (!impl_) {
      report_error("Decompose a matrix before calling LU::solve.");
    }
    return impl_->solve(rhs);
  }

  int LU::nrow() const {
    if (!impl_) return 0;
    return impl_->nrow();
  }

  int LU::ncol() const {
    if (!impl_) return 0;
    return impl_->ncol();
  }

  void LU::clear() {
    impl_.reset(nullptr);
  }

  double LU::det() const {
    if (!impl_) {
      report_error("Decompose a matrix before calling LU::det().");
    }
    return impl_->det();
  }

  double LU::logdet() const {
    if (!impl_) {
      report_error("Decompose a matrix before calling LU::logdet().");
    }
    return impl_->logdet();
  }


}  // namespace BOOM
