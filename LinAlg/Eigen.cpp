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

#include "LinAlg/Eigen.hpp"
#include <sstream>
#include "Eigen/Eigenvalues"
#include "LinAlg/EigenMap.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {
  namespace {
    using Eigen::MatrixXd;
  }  // namespace

  EigenDecomposition::EigenDecomposition(const Matrix &mat, bool vectors)
      : eigenvalues_(mat.nrow()),
        real_eigenvalues_(mat.nrow()),
        imaginary_eigenvalues_(mat.nrow()),
        real_eigenvectors_(0, 0),
        imaginary_eigenvectors_(0, 0) {
    Eigen::EigenSolver<MatrixXd> eigen(EigenMap(mat), vectors);
    const auto &eigen_values = eigen.eigenvalues();
    int dim = mat.nrow();
    for (int i = 0; i < dim; ++i) {
      eigenvalues_[i] = eigen_values(i);
      real_eigenvalues_[i] = eigenvalues_[i].real();
      imaginary_eigenvalues_[i] = eigenvalues_[i].imag();
    }
    if (vectors) {
      real_eigenvectors_ = Matrix(dim, dim);
      imaginary_eigenvectors_ = Matrix(dim, dim);
      const auto &eigen_vectors = eigen.eigenvectors();
      for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
          real_eigenvectors_(i, j) = eigen_vectors(i, j).real();
          imaginary_eigenvectors_(i, j) = eigen_vectors(i, j).imag();
        }
      }
    }
  }

  ConstVectorView EigenDecomposition::real_eigenvector(int i) const {
    if (real_eigenvectors_.nrow() == 0) {
      report_error("Eigenvectors were not requested by the constructor.");
    }
    return real_eigenvectors_.col(i);
  }

  ConstVectorView EigenDecomposition::imaginary_eigenvector(int i) const {
    if (imaginary_eigenvectors_.nrow() == 0) {
      report_error("Eigenvectors were not requested by the constructor.");
    }
    return imaginary_eigenvectors_.col(i);
  }

  namespace {
    std::vector<std::complex<double>> complex_vector(
        const ConstVectorView &real, const ConstVectorView &imag) {
      std::vector<std::complex<double>> ans;
      if (real.size() != imag.size()) {
        report_error("Real and imaginary parts must be the same size.");
      }
      for (int i = 0; i < real.size(); ++i) {
        std::complex<double> value(real[i], imag[i]);
        ans.push_back(value);
      }
      return ans;
    }
  }  // namespace

  std::vector<std::complex<double>> EigenDecomposition::eigenvector(
      int i) const {
    if (real_eigenvectors_.size() == 0) {
      report_error("Eigenvectors not requested by the constructor.");
    }
    return complex_vector(real_eigenvectors_.col(i),
                          imaginary_eigenvectors_.col(i));
  }

  //======================================================================
  SpdEigen::SpdEigen(const SpdMatrix &matrix, bool compute_vectors)
      : eigenvalues_(matrix.nrow()),
        right_vectors_(0, 0)
  {
    ::Eigen::SelfAdjointEigenSolver<::Eigen::MatrixXd> solver(
        EigenMap(matrix),
        compute_vectors ? ::Eigen::ComputeEigenvectors : ::Eigen::EigenvaluesOnly);
    EigenMap(eigenvalues_) = solver.eigenvalues();
    if (compute_vectors) {
      right_vectors_.resize(matrix.nrow(), matrix.ncol());
      EigenMap(right_vectors_) = solver.eigenvectors();
    }
  }

}  // namespace BOOM
