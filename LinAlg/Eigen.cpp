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

#include <LinAlg/Eigen.hpp>
#include <cpputil/report_error.hpp>
#include <sstream>

extern "C" {
  // LAPACK routine for computing eigenvalues from a square matrix.
  void dgeev_(const char *,  // JOBVL
              const char *,  // JOBVR
              const int *,   // N (order of matrix... number of columns)
              double *,      // A (matrix data)
              const int *,   // LDA (number of rows in A)
              double *,      // WR
              double *,      // WI
              double *,      // VL (data for left eigenvectors)
              const int *,   // nrow(VL)
              double *,      // VR (data for right eigenvectors)
              const int *,   // nrow(VR)
              double *,      // WORK (workspace)
              const int *,   // size of workspace (or -1 for query)
              int *);        // INFO (return value)
}

namespace BOOM {

  Eigen::Eigen(const Matrix &mat,
               bool right_vectors,
               bool left_vectors)
      : real_eigenvalues_(mat.nrow()),
        imaginary_eigenvalues_(mat.nrow()),
        imaginary_sign_(mat.nrow(), 0),
        zero_(mat.nrow(), 0.0),
        left_vectors_(0, 0),
        right_vectors_(0, 0)
  {
    int lda = mat.nrow();
    int n = mat.ncol();
    if (lda != n) {
      report_error("Eigenvalues can only be computed for a square matrix.");
    }
    const char * JOBVL = left_vectors ? "V" : "N";
    const char * JOBVR = right_vectors ? "V" : "N";

    int left_eigenvector_rows = 1;
    if (left_vectors) {
      left_vectors_.resize(n, n);
      left_eigenvector_rows = n;
    }

    int right_eigenvector_rows = 1;
    if (right_vectors) {
      right_vectors_.resize(n, n);
      right_eigenvector_rows = n;
    }

    Matrix tmp(mat);
    std::vector<double> work(1);
    int work_query = -1;
    int info = 0;

    dgeev_(JOBVL,                         // left eigenvectors?
           JOBVR,                         // right eigenvectors?
           &n,                            // ncol(A)
           tmp.data(),                    // data for input matrix
           &lda,                          // leading dimension of a
           real_eigenvalues_.data(),      // data for output
           imaginary_eigenvalues_.data(), // data for output
           left_vectors_.data(),          // space for left eigenvectors
           &left_eigenvector_rows,        // number of left ev rows
           right_vectors_.data(),         // space for right eigenvectors
           &right_eigenvector_rows,       // number of right ev rows
           work.data(),                   // data for workspace
           &work_query,                   // query workspace size
           &info);                        // exit status.  0 == success

    if (info < 0) {
      std::ostringstream err;
      err << "Argument " << -info <<
          " had an illegal value in the LAPACK routine for finding eigenvalues"
          << std::endl;
      report_error(err.str());
    } else if (info > 0) {
      report_error("Eigenvalue computation failed for numerical reasons "
                   "during initial workspace query.");
    }

    int work_size = work[0];
    work.resize(work_size);
    dgeev_(JOBVL,                         // left eigenvectors?
           JOBVR,                         // right eigenvectors?
           &n,                            // ncol(A)
           tmp.data(),                    // data for input matrix
           &lda,                          // leading dimension of a
           real_eigenvalues_.data(),      // data for output
           imaginary_eigenvalues_.data(), // data for output
           left_vectors_.data(),          // space for left eigenvectors
           &left_eigenvector_rows,        // number of left ev rows
           right_vectors_.data(),         // space for right eigenvectors
           &right_eigenvector_rows,       // number of right ev rows
           work.data(),                   // data for workspace
           &work_size,                    // query workspace size
           &info);                        // exit status.  0 == success

    if (info < 0) {
      std::ostringstream err;
      err << "Argument " << -info <<
          " had an illegal value in the LAPACK routine for finding eigenvalues"
          << std::endl;
      report_error(err.str());
    } else if (info > 0) {
      report_error("Eigenvalue computation failed for numerical reasons "
                   "during computation phase.");
    }

    for (int i = 0; i < n; ++i) {
      if ((i+1 < n)
          && (fabs(real_eigenvalues_[i] -
                   real_eigenvalues_[i+1]) < 1e-8)
          && (fabs(imaginary_eigenvalues_[i] +
                   imaginary_eigenvalues_[i+1]) < 1e-8)) {
          imaginary_sign_[i] = 1;
          imaginary_sign_[i+1] = -1;
          ++i;
      }
    }
  }

  std::vector<std::complex<double> > Eigen::eigenvalues()const{
    std::vector<std::complex<double> > ans;
    int n = real_eigenvalues_.size();
    ans.reserve(n);
    for (int i = 0; i < n; ++i) {
      std::complex<double> value(real_eigenvalues_[i],
                                 imaginary_eigenvalues_[i]);
      ans.push_back(value);
    }
    return ans;
  }

  const Vector & Eigen::real_eigenvalues()const{
    return real_eigenvalues_;
  }

  const Vector & Eigen::imaginary_eigenvalues()const{
    return imaginary_eigenvalues_;
  }

  const ConstVectorView Eigen::right_real_eigenvector(int i)const{
    if(right_vectors_.nrow() == 0){
      report_error("Right eigenvectors were not requested by the constructor.");
    }
    if (imaginary_sign_[i] > -1) return right_vectors_.col(i);
    return right_vectors_.col(i-1);
  }

  Vector Eigen::right_imaginary_eigenvector(int i)const{
    if(right_vectors_.nrow() == 0){
      report_error("Right eigenvectors were not requested by the constructor.");
    }
    if (imaginary_sign_[i] == 0) return zero_;
    else if(imaginary_sign_[i] == 1) return right_vectors_.col(i+1);
    else if(imaginary_sign_[i] == -1) return -1 * right_vectors_.col(i);
    report_error("Should never get here.  "
                 "The imaginary_sign_ structure contains illegal values");
    return Vector(0);
  }

  std::vector<std::complex<double> > Eigen::right_eigenvector(int i)const{
    std::vector<std::complex<double> > ans;
    int n = real_eigenvalues_.size();
    ans.reserve(n);
    for (int j = 0; j < n; ++j) {
      double real=0;
      double imaginary=0;
      if (imaginary_sign_[i] == 0) {
        real = right_vectors_(j, i);
      } else if (imaginary_sign_[i] == 1) {
        real = right_vectors_(j, i);
        imaginary = right_vectors_(j, i+1);
      } else if (imaginary_sign_[i] == -1) {
        real = right_vectors_(j, i-1);
        imaginary = -1 * right_vectors_(j, i);
      } else {
        report_error("Should never get here.  "
                     "The imaginary_sign_ structure contains illegal values");
      }
      std::complex<double> value(real, imaginary);
      ans.push_back(value);
    }
    return ans;
  }

  int Eigen::imaginary_sign(int i)const{return imaginary_sign_[i];}

}
