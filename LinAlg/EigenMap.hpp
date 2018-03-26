#ifndef BOOM_EIGEN_MAP_WRAPPER_
#define BOOM_EIGEN_MAP_WRAPPER_

/*
  Copyright (C) 2005-2018 Steven L. Scott

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

#include "Eigen/Core"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/Vector.hpp"

namespace BOOM {
  // EigenMap(foo) takes a BOOM linear algebra object foo and maps it into an
  // Eigen object.

  // Maps for Matrices.
  inline ::Eigen::Map<::Eigen::MatrixXd> EigenMap(Matrix &m) {
    return ::Eigen::Map<::Eigen::MatrixXd>(m.data(), m.nrow(), m.ncol());
  }

  inline const ::Eigen::Map<const ::Eigen::MatrixXd> EigenMap(const Matrix &m) {
    return ::Eigen::Map<const ::Eigen::MatrixXd>(m.data(), m.nrow(), m.ncol());
  }

  // Maps for Vectors
  inline ::Eigen::Map<::Eigen::VectorXd> EigenMap(Vector &v) {
    return ::Eigen::Map<::Eigen::VectorXd>(v.data(), v.size());
  }

  inline const ::Eigen::Map<const ::Eigen::VectorXd> EigenMap(const Vector &v) {
    return ::Eigen::Map<const ::Eigen::VectorXd>(v.data(), v.size());
  }

  // Maps for VectorViews and ConstVectorViews
  inline ::Eigen::Map<::Eigen::VectorXd, ::Eigen::Unaligned,
                      ::Eigen::InnerStride<::Eigen::Dynamic>>
  EigenMap(VectorView &view) {
    return ::Eigen::Map<::Eigen::VectorXd,
                        ::Eigen::Unaligned,
                        ::Eigen::InnerStride<::Eigen::Dynamic>>(
        view.data(),
        view.size(),
        ::Eigen::InnerStride<::Eigen::Dynamic>(view.stride()));
  }

  inline ::Eigen::Map<const ::Eigen::VectorXd, ::Eigen::Unaligned,
                      ::Eigen::InnerStride<::Eigen::Dynamic>>
  EigenMap(const VectorView &view) {
    return ::Eigen::Map<const ::Eigen::VectorXd,
                        ::Eigen::Unaligned,
                        ::Eigen::InnerStride<::Eigen::Dynamic>>(
        view.data(),
        view.size(),
        ::Eigen::InnerStride<::Eigen::Dynamic>(view.stride()));
  }

  inline ::Eigen::Map<const ::Eigen::VectorXd, ::Eigen::Unaligned,
                      ::Eigen::InnerStride<::Eigen::Dynamic>>
  EigenMap(const ConstVectorView &view) {
    return ::Eigen::Map<const ::Eigen::VectorXd,
                        ::Eigen::Unaligned,
                        ::Eigen::InnerStride<::Eigen::Dynamic>>(
        view.data(),
        view.size(),
        ::Eigen::InnerStride<::Eigen::Dynamic>(view.stride()));
  }

}  // namespace BOOM

#endif  // BOOM_EIGEN_MAP_WRAPPER_
