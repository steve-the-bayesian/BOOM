#ifndef BOOM_MODELS_GP_KERNELS_HPP_
#define BOOM_MODELS_GP_KERNELS_HPP_
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

#include "Models/ParamTypes.hpp"
#include <ostream>
#include "LinAlg/Vector.hpp"


namespace BOOM {

  // A "kernel" is the parameter to a Gaussian process.  A kernel is a function
  // of two vector arguments: k(x1, x2), subject to the condition that a matrix
  // K with elements K_{ij} = k(xi, xj) must be positive defininte for arbitrary
  // elements xi, xj.
  //
  // Kernels may depend on parameters, in which case the 'vectorize' and
  // 'unvectorize' methods should return or consume those parameters.
  class KernelParams : public Params {
   public:
    KernelParams * clone() const override = 0;
    virtual double operator()(const ConstVectorView &x1,
                              const ConstVectorView &x2) const = 0;

    virtual SpdMatrix operator()(const Matrix &predictors) const;
  };

  //===========================================================================
  // A radial basis function kernel.
  // K(x1, x2) = exp(-.5 * ||x1 - x2||^2 / scale^2)
  class RadialBasisFunction : public KernelParams {
   public:

    // Args:
    //   scale: The size of a "standard deviation" over which the kernel should
    //     reach.
    explicit RadialBasisFunction(double scale = 1.0);
    explicit RadialBasisFunction(const Vector &scale);
    RadialBasisFunction *clone() const override;

    uint size(bool = true) const override {return 1;}

    const Vector &scale() const {return scale_;}
    void set_scale(const Vector &scale);

    double operator()(const ConstVectorView &x1,
                      const ConstVectorView &x2) const override;
    using KernelParams::operator();

    std::ostream &display(std::ostream &out) const override;
    Vector vectorize(bool minimal=true) const override;
    Vector::const_iterator unvectorize(Vector::const_iterator &v,
                                       bool minimal = true) override;
    Vector::const_iterator unvectorize(const Vector &v,
                                       bool minimal = true) override;

   private:
    mutable Vector scale_;
  };

  //===========================================================================
  // TODO: implement the other kernels from :
  //
  // https://scikit-learn.org/stable/modules/gaussian_process.html#kernels-for-gaussian-processes


  // A kernel using the design matrix X to create a distance metric:
  //
  //   exp(-0.5 * scale_factor * D(X'X, a) / n)
  //
  // Where D(S, a) = (1-a) * S + a * diag(S) is an operator that averages the
  // matrix S with its diagonal (using weight a applied to the diagonal).
  class MahalanobisKernel : public KernelParams {
   public:
    explicit MahalanobisKernel(int dim, double scale = 1.0);
    explicit MahalanobisKernel(const Matrix &X,
                               double scale = 1.0,
                               double diagonal_shrinkage = 0.05);
    MahalanobisKernel *clone() const override;

    uint size(bool = true) const override;
    double scale() const {
      return scale_;
    }
    void set_scale(double scale);

    double operator()(const ConstVectorView &x1,
                      const ConstVectorView &x2) const override;
    using KernelParams::operator();

    std::ostream &display(std::ostream &out) const override;
    Vector vectorize(bool minimal=true) const override;
    Vector::const_iterator unvectorize(Vector::const_iterator &v,
                                       bool minimal = true) override;
    Vector::const_iterator unvectorize(const Vector &v,
                                       bool minimal = true) override;

   private:
    double scale_;
    double sample_size_;

    SpdMatrix scaled_shrunk_xtx_inv_;

  };


}  // namespace BOOM


#endif  //  BOOM_MODELS_GP_KERNELS_HPP_
