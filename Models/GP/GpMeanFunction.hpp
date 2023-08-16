#ifndef BOOM_MODELS_GP_GP_MEAN_FUNCTION_HPP_
#define BOOM_MODELS_GP_GP_MEAN_FUNCTION_HPP_
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

#include "Models/GP/kernels.hpp"

namespace BOOM {

  // Forward declarations needed to prevent circular inclusions.
  class GaussianProcessRegressionModel;
  class RegressionModel;
  class Encoder;

  // FunctionParams describes a function mapping a vector of predictors X to a
  // real number yhat.
  class FunctionParams : public Params {
   public:
    virtual FunctionParams * clone() const override = 0;
    virtual double operator()(const ConstVectorView &x) const = 0;
    virtual Vector operator()(const Matrix &X) const;
  };


  // A FunctionParams that alwas returns 0.
  class ZeroFunction : public FunctionParams {
   public:
    ZeroFunction * clone() const override;
    uint size(bool) const override {return 0;}

    virtual double operator()(const ConstVectorView &x) const override {
      return 0;
    }

    using FunctionParams::operator();

    std::ostream & display(std::ostream &out) const override {
      out << "ZeroFunction";
      return out;
    }

    Vector vectorize(bool minimal = true) const override {
      return Vector();
    }
    Vector::const_iterator unvectorize(Vector::const_iterator &v,
                                       bool minimal = true) override {return v;}
    Vector::const_iterator unvectorize(const Vector &v,
                                       bool minimal = true) override {
      return v.cbegin();
    }
  };

  // A FunctionParams where the function is a linear regression.
  class LinearMeanFunction : public FunctionParams {
   public:
    LinearMeanFunction(const Ptr<RegressionModel> &model);

   private:
    Ptr<RegressionModel> model_;
  };

  //
  class ExpandedLinearMeanFunction : public FunctionParams {
  };


  // A mean function that can be used with a GaussianProcessRegressionModel, where
  // the mean is a GP regression.
  class GpMeanFunction : public FunctionParams {
   public:
    GpMeanFunction(const Ptr<GaussianProcessRegressionModel> &gp);

    GpMeanFunction * clone() const override;
    double operator()(const ConstVectorView &x) const override;
    using FunctionParams::operator();

    uint size(bool minimal = true) const override;

   private:
    Ptr<GaussianProcessRegressionModel> gp_;
  };

}  // namespace BOOM
#endif //  BOOM_MODELS_GP_GP_MEAN_FUNCTION_HPP_
