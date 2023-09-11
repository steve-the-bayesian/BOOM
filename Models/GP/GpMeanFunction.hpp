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

#include "Models/ParamTypes.hpp"
#include "LinAlg/Vector.hpp"
#include "Models/Glm/GlmCoefs.hpp"
#include <ostream>

// #include "stats/Encoders.hpp"

namespace BOOM {

  // TODO: move this in parallel to other 'Params' files.

  // Forward declarations needed to prevent circular inclusions.
  class GaussianProcessRegressionModel;

  // FunctionParams describes a function mapping a vector of predictors X to a
  // real number yhat.
  class FunctionParams : public Params {
   public:
    virtual FunctionParams * clone() const override = 0;
    virtual double operator()(const ConstVectorView &x) const = 0;
    virtual Vector operator()(const Matrix &X) const;
  };

  //===========================================================================
  // A FunctionParams that alwas returns 0.
  class ZeroFunction : public FunctionParams {
   public:
    ZeroFunction * clone() const override;
    uint size(bool) const override {return 0;}

    double operator()(const ConstVectorView &x) const override {
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
    using Params::unvectorize;
  };

  //===========================================================================
  // A FunctionParams where the function is a linear regression.
  class LinearMeanFunction : public FunctionParams {
   public:
    explicit LinearMeanFunction(const Ptr<GlmCoefs> &coefficients);
    LinearMeanFunction(const LinearMeanFunction &rhs);
    LinearMeanFunction & operator=(const LinearMeanFunction &rhs);
    LinearMeanFunction(LinearMeanFunction &&rhs) = default;
    LinearMeanFunction & operator=(LinearMeanFunction &&rhs) = default;
    LinearMeanFunction * clone() const override;

    double operator()(const ConstVectorView &x) const override;
    using FunctionParams::operator();

    uint size(bool minimal = true) const override;
    std::ostream &display(std::ostream &out) const override;

    // The vectorized parameters don't include the model's residual variance
    // parameter.
    Vector vectorize(bool minimal = true) const override;
    Vector::const_iterator unvectorize(Vector::const_iterator &v,
                                       bool minimal = true) override;
    using Params::unvectorize;

    Ptr<GlmCoefs> & coef() {return coefficients_;}
    const Ptr<GlmCoefs> & coef() const {return coefficients_;}

   private:
    Ptr<GlmCoefs> coefficients_;
  };

  //===========================================================================
  // A FunctionParams where the function is a linear regression preceeded by a
  // basis expansion or transformation.
  //
  // TODO: Need to work out a VectorEncoder encodign scheme before actually
  // rolling this out.
  //
  // class ExpandedLinearMeanFunction : public FunctionParams {
  //  public:
  //   ExpandedLinearMeanFunction(
  //       const Ptr<RegressionModel> &model,
  //       const Ptr<VectorEncoder> &encoder);

  //   double operator()(const ConstVectorView &x) const override;
  //   using FunctionParams::operator();

  //   std::ostream &display(std::ostream &out) const override;
  //   Vector vectorize(bool minimal = true) const override;
  //   Vector::const_iterator unvectorize(Vector::const_iterator &v,
  //                                      bool minimal = true) override;
  //   using Params::unvectorize;

  // };


  //===========================================================================
  // A mean function that can be used with a GaussianProcessRegressionModel,
  // where the mean is a GP regression.
  //
  // This mean function is useful in the context of a hierarchical Gaussian
  // process.  It really can't be used as the mean function of a regular GP
  // because it wouldn't be identified.
  class GpMeanFunction : public FunctionParams {
   public:
    explicit GpMeanFunction(const Ptr<GaussianProcessRegressionModel> &gp);
    GpMeanFunction(const GpMeanFunction &rhs);
    GpMeanFunction &operator=(const GpMeanFunction &rhs);
    GpMeanFunction(GpMeanFunction &&rhs) = default;
    GpMeanFunction & operator=(GpMeanFunction &&rhs) = default;
    GpMeanFunction * clone() const override;
    double operator()(const ConstVectorView &x) const override;
    using FunctionParams::operator();

    uint size(bool minimal = true) const override;

    std::ostream &display(std::ostream &out) const override;
    Vector vectorize(bool minimal = true) const override;
    Vector::const_iterator unvectorize(Vector::const_iterator &v,
                                       bool minimal = true) override;
    using Params::unvectorize;

   private:
    Ptr<GaussianProcessRegressionModel> gp_;
  };

}  // namespace BOOM

#endif //  BOOM_MODELS_GP_GP_MEAN_FUNCTION_HPP_
