// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2015 Steven L. Scott

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
#ifndef BOOM_GAMMA_REGRESSION_MODEL_HPP_
#define BOOM_GAMMA_REGRESSION_MODEL_HPP_

#include "Models/GammaModel.hpp"
#include "Models/Glm/Glm.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/ParamPolicy_2.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/Sufstat.hpp"

namespace BOOM {

  // Base class to describe Gamma regression models.  Child classes
  // may differ based on how they store their data.
  //
  // Gamma regression models
  //    y[i] ~ Ga(alpha,  alpha / mu[i])
  //    log(mu[i]) = beta.dot(x[i])
  // E(y[i] | x[i]) = mu
  // V(y[i] | x[i]) = mu^2 / alpha
  // Coefficient of variation:  sd(y) / mean(y) = 1/sqrt(alpha).
  // Thus priors on alpha induce priors on the coefficient of
  // variation.
  //
  // In the code below, alpha is the "shape parameter" and beta are
  // the "coefficients".
  class GammaRegressionModelBase : public GlmModel,
                                   public ParamPolicy_2<UnivParams, GlmCoefs>,
                                   public PriorPolicy,
                                   public NumOptModel,
                                   virtual public MixtureComponent {
   public:
    explicit GammaRegressionModelBase(int xdim);
    GammaRegressionModelBase(double shape_parameter,
                             const Vector &coefficients);
    GammaRegressionModelBase(const Ptr<UnivParams> &alpha,
                             const Ptr<GlmCoefs> &coefficients);
    GammaRegressionModelBase *clone() const override = 0;

    GlmCoefs &coef() override { return ParamPolicy::prm2_ref(); }
    const GlmCoefs &coef() const override { return ParamPolicy::prm2_ref(); }
    Ptr<GlmCoefs> coef_prm() override { return ParamPolicy::prm2(); }
    const Ptr<GlmCoefs> coef_prm() const override {
      return ParamPolicy::prm2();
    }

    Ptr<UnivParams> shape_prm() { return prm1(); }
    double shape_parameter() const { return prm1_ref().value(); }
    void set_shape_parameter(double alpha);

    double expected_value(const Vector &x) const;
    double expected_value(const VectorView &x) const;
    double expected_value(const ConstVectorView &x) const;

    double pdf(const Data *dp, bool logscale) const override;

    double sim(const Vector &x, RNG &rng = GlobalRng::rng) const;
  };
  //======================================================================
  class GammaRegressionModel : public GammaRegressionModelBase,
                               public IID_DataPolicy<RegressionData> {
   public:
    explicit GammaRegressionModel(int xdim);
    GammaRegressionModel(double shape_parameter, const Vector &coefficients);
    GammaRegressionModel(const Ptr<UnivParams> &alpha,
                         const Ptr<GlmCoefs> &coefficients);
    GammaRegressionModel *clone() const override;

    // Returns log likelihood and its derivatives as a function of the
    // concatenated vector [shape_parameter(), coef()].
    double Loglike(const Vector &alpha_beta, Vector &gradient, Matrix &Hessian,
                   uint nderiv) const override;
    int number_of_observations() const override { return dat().size(); }
  };

  //======================================================================

  // A comparator ("less") to be used with the map in
  // GammaRegressionConditionalSuf.
  struct VectorPtrLess {
    bool operator()(const Ptr<VectorData> &lhs,
                    const Ptr<VectorData> &rhs) const {
      if (!lhs || !rhs) return false;
      return lhs->value() < rhs->value();
    }
  };

  // Conditional (on x) sufficient statistics for a gamma regression
  // model.  The data are organized using a map, so that unique
  // covariate patterns accumulate sufficient statistics for that
  // pattern.
  class GammaRegressionConditionalSuf : public SufstatDetails<RegressionData> {
   public:
    typedef std::map<Ptr<VectorData>, Ptr<GammaSuf>, VectorPtrLess> MapType;
    GammaRegressionConditionalSuf();
    GammaRegressionConditionalSuf *clone() const override;
    void Update(const RegressionData &data) override;
    void clear() override;
    void increment(double n, double sum, double sumlog,
                   const Ptr<VectorData> &predictors);

    GammaRegressionConditionalSuf *abstract_combine(Sufstat *rhs) override;

    void combine(const Ptr<GammaRegressionConditionalSuf> &rhs);
    void combine(const GammaRegressionConditionalSuf &rhs);

    Vector vectorize(bool minimal = true) const override;
    Vector::const_iterator unvectorize(Vector::const_iterator &v,
                                       bool minimal = true) override;
    Vector::const_iterator unvectorize(const Vector &v,
                                       bool minimal = true) override;
    std::ostream &print(std::ostream &out) const override;

    // This is the primay accessor for the sufficient statistics.
    const MapType &map() const { return suf_; }

    // Set the dimensions of the sufficient statistics.  If the
    // sufstat object has been built natively using update() or
    // combine() this is not necessary.  If a fresh object is to be
    // populated from vectorized data then this function must be
    // called first.
    void set_dimensions(int number_of_rows, int xdim);

   private:
    // If predictors exists in the sufficient statistics map, then
    // return the GammaSuf that it is associated with.  Otherwise,
    // allocate a new, empty GammaSuf, associate it with predictors,
    // and return it.
    Ptr<GammaSuf> get(const Ptr<VectorData> &predictors);

    // This is the main object holding the conditional sufficient
    // statistics and the vectors of predictors associated with them.
    MapType suf_;

    // Dimension of the predictor variable.  Set during calls to Update.
    int xdim_;

    // Number of distinct covariate (x) patterns.
    int nrow_;
  };

  // A GammaRegressionModel where the data are stored using
  // conditional sufficient statistics.  If multiple observations will
  // be observed for each covariate pattern then this is a more
  // efficient class.  If each covariate pattern will be unique then
  // this class will be less space efficient than
  // GammaRegressionModel.
  class GammaRegressionModelConditionalSuf
      : public GammaRegressionModelBase,
        public SufstatDataPolicy<RegressionData,
                                 GammaRegressionConditionalSuf> {
   public:
    explicit GammaRegressionModelConditionalSuf(int xdim);
    GammaRegressionModelConditionalSuf(double shape_parameter,
                                       const Vector &coefficients);
    GammaRegressionModelConditionalSuf(const Ptr<UnivParams> &alpha,
                                       const Ptr<GlmCoefs> &coefficients);
    GammaRegressionModelConditionalSuf *clone() const override;

    // Returns log likelihood and its derivatives as a function of the
    // concatenated vector [shape_parameter(), coef()].
    double Loglike(const Vector &alpha_beta, Vector &gradient, Matrix &Hessian,
                   uint nderiv) const override;

    void increment_sufficient_statistics(double n, double sum, double sumlog,
                                         const Ptr<VectorData> &predictors);

    int number_of_observations() const override { return dat().size(); }
  };

}  // namespace BOOM

#endif  //  BOOM_GAMMA_REGRESSION_MODEL_HPP_
