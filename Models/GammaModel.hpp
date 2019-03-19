// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005 Steven L. Scott

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

#ifndef BOOM_GAMMA_MODEL_HPP
#define BOOM_GAMMA_MODEL_HPP

#include <iosfwd>
#include "Models/DoubleModel.hpp"
#include "Models/EmMixtureComponent.hpp"
#include "Models/ModelTypes.hpp"
#include "Models/Policies/ParamPolicy_2.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/Policies/SufstatDataPolicy.hpp"
#include "Models/Sufstat.hpp"
#include "cpputil/Ptr.hpp"

//======================================================================
namespace BOOM {
  class GammaSuf : public SufstatDetails<DoubleData> {
   public:
    GammaSuf();
    GammaSuf *clone() const override;

    void set(double sum, double sumlog, double n);
    void clear() override;
    void Update(const DoubleData &dat) override;
    void update_raw(double x);
    void add_mixture_data(double y, double prob);

    // Add the given sufficient components to the sufficient statistics.
    void increment(double n, double sum, double sumlog);

    double sum() const;
    double sumlog() const;
    double n() const;
    std::ostream &display(std::ostream &out) const override;

    virtual void combine(const Ptr<GammaSuf> &s);
    virtual void combine(const GammaSuf &s);
    GammaSuf *abstract_combine(Sufstat *s) override;
    Vector vectorize(bool minimal = true) const override;
    Vector::const_iterator unvectorize(Vector::const_iterator &v,
                                       bool minimal = true) override;
    Vector::const_iterator unvectorize(const Vector &v,
                                       bool minimal = true) override;
    std::ostream &print(std::ostream &out) const override;

   private:
    // Sum of the observations.
    double sum_;

    // Sum of the logs of the observations.
    double sumlog_;

    // Number of observations.
    double n_;
  };

  //======================================================================
  class GammaModelBase  // Gamma Model, Chi-Square Model, Scaled Chi-Square
      : public SufstatDataPolicy<DoubleData, GammaSuf>,
        public DiffDoubleModel,
        public LocationScaleDoubleModel,
        public NumOptModel,
        public EmMixtureComponent {
   public:
    GammaModelBase();
    GammaModelBase *clone() const override = 0;

    virtual double alpha() const = 0;
    virtual double beta() const = 0;
    double mean() const override;
    double variance() const override;

    void add_mixture_data(const Ptr<Data> &, double prob) override;
    double pdf(const Ptr<Data> &dp, bool logscale) const override;
    double pdf(const Data *dp, bool logscale) const override;
    int number_of_observations() const override { return dat().size(); }

    double Logp(double x, double &g, double &h, uint nd) const override;
    double sim(RNG &rng = GlobalRng::rng) const override;

    //  p(1/sigsq) = beta^alpha * (1/sigsq)^(alpha - 1) * exp(-beta/sigsq)
    //               -----------------------------------------------------
    //                                    Gamma(alpha)
    //
    // p(sigsq) = beta^alpha * (1/sigsq)^(alpha + 1) * exp(-beta / sigsq)
    //               -----------------------------------------------------
    //                                    Gamma(alpha)
    //
    // This function returns log(p(sigsq)).  If gradient is non-NULL then it is
    // filled with the derivative with respect to sigsq.  If the gradient _and_
    // hessian are both non-NULL then hessian is filled with the second
    // derivative with respect to sigsq.
    double logp_reciprocal(double sigsq, double *gradient = nullptr,
                           double *hessian = nullptr) const;
  };
  //======================================================================

  class GammaModel : public GammaModelBase,
                     public ParamPolicy_2<UnivParams, UnivParams>,
                     public PriorPolicy {
   public:
    // The usual parameterization of the Gamma distribution a =
    // shape, b = scale, mean = a/b.
    explicit GammaModel(double a = 1.0, double b = 1.0);

    // To initialize a GammaModel with shape (a) and mean parameters,
    // simply include a third argument that is an int.
    GammaModel(double shape, double mean, int);

    GammaModel *clone() const override;

    Ptr<UnivParams> Alpha_prm();
    Ptr<UnivParams> Beta_prm();
    const Ptr<UnivParams> Alpha_prm() const;
    const Ptr<UnivParams> Beta_prm() const;

    double alpha() const override;
    double beta() const override;
    void set_alpha(double);
    void set_beta(double);

    // Three different ways to set parameters, depending on the
    // parameterization.
    void set_shape_and_scale(double a, double b);
    void set_shape_and_mean(double a, double mean);
    void set_mean_and_scale(double mean, double b);

    double mean() const override;

    // probability calculations
    double Loglike(const Vector &shape_scale, Vector &gradient, Matrix &hessian,
                   uint number_of_derivatives) const override;
    double loglikelihood(double shape, double scale) const;
    double loglikelihood(const Vector &ab, Vector *gradient,
                         Matrix *hessian) const;
    void mle() override;
  };

}  // namespace BOOM

#endif  // GAMMA_MODEL_H
