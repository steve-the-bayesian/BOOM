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

#ifndef EXPONENTIAL_MODEL_H
#define EXPONENTIAL_MODEL_H
#include <iosfwd>
#include "Models/DoubleModel.hpp"
#include "Models/EmMixtureComponent.hpp"
#include "Models/ModelTypes.hpp"
#include "Models/Policies/ParamPolicy_1.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/Policies/SufstatDataPolicy.hpp"
#include "Models/Sufstat.hpp"
#include "cpputil/Ptr.hpp"

namespace BOOM {
  class ExpSuf : public SufstatDetails<DoubleData> {
   public:
    ExpSuf();
    ExpSuf(const ExpSuf &);
    ExpSuf *clone() const override;

    void clear() override;
    void Update(const DoubleData &x) override;
    void add_mixture_data(double y, double prob);
    double sum() const;
    double n() const;
    void combine(const Ptr<ExpSuf> &);
    void combine(const ExpSuf &);
    ExpSuf *abstract_combine(Sufstat *s) override;
    Vector vectorize(bool minimal = true) const override;
    Vector::const_iterator unvectorize(Vector::const_iterator &v,
                                       bool minimal = true) override;
    Vector::const_iterator unvectorize(const Vector &v,
                                       bool minimal = true) override;
    std::ostream &print(std::ostream &out) const override;

   private:
    double sum_, n_;
  };
  //======================================================================
  class GammaModel;
  class ExponentialGammaSampler;

  class ExponentialModel : public ParamPolicy_1<UnivParams>,
                           public SufstatDataPolicy<DoubleData, ExpSuf>,
                           public PriorPolicy,
                           public DiffDoubleModel,
                           public LocationScaleDoubleModel,
                           public NumOptModel,
                           public EmMixtureComponent {
   public:
    ExponentialModel();
    explicit ExponentialModel(double lam);
    ExponentialModel(const ExponentialModel &rhs);
    ExponentialModel *clone() const override;

    Ptr<UnivParams> Lam_prm();
    const Ptr<UnivParams> Lam_prm() const;
    const double &lam() const;
    void set_lam(double);

    double mean() const override { return 1.0 / lam(); }

    double variance() const override {
      double m = mean();
      return m * m;
    }

    // probability calculations
    double pdf(const Ptr<Data> &dp, bool logscale) const override;
    double pdf(const Data *dp, bool logscale) const override;
    double Loglike(const Vector &lambda_vector, Vector &g, Matrix &h,
                   uint nd) const override;
    double Logp(double x, double &g, double &h, uint nd) const override;
    void mle() override;

    double sim(RNG &rng = GlobalRng::rng) const override;
    int number_of_observations() const override { return dat().size(); }
    void add_mixture_data(const Ptr<Data> &, double prob) override;
  };
}  // namespace BOOM
#endif  // EXPONENTIALMODEL_H
