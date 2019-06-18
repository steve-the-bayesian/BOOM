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
#ifndef BOOM_BETA_MODEL_HPP
#define BOOM_BETA_MODEL_HPP

#include "Models/DoubleModel.hpp"
#include "Models/ModelTypes.hpp"
#include "Models/ParamTypes.hpp"
#include "Models/Policies/ParamPolicy_2.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/Policies/SufstatDataPolicy.hpp"
#include "Models/Sufstat.hpp"

namespace BOOM {
  class BetaSuf : public SufstatDetails<DoubleData> {
   public:
    BetaSuf();
    BetaSuf(const BetaSuf &rhs);
    BetaSuf *clone() const override;
    void clear() override { n_ = sumlog_ = sumlogc_ = 0.0; }
    void Update(const DoubleData &) override;
    void update_raw(double p);
    double n() const { return n_; }
    double sumlog() const { return sumlog_; }
    double sumlogc() const { return sumlogc_; }
    BetaSuf *abstract_combine(Sufstat *s) override;
    void combine(const Ptr<BetaSuf> &s);
    void combine(const BetaSuf &s);
    std::ostream &print(std::ostream &out) const override;

    Vector vectorize(bool minimal = true) const override;
    Vector::const_iterator unvectorize(Vector::const_iterator &v,
                                       bool minimal = true) override;
    Vector::const_iterator unvectorize(const Vector &v,
                                       bool minimal = true) override;

   private:
    double n_, sumlog_, sumlogc_;
  };

  class BetaModel : public ParamPolicy_2<UnivParams, UnivParams>,
                    public SufstatDataPolicy<DoubleData, BetaSuf>,
                    public PriorPolicy,
                    public NumOptModel,
                    public DiffDoubleModel,
                    public LocationScaleDoubleModel {
   public:
    // Initialize with the prior number of "successes" (a) and
    // "failures" (b).  This is the usual parameterization of the Beta
    // model.
    explicit BetaModel(double a = 1.0, double b = 1.0);

    // Initialize the Beta model with a mean and a sample size.  In
    // the standard parameterization, the mean maps to a/(a+b) and the
    // sample size is (a+b).
    BetaModel(double mean, double sample_size, int);

    BetaModel(const BetaModel &rhs);

    BetaModel *clone() const override;

    Ptr<UnivParams> Alpha();
    Ptr<UnivParams> Beta();
    const Ptr<UnivParams> Alpha() const;
    const Ptr<UnivParams> Beta() const;

    const double &a() const;
    const double &b() const;
    void set_a(double alpha);
    void set_b(double beta);
    void set_params(double a, double b);

    // An alternative parameterization:
    //        mean = a/(a+b)
    // sample_size = (a+b)
    double mean() const override;
    void set_mean(double a_over_a_plus_b);
    double variance() const override;
    double sample_size() const;
    void set_sample_size(double a_plus_b);

    // probability calculations
    double Loglike(const Vector &ab, Vector &g, Matrix &h,
                   uint nd) const override;
    double log_likelihood(double a, double b) const;
    using LoglikeModel::log_likelihood;

    double Logp(double x, double &d1, double &d2, uint nd) const override;
    double sim(RNG &rng = GlobalRng::rng) const override;

    int number_of_observations() const override { return dat().size(); }

   private:
    double Logp_degenerate(double x, double &d1, double &d2, uint nd) const;
  };

  double beta_log_likelihood(double a, double b, const BetaSuf &);

}  // namespace BOOM

#endif  // BOOM_BETA_MODEL_HPP
