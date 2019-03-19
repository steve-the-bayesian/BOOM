// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2007 Steven L. Scott

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
#ifndef BOOM_UNIFORM_MODEL_HPP
#define BOOM_UNIFORM_MODEL_HPP

#include <vector>
#include "Models/DoubleModel.hpp"
#include "Models/ModelTypes.hpp"
#include "Models/Policies/ParamPolicy_2.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/Policies/SufstatDataPolicy.hpp"
#include "Models/Sufstat.hpp"

namespace BOOM {

  class UniformSuf : public SufstatDetails<DoubleData> {
   public:
    UniformSuf();
    explicit UniformSuf(const std::vector<double> &d);
    UniformSuf(double low, double high);
    UniformSuf(const UniformSuf &rhs);
    UniformSuf *clone() const override;

    void clear() override;
    void Update(const DoubleData &d) override;
    void update_raw(double x);

    double lo() const;
    double hi() const;

    void set_lo(double a);
    void set_hi(double b);
    void combine(const Ptr<UniformSuf> &s);
    void combine(const UniformSuf &);
    UniformSuf *abstract_combine(Sufstat *s) override;

    Vector vectorize(bool minimal = true) const override;
    Vector::const_iterator unvectorize(Vector::const_iterator &v,
                                       bool minimal = true) override;
    Vector::const_iterator unvectorize(const Vector &v,
                                       bool minimal = true) override;
    std::ostream &print(std::ostream &out) const override;

   private:
    double lo_, hi_;
  };

  class UniformModel : public ParamPolicy_2<UnivParams, UnivParams>,
                       public SufstatDataPolicy<DoubleData, UniformSuf>,
                       public PriorPolicy,
                       public DiffDoubleModel,
                       public LocationScaleDoubleModel,
                       public LoglikeModel {
   public:
    explicit UniformModel(double a = 0, double b = 1);
    explicit UniformModel(const std::vector<double> &data);
    UniformModel(const UniformModel &rhs);
    UniformModel *clone() const override;

    double lo() const;
    double hi() const;
    double nc() const;  // 1.0/(hi - lo);
    void set_lo(double a);
    void set_hi(double b);
    void set_ab(double a, double b);

    double mean() const override;
    double variance() const override;

    Ptr<UnivParams> LoParam();
    Ptr<UnivParams> HiParam();
    const Ptr<UnivParams> LoParam() const;
    const Ptr<UnivParams> HiParam() const;
    double Logp(double x, double &g, double &h, uint nd) const override;
    double loglike(const Vector &support_lower_and_upper_limits) const override;
    void mle() override;
    double sim(RNG &rng = GlobalRng::rng) const override;
    int number_of_observations() const override { return dat().size(); }
  };
}  // namespace BOOM
#endif  // BOOM_UNIFORM_MODEL_HPP;
