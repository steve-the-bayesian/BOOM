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

#include "Models/ExponentialModel.hpp"
#include <cmath>
#include "Models/GammaModel.hpp"
#include "Models/PosteriorSamplers/ExponentialGammaSampler.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "Models/SufstatAbstractCombineImpl.hpp"
#include "cpputil/math_utils.hpp"
#include "distributions.hpp"

namespace BOOM {

  ExpSuf::ExpSuf() = default;

  ExpSuf::ExpSuf(const ExpSuf &rhs)
      : Sufstat(rhs),
        SufstatDetails<DoubleData>(rhs),
        sum_(rhs.sum_),
        n_(rhs.n_) {}

  ExpSuf *ExpSuf::clone() const { return new ExpSuf(*this); }

  double ExpSuf::sum() const { return sum_; }
  double ExpSuf::n() const { return n_; }

  void ExpSuf::Update(const DoubleData &x) {
    n_ += 1.0;
    sum_ += x.value();
  }

  void ExpSuf::add_mixture_data(double y, double prob) {
    n_ += prob;
    sum_ += y * prob;
  }

  void ExpSuf::clear() { n_ = sum_ = 0; }

  void ExpSuf::combine(const Ptr<ExpSuf> &s) {
    n_ += s->n_;
    sum_ += s->sum_;
  }

  void ExpSuf::combine(const ExpSuf &s) {
    n_ += s.n_;
    sum_ += s.sum_;
  }

  ExpSuf *ExpSuf::abstract_combine(Sufstat *s) {
    return abstract_combine_impl(this, s);
  }

  Vector ExpSuf::vectorize(bool) const {
    Vector ans(2);
    ans[0] = sum_;
    ans[1] = n_;
    return ans;
  }

  Vector::const_iterator ExpSuf::unvectorize(Vector::const_iterator &v, bool) {
    sum_ = *v;
    ++v;
    n_ = *v;
    ++v;
    return v;
  }

  Vector::const_iterator ExpSuf::unvectorize(const Vector &v, bool minimal) {
    Vector::const_iterator it = v.begin();
    return unvectorize(it, minimal);
  }

  std::ostream &ExpSuf::print(std::ostream &out) const {
    return out << n_ << " " << sum_;
  }
  //======================================================================
  using EM = BOOM::ExponentialModel;

  EM::ExponentialModel()
      : ParamPolicy(new UnivParams(1.0)), DataPolicy(new ExpSuf()) {}

  EM::ExponentialModel(double lam)
      : ParamPolicy(new UnivParams(lam)), DataPolicy(new ExpSuf()) {}

  EM::ExponentialModel(const EM &rhs)
      : Model(rhs),
        ParamPolicy(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs),
        DiffDoubleModel(rhs),
        NumOptModel(rhs),
        EmMixtureComponent(rhs) {}

  ExponentialModel *ExponentialModel::clone() const {
    return new ExponentialModel(*this);
  }

  Ptr<UnivParams> EM::Lam_prm() { return ParamPolicy::prm(); }
  const Ptr<UnivParams> EM::Lam_prm() const { return ParamPolicy::prm(); }

  const double &EM::lam() const { return Lam_prm()->value(); }
  void EM::set_lam(double x) { return Lam_prm()->set(x); }

  double ExponentialModel::Loglike(const Vector &lambda_vector, Vector &g,
                                   Matrix &h, uint nd) const {
    if (lambda_vector.size() != 1) {
      report_error("Wrong size argument.");
    }
    double lam = lambda_vector[0];
    double ans = 0;
    if (lam <= 0) {
      ans = negative_infinity();
      if (nd > 0) {
        g[0] = std::max(fabs(lam), .10);
        if (nd > 1) {
          h(0, 0) = -1;
        }
      }
      return ans;
    }

    double n = suf()->n();
    double sum = suf()->sum();
    ans = n * log(lam) - lam * sum;
    if (nd > 0) {
      g[0] = n / lam - sum;
      if (nd > 1) {
        h(0, 0) = -n / (lam * lam);
      }
    }
    return ans;
  }

  void ExponentialModel::mle() {
    double number_of_observations = suf()->n();
    double sum_of_durations = suf()->sum();
    set_lam(number_of_observations / sum_of_durations);
  }

  double ExponentialModel::pdf(const Ptr<Data> &dp, bool logscale) const {
    double ans = logp(DAT(dp)->value());
    return logscale ? ans : exp(ans);
  }

  double ExponentialModel::pdf(const Data *dp, bool logscale) const {
    double ans = logp(DAT(dp)->value());
    return logscale ? ans : exp(ans);
  }

  double ExponentialModel::Logp(double x, double &g, double &h, uint nd) const {
    double lam = this->lam();
    if (lam <= 0) {
      return negative_infinity();
    }
    double ans = x < 0 ? negative_infinity() : log(lam) - lam * x;
    if (nd > 0) {
      if (lam > 0) {
        g = 1.0 / lam - x;
      } else {
        g = 1.0;
      }
      if (nd > 1) {
        if (lam > 0) {
          h = -1.0 / (lam * lam);
        } else {
          h = -1.0;
        }
      }
    }
    return ans;
  }

  double ExponentialModel::sim(RNG &rng) const { return rexp_mt(rng, lam()); }

  void ExponentialModel::add_mixture_data(const Ptr<Data> &dp, double prob) {
    double y = DAT(dp)->value();
    suf()->add_mixture_data(y, prob);
  }
}  // namespace BOOM
