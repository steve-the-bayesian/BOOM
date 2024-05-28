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
#include "Models/MultinomialModel.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "Models/SufstatAbstractCombineImpl.hpp"

#include "distributions.hpp"

#include <cmath>
#include <stdexcept>

namespace BOOM {
  typedef MultinomialSuf MS;
  MS::MultinomialSuf(const uint p) : counts_(p, 0.0) {}

  MS::MultinomialSuf(const Vector &counts) : counts_(counts) {
    if (counts.min() < 0.0) {
      report_error("All elements of counts must be non-negative.");
    }
  }

  MS::MultinomialSuf(const MultinomialSuf &rhs)
      : Sufstat(rhs),
        SufstatDetails<CategoricalData>(rhs),
        counts_(rhs.counts_)
  {}

  MS *MS::clone() const { return new MS(*this); }

  void MS::Update(const CategoricalData &d) {
    uint i = d.value();
    while (i >= counts_.size()) {
      counts_.push_back(0);  // counts_ grows when needed
    }
    ++counts_[i];
  }

  void MS::add_mixture_data(uint y, double prob) { counts_[y] += prob; }
  void MS::add_mixture_data(const Vector &weights) { counts_ += weights; }
  void MS::update_raw(uint k) { ++counts_[k]; }
  void MS::clear() { counts_ = 0.0; }
  const Vector &MS::n() const { return counts_; }
  uint MS::dim() const { return counts_.size(); }
  void MS::combine(const Ptr<MS> &s) { counts_ += s->counts_; }
  void MS::combine(const MS &s) { counts_ += s.counts_; }

  MultinomialSuf *MS::abstract_combine(Sufstat *s) {
    return abstract_combine_impl(this, s);
  }
  Vector MS::vectorize(bool) const { return counts_; }

  Vector::const_iterator MS::unvectorize(Vector::const_iterator &v, bool) {
    uint dim = counts_.size();
    counts_.assign(v, v + dim);
    v += dim;
    return v;
  }

  Vector::const_iterator MS::unvectorize(const Vector &v, bool minimal) {
    Vector::const_iterator it = v.begin();
    return unvectorize(it, minimal);
  }

  std::ostream &MS::print(std::ostream &out) const { return out << counts_; }
  //======================================================================

  typedef MultinomialModel MM;
  typedef std::vector<std::string> StringVector;

  MM::MultinomialModel(uint p)
      : ParamPolicy(new VectorParams(p, 1.0 / p)),
        DataPolicy(new MS(p)),
        PriorPolicy(),
        logp_current_(false) {
    set_observer();
  }

  uint count_levels(const StringVector &sv) {
    std::set<std::string> s;
    for (uint i = 0; i < sv.size(); ++i) s.insert(sv[i]);
    return s.size();
  }

  MM::MultinomialModel(const Vector &probs)
      : ParamPolicy(new VectorParams(probs)),
        DataPolicy(new MS(probs.size())),
        PriorPolicy(),
        logp_current_(false) {
    set_observer();
  }

  MM::MultinomialModel(const StringVector &names)
      : ParamPolicy(new VectorParams(1)),
        DataPolicy(new MS(1)),
        PriorPolicy(),
        logp_current_(false) {
    std::vector<Ptr<CategoricalData>> data_vector(
        create_categorical_data(names));

    uint nlevels = data_vector[0]->nlevels();
    Vector probs(nlevels, 1.0 / nlevels);
    set_pi(probs);

    set_data(data_vector);
    mle();
    set_observer();
  }

  MM::MultinomialModel(const MultinomialSuf &suf)
      : ParamPolicy(new VectorParams(suf.dim())),
        DataPolicy(new MS(suf)),
        PriorPolicy(),
        logp_current_(false) {
    set_observer();
    mle();
  }

  MM::MultinomialModel(const MM &rhs)
      : Model(rhs),
        ParamPolicy(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs),
        LoglikeModel(rhs),
        MixtureComponent(rhs),
        logp_(rhs.logp_),
        logp_current_(false) {
    set_observer();
  }

  MM *MM::clone() const { return new MM(*this); }

  uint MM::nlevels() const { return pi().size(); }
  uint MM::dim() const { return pi().size(); }

  Ptr<VectorParams> MM::Pi_prm() { return ParamPolicy::prm(); }
  const Ptr<VectorParams> MM::Pi_prm() const { return ParamPolicy::prm(); }

  Vector MM::vectorize_params(bool minimal) const {
    const Vector &prob(pi());
    if (minimal) {
      return Vector(ConstVectorView(prob, 1));
    } else {
      return prob;
    }
  }

  void MM::unvectorize_params(const Vector &v, bool minimal) {
    if (minimal) {
      set_pi(concat(v.sum(), v));
    } else {
      set_pi(v);
    }
  }

  const double &MM::pi(int s) const { return pi()[s]; }
  const Vector &MM::pi() const { return Pi_prm()->value(); }
  const Vector &MM::logpi() const {
    check_logp();
    return logp_;
  }

  void MM::set_pi(const Vector &probs) {
    Pi_prm()->set(probs);
    check_logp();
  }

  double MM::entropy() const {
    double ans = pi().dot(logpi());
    if (!std::isnan(ans)) {
      return ans;
    } else {
      Selector inc(dim(), true);
      const Vector &probs = pi();
      for (int i = 0; i < probs.size(); ++i) {
        if (std::isfinite(probs[i])) {
          inc.add(i);
        } else {
          inc.drop(i);
        }
      }
      if (inc.empty()) {
        report_error("There are no finite elements of pi().");
      }
      return inc.select(pi()).dot(inc.select(logpi()));
    }
  }

  double MM::loglike(const Vector &probs) const {
    double ans(0.0);
    const Vector &n(suf()->n());
    for (uint i = 0; i < dim(); ++i) ans += n[i] * log(probs[i]);
    return ans;
  }

  void MM::mle() {
    const Vector &n(suf()->n());
    double tot = sum(n);
    if (tot == 0) {
      Vector probs(dim(), 1.0 / dim());
      set_pi(probs);
      return;
    }
    set_pi(n / tot);
  }

  double MM::pdf(const Data *dp, bool logscale) const {
    check_logp();
    uint i = DAT(dp)->value();
    if (i >= dim()) {
      std::string msg = "too large a value passed to MultinomialModel::pdf";
      report_error(msg);
    }
    return logscale ? logp_[i] : pi(i);
  }

  double MM::pdf(const Ptr<Data> &dp, bool logscale) const {
    check_logp();
    uint i = DAT(dp)->value();
    if (i >= dim()) {
      std::string msg = "too large a value passed to MultinomialModel::pdf";
      report_error(msg);
    }
    return logscale ? logp_[i] : pi(i);
  }

  uint MM::sim(RNG &rng) const { return rmulti_mt(rng, pi()); }

  void MM::add_mixture_data(const Ptr<Data> &dp, double prob) {
    uint i = DAT(dp)->value();
    suf()->add_mixture_data(i, prob);
  }

  void MM::check_logp() const {
    if (logp_current_) return;
    logp_ = log(pi());
    logp_current_ = true;
  }

  void MM::set_observer() {
    Pi_prm()->add_observer(this, [this]() { this->logp_current_ = false;});
  }


}  // namespace BOOM
