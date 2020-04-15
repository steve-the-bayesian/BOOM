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

#include "Models/MarkovModel.hpp"
#include <cmath>
#include <iostream>
#include "LinAlg/Matrix.hpp"
#include "LinAlg/VectorView.hpp"
#include "Models/DirichletModel.hpp"
#include "Models/PosteriorSamplers/MarkovConjSampler.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "Models/ProductDirichletModel.hpp"
#include "Models/SufstatAbstractCombineImpl.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"
#include "distributions/Markov.hpp"

namespace BOOM {


  MarkovData::MarkovData(uint val, uint Nlev)
      : CategoricalData(val, Nlev),
        prev_(nullptr),
        next_(nullptr) {}

  MarkovData::MarkovData(uint val, const Ptr<CatKeyBase> &key)
      : CategoricalData(val, key),
        prev_(nullptr),
        next_(nullptr) {}

  MarkovData::MarkovData(uint val, Ptr<MarkovData> last)
      : CategoricalData(val, last->key()) {
    set_prev(last.get());
  }

  MarkovData::MarkovData(const std::string &value, const Ptr<CatKey> &key)
      : CategoricalData(value, key) {}

  MarkovData::MarkovData(const MarkovData &rhs) :
      Data(rhs), CategoricalData(rhs) {
      clear_links();
  }

  MarkovData *MarkovData::clone() const {
    return new MarkovData(*this);
  }

  void MarkovData::set_prev(MarkovData *prev, bool reciprocate) {
    prev_ = prev;
    if (prev && reciprocate) {
      prev->set_next(this, false);
    }
  }

  void MarkovData::set_next(MarkovData *next, bool reciprocate) {
    next_ = next;
    if (next && reciprocate) {
      next->set_prev(this, false);
    }
  }

  void MarkovData::unset_prev() {
    if (!!prev_) {
      prev_->unset_next();
    }
    prev_ = nullptr;
  }

  void MarkovData::unset_next() {
    if (!!next_) {
      next_->unset_prev();
    }
    next_ = nullptr;
  }

  void MarkovData::clear_links() {
    unset_prev();
    unset_next();
  }

  std::ostream &MarkovData::display(std::ostream &out) const {
    return CategoricalData::display(out);
  }

  //------------------------------------------------------------
  Ptr<TimeSeries<MarkovData>> make_markov_data(
      const std::vector<uint> &raw_data) {
    int max = *max_element(raw_data.begin(), raw_data.end());
    NEW(TimeSeries<MarkovData>, series)();
    series->reserve(raw_data.size());
    for (int i = 0; i < raw_data.size(); ++i) {
      if (i > 0) {
        Ptr<MarkovData> prev = series->back();
        NEW(MarkovData, dp)(raw_data[i], prev);
        series->push_back(dp);
      } else {
        // nlevels = max + 1
        NEW(MarkovData, dp)(raw_data[i], max + 1);
        series->push_back(dp);
      }
    }
    return series;
  }

  Ptr<TimeSeries<MarkovData>> make_markov_data(
      const std::vector<std::string> &raw_data) {
    if (raw_data.empty()) return nullptr;
    Ptr<CatKey> key = make_catkey(raw_data);
    NEW(TimeSeries<MarkovData>, series)();
    for (int i = 0; i < raw_data.size(); ++i) {
        NEW(MarkovData, dp)(raw_data[i], key);
        if (i > 0) {
          dp->set_prev(series->back().get());
        }
        series->push_back(dp);
    }
    return series;
  }

  //------------------------------------------------------------
  std::ostream &operator<<(std::ostream &out, const Ptr<MarkovSuf> &sf) {
    out << "markov initial counts:" << endl
        << sf->init() << endl
        << " transition counts:" << endl
        << sf->trans() << endl;
    return out;
  }

  MarkovSuf::MarkovSuf(uint S) : trans_(S, S, 0.0), init_(S, 0.0) {}

  MarkovSuf::MarkovSuf(const MarkovSuf &rhs)
      : Sufstat(rhs), SufTraits(rhs), trans_(rhs.trans_), init_(rhs.init_) {}

  MarkovSuf *MarkovSuf::clone() const { return new MarkovSuf(*this); }

  void MarkovSuf::Update(const MarkovData &dat) {
    const MarkovData *prev = dat.prev();
    if (!prev)
      init_(dat.value()) += 1;
    else {
      int oldx = prev->value();
      int newx = dat.value();
      trans_(oldx, newx) += 1;
    }
  }

  void MarkovSuf::add_transition_distribution(const Matrix &P) { trans_ += P; }

  void MarkovSuf::add_initial_distribution(const Vector &pi) { init_ += pi; }

  void MarkovSuf::add_transition(uint from, uint to) { ++trans_(from, to); }

  void MarkovSuf::add_initial_value(uint h) { ++init_[h]; }

  void MarkovSuf::add_mixture_data(const Ptr<MarkovData> &dp, double prob) {
    uint now = dp->value();
    MarkovData *prev = dp->prev();
    if (!prev)
      init_(now) += prob;
    else {
      uint then = prev->value();
      trans_(then, now) += prob;
    }
  }

  std::ostream &MarkovSuf::print(std::ostream &out) const {
    trans_.write(out, false);
    out << " ";
    init_.write(out, true);
    return out;
  }

  void MarkovSuf::resize(uint p) {
    if (state_space_size() != p) {
      trans_ = Matrix(p, p, 0.0);
      init_ = Vector(p, 0.0);
    }
  }

  void MarkovSuf::combine(const Ptr<MarkovSuf> &s) {
    trans_ += s->trans_;
    init_ += s->init_;
  }
  void MarkovSuf::combine(const MarkovSuf &s) {
    trans_ += s.trans_;
    init_ += s.init_;
  }

  MarkovSuf *MarkovSuf::abstract_combine(Sufstat *s) {
    return abstract_combine_impl(this, s);
  }

  Vector MarkovSuf::vectorize(bool) const {
    Vector ans(trans_.begin(), trans_.end());
    ans.concat(init_);
    return ans;
  }

  Vector::const_iterator MarkovSuf::unvectorize(Vector::const_iterator &v,
                                                bool) {
    uint d = trans_.nrow();
    Matrix tmp(v, v + d * d, d, d);
    trans_ = tmp;
    v += d * d;
    init_.assign(v, v + d);
    v += d;
    return v;
  }

  Vector::const_iterator MarkovSuf::unvectorize(const Vector &v, bool minimal) {
    Vector::const_iterator it = v.begin();
    return unvectorize(it, minimal);
  }

  //------------------------------------------------------------

  typedef MatrixRowsObserver MRO;

  MRO::MatrixRowsObserver(Rows &r) : rows(r) {}

  void MRO::operator()(const Matrix &m) {
    uint n = m.nrow();
    assert(rows.size() == n);
    Vector x;
    for (uint i = 0; i < n; ++i) {
      x = m.row(i);
      rows[i]->set(x, false);
    }
  }

  //------------------------------------------------------------
  typedef StationaryDistObserver SDO;

  SDO::StationaryDistObserver(const Ptr<VectorParams> &p) : stat(p) {}

  void SDO::operator()(const Matrix &m) {
    Vector x = get_stat_dist(m);
    stat->set(x);
  }

  //------------------------------------------------------------

  RowObserver::RowObserver(const Ptr<MatrixParams> &M, uint I) : mp(M), i(I) {
    m = mp->value();
  }

  void RowObserver::operator()(const Vector &v) {
    assert(v.size() == m.ncol());
    m = mp->value();
    std::copy(v.begin(), v.end(), m.row_begin(i));
    mp->set(m, false);
  }

  //======================================================================

  MarkovModel::MarkovModel(uint state_size)
      : ParamPolicy(new MatrixParams(state_size, state_size),
                    new VectorParams(state_size)),
        DataPolicy(new MarkovSuf(state_size)),
        PriorPolicy(),
        LoglikeModel(),
        log_transition_probabilities_current_(false)
  {
    fix_pi0(Vector(state_size, 1.0 / state_size));
    Matrix transition_probabilities = Q();
    for (uint s = 0; s < state_size; ++s) {
      transition_probabilities.row(s) = pi0();
    }
    set_Q(transition_probabilities);
  }

  MarkovModel::MarkovModel(const Matrix &Q)
      : MarkovModel(Q, Vector(Q.nrow(), 1.0 / Q.nrow())) {
    fix_pi0(pi0());
  }

  MarkovModel::MarkovModel(const Matrix &Q, const Vector &Pi0)
      : ParamPolicy(new MatrixParams(Q), new VectorParams(Pi0)),
        DataPolicy(new MarkovSuf(Q.nrow())),
        log_transition_probabilities_current_(false){}

  template <class T>
  uint number_of_unique_elements(const std::vector<T> &v) {
    std::set<T> s(v.begin(), v.end());
    return s.size();
  }

  MarkovModel::MarkovModel(const std::vector<uint> &idata)
      : DataPolicy(new MarkovSuf(number_of_unique_elements(idata))) {
    uint S = suf()->state_space_size();
    NEW(MatrixParams, Q1)(S, S);
    NEW(VectorParams, Pi0)(S);
    ParamPolicy::set_params(Q1, Pi0);

    Ptr<TimeSeries<MarkovData>> ts = make_markov_data(idata);
    add_data_series(ts);
    mle();
  }

  MarkovModel::MarkovModel(const std::vector<std::string> &sdata)
      : DataPolicy(new MarkovSuf(number_of_unique_elements(sdata))) {
    uint S = suf()->state_space_size();
    NEW(MatrixParams, Q1)(S, S);
    NEW(VectorParams, Pi0)(S);
    ParamPolicy::set_params(Q1, Pi0);

    Ptr<TimeSeries<MarkovData>> ts = make_markov_data(sdata);
    add_data_series(ts);
    mle();
  }

  MarkovModel::MarkovModel(const MarkovModel &rhs)
      : Model(rhs),
        ParamPolicy(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs),
        LoglikeModel(rhs),
        EmMixtureComponent(rhs),
        initial_distribution_status_(rhs.initial_distribution_status_) {}

  MarkovModel *MarkovModel::clone() const { return new MarkovModel(*this); }

  double MarkovModel::pdf(const Ptr<DataPointType> &dp, bool logscale) const {
    double ans = 0;
    if (!!dp->prev()) {
      ans = Q(dp->prev()->value(), dp->value());
    } else
      ans = pi0(dp->value());
    return logscale ? safelog(ans) : ans;
  }

  inline void BadMarkovData() {
    report_error("Bad data type passed to MarkovModel::pdf");
  }

  double MarkovModel::pdf(const Ptr<Data> &dp, bool logscale) const {
    Ptr<MarkovData> dp1 = dp.dcast<MarkovData>();
    double ans = 0;
    if (!!dp1)
      ans = pdf(*dp1, logscale);
    else {
      Ptr<TimeSeries<MarkovData>> dpn = dp.dcast<TimeSeries<MarkovData>>();
      if (!!dpn)
        ans = pdf(*dpn, logscale);
      else
        BadMarkovData();
    }
    return ans;
  }

  double MarkovModel::pdf(const Data *dp, bool logscale) const {
    const MarkovData *dp1 = dynamic_cast<const MarkovData *>(dp);
    if (dp1) return pdf(*dp1, logscale);

    const TimeSeries<MarkovData> *dp2 =
        dynamic_cast<const TimeSeries<MarkovData> *>(dp);
    if (dp2) return pdf(*dp2, logscale);
    BadMarkovData();
    return 0;
  }

  double MarkovModel::pdf(const MarkovData &dat, bool logscale) const {
    double ans;
    if (!!dat.prev()) {
      const MarkovData *prev = dat.prev();
      ans = Q(prev->value(), dat.value());
    } else
      ans = pi0(dat.value());
    return logscale ? safelog(ans) : ans;
  }

  double MarkovModel::pdf(const TimeSeries<MarkovData> &dat,
                          bool logscale) const {
    double ans = 0.0;
    for (uint i = 0; i != dat.length(); ++i) {
      ans += pdf(*(dat[i]), true);
    }
    return logscale ? ans : exp(ans);
  }

  void MarkovModel::mle() {
    Matrix Q(this->Q());
    for (uint i = 0; i < Q.nrow(); ++i) {
      Vector tmp(suf()->trans().row(i));
      Q.set_row(i, tmp / tmp.sum());
    }
    set_Q(Q);

    if (initial_distribution_status_ == Free) {
      const Vector &tmp(suf()->init());
      set_pi0(tmp / sum(tmp));
    } else if (initial_distribution_status_ == Stationary) {
      set_pi0(get_stat_dist(Q));
    }
  }

  double MarkovModel::loglike(const Vector &serialized_params) const {
    const Vector &initial_state_count(suf()->init());
    const Matrix &transition_counts(suf()->trans());

    Vector logpi0(log(pi0()));

    double ans = initial_state_count.dot(logpi0);
    ans += el_mult_sum(transition_counts,
                       log_transition_probabilities());
    return ans;
  }

  Vector MarkovModel::stat_dist() const { return get_stat_dist(Q()); }

  void MarkovModel::fix_pi0(const Vector &Pi0) {
    set_pi0(Pi0);
    initial_distribution_status_ = Known;
  }

  void MarkovModel::fix_pi0_stationary() {
    Q_prm()->add_observer(
        [this]() {
          this->set_pi0(this->stat_dist());
        });
    initial_distribution_status_ = Stationary;
  }

  uint MarkovModel::state_space_size() const { return Q().nrow(); }

  Ptr<MatrixParams> MarkovModel::Q_prm() { return ParamPolicy::prm1(); }

  const Ptr<MatrixParams> MarkovModel::Q_prm() const {
    return ParamPolicy::prm1();
  }

  const Matrix &MarkovModel::Q() const { return Q_prm()->value(); }
  void MarkovModel::set_Q(const Matrix &Q) const { Q_prm()->set(Q); }
  double MarkovModel::Q(uint i, uint j) const { return Q()(i, j); }

  const Matrix &MarkovModel::log_transition_probabilities() const {
    ensure_log_probabilities_are_current();
    return log_transition_probabilities_;
  }

  double MarkovModel::log_transition_probability(int from, int to) const {
    ensure_log_probabilities_are_current();
    return log_transition_probabilities_(from, to);
  }

  Ptr<VectorParams> MarkovModel::Pi0_prm() { return ParamPolicy::prm2(); }
  const Ptr<VectorParams> MarkovModel::Pi0_prm() const {
    return ParamPolicy::prm2();
  }

  const Vector &MarkovModel::pi0() const { return Pi0_prm()->value(); }

  void MarkovModel::set_pi0(const Vector &pi0) { Pi0_prm()->set(pi0); }

  double MarkovModel::pi0(int i) const { return pi0()(i); }

  bool MarkovModel::pi0_fixed() const {
    return initial_distribution_status_ != Free;
  }

  void MarkovModel::resize(uint S) {
    suf()->resize(S);
    set_pi0(Vector(S, 1.0 / S));
    set_Q(Matrix(S, S, 1.0 / S));
  }

  void MarkovModel::observe_transition_probabilities() {
    log_transition_probabilities_current_ = false;
  }

  void MarkovModel::ensure_log_probabilities_are_current() const {
    if (!log_transition_probabilities_current_) {
      log_transition_probabilities_ = log(Q());
      log_transition_probabilities_current_ = true;
    }
  }

  //______________________________________________________________________

  void MarkovModel::add_mixture_data(const Ptr<Data> &dp, double prob) {
    suf()->add_mixture_data(DAT_1(dp), prob);
  }
}  // namespace BOOM
