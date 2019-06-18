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

  typedef MarkovData MD;
  typedef CategoricalData CD;

  MD::MarkovData(uint val, uint Nlev) : CD(val, Nlev) { clear_links(); }

  MD::MarkovData(uint val, const Ptr<CatKeyBase> &key) : CD(val, key) {
    clear_links();
  }

  MD::MarkovData(uint val, const Ptr<MarkovData> &last) : CD(val, last->key()) {
    set_prev(last);
    last->set_next(this);
  }

  MD::MarkovData(const std::string &value, const Ptr<CatKey> &key)
      : CD(value, key) {}

  MD::MarkovData(const MarkovData &rhs, bool copy_links) : Data(rhs), CD(rhs) {
    if (copy_links) {
      links = rhs.links;
    } else {
      clear_links();
    }
  }

  MD *MD::create() const {
    return new MD(*this, false);
  }  // does not copy links

  MD *MD::clone() const { return new MD(*this, true); }  // copies links

  void MD::unset_prev() { links.unset_prev(); }
  void MD::unset_next() { links.unset_next(); }
  void MD::clear_links() { links.clear_links(); }

  MD *MD::prev() const { return links.prev(); }
  MD *MD::next() const { return links.next(); }

  void MD::set_prev(const Ptr<MD> &p) { links.set_prev(p); }
  void MD::set_next(const Ptr<MD> &p) { links.set_next(p); }

  std::ostream &MD::display(std::ostream &out) const {
    return CategoricalData::display(out);
  }

  //------------------------------------------------------------
  typedef MarkovDataSeries MDS;
  typedef TimeSeries<MarkovData> TS;

  template <class STRING_OR_UINT>
  Ptr<MarkovDataSeries> make_markov_data_series(
      const std::vector<uint> &raw_data, const Ptr<CatKeyBase> &key) {
    NEW(MarkovData, last)(raw_data[0], key);
    uint n = raw_data.size();
    std::vector<Ptr<MarkovData>> dvec;
    dvec.reserve(n);
    dvec.push_back(last);
    for (uint i = 1; i < n; ++i) {
      NEW(MarkovData, dp)(raw_data[i], last);
      dvec.push_back(dp);
      last = dp;
    }
    return new TimeSeries<MarkovData>(dvec, false);
  }

  Ptr<MarkovDataSeries> make_markov_data(const std::vector<uint> &raw_data) {
    int max = *max_element(raw_data.begin(), raw_data.end());
    Ptr<CatKeyBase> key = new FixedSizeIntCatKey(max + 1);
    std::vector<Ptr<MarkovData>> data;
    for (int i = 0; i < raw_data.size(); ++i) {
      NEW(MarkovData, dp)(raw_data[i], key);
    }
    return new TimeSeries<MarkovData>(data, true);
  }

  Ptr<MarkovDataSeries> make_markov_data(
      const std::vector<std::string> &raw_data) {
    if (raw_data.empty()) return nullptr;
    Ptr<CatKey> key = make_catkey(raw_data);
    std::vector<Ptr<MarkovData>> data;
    for (int i = 0; i < raw_data.size(); ++i) {
      NEW(MarkovData, dp)(raw_data[i], key);
    }
    return new TimeSeries<MarkovData>(data, true);
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

  void MarkovSuf::Update(const MD &dat) {
    MD *prev = dat.prev();
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
    MD *prev = dp->prev();
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

  typedef TransitionProbabilityMatrix TPM;
  typedef MatrixParams MP;

  TPM::TransitionProbabilityMatrix(uint S) : MP(S, S, 1.0 / S) {}

  TPM::TransitionProbabilityMatrix(const Matrix &M) : MP(M) {}

  TPM::TransitionProbabilityMatrix(const TPM &rhs)
      : Data(rhs), Params(rhs), MP(rhs) {}

  TPM *TPM::clone() const { return new TPM(*this); }

  Vector::const_iterator TPM::unvectorize(Vector::const_iterator &v,
                                          bool minimal) {
    Vector::const_iterator ans = MP::unvectorize(v, minimal);
    notify();
    return ans;
  }

  Vector::const_iterator TPM::unvectorize(const Vector &v, bool minimal) {
    Vector::const_iterator ans = MP::unvectorize(v, minimal);
    notify();
    return ans;
  }

  void TPM::set(const Matrix &m, bool) {
    MP::set(m);
    notify();
  }

  void TPM::add_observer(const Ptr<VectorParams> &vp) const {
    observers.insert(vp);
  }

  void TPM::delete_observer(const Ptr<VectorParams> &vp) const {
    observers.erase(vp);
  }

  void TPM::notify() const {
    for (const auto &el : observers) {
      el->set(get_stat_dist(value()));
    }
  }

  //------------------------------------------------------------
  MarkovModel::MarkovModel(uint StateSize)
      : ParamPolicy(new TPM(StateSize), new VectorParams(StateSize)),
        DataPolicy(new MarkovSuf(StateSize)),
        PriorPolicy(),
        LoglikeModel() {
    fix_pi0_uniform();
  }

  MarkovModel::MarkovModel(const Matrix &Q)
      : ParamPolicy(new TPM(Q), new VectorParams(Q.nrow())),
        DataPolicy(new MarkovSuf(Q.nrow())) {
    fix_pi0_uniform();
  }

  MarkovModel::MarkovModel(const Matrix &Q, const Vector &Pi0)
      : ParamPolicy(new TPM(Q), new VectorParams(Pi0)),
        DataPolicy(new MarkovSuf(Q.nrow())) {}

  template <class T>
  uint number_of_unique_elements(const std::vector<T> &v) {
    std::set<T> s(v.begin(), v.end());
    return s.size();
  }

  MarkovModel::MarkovModel(const std::vector<uint> &idata)
      : DataPolicy(new MarkovSuf(number_of_unique_elements(idata))) {
    uint S = suf()->state_space_size();
    NEW(TPM, Q1)(S);
    NEW(VectorParams, Pi0)(S);
    ParamPolicy::set_params(Q1, Pi0);

    Ptr<MarkovDataSeries> ts = make_markov_data(idata);
    add_data_series(ts);
    mle();
  }

  MarkovModel::MarkovModel(const std::vector<std::string> &sdata)
      : DataPolicy(new MarkovSuf(number_of_unique_elements(sdata))) {
    uint S = suf()->state_space_size();
    NEW(TPM, Q1)(S);
    NEW(VectorParams, Pi0)(S);
    ParamPolicy::set_params(Q1, Pi0);

    Ptr<MarkovDataSeries> ts = make_markov_data(sdata);
    add_data_series(ts);
    mle();
  }

  MarkovModel::MarkovModel(const MarkovModel &rhs)
      : Model(rhs),
        DataInfoPolicy(rhs),
        ParamPolicy(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs),
        LoglikeModel(rhs),
        EmMixtureComponent(rhs),
        pi0_status(rhs.pi0_status) {}

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
      Ptr<MarkovDataSeries> dpn = dp.dcast<MarkovDataSeries>();
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

    const MarkovDataSeries *dp2 = dynamic_cast<const MarkovDataSeries *>(dp);
    if (dp2) return pdf(*dp2, logscale);
    BadMarkovData();
    return 0;
  }

  double MarkovModel::pdf(const MarkovData &dat, bool logscale) const {
    double ans;
    if (!!dat.prev()) {
      MD *prev = dat.prev();
      ans = Q(prev->value(), dat.value());
    } else
      ans = pi0(dat.value());
    return logscale ? safelog(ans) : ans;
  }

  double MarkovModel::pdf(const MarkovDataSeries &dat, bool logscale) const {
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

    if (pi0_status == Free) {
      const Vector &tmp(suf()->init());
      set_pi0(tmp / sum(tmp));
    } else if (pi0_status == Stationary) {
      set_pi0(get_stat_dist(Q));
    }
  }

  double MarkovModel::loglike(const Vector &serialized_params) const {
    const Vector &icount(suf()->init());
    const Matrix &tcount(suf()->trans());

    int S = state_space_size();
    TransitionProbabilityMatrix transition_probabilities(S);

    Vector logpi0(log(pi0()));
    Matrix logQ(log(Q()));

    double ans = icount.dot(logpi0);
    ans += el_mult_sum(tcount, logQ);
    return ans;
  }

  Vector MarkovModel::stat_dist() const { return get_stat_dist(Q()); }

  void MarkovModel::fix_pi0(const Vector &Pi0) {
    set_pi0(Pi0);
    pi0_status = Known;
  }
  void MarkovModel::fix_pi0_uniform() {
    uint S = state_space_size();
    set_pi0(Vector(S, 1.0 / S));
    pi0_status = Uniform;
  }
  void MarkovModel::fix_pi0_stationary() {
    // need to observe Q and change when it changes
    Q_prm()->add_observer(Pi0_prm());
    set_pi0(stat_dist());
    pi0_status = Stationary;
  }

  void MarkovModel::free_pi0() {
    if (pi0_status == Stationary) {
      Q_prm()->delete_observer(Pi0_prm());
    }
    pi0_status = Free;
  }

  uint MarkovModel::state_space_size() const { return Q().nrow(); }

  Ptr<TPM> MarkovModel::Q_prm() { return ParamPolicy::prm1(); }

  const Ptr<TPM> MarkovModel::Q_prm() const { return ParamPolicy::prm1(); }

  const Matrix &MarkovModel::Q() const { return Q_prm()->value(); }
  void MarkovModel::set_Q(const Matrix &Q) const { Q_prm()->set(Q); }
  double MarkovModel::Q(uint i, uint j) const { return Q()(i, j); }

  Ptr<VectorParams> MarkovModel::Pi0_prm() { return ParamPolicy::prm2(); }
  const Ptr<VectorParams> MarkovModel::Pi0_prm() const {
    return ParamPolicy::prm2();
  }

  const Vector &MarkovModel::pi0() const { return Pi0_prm()->value(); }

  void MarkovModel::set_pi0(const Vector &pi0) { Pi0_prm()->set(pi0); }

  double MarkovModel::pi0(int i) const { return pi0()(i); }

  bool MarkovModel::pi0_fixed() const { return pi0_status != Free; }

  void MarkovModel::resize(uint S) {
    suf()->resize(S);
    set_pi0(Vector(S, 1.0 / S));
    set_Q(Matrix(S, S, 1.0 / S));
  }

  //______________________________________________________________________

  void MarkovModel::add_mixture_data(const Ptr<Data> &dp, double prob) {
    suf()->add_mixture_data(DAT_1(dp), prob);
  }
}  // namespace BOOM
