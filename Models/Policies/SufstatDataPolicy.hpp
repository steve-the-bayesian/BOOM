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
#ifndef BOOM_IID_SUFSTAT_DATA_POLICY_HPP
#define BOOM_IID_SUFSTAT_DATA_POLICY_HPP

#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Sufstat.hpp"
#include "cpputil/Ptr.hpp"

namespace BOOM {

  template <class D, class S>
  class SufstatDataPolicy : public IID_DataPolicy<D> {
   public:
    typedef IID_DataPolicy<D> DPBase;
    typedef D DataType;
    typedef SufstatDataPolicy<D, S> DataPolicy;
    typedef std::vector<Ptr<DataType> > DatasetType;

    explicit SufstatDataPolicy(const Ptr<S> &);
    SufstatDataPolicy(const Ptr<S> &, const DatasetType &d);
    template <class FwdIt>
    SufstatDataPolicy(const Ptr<S> &, FwdIt Begin, FwdIt End);
    SufstatDataPolicy(const SufstatDataPolicy &);
    SufstatDataPolicy *clone() const = 0;
    SufstatDataPolicy &operator=(const SufstatDataPolicy &);

    virtual void clear_data();
    void only_keep_sufstats(bool tf = true);
    void clear_data_but_not_sufstats();
    bool is_raw_data_kept() const { return !only_keep_suf_; }

    //---- setting the entire data vector  ----------
    virtual void set_data(const DatasetType &d);

    template <class FwdIt>
    void set_data(FwdIt Beg, FwdIt End);

    // for automatic conversions from raw data types, e.g. double data
    template <class FwdIt>
    void set_data_raw(FwdIt Beg, FwdIt End);

    //------ adding data incrementally
    virtual void add_data(const Ptr<Data> &dp);
    virtual void add_data(const Ptr<DataType> &dp);
    virtual void add_data(DataType *dp) { add_data(Ptr<DataType>(dp)); }

    virtual void combine_data(const Model &, bool just_suf = true);

    const Ptr<S> suf() const { return suf_; }
    Ptr<S> suf() {return suf_;}
    void clear_suf() { suf_->clear(); }
    void update_suf(const Ptr<DataType> &d) { suf_->update(d); }
    void refresh_suf();
    void set_suf(const Ptr<S> &s) { suf_ = s; }

   private:
    Ptr<S> suf_;
    bool only_keep_suf_;
  };
  //======================================================================
  template <class D, class S>
  void SufstatDataPolicy<D, S>::refresh_suf() {
    if (only_keep_suf_) return;
    suf()->clear();
    const DatasetType &d(this->dat());
    for (uint i = 0; i < d.size(); ++i) suf_->update(d[i]);
  }

  template <class D, class S>
  SufstatDataPolicy<D, S>::SufstatDataPolicy(const Ptr<S> &s)
      : DPBase(), suf_(s), only_keep_suf_(false) {}

  template <class D, class S>
  SufstatDataPolicy<D, S>::SufstatDataPolicy(const Ptr<S> &s,
                                             const DatasetType &d)
      : DPBase(d), suf_(s), only_keep_suf_(false) {
    refresh_suf();
  }

  template <class D, class S>
  template <class FwdIt>
  SufstatDataPolicy<D, S>::SufstatDataPolicy(const Ptr<S> &s, FwdIt b, FwdIt e)
      : DPBase(b, e), suf_(s), only_keep_suf_(false) {
    refresh_suf();
  }

  template <class D, class S>
  SufstatDataPolicy<D, S>::SufstatDataPolicy(const SufstatDataPolicy &rhs)
      : Model(rhs),
        DPBase(rhs),
        suf_(rhs.suf_->clone()),
        only_keep_suf_(rhs.only_keep_suf_) {
    refresh_suf();
  }

  template <class D, class S>
  SufstatDataPolicy<D, S> &SufstatDataPolicy<D, S>::operator=(
      const SufstatDataPolicy &rhs) {
    if (&rhs != this) {
      DPBase::operator=(rhs);
      suf_ = rhs.suf_->clone();
      only_keep_suf_ = rhs.only_keep_suf_;
      refresh_suf();
    }
    return *this;
  }

  template <class D, class S>
  void SufstatDataPolicy<D, S>::only_keep_sufstats(bool tf) {
    only_keep_suf_ = tf;
    if (tf) clear_data_but_not_sufstats();
  }

  template <class D, class S>
  void SufstatDataPolicy<D, S>::clear_data() {
    DPBase::clear_data();
    suf()->clear();
  }

  template <class D, class S>
  void SufstatDataPolicy<D, S>::clear_data_but_not_sufstats() {
    DPBase::clear_data();
  }

  template <class D, class S>
  void SufstatDataPolicy<D, S>::set_data(const DatasetType &d) {
    DPBase::set_data(d);
    refresh_suf();
  }

  template <class D, class S>
  template <class Fwd>
  void SufstatDataPolicy<D, S>::set_data(Fwd b, Fwd e) {
    DPBase::set_data(b, e);
    refresh_suf();
  }

  template <class D, class S>
  template <class Fwd>
  void SufstatDataPolicy<D, S>::set_data_raw(Fwd b, Fwd e) {
    DPBase::set_data_raw(b, e);
    refresh_suf();
  }

  template <class D, class S>
  void SufstatDataPolicy<D, S>::add_data(const Ptr<DataType> &d) {
    if (!only_keep_suf_) DPBase::add_data(d);
    // Add data to the vector of pointers in the data policy, but
    // don't update the sufficient statistics if d is missing.
    if (!d->missing()) suf()->update(d);
  }

  template <class D, class S>
  void SufstatDataPolicy<D, S>::add_data(const Ptr<Data> &d) {
    add_data(this->DAT(d));
  }

  template <class D, class S>
  void SufstatDataPolicy<D, S>::combine_data(const Model &other,
                                             bool just_suf) {
    const DataPolicy &m(dynamic_cast<const DataPolicy &>(other));
    suf_->combine(m.suf_);
    if (!just_suf) IID_DataPolicy<D>::combine_data(other, just_suf);
  }

}  // namespace BOOM
#endif  // BOOM_IID_SUFSTAT_DATA_POLICY_HPP
