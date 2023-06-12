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

#ifndef BOOM_MARKOV_MODEL_HPP
#define BOOM_MARKOV_MODEL_HPP

#include <vector>
#include "uint.hpp"

#include "Models/CategoricalData.hpp"
#include "Models/EmMixtureComponent.hpp"
#include "Models/ModelTypes.hpp"
#include "Models/ParamTypes.hpp"

#include "Models/Policies/ParamPolicy_2.hpp"
#include "Models/Policies/PriorPolicy.hpp"

#include "Models/TimeSeries/TimeSeriesSufstatDataPolicy.hpp"
#include "Models/TimeSeries/TimeSeries.hpp"

#include "Models/Sufstat.hpp"

#include "LinAlg/Matrix.hpp"
#include "LinAlg/Vector.hpp"

namespace BOOM {

  //====================================================================
  // MarkovData is a CategoricalData node in a linked list.
  class MarkovData : public CategoricalData {
   public:

    // Clear links to neighboring data points before destruction.
    ~MarkovData() {}

    // Create MarkovData that needs to be linked to other data.
    MarkovData(uint val, uint Nlevels);
    explicit MarkovData(uint val, const Ptr<CatKeyBase> &key);
    explicit MarkovData(const std::string &value, const Ptr<CatKey> &key);

    // Create MarkovData.  The use of an object for the second argument (instead
    // of a const ref) is intentional.
    explicit MarkovData(uint val, Ptr<MarkovData> last);

    // Copies of MarkovData will need to reset links to neighbors.
    MarkovData(const MarkovData &);

    MarkovData *clone() const override;  // copies links

    // Links to neighbors.
    MarkovData *prev() {return prev_;}
    const MarkovData *prev() const {return prev_;}
    MarkovData *next() {return next_;}
    const MarkovData *next() const {return next_;}

    // Args:
    //   next, prev:  Links to the next or previous data point.
    //   reciprocate: If true then also set the link in the reverse dicrection.
    void set_prev(MarkovData *prev, bool reciprocate = true);
    void set_next(MarkovData *next, bool reciprocate = true);

    // Unlink *this from neighboring observations.  This also adjusts the links
    // in the neighbors.
    void clear_links();
    void unset_prev();
    void unset_next();

    std::ostream &display(std::ostream &) const override;

   private:
    // Data stored as raw pointers to avoid ownership cycles.
    MarkovData *prev_;
    MarkovData *next_;
  };
  //=====================================================================
  // Create series of MarkovData from a vector of uint's or strings.
  Ptr<TimeSeries<MarkovData>> make_markov_data(
      const std::vector<uint> &raw_data);

  Ptr<TimeSeries<MarkovData>> make_markov_data(
      const std::vector<std::string> &raw_data);

  // Copy constructor for TimeSeries<MarkovData> copies underlying data and
  // resets links.
  template<>
  inline TimeSeries<MarkovData>::TimeSeries(const TimeSeries<MarkovData> &rhs)
      : Data(rhs), std::vector<Ptr<MarkovData>>() {
    reserve(rhs.size());
    for (size_t i = 0; i < rhs.size(); ++i) {
      Ptr<MarkovData> dp = rhs[i]->clone();
      if (i > 0) {
        dp->set_prev(back().get());
      }
      push_back(dp);
    }
  }

  // Assignment operator for TimeSeries<MarkovData> copies underlying data and
  // resets links.
  template <>
  inline TimeSeries<MarkovData> &TimeSeries<MarkovData>::operator=(
      const TimeSeries<MarkovData> &rhs) {
    if (&rhs != this) {
      this->clear();
      this->reserve(rhs.size());
      for (size_t i = 0; i < rhs.size(); ++i) {
        Ptr<MarkovData> dp = rhs[i]->clone();
        if (i > 0) {
          dp->set_prev(this->back().get());
        }
        this->push_back(dp);
      }
    }
    return *this;
  }

  //=====================================================================
  const bool debug_markov_update_suf(false);
  class MarkovSuf
      : public TimeSeriesSufstatDetails<MarkovData, TimeSeries<MarkovData>> {
   public:
    explicit MarkovSuf(uint S);
    MarkovSuf(const MarkovSuf &sf);
    MarkovSuf *clone() const override;

    uint state_space_size() const { return trans().nrow(); }
    void resize(uint p);
    void clear() override {
      trans_ = 0.0;
      init_ = 0.0;
    }
    void Update(const MarkovData &) override;
    void add_mixture_data(const Ptr<MarkovData> &, double prob);
    void add_transition_distribution(const Matrix &P);
    void add_initial_distribution(const Vector &pi);
    void add_transition(uint from, uint to);
    void add_initial_value(uint val);
    const Matrix &trans() const { return trans_; }
    const Vector &init() const { return init_; }
    std::ostream &print(std::ostream &) const override;
    void combine(const Ptr<MarkovSuf> &);
    void combine(const MarkovSuf &);
    MarkovSuf *abstract_combine(Sufstat *s) override;

    Vector vectorize(bool minimal = true) const override;
    Vector::const_iterator unvectorize(Vector::const_iterator &v,
                                       bool minimal = true) override;
    Vector::const_iterator unvectorize(const Vector &v,
                                       bool minimal = true) override;

   private:
    Matrix trans_;  // transition counts
    Vector init_;   // initial count, typically one 1 and rest 0's
  };
  //=====================================================================
  std::ostream &operator<<(std::ostream &out, const Ptr<MarkovSuf> &sf);
  //=====================================================================

  //------ observer classes ------------------
  class MatrixRowsObserver {
   public:
    typedef std::vector<Ptr<VectorParams> > Rows;
    explicit MatrixRowsObserver(Rows &);
    void operator()(const Matrix &);

   private:
    Rows &rows;
  };

  class StationaryDistObserver {
   public:
    explicit StationaryDistObserver(const Ptr<VectorParams> &);
    void operator()(const Matrix &);

   private:
    Ptr<VectorParams> stat;
  };

  class RowObserver {
   public:
    RowObserver(const Ptr<MatrixParams> &M, uint I);
    void operator()(const Vector &v);

   private:
    Ptr<MatrixParams> mp;
    Matrix m;
    uint i;
  };

  //======================================================================

  class ProductDirichletModel;
  class DirichletModel;
  class MarkovConjSampler;

  class MarkovModel
      : public ParamPolicy_2<MatrixParams, VectorParams>,
        public TimeSeriesSufstatDataPolicy<MarkovData,
                                           TimeSeries<MarkovData>,
                                           MarkovSuf>,
        public PriorPolicy,
        public LoglikeModel,
        public EmMixtureComponent {
   public:
    typedef MarkovData DataPointType;
    typedef TimeSeries<MarkovData> DataSeriesType;

    // Initialize model parameters to the uniform distribution.
    explicit MarkovModel(uint state_size);

    // Initialize the transition probability matrix to Q.  Fix the initial state
    // distribution to uniform.
    explicit MarkovModel(const Matrix &Q);

    // Set the transition probability matrix to Q, and the initial state
    // distribution to pi0.
    MarkovModel(const Matrix &Q, const Vector &pi0);

    explicit MarkovModel(const std::vector<uint> &);
    explicit MarkovModel(const std::vector<std::string> &);

    MarkovModel(const MarkovModel &rhs);
    MarkovModel *clone() const override;

    void fix_pi0(const Vector &Pi0);
    void fix_pi0_stationary();
    bool pi0_fixed() const;

    double pdf(const Ptr<Data> &dp, bool logscale) const;
    double pdf(const Data *dp, bool logscale) const override;
    double pdf(const Ptr<DataPointType> &dp, bool logscale) const;
    double pdf(const Ptr<DataSeriesType> &dp, bool logscale) const;
    double pdf(const DataPointType &dat, bool logscale) const;
    double pdf(const DataSeriesType &dat, bool logscale) const;

    int number_of_observations() const override { return dat().size(); }

    void add_mixture_data(const Ptr<Data> &, double prob) override;

    uint state_space_size() const;

    Ptr<MatrixParams> Q_prm();
    const Ptr<MatrixParams> Q_prm() const;
    virtual const Matrix &Q() const;
    virtual void set_Q(const Matrix &Q) const;
    double Q(uint, uint) const;
    double log_transition_probability(int from, int to) const;
    const Matrix &log_transition_probabilities() const;

    Ptr<VectorParams> Pi0_prm();
    const Ptr<VectorParams> Pi0_prm() const;
    virtual const Vector &pi0() const;
    void set_pi0(const Vector &pi0);
    double pi0(int) const;

    void mle() override;

    // The argument is a Vector with S^2 - 1 elements, where S is the
    // state space size.  The final S-1 are the initial distribution.
    // The first S * (S-1) elements are the first (S-1) columns of the
    // transition probability matrix.  The final
    double loglike(const Vector &serialized_params) const override;
    Vector stat_dist() const;

   protected:
    virtual void resize(uint S);

   private:
    // An observer to be placed on the transition probability matrix Q.  When Q
    // changes it flips the flag so that we know that
    // log_transition_probabilities_ needs to be refreshed.
    void observe_transition_probabilities();

    // Refresh the log_transition_probabilities_, if needed.
    void ensure_log_probabilities_are_current() const;

    Ptr<MarkovData> dpp;  // data point prototype

    // How should the stationary distribution be treated:
    //   Free: It is a free parameter to be estimated.
    //   Stationary: It is the stationary distribution of the transition
    //     probability matrix.
    //   Known:  It is set to known values.
    enum InitialDistributionStrategy { Free, Stationary, Known };
    InitialDistributionStrategy initial_distribution_status_;

    // If the initial distribution is set to be the stationary distribution of Q
    // then the following workspace is needed to ensure its current value.
    mutable bool pi0_current_;
    mutable Vector pi0_workspace_;

    // The log of the transition probability matrix Q, and a flag to keep the
    // two in sync.
    mutable bool log_transition_probabilities_current_;
    mutable Matrix log_transition_probabilities_;
  };

}  // namespace BOOM

#endif  // MARKOV_MODEL_H
