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
#ifndef BOOM_MULTINOMIAL_MODEL_HPP
#define BOOM_MULTINOMIAL_MODEL_HPP
#include "Models/CategoricalData.hpp"
#include "Models/ModelTypes.hpp"
#include "Models/ParamTypes.hpp"
#include "Models/Policies/ParamPolicy_1.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/Policies/SufstatDataPolicy.hpp"
#include "Models/Sufstat.hpp"

namespace BOOM {

  class MultinomialSuf : public SufstatDetails<CategoricalData> {
   public:
    explicit MultinomialSuf(uint p);
    explicit MultinomialSuf(const Vector &counts);
    MultinomialSuf(const MultinomialSuf &rhs);
    MultinomialSuf *clone() const override;

    void Update(const CategoricalData &d) override;
    void add_mixture_data(uint y, double prob);
    void add_mixture_data(const Vector &weights);
    void update_raw(uint k);
    void clear() override;

    const Vector &n() const;
    uint dim() const;
    void combine(const Ptr<MultinomialSuf> &);
    void combine(const MultinomialSuf &);
    MultinomialSuf *abstract_combine(Sufstat *s) override;

    Vector vectorize(bool minimal = true) const override;
    Vector::const_iterator unvectorize(Vector::const_iterator &v,
                                       bool minimal = true) override;
    Vector::const_iterator unvectorize(const Vector &v,
                                       bool minimal = true) override;
    std::ostream &print(std::ostream &out) const override;

   private:
    Vector counts_;
  };

  //======================================================================
  class MultinomialModel
      : public ParamPolicy_1<VectorParams>,
        public SufstatDataPolicy<CategoricalData, MultinomialSuf>,
        public PriorPolicy,
        public LoglikeModel,
        public MixtureComponent {
   public:
    explicit MultinomialModel(uint Nlevels);
    explicit MultinomialModel(const Vector &probs);

    // The argument is a vector of names to use for factor levels to
    // be modeled.
    explicit MultinomialModel(const std::vector<std::string> &);

    explicit MultinomialModel(const MultinomialSuf &suf);

    MultinomialModel(const MultinomialModel &rhs);
    MultinomialModel *clone() const override;

    uint nlevels() const;
    uint dim() const;

    Ptr<VectorParams> Pi_prm();
    const Ptr<VectorParams> Pi_prm() const;

    // If 'minimal' then the first element of pi is omitted.
    Vector vectorize_params(bool minimal = true) const override;
    void unvectorize_params(const Vector &v, bool minimal = true) override;

    const double &pi(int s) const;
    const Vector &pi() const;
    const Vector &logpi() const;
    void set_pi(const Vector &probs);

    // Return sum(pi * logpi)
    double entropy() const;

    double loglike(const Vector &probs) const override;
    double log_likelihood() const override { return loglike(pi()); }
    void mle() override;
    double pdf(const Data *dp, bool logscale) const override;
    double pdf(const Ptr<Data> &dp, bool logscale) const;
    void add_mixture_data(const Ptr<Data> &, double prob);
    int number_of_observations() const override { return suf()->n().sum(); }

    uint sim(RNG &rng = GlobalRng::rng) const;

   private:
    mutable Vector logp_;
    mutable bool logp_current_;
    void set_observer();
    void check_logp() const;

  };

}  // namespace BOOM
#endif  // BOOM_MULTINOMIAL_MODEL_HPP
