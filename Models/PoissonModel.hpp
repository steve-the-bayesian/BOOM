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

#ifndef POISSON_MODEL_H
#define POISSON_MODEL_H

#include "Models/ModelTypes.hpp"
#include "Models/Policies/ParamPolicy_1.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/Policies/SufstatDataPolicy.hpp"
#include "Models/Sufstat.hpp"

//----------------------------------------------------------------------//
namespace BOOM {

  class PoissonSuf : public SufstatDetails<IntData> {
   public:
    // constructor
    PoissonSuf();

    // If this constructor is used, then the normalizing constant will
    // not be correctly set.  That's probably okay for most
    // applications.
    PoissonSuf(double event_count, double exposure);

    PoissonSuf(const PoissonSuf &rhs);
    PoissonSuf *clone() const override;

    // If this function is used to set the value of the sufficient
    // statistics, then the normalizing constant will be wrong, which
    // is probably fine for most applications.
    void set(double event_count, double exposure);

    // Adds increment_event_count to the event count (sum_), and
    // incremental_exposure to exposure (n_).
    void add_incremental_counts(double incremental_event_count,
                                double incremental_exposure);

    void clear() override;
    double sum() const;
    double n() const;
    double lognc() const;

    void Update(const IntData &dat) override;
    void add_mixture_data(double y, double prob);
    void combine(const Ptr<PoissonSuf> &);
    void combine(const PoissonSuf &);
    PoissonSuf *abstract_combine(Sufstat *s) override;

    Vector vectorize(bool minimal = true) const override;
    Vector::const_iterator unvectorize(Vector::const_iterator &v,
                                       bool minimal = true) override;
    Vector::const_iterator unvectorize(const Vector &v,
                                       bool minimal = true) override;
    std::ostream &print(std::ostream &out) const override;

   private:
    double sum_;    // event_count
    double n_;      // exposure
    double lognc_;  // log nc is the log product of x-factorials
  };
  //----------------------------------------------------------------------//

  class PoissonModel : public ParamPolicy_1<UnivParams>,
                       public SufstatDataPolicy<IntData, PoissonSuf>,
                       public PriorPolicy,
                       public NumOptModel,
                       public IntModel,
                       virtual public MixtureComponent {
   public:
    explicit PoissonModel(double lam = 1.0);
    explicit PoissonModel(const std::vector<uint> &);
    PoissonModel(const PoissonModel &m);
    PoissonModel *clone() const override;

    void mle() override;
    double Loglike(const Vector &lambda, Vector &g, Matrix &h,
                   uint nd) const override;

    Ptr<UnivParams> Lam();
    const Ptr<UnivParams> Lam() const;
    double lam() const;
    void set_lam(double);

    // probability calculations
    virtual double pdf(const Ptr<Data> &dp, bool logscale) const;
    double pdf(const Data *x, bool logscale) const override;
    double pdf(uint x, bool logscale) const;
    double logp(int x) const override;
    int number_of_observations() const override { return dat().size(); }

    // moments and summaries:
    double mean() const;
    double var() const;
    double sd() const;
    double sim(RNG &rng = GlobalRng::rng) const;

    virtual void add_mixture_data(const Ptr<Data> &, double prob);
  };

}  // namespace BOOM
#endif  // POISSON_MODEL_H
