// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2012 Steven L. Scott

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

#ifndef BOOM_INDEPENDENT_MVN_MODEL_HPP
#define BOOM_INDEPENDENT_MVN_MODEL_HPP

#include "Models/EmMixtureComponent.hpp"
#include "Models/GaussianModelBase.hpp"
#include "Models/MvnBase.hpp"
#include "Models/Policies/ParamPolicy_1.hpp"
#include "Models/Policies/ParamPolicy_2.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/Policies/SufstatDataPolicy.hpp"
#include "LinAlg/DiagonalMatrix.hpp"
#include "stats/moments.hpp"

namespace BOOM {
  class IndependentMvnSuf : public SufstatDetails<VectorData> {
   public:
    explicit IndependentMvnSuf(int dim);
    IndependentMvnSuf *clone() const override;

    void clear() override;
    void resize(int dim);
    void Update(const VectorData &) override;
    void update_raw(const Vector &y);
    void update_single_dimension(double y, int position);
    void add_mixture_data(const Vector &v, double prob);

    // Increment the sample size, sum, and sum_of_squares by specified amounts.
    void update_expected_value(double sample_size, const Vector &expected_sum,
                               const Vector &expected_sum_of_squares);

    // The sum of observations for the variable in position i.
    double sum(int i) const;

    // The sum of squares of observations for the variable in position i.
    double sumsq(int i) const;  // uncentered sum of squares

    // The centered sum of squared observations for the variable in position i.
    double centered_sumsq(int i, double mu) const;

    // The number of observations in position i.
    double n(int i = 0) const;

    // The sample mean of variable in position i.
    double ybar(int i) const;

    // The sample variance of the variable in position i.
    double sample_var(int i) const;

    IndependentMvnSuf *abstract_combine(Sufstat *s) override;
    void combine(const Ptr<IndependentMvnSuf> &);
    void combine(const IndependentMvnSuf &);
    Vector vectorize(bool minimal = true) const override;
    Vector::const_iterator unvectorize(Vector::const_iterator &v,
                                       bool minimal = true) override;
    Vector::const_iterator unvectorize(const Vector &v,
                                       bool minimal = true) override;
    std::ostream &print(std::ostream &out) const override;

   private:
    std::vector<GaussianSuf> suf_;
  };

  //===========================================================================
  // A base class containing code shared by IndependentMvnModel and
  // ZeroMeanIndependentMvnModel.
  class IndependentMvnBase
      : public MvnBase,
        public SufstatDataPolicy<VectorData, IndependentMvnSuf>,
        virtual public EmMixtureComponent         
  {
   public:
    explicit IndependentMvnBase(int dim);
    IndependentMvnBase * clone() const override = 0;
    
    void add_mixture_data(const Ptr<Data> &dp, double weight) override;

    double Logp(const Vector &x, Vector &g, Matrix &h,
                uint nderivs) const override;

    // mu() is inherited from mvnbase
    const Vector &mu() const override = 0;
    virtual double mu(int i) const {return mu()[i];}
    
    virtual const Vector &sigsq() const = 0;
    virtual double sigsq(int i) const {return sigsq()[i];}
    virtual double sigma(int i) const {return sqrt(sigsq(i));}

    const SpdMatrix &Sigma() const override;
    const SpdMatrix &siginv() const override;
    DiagonalMatrix diagonal_variance() const;
    double ldsi() const override;
    Vector sim(RNG &rng = GlobalRng::rng) const override;

    double pdf(const Data *dp, bool logscale) const override;
    int number_of_observations() const override { return dat().size(); }

   private:
    // Scratch space for computing variance and precision matrices.
    mutable SpdMatrix sigma_scratch_;

    // Scratch space for computing gradients and hessians.
    mutable Vector g_;
    mutable Matrix h_;
  };

  //===========================================================================
  class IndependentMvnModel
      : public IndependentMvnBase,
        public ParamPolicy_2<VectorParams, VectorParams>,
        public PriorPolicy {
   public:
    explicit IndependentMvnModel(int dim);
    IndependentMvnModel(const Vector &mean, const Vector &variance);
    IndependentMvnModel(const IndependentMvnModel &rhs);
    IndependentMvnModel *clone() const override;

    void mle() override;
    
    // Several virtual functions from MvnBase are re-implemented here
    // for efficiency.
    Ptr<VectorParams> Mu_prm() {return prm1();}
    const Ptr<VectorParams> Mu_prm() const {return prm1();}
    const VectorParams &Mu_ref() const {return prm1_ref();}
    const Vector &mu() const override {return Mu_ref().value();}
    void set_mu(const Vector &mu);
    void set_mu_element(double value, int position);

    Ptr<VectorParams> Sigsq_prm() {return prm2();}
    const Ptr<VectorParams> Sigsq_prm() const {return prm2();}
    const VectorParams &Sigsq_ref() const {return prm2_ref();}
    const Vector &sigsq() const override {return Sigsq_ref().value();}
    double sigsq(int i) const override {return sigsq()[i];}
    void set_sigsq(const Vector &sigsq);
    void set_sigsq_element(double sigsq, int position);
  };

  //===========================================================================
  class ZeroMeanIndependentMvnModel
      : public IndependentMvnBase,
        public ParamPolicy_1<VectorParams>,
        public PriorPolicy
  {
   public:
    explicit ZeroMeanIndependentMvnModel(int dim);
    explicit ZeroMeanIndependentMvnModel(const Vector &variance);
    ZeroMeanIndependentMvnModel(const ZeroMeanIndependentMvnModel &rhs);
    ZeroMeanIndependentMvnModel *clone() const override;

    void mle() override;

    const Vector &mu() const override {return zero_;}

    Ptr<VectorParams> Sigsq_prm() {return prm();}
    const Ptr<VectorParams> Sigsq_prm() const {return prm();}
    const VectorParams &Sigsq_ref() const {return prm_ref();}
    const Vector &sigsq() const override {return Sigsq_ref().value();}
    double sigsq(int i) const override {return sigsq()[i];}
    void set_sigsq(const Vector &sigsq) {Sigsq_prm()->set(sigsq);}
    void set_sigsq_element(double sigsq, int position) {
      Sigsq_prm()->set_element(sigsq, position);
    }
    
   private:
    Vector zero_;
    mutable SpdMatrix sigma_scratch_;
    mutable Vector g_;
    mutable Matrix h_;
  };
  
}  // namespace BOOM
#endif  //  BOOM_INDEPENDENT_MVN_MODEL_HPP
