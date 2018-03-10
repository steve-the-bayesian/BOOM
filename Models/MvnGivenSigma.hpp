// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2006 Steven L. Scott

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
#ifndef BOOM_MVN_GIVEN_SIGMA_HPP
#define BOOM_MVN_GIVEN_SIGMA_HPP

#include "Models/MvnBase.hpp"
#include "Models/Policies/ParamPolicy_2.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/Policies/SufstatDataPolicy.hpp"

namespace BOOM {

  class MvnGivenSigma : public MvnBase,
                        public LoglikeModel,
                        public ParamPolicy_2<VectorParams, UnivParams>,
                        public SufstatDataPolicy<VectorData, MvnSuf>,
                        public PriorPolicy {
   public:
    //  This model is y | mu, kappa Sigma ~ N(mu, Sigma/kappa)

    // Sigma is viewed as a fixed constant, not a parameter of the
    // model.  This model is intended for use as a prior for
    // regression coefficients, where Siginv is proportional to XTX.

    // This model can also be a conjugate prior for the mean of an
    // MvnModel, where Sigma is a parameter of that model (but not of
    // this class).

    // comment and changes made on 4/4/2008.  This might break earlier
    // code e.g. for hierarchical models with same mean but different
    // Sigma?

    // See also MvnGivenX.  These priors are similar.  Use this one if
    // X is going to remain fixed, and MvnGivenX if X will change.
    MvnGivenSigma(const Vector &mu, double kappa);
    MvnGivenSigma(const Vector &mu, double kappa, const SpdMatrix &Siginv);
    MvnGivenSigma(const Vector &mu, double kappa, const Ptr<SpdData> &Sigma);
    MvnGivenSigma(const Ptr<VectorParams> &mu, const Ptr<UnivParams> &kappa);
    MvnGivenSigma(const Ptr<VectorParams> &mu, const Ptr<UnivParams> &kappa,
                  const Ptr<SpdData> &Sigma);

    MvnGivenSigma *clone() const override;

    void set_Sigma(const Ptr<SpdData> &Sigma);
    void set_Sigma(const SpdMatrix &V, bool ivar = false);

    Ptr<VectorParams> Mu_prm();
    const Ptr<VectorParams> Mu_prm() const;
    Ptr<UnivParams> Kappa_prm();
    const Ptr<UnivParams> Kappa_prm() const;

    uint dim() const override;
    const Vector &mu() const override;
    double kappa() const;

    void set_mu(const Vector &);
    void set_kappa(double k);
    void mle() override;
    // The argument is a vector with the mean vector first, and then
    // the proportionalyity facctor 'kappa' second.
    double loglike(const Vector &mu_kappa) const override;
    double pdf(const Ptr<Data> &dp, bool) const;
    double pdf(const Ptr<DataType> &dp, bool) const;

    double Logp(const Vector &x, Vector &g, Matrix &h, uint nd) const override;
    Vector sim(RNG &rng = GlobalRng::rng) const override;

    // overloads required to conform with the MvnBase interface The
    // 'Sigma' here refers to the vaiance, not the Sigma paramter.
    // i.e. Sigma() returns  Sigma/kappa

    const SpdMatrix &Sigma() const override;
    const SpdMatrix &siginv() const override;
    double ldsi() const override;

   private:
    Ptr<SpdData> Sigma_;
    void check_Sigma() const;
    mutable SpdMatrix S;
  };

}  // namespace BOOM
#endif  // BOOM_MVN_GIVEN_SIGMA_HPP
