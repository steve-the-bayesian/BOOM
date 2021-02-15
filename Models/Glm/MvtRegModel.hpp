// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2007 Steven L. Scott

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
#ifndef BOOM_MVT_REG_HPP
#define BOOM_MVT_REG_HPP
#include "Models/Glm/Glm.hpp"  // for MvRegData
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/ParamPolicy_3.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/SpdParams.hpp"
#include "cpputil/Ptr.hpp"

namespace BOOM {
  class MvtRegModel
      : public ParamPolicy_3<MatrixGlmCoefs, SpdParams, UnivParams>,
        public IID_DataPolicy<MvRegData>,
        public PriorPolicy,
        public LoglikeModel {
   public:
    MvtRegModel(uint xdim, uint ydim);
    MvtRegModel(const Matrix &X, const Matrix &Y, bool add_intercept = false);
    MvtRegModel(const Matrix &B, const SpdMatrix &Sigma, double nu);

    MvtRegModel(const MvtRegModel &rhs);
    MvtRegModel *clone() const override;

    uint xdim() const;  // x includes intercept
    uint ydim() const;

    const Matrix &Beta() const;  // [xdim rows, ydim columns]
    const SpdMatrix &Sigma() const;
    const SpdMatrix &Siginv() const;
    double ldsi() const;
    double nu() const;

    Ptr<MatrixGlmCoefs> Beta_prm();
    Ptr<SpdParams> Sigma_prm();
    Ptr<UnivParams> Nu_prm();
    const Ptr<MatrixGlmCoefs> Beta_prm() const;
    const Ptr<SpdParams> Sigma_prm() const;
    const Ptr<UnivParams> Nu_prm() const;

    void set_Beta(const Matrix &B);
    void set_beta(const Vector &b, uint m);
    void set_Sigma(const SpdMatrix &V);
    void set_Siginv(const SpdMatrix &iV);

    void set_nu(double nu);

    //--- estimation and probability calculations
    void mle() override;
    double loglike(
        const Vector &beta_columns_siginv_triangle_nu) const override;
    virtual double pdf(const Ptr<Data> &, bool) const;
    virtual Vector predict(const Vector &x) const;

    //---- simulate MV regression data ---
    virtual MvRegData *sim(RNG &rng = GlobalRng::rng) const;
    virtual MvRegData *sim(const Vector &X, RNG &rng = GlobalRng::rng) const;
    Vector simulate_fake_x(RNG &rng = GlobalRng::rng) const;  // no intercept
  };
}  // namespace BOOM

#endif  // BOOM_MVT_REG_HPP
