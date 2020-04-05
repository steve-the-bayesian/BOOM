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

#ifndef MVT_MODEL_H
#define MVT_MODEL_H

#include "Models/ModelTypes.hpp"
#include "Models/Policies/CompositeParamPolicy.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/ScaledChisqModel.hpp"
#include "Models/SpdParams.hpp"
#include "Models/VectorModel.hpp"
#include "Models/WeightedMvnModel.hpp"
#include "distributions/rng.hpp"

namespace BOOM {
  class ScaledChisqModel;
  class WeightedMvnModel;

  class MvtModel : public CompositeParamPolicy,
                   public IID_DataPolicy<VectorData>,
                   public PriorPolicy,
                   public LatentVariableModel,
                   public LoglikeModel,
                   public LocationScaleVectorModel,
                   public MixtureComponent {
   public:
    explicit MvtModel(uint p, double mu = 0.0, double sig = 1.0,
                      double nu = 30.0);
    MvtModel(const Vector &mean, const SpdMatrix &Var, double Nu);
    MvtModel(const MvtModel &m);

    MvtModel *clone() const override;

    void initialize_params() override;

    Ptr<VectorParams> Mu_prm();
    Ptr<SpdParams> Sigma_prm();
    Ptr<UnivParams> Nu_prm();

    const Ptr<VectorParams> Mu_prm() const;
    const Ptr<SpdParams> Sigma_prm() const;
    const Ptr<UnivParams> Nu_prm() const;

    int dim() const;
    const Vector &mu() const override;
    const SpdMatrix &Sigma() const override;
    const SpdMatrix &siginv() const override;
    Matrix Sigma_chol() const;
    double ldsi() const override;
    double nu() const;

    void set_mu(const Vector &);
    void set_Sigma(const SpdMatrix &);
    void set_siginv(const SpdMatrix &);
    void set_nu(double);

    double logp(const Vector &x) const override;

    double pdf(const Data *dp, bool logscale) const override;
    double pdf(const VectorData *dp, bool logscale) const;
    double pdf(const Vector &x, bool logscale) const;
    int number_of_observations() const override { return dat().size(); }

    void add_data(const Ptr<Data> &) override;
    void add_data(const Ptr<VectorData> &) override;

    void mle() override;  // ECME
    double loglike(const Vector &mu_siginv_triangle_nu) const override;
    void impute_latent_data(RNG &rng) override;
    void Estep();  // E step for EM/ECME

    virtual double complete_data_loglike() const;
    Vector sim(RNG &rng = GlobalRng::rng) const override;

   private:
    void Impute(bool sample, RNG &rng = GlobalRng::rng);
    Ptr<WeightedMvnModel> mvn;
    Ptr<ScaledChisqModel> wgt;
  };

}  // namespace BOOM

#endif  // MVT_MODEL_H
