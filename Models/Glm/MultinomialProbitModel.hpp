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

#ifndef BOOM_MULTINOMIAL_PROBIT_MODEL_HPP
#define BOOM_MULTINOMIAL_PROBIT_MODEL_HPP

#include "Models/Glm/ChoiceData.hpp"
#include "Models/Glm/Glm.hpp"  // for GlmCoefs
#include "Models/Glm/MultivariateRegression.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/ParamPolicy_2.hpp"
#include "Models/Policies/PriorPolicy.hpp"

namespace BOOM {
  class TrunMvnTF;
  class MultinomialProbitModel : public ParamPolicy_2<GlmCoefs, SpdParams>,
                                 public IID_DataPolicy<ChoiceData>,
                                 public PriorPolicy,
                                 public LatentVariableModel {
   public:
    typedef std::vector<Ptr<CategoricalData> > ResponseVec;
    enum ImputationMethod { Slice, Gibbs };

    // each column of beta_subject corresponds to a different choice.
    MultinomialProbitModel(const Matrix &beta_subject,
                           const Vector &beta_choice,
                           const SpdMatrix &utility_covariance);

    //     // the function create_categorical_data can make a ResponseVector
    //     // out of a vector of strings or uints
    //     MultinomialProbitModel(ResponseVector responses,
    //                        const Matrix &Xsubject_info,
    //                        const Arr3 &Xchoice_info);
    //     // dim(Xchoice_info) = [#obs, #choices, #choice x's]

    //     MultinomialProbitModel(ResponseVector responses,    // no choice
    //     information
    //                        const Matrix &Xsubject_info);

    explicit MultinomialProbitModel(const std::vector<Ptr<ChoiceData> > &);
    MultinomialProbitModel(const MultinomialProbitModel &rhs);
    MultinomialProbitModel *clone() const override;

    void use_slice_sampling() { imp_method = Slice; }
    void use_Gibbs_sampling() { imp_method = Gibbs; }
    void impute_latent_data(RNG &rng) override;
    virtual double complete_data_loglike() const;

    double pdf(const Ptr<Data> &dp, bool logscale) const;
    double pdf(const Ptr<ChoiceData> &dp, bool logscale) const;
    virtual void initialize_params();

    const Vector &beta() const;
    Vector beta_subject(uint choice) const;
    Vector beta_choice() const;

    const SpdMatrix &Sigma() const;
    const SpdMatrix &siginv() const;
    double ldsi() const;

    // eta is the value of the linear predictor when evaluated at X
    Vector eta(const Ptr<ChoiceData> &dp) const;
    Vector &eta(const Ptr<ChoiceData> &dp, Vector &ans) const;

    uint n() const;
    uint xdim() const;
    uint subject_nvars() const;
    uint choice_nvars() const;
    uint Nchoices() const;

    void set_beta(const Vector &b);
    void set_included_coefficients(const Vector &b);
    void set_Sigma(const SpdMatrix &Sig);
    void set_siginv(const SpdMatrix &siginv);

    Ptr<GlmCoefs> Beta_prm() { return ParamPolicy::prm1(); }
    const Ptr<GlmCoefs> Beta_prm() const { return ParamPolicy::prm1(); }
    Ptr<SpdParams> Sigma_prm() { return ParamPolicy::prm2(); }
    const Ptr<SpdParams> Sigma_prm() const { return ParamPolicy::prm2(); }

    const SpdMatrix &xtx() const;
    const SpdMatrix &yyt() const;
    double yty() const;
    const Vector &xty() const;

    void add_data(const Ptr<Data> &dp) override;
    void add_data(const Ptr<ChoiceData> &) override;

   private:
    ImputationMethod imp_method;
    mutable Vector wsp;
    std::vector<Vector> U;
    uint nchoices_, subject_xdim_, choice_xdim_;
    SpdMatrix yyt_;  // sum y*y^T
    SpdMatrix xtx_;  // sum
    Vector xty_;

    Ptr<GlmCoefs> make_beta(const Matrix &beta_subject,
                            const Vector &beta_choice);
    Ptr<GlmCoefs> make_beta(const std::vector<Ptr<ChoiceData> > &);
    void setup_suf();
    void impute_u(RNG &rng, Vector &u, const Ptr<ChoiceData> &data,
                  TrunMvnTF &);
    void impute_u_slice(Vector &u, const Ptr<ChoiceData> &data, TrunMvnTF &);
    void impute_u_Gibbs(RNG &rng, Vector &u, const Ptr<ChoiceData> &data,
                        TrunMvnTF &);
    void update_suf(const Vector &u, const Ptr<ChoiceData> &data);
  };

}  // namespace BOOM
#endif  // BOOM_MULTINOMIAL_PROBIT_MODEL_HPP
