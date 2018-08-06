// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2011 Steven L. Scott

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

#ifndef BOOM_MULTINOMIAL_LOGIT_MODEL_HPP
#define BOOM_MULTINOMIAL_LOGIT_MODEL_HPP

#include "Models/EmMixtureComponent.hpp"
#include "Models/Glm/ChoiceData.hpp"
#include "Models/Glm/GlmCoefs.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/ParamPolicy_1.hpp"
#include "Models/Policies/PriorPolicy.hpp"

namespace BOOM {

  class MultinomialLogitModel : public ParamPolicy_1<GlmCoefs>,
                                public IID_DataPolicy<ChoiceData>,
                                public PriorPolicy,
                                public NumOptModel,
                                virtual public MixtureComponent {
   public:
    // Initialize the model with a set of regression coefficients.
    // Args:
    //   beta_subject: The subject-level parameters.  Each column of
    //     beta_subject corresponds to a different choice level.
    //     There should be (Nchoices - 1) total columns, because the
    //     choice value 0 is assumed to have all 0's for its
    //     subject-level coefficients.  The implicit column of 0's
    //     should not be included.
    //   beta_choice: A vector of coefficients describing the impact
    //     of the choice-level predictors.
    MultinomialLogitModel(const Matrix &beta_subject,
                          const Vector &beta_choice);

    // Args:
    //   Nchoices:  The number of possible choices in the response variable.
    //   subject_xdim: The dimension of the predictor variables
    //     measuring the characteristics of the subjects doing the
    //     choosing.  Ususally this includes an intercept, but an
    //     intercept term is not strictly necessary.
    //   choice_xdim: The dimension of the predictor variables
    //     describing characteristics of the items being chosen.  This
    //     is for a single object.  The complete set of choice level
    //     predictors will be an array of (Nchoices X choice_xdim)
    //     values.
    MultinomialLogitModel(uint Nchoices, uint subject_xdim, uint choice_xdim);

    // Args:
    //   responses:  The vector of responses
    //   Xsubject_info: A matrix of subject level predictors.  Each
    //     row describes a subject associated with that observation.
    //   Xchoice_info: Each entry is a matrix of predictors describing
    //     the characteristics of the choices for that observation.
    //     Rows of the matrix correspond to choices.  Columns
    //     correspond to predictors.  The vector can be empty to
    //     signify that there is no choice-level data available.
    MultinomialLogitModel(
        const std::vector<Ptr<CategoricalData> > &responses,
        const Matrix &Xsubject_info,
        const std::vector<Matrix> &Xchoice_info = std::vector<Matrix>());

    MultinomialLogitModel(const MultinomialLogitModel &rhs);
    MultinomialLogitModel *clone() const override;

    // coefficient vector: elements corresponding to choice level 0
    // (which are constrained to 0 for identifiability) are omitted.
    // Thus beta() is of dimension ((num_choices-1)*psub + pch)

    // If the choices are labelled 0, 1, 2, ..., M-1 then the elements
    // of beta are
    // [ subject_characeristic_beta_for_choice_1,
    //   subject_characeristic_beta_for_choice_2
    //   ...
    //   subject_characeristic_beta_for_choice_M-1
    //   choice_characteristic_beta ]
    const Vector &beta() const;

    // Returns the vector of logistic regression coefficients
    // described above (see beta()), but with a vector of 0's
    // prepended, corresponding to the subject parameters for choice
    // level 0.
    const Vector &beta_with_zeros() const;

    // Returns the vector of subject specific coefficients for the
    // given choice level.  If 'choice' is 0 then a vector of all 0's
    // is returned.
    Vector beta_subject(uint choice) const;

    // Returns the vector of choice specific coefficients.
    Vector beta_choice() const;

    void set_beta(const Vector &b);

    // Args:
    //   b: The vector of coefficients to use for the specified choice
    //     level.  The dimension of b must match subject_nvars().
    //   choice_level: The choice level that b refers to.  Must be >=1
    //     and < Nchoices().
    void set_beta_subject(const Vector &b, uint choice_level);

    // Args:
    //   b: The vector of choice-specific coefficients.  The size of b
    //     must match choice_nvars().
    void set_beta_choice(const Vector &b);

    virtual GlmCoefs &coef();
    virtual const GlmCoefs &coef() const;
    virtual Ptr<GlmCoefs> coef_prm();
    virtual const Ptr<GlmCoefs> coef_prm() const;

    // Returns a Selector of the same dimension as beta().  The
    // structural zeros for beta_subject(0) are not considered.
    const Selector &inc() const;

    // If keep_intercepts is true, all the slopes will be dropped but
    // the subject level intercepts will remain in the model.  This
    // function assumes that the first element of each choice data's
    // subject level predictors is the number 1.
    void drop_all_slopes(bool keep_intercepts = true);
    void add_all_slopes();

    // 'beta' refers to the vector of nonzero "included" coefficients.
    double Loglike(const Vector &beta, Vector &g, Matrix &H,
                   uint nd) const override;

    // Args:
    //   beta: The vector of logistic regression coefficients, with
    //     structural zeros omitted.
    //   gradient: If nd > 0 then 'gradient' is filled with the
    //     gradient of log_likelihood.  Otherwise it is not
    //     referenced.
    //   Hessian: If nd > 1 then 'Hessian' is filled with the matrix
    //     of second derivatives of log_likelihood.  Otherwise it is
    //     not referenced.
    //   nd:  The number of derivatives to take.
    // Returns:
    //   The log likelihood evaluated at beta.
    double log_likelihood(const Vector &beta, Vector &gradient, Matrix &Hessian,
                          int nd) const;

    double log_likelihood() const override {
      Vector g;
      Matrix h;
      return Loglike(beta(), g, h, 0);
    }

    // Compute beta^Tx for the choice and subject portions of X.
    double predict_choice(const ChoiceData &, uint m) const;
    double predict_subject(const ChoiceData &, uint m) const;

    // Fill in the linear predictor.  The dimension of eta is
    // Nchoices(), so the baseline choice is filled in as well.
    Vector &fill_eta(const ChoiceData &, Vector &ans,
                     const Vector &full_beta) const;
    Vector &fill_eta(const ChoiceData &, Vector &ans) const;

    //----------------------------------------------------------------------
    virtual double pdf(const Ptr<Data> &dp, bool logscale) const;
    double pdf(const Data *dp, bool logscale) const override;
    virtual double logp(const ChoiceData &dp) const;
    int number_of_observations() const override { return dat().size(); }

    // Returns the dimension of the non-sparse set of regression coefficients.
    // Args:
    //   include_zeros: If true then the identically zero coefficients
    //     corresponding to the reference category are included in the
    //     count.
    uint beta_size(bool include_zeros = false) const;

    // simulate an outcome
    uint sim(const Ptr<ChoiceData> &dp, RNG &rng = GlobalRng::rng) const;
    uint sim(const Ptr<ChoiceData> &dp, Vector &eta,
             RNG &rng = GlobalRng::rng) const;

    // compute all choice probabilities
    Vector predict(const Ptr<ChoiceData> &) const;
    // returns choice probabilities
    Vector &predict(const Ptr<ChoiceData> &, Vector &ans) const;

    uint subject_nvars() const;
    uint choice_nvars() const;
    uint Nchoices() const;

    // Args:
    //   probs: Gives the probability of keeping in the sample an
    //     observation with response level m.  For a prospective study
    //     all elements of probs would be 1.
    void set_sampling_probs(const Vector &probs);
    const Vector &log_sampling_probs() const;

   private:
    mutable Vector beta_with_zeros_;
    mutable bool beta_with_zeros_current_;

    void watch_beta();
    void setup();
    void setup_observers();
    void fill_extended_beta() const;
    void index_out_of_bounds(uint m) const;

    mutable Vector wsp_;
    uint nch_;   // number of choices
    uint psub_;  // number of subject X variables
    uint pch_;   // number of choice X variables
    Vector log_sampling_probs_;
  };
}  // namespace BOOM
#endif  // BOOM_MULTINOMIAL_LOGIT_MODEL_HPP
