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
#ifndef BOOM_MVN_GIVEN_X_HPP
#define BOOM_MVN_GIVEN_X_HPP

#include "Models/MvnBase.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/ParamPolicy_2.hpp"
#include "Models/Policies/PriorPolicy.hpp"

#include "Models/Glm/RegressionModel.hpp"
#include "Models/Glm/WeightedRegressionModel.hpp"
#include "Models/Glm/MultivariateRegression.hpp"

namespace BOOM {

  class GlmModel;
  class GlmCoefs;
  class RegressionModel;
  class LogisticRegressionModel;
  class ProbitRegressionModel;

  // This model is intended to serve as a prior for probit and logit
  // regression models p(y | beta, X).  For a prior on Gaussian
  // regression models look at MvnGivenXAndSigma
  //
  // The model is
  // beta ~ N(mu, Ivar)
  // Ivar = Lambda + kappa * [(1-diag_wgt) * xtx
  //                               + diag_wgt*diag(xtx)]) / sumw
  // sumw = sum of w's from add_x
  // Lambda is a diagonal matrix that defaults to 0
  // Ivar = Lambda if sumw==0
  //
  // The justification for this prior is that xtwx is the information
  // in the full complete-data likelihood, so xtwx / sumw is the
  // average information available from a single observation.  If xtwx
  // is highly collinear then it may not be positive definite, so this
  // class offers the option to shrink away from xtwx and towards
  // diag(xtwx).  The amount of shrinkage towards the diagonal.  The
  // Lambda matrix is there to handle cases where no X's have been
  // observed.
  //
  // The "kappa" parameter is a prior sample size.
  // The "Mu" parameter is the prior mean.  Mu decreases in
  // relevance as kappa->0
  //
  class MvnGivenXBase : public MvnBase,
                        public ParamPolicy_2<VectorParams, UnivParams>,
                        public IID_DataPolicy<GlmCoefs>,
                        public PriorPolicy {
   public:
    // Args:
    //   mean:  The mean of the distribution.
    //   prior_sample_size:  The number of observations worth of prior weight.
    //   precision_diagonal:  The diagonal that X'X is to be averaged with.  
    //   diagonal_weight: A number between 0 and 1 indicating the weight to put
    //     on the diagonal when X'X is averaged with its diagonal.
    MvnGivenXBase(const Ptr<VectorParams> &mean,
                  const Ptr<UnivParams> &prior_sample_size,
                  const Vector &precision_diagonal = Vector(),
                  double diagonal_weight = 0);

    MvnGivenXBase(const MvnGivenXBase &rhs) = default;
    MvnGivenXBase(MvnGivenXBase &&rhs) = default;

    MvnGivenXBase *clone() const override = 0;

    //---------------------------------------------------------------------------
    // MvnBase overrides
    //---------------------------------------------------------------------------
    uint dim() const override {return mu().size();}
    const Vector &mu() const override {return Mu_prm()->value();}
    double kappa() const {return Kappa_prm()->value();}

    const SpdMatrix &Sigma() const override;
    const SpdMatrix &siginv() const override;
    double ldsi() const override;
    Vector sim(RNG &rng = GlobalRng::rng) const override;

    //---------------------------------------------------------------------------
    // Parameter access
    //---------------------------------------------------------------------------
    const Ptr<VectorParams> Mu_prm() const {return prm1();}
    Ptr<VectorParams> Mu_prm() {return prm1();}
    
    const Ptr<UnivParams> Kappa_prm() const {return prm2();}
    Ptr<UnivParams> Kappa_prm() {return prm2();}

    double diagonal_weight() const { return diagonal_weight_; }

    // Note that diagonal might be empty.
    const Vector &diagonal() const { return diagonal_; }
    
    // An observer that can be set when the precision matrix is determined by
    // outside data that might change.
    std::function<void(void)> observer() {
      return [this]() {this->current_ = false;};
    }

   protected:
    bool current() const {return current_;}
    void mark_not_current() const {current_ = false;}

    void store_precision_matrix(SpdMatrix &&xtx) const;
   private:
    // Sets the value of precision_, and sets current_ to true.
    virtual void set_precision_matrix() const = 0;
    
    // The weight to give to the diagonal in the average of diagonal_ and X'X.
    double diagonal_weight_;

    // The diagonal is the prior precision if there are no X's.  It may be
    // unallocated in which case the diagonal is taken to be diag(X'X).
    Vector diagonal_;    

    // The actual matrix used to hold the precision.
    mutable Ptr<SpdData> precision_;

    // An observer flag.
    mutable bool current_;
  };

  //---------------------------------------------------------------------------
  // MvnGivenX that directly stores the matrix.
  //
  // Users of this class must manage xtwx_ explicitly.  For problems where X and
  // weights are fixed then add_x should be called once per row of X shortly
  // after initialization.  For problems where X or w change frequently then
  // clear_xtwx() and add_x() should be called as needed to manage the changes.
  class MvnGivenX : public MvnGivenXBase {
   public:
    MvnGivenX(const Ptr<VectorParams> &mean,
              const Ptr<UnivParams> &prior_sample_size,
              const Vector &precision_diagonal = Vector(),
              double diagonal_weight = 0);

    MvnGivenX * clone() const override {
      return new MvnGivenX(*this);
    }
    
    void add_x(const Vector &x, double w = 1.0);
    void clear_xtwx();
    void set_xtwx(const SpdMatrix &xtwx);

   private:
    void set_precision_matrix() const override; 

    mutable SpdMatrix xtwx_;
    double sumw_;
  };

  //---------------------------------------------------------------------------
  // The link to X is provided through a pointer to regression sufficient
  // statistics.
  class MvnGivenXRegSuf : public MvnGivenXBase {
   public:
    MvnGivenXRegSuf(const Ptr<VectorParams> &mean,
                    const Ptr<UnivParams> &prior_sample_size,
                    const Vector &precision_diagonal = Vector(),
                    double diagonal_weight = 0,
                    const Ptr<RegSuf> &suf = Ptr<RegSuf>(nullptr));

    MvnGivenXRegSuf(const MvnGivenXRegSuf &rhs);
    
    MvnGivenXRegSuf * clone() const override {
      return new MvnGivenXRegSuf(*this);
    }
  
    void set_suf(const Ptr<RegSuf> &suf) { suf_ = suf; }
    
   private:
    void set_precision_matrix() const override;
    Ptr<RegSuf> suf_;
  };

  //---------------------------------------------------------------------------
  // When the information about X comes from the sufficient statistics from a
  // multivariate regression.
  class MvnGivenXMvRegSuf : public MvnGivenXBase {
   public:
    MvnGivenXMvRegSuf(const Ptr<VectorParams> &mean,
                      const Ptr<UnivParams> &prior_sample_size,
                      const Vector &precision_diagonal = Vector(),
                      double diagonal_weight = 0,
                      const Ptr<MvRegSuf> &suf = Ptr<MvRegSuf>(nullptr));

    MvnGivenXMvRegSuf(const MvnGivenXMvRegSuf &rhs);
    
    MvnGivenXMvRegSuf * clone() const override {
      return new MvnGivenXMvRegSuf(*this);
    }
  
    void set_suf(const Ptr<MvRegSuf> &suf) { suf_ = suf; }
    
   private:
    void set_precision_matrix() const override;
    Ptr<MvRegSuf> suf_;
  };

  //---------------------------------------------------------------------------
  // The link to X is provided through a pointer to weighted regression
  // sufficient statistics.
  class MvnGivenXWeightedRegSuf : public MvnGivenXBase {
   public:
    MvnGivenXWeightedRegSuf(
        const Ptr<VectorParams> &mean,
        const Ptr<UnivParams> &prior_sample_size,
        const Vector &precision_diagonal = Vector(),
        double diagonal_weight = 0,
        const Ptr<WeightedRegSuf> &suf = Ptr<WeightedRegSuf>(nullptr));

    MvnGivenXWeightedRegSuf(const MvnGivenXWeightedRegSuf &rhs);
    
    MvnGivenXWeightedRegSuf * clone() const override {
      return new MvnGivenXWeightedRegSuf(*this);
    }
    
    void set_suf(const Ptr<WeightedRegSuf> &suf) { suf_ = suf; }
    
   private:
    void set_precision_matrix() const override;
    Ptr<WeightedRegSuf> suf_;
  };

  //----------------------------------------------------------------------
  // For multinomial logit models there are separate X's for subject
  // characteristics and choice characteristics.  The intercept term
  // is always considered a subject characeristic.  The prior is
  //
  //           beta ~ N(b0, B / prior_sample_size),
  //
  // where b0 and prior_sample_size are specified, and
  //
  // B^{-1} = (1-diagonal_weight) * U + diagonal_weight * diag(U).
  //
  // The matrix U is a block diagonal matrix corresponding roughly to
  // 'unit information'.  There is one block for each choice level
  // (other than choice level 0) corresponding to characteristic of
  // the subject making the choice.  These blocks are identical and
  // equal to X'X/(number of observations * number of choices), where
  // X is the matrix of subject characteristics (or subject
  // covariates).
  //
  // There is an additional block (in the lower right corner) equal to
  //
  // sum_i sum_m w_{im} w_{im}' / (number of observations * number of choices)
  //
  // where w_{im} is the vector of predictors describing choice m
  // faced by subject i.
  class MvnGivenXMultinomialLogit
      : public MvnBase,
        public ParamPolicy_2<VectorParams, UnivParams>,
        public IID_DataPolicy<GlmCoefs>,
        public PriorPolicy {
   public:
    MvnGivenXMultinomialLogit(const Vector &beta_prior_mean,
                              double prior_sample_size,
                              double diagonal_weight = 0);
    MvnGivenXMultinomialLogit(const Ptr<VectorParams> &beta_prior_mean,
                              const Ptr<UnivParams> &prior_sample_size,
                              double diagonal_weight = 0);
    MvnGivenXMultinomialLogit(const MvnGivenXMultinomialLogit &rhs);
    MvnGivenXMultinomialLogit *clone() const override;

    // Args:
    //   subject_characeristics: An n x p array with rows
    //     corresponding to subjects, and columns to measurements of
    //     each subject.
    //   choice_characteristics: a vector of [nchoices x p] matrices
    //     containing characteristics of the objects to be chosen.
    //   number_of_choices: The number of choices available for the
    //     response.  If choice_characteristics is provided, this
    //     argument must match.
    void set_x(const Matrix &subject_characeristics,
               const std::vector<Matrix> &choice_characteristics,
               int number_of_choices);

    Ptr<VectorParams> Mu_prm();
    const Ptr<VectorParams> Mu_prm() const;
    void set_mu(const Vector &mu);

    Ptr<UnivParams> Kappa_prm();
    const Ptr<UnivParams> Kappa_prm() const;
    double kappa() const;
    void set_kappa(double kappa);

    const Vector &mu() const override;
    const SpdMatrix &Sigma() const override;
    const SpdMatrix &siginv() const override;
    double ldsi() const override;

   private:
    double diagonal_weight_;

    SpdMatrix scaled_subject_xtx_;
    SpdMatrix scaled_choice_xtx_;

    mutable SpdMatrix overall_xtx_;
    mutable bool current_;
    mutable Ptr<SpdData> Sigma_storage_;

    void make_current() const;
  };

}  // namespace BOOM

#endif  // BOOM_MVN_GIVEN_X_HPP
