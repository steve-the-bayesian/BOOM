// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2014 Steven L. Scott

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

#ifndef BOOM_REGRESSION_MODEL_H
#define BOOM_REGRESSION_MODEL_H

#include "uint.hpp"
#include <cstdint>

#include "LinAlg/QR.hpp"
#include "Models/EmMixtureComponent.hpp"
#include "Models/Glm/Glm.hpp"
#include "Models/ParamTypes.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/ParamPolicy_2.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/Policies/SufstatDataPolicy.hpp"
#include "Models/Sufstat.hpp"
#include "uint.hpp"

namespace BOOM {

  class AnovaTable {
   public:
    double SSE, SSM, SST;
    double MSM, MSE;
    double df_error, df_model, df_total;
    double F, p_value;
    std::ostream &display(std::ostream &out) const;

    double Rsquare() const {
      return SSM / SST;
    }
  };

  std::ostream &operator<<(std::ostream &out, const AnovaTable &tab);

  Matrix add_intercept(const Matrix &X);
  Vector add_intercept(const Vector &X);

  //------- virtual base for regression sufficient statistics ----
  class RegSuf : virtual public Sufstat {
   public:
    typedef std::vector<Ptr<RegressionData> > dataset_type;

    RegSuf *clone() const override = 0;

    virtual void fix_xtx(bool fixed = true) = 0;
    virtual uint size() const = 0;  // dimension of beta
    virtual double yty() const = 0;
    virtual Vector xty() const = 0;
    virtual SpdMatrix xtx() const = 0;

    virtual Vector xty(const Selector &) const = 0;
    virtual SpdMatrix xtx(const Selector &) const = 0;

    // (X - Xbar)^T * (X - Xbar)
    //  = xtx - n * xbar xbar^T
    SpdMatrix centered_xtx() const;

    // return least squares estimates of regression params
    virtual Vector beta_hat() const = 0;
    virtual double SSE() const = 0;  // SSE measured from ols beta
    virtual double SST() const = 0;
    virtual double ybar() const = 0;
    // Column means of the design matrix.
    virtual Vector xbar() const = 0;
    virtual double n() const = 0;
    double sample_variance() const;
    double sample_sd() const { return sqrt(sample_variance()); }

    // Compute the sum of square errors using the given set of
    // coefficients, taking advantage of sparsity.
    double relative_sse(const GlmCoefs &beta) const;
    double relative_sse(const Vector &beta) const;

    AnovaTable anova() const;

    virtual void add_mixture_data(double y, const Vector &x, double prob) = 0;
    virtual void add_mixture_data(double y, const ConstVectorView &x,
                                  double prob) = 0;
    virtual void combine(const Ptr<RegSuf> &) = 0;

    std::ostream &print(std::ostream &out) const override;
  };

  inline std::ostream &operator<<(std::ostream &out, const RegSuf &suf) {
    return suf.print(out);
  }

  //------------------------------------------------------------------
  class QrRegSuf : public RegSuf, public SufstatDetails<RegressionData> {
   public:
    QrRegSuf(const Matrix &X, const Vector &y);

    QrRegSuf *clone() const override;
    void clear() override;
    void Update(const DataType &) override;
    void add_mixture_data(double y, const Vector &x, double prob) override;
    void add_mixture_data(double y, const ConstVectorView &x,
                          double prob) override;
    void fix_xtx(bool fixed = true) override;
    uint size() const override;  // dimension of beta
    double yty() const override;
    Vector xty() const override;
    SpdMatrix xtx() const override;

    Vector xty(const Selector &) const override;
    SpdMatrix xtx(const Selector &) const override;

    Vector beta_hat() const override;
    virtual Vector beta_hat(const Vector &y) const;
    double SSE() const override;
    double SST() const override;
    double ybar() const override;
    Vector xbar() const override;
    double n() const override;
    void refresh_qr(const std::vector<Ptr<DataType> > &) const;
    //    void check_raw_data(const Matrix &X, const Vector &y);
    void combine(const Ptr<RegSuf> &) override;
    virtual void combine(const RegSuf &);
    QrRegSuf *abstract_combine(Sufstat *s) override;

    Vector vectorize(bool minimal = true) const override;
    Vector::const_iterator unvectorize(Vector::const_iterator &v,
                                       bool minimal = true) override;
    Vector::const_iterator unvectorize(const Vector &v,
                                       bool minimal = true) override;
    std::ostream &print(std::ostream &out) const override;

   private:
    mutable QR qr;
    mutable Vector Qty;
    mutable double sumsqy_;
    mutable bool current;
    mutable Vector x_column_sums_;
  };

  //------------------------------------------------------------------
  class NeRegSuf
      : public RegSuf,
        public SufstatDetails<RegressionData> {  // directly solves 'normal
                                                 // equations'
   public:
    // An empty, but right-sized set of sufficient statistics.
    explicit NeRegSuf(uint p);

    // Build from the design matrix X and response vector y.
    NeRegSuf(const Matrix &X, const Vector &y);

    // Build from the indiviudal sufficient statistic components.  The
    // 'n' is needed because X might not have an intercept term.
    NeRegSuf(const SpdMatrix &xtx,
             const Vector &xty,
             double yty,
             double n,
             double ybar,
             const Vector &xbar);

    // Build from a sequence of Ptr<RegressionData>
    template <class Fwd>
    NeRegSuf(Fwd b, Fwd e);
    NeRegSuf *clone() const override;

    // If fixed, then xtx will not be changed by a call to clear(),
    // add_mixture_data(), or any of the flavors of Update().
    void fix_xtx(bool fixed = true) override;

    void clear() override;
    void add_mixture_data(double y, const Vector &x, double prob) override;
    void add_mixture_data(double y, const ConstVectorView &x,
                          double prob) override;
    void Update(const RegressionData &rdp) override;
    uint size() const override;  // dimension of beta
    double yty() const override;
    Vector xty() const override;
    SpdMatrix xtx() const override;
    Vector xty(const Selector &) const override;
    SpdMatrix xtx(const Selector &) const override;
    Vector beta_hat() const override;
    double SSE() const override;
    double SST() const override;
    double ybar() const override;
    Vector xbar() const override;
    double n() const override;
    void combine(const Ptr<RegSuf> &) override;
    void combine(const RegSuf &);
    NeRegSuf *abstract_combine(Sufstat *s) override;

    Vector vectorize(bool minimal = true) const override;
    Vector::const_iterator unvectorize(Vector::const_iterator &v,
                                       bool minimal = true) override;
    Vector::const_iterator unvectorize(const Vector &v,
                                       bool minimal = true) override;
    std::ostream &print(std::ostream &out) const override;

    // Adding data only updates the upper triangle of xtx_.  Calling
    // reflect() fills the lower triangle as well, if needed.
    void reflect() const;

    void allow_non_finite_responses(bool allow) {
      allow_non_finite_responses_ = allow;
    }

   private:
    mutable SpdMatrix xtx_;
    mutable bool needs_to_reflect_;
    Vector xty_;
    bool xtx_is_fixed_;
    double sumsqy_;
    double n_;
    double sumy_;
    Vector x_column_sums_;
    bool allow_non_finite_responses_;
  };

  template <class Fwd>
  NeRegSuf::NeRegSuf(Fwd b, Fwd e)
      : needs_to_reflect_(true),
        xtx_is_fixed_(false),
        sumsqy_(0.0),
        n_(0.0),
        sumy_(0.0),
        allow_non_finite_responses_(false)
  {
    Ptr<RegressionData> dp = *b;
    uint p = dp->xdim();
    xtx_ = SpdMatrix(p, 0.0);
    xty_ = Vector(p, 0.0);
    sumsqy_ = 0.0;
    while (b != e) {
      update(*b);
      ++b;
    }
  }

  //------------------------------------------------------------------
  using RegressionDataPolicy = SufstatDataPolicy<RegressionData, RegSuf>;
  
  class RegressionModel : public GlmModel,
                          public ParamPolicy_2<GlmCoefs, UnivParams>,
                          public RegressionDataPolicy,
                          public PriorPolicy,
                          public NumOptModel,
                          public EmMixtureComponent {
   public:
    explicit RegressionModel(uint xdim);

    // Args:
    //   coefficients: The vector of regression coefficients.  All are included.
    //   residual_sd: The standard deviation of the Gaussian errors around the
    //     regression line.
    RegressionModel(const Vector &coefficients, double residual_sd);

    // Use this constructor if the model needs to share parameters
    // with another model.  E.g. a mixture model with shared variance
    // parameter.
    // Args:
    //   coefficients:  The vector of regression coefficients.
    //   residual_variance: The residual variance parameter.  Note the
    //     constructor above is parameterized in terms of residual sd instead of
    //     residual variance.
    RegressionModel(const Ptr<GlmCoefs> &coefficients,
                    const Ptr<UnivParams> &residual_variance);

    // Args:
    //   X: The design matrix of predictor variables.  Must contain an
    //     explicit column of 1's if an intercept term is desired.
    //   y: The vector of responses.  The length of y must match the
    //     number of rows in X.
    //   start_at_mle: If true then the regression coefficients will begin at
    //     their maximum likelihood estimate.  Otherwise the coefficients begin
    //     at zero.
    RegressionModel(const Matrix &X, const Vector &y, bool start_at_mle = true);


    // Initialize a regression model using the sufficient statistics.  The
    // coefficients wll be initialized at zero except for the intercept term,
    // set to the mean.
    //
    // Args:
    //   suf: An object containing the sufficient statistics for the model.
    explicit RegressionModel(const Ptr<RegSuf> &suf);

    RegressionModel(const RegressionModel &rhs);
    RegressionModel *clone() const override;

    // The number of variables currently included in the model, including the
    // intercept, if present.
    uint nvars() const;

    // The number of potential variables, including the intercept.
    uint nvars_possible() const;

    //---- parameters ----
    GlmCoefs &coef() override;
    const GlmCoefs &coef() const override;
    Ptr<GlmCoefs> coef_prm() override;
    const Ptr<GlmCoefs> coef_prm() const override;
    Ptr<UnivParams> Sigsq_prm();
    const Ptr<UnivParams> Sigsq_prm() const;

    void set_sigsq(double s2);

    double sigsq() const;
    double sigma() const;

    //---- simulate regression data  ---
    virtual RegressionData *sim(RNG &rng = GlobalRng::rng) const;
    virtual RegressionData *sim(const Vector &X,
                                RNG &rng = GlobalRng::rng) const;
    Vector simulate_fake_x(RNG &rng = GlobalRng::rng) const;  // no intercept

    //---- estimation ---
    SpdMatrix xtx(const Selector &inc) const;
    Vector xty(const Selector &inc) const;
    SpdMatrix xtx() const;  // adjusts for covariate inclusion-
    Vector xty() const;     // exclusion, and includes weights,
    double yty() const;     // if used

    void make_X_y(Matrix &X, Vector &y) const;

    //--- probability calculations ----
    void mle() override;
    // The argument 'sigsq_beta' is a Vector with the first element
    // corresponding to the residual variance parameter, and the
    // remaining elements corresponding to the set of included
    // coefficients.
    double Loglike(const Vector &sigsq_beta, Vector &g, Matrix &h,
                   uint nd) const override;

    // This implementation of log_likelihood allows for a vector beta that might
    // be of smaller dimension than the predictors, so long as those predictors
    // have been dropped from the model using the drop() method of this model's
    // GlmCoefs.
    double log_likelihood(const Vector &beta, double sigsq) const;

    // Avoid hiding the 'double log_likelihood()' implementation from
    // LoglikeModel.
    using LoglikeModel::log_likelihood;

    // This implementation of log_likelihood assumes that beta is the same
    // dimension as the predictor matrix used to construct suf.
    static double log_likelihood(const Vector &beta, double sigsq,
                                 const RegSuf &suf);

    // The marginal log likelihood of a regression model, given sigsq.  In a
    // context involving variable selection, this function assumes that all
    // inputs have already been selected down.
    //
    // Args:
    //   sigsq:  The residual variance parameter.
    //   xtx:  The predictor cross product matrix X'X.
    //   xty:  The inner product of the predictor and response: X'y.
    //   yty:  The sum of the squares of the response: y'y.
    //   n:  The sample size.
    //   prior_mean:  The prior mean of the regression coefficients.
    //   unscaled_prior_precision_lower_cholesky: The lower Cholesky triangle of
    //     the unscaled prior precision of the regression coefficients.  The
    //     actual precision matrix is unscaled_prior_precision / sigsq.
    //   posterior_mean:  The posterior mean of the regression coefficients.
    //   unscaled_posterior_precision_cholesky: The Cholesky triangle (either
    //     upper or lower) of the unscaled posterior precision.  Dividing the
    //     unscaled precision by sigsq gives the actual precision.
    //
    // Returns:
    //   The marginal log likelihood log p(data | sigsq), integrating out the
    //   regression coefficients.
    //
    // Notes:
    //   This function is designed to be called in the middle of a
    //   prior-to-posterior computation, when Cholesky triangles of all the
    //   relevant matrices have already been computed.  The design sacrifices a
    //   bit of convenience to avoid duplicating work.
    static double marginal_log_likelihood(
        double sigsq,
        const SpdMatrix &xtx,
        const Vector &xty,
        double yty,
        double n,
        const Vector &prior_mean,
        const Matrix &unscaled_prior_precision_lower_cholesky,
        const Vector &posterior_mean,
        const Matrix &unscaled_posterior_precision_cholesky);

    virtual double pdf(const Ptr<Data> &, bool) const;
    double pdf(const Data *, bool) const override;

    int number_of_observations() const override { return dat().size(); }

    // The log likelihood when beta is empty (i.e. all coefficients,
    // including the intercept, are zero).
    double empty_loglike(Vector &g, Matrix &h, uint nd) const;

    // If the model was formed using the QR decomposition, switch to using the
    // normal equations.  The normal equations are computationally more
    // efficient when doing variable selection or when the data is changing
    // between MCMC iterations (as in finite mixtures).
    void use_normal_equations();

    void add_mixture_data(const Ptr<Data> &, double prob) override;

    //--- diagnostics ---
    AnovaTable anova() const { return suf()->anova(); }
  };


  // A BigRegressionModel is a regression model where the number of predictors is
  // too large to use the sufficient statistics in the ordinary RegressionModel.
  class BigRegressionModel
      : public GlmModel,
        public ParamPolicy_2<GlmCoefs, UnivParams>,
        public IID_DataPolicy<RegressionData>,
        public PriorPolicy
  {
    friend class BigAssSpikeSlabSampler;
   public:
    // Args:
    //   xdim:  The dimension of the full (very large) predictor vector.
    //   subordinate_model_max_dim:  The largest predictor dimension for each model.
    //   force_intercept:  If true then the intercep
    explicit BigRegressionModel(uint xdim,
                                int subordinate_model_max_dim = 500,
                                bool force_intercept = true);

    BigRegressionModel * clone() const override;

    uint xdim() const {return coef().nvars_possible();}

    GlmCoefs &coef() override {return prm1_ref();}
    const GlmCoefs &coef() const override {return prm1_ref();}
    Ptr<GlmCoefs> coef_prm() override {return prm1();}
    const Ptr<GlmCoefs> coef_prm() const override {return prm1();}

    double sigsq() const {return prm2_ref().value();}
    double sigma() const {return std::sqrt(sigsq());}
    void set_sigsq(double sigsq) {prm2_ref().set(sigsq);}
    Ptr<UnivParams> Sigsq_prm() { return prm2(); }

    double predict(const Vector &x) const override {
      return coef().predict(x);
    }

    // Pass data to the subordinate models.  The data are not kept, but are
    // added to the subordinate model's sufficient statistics.
    void stream_data_for_initial_screen(const RegressionData &data_point);

    // Pass data to the primary model.  The set of candidate values are
    void stream_data_for_restricted_model(const RegressionData &data_point);

    // Set the subset of variables to use in the final spike-and-slab run.
    void set_candidates(const Selector &candidates);

    const Selector & candidate_selector() const {
      return predictor_candidates_;
    }

    // To handle predictors of very high dimension, the model maintains several
    // smaller regression models, each of moderate dimension.  The data in each
    // of the smaller models is independent
    int number_of_subordinate_models() const {
      return subordinate_models_.size();
    }

    RegressionModel *subordinate_model(int i) {
      return subordinate_models_[i].get();
    }

    // The dimension of the largest subordinate model (which is always the first
    // one).
    int worker_dim_upper_limit() const {
      return subordinate_models_[0]->xdim();
    }

    RegressionModel *restricted_model() {
      if (!!restricted_model_) {
        return restricted_model_.get();
      } else {
        return nullptr;
      }
    }

    // Write the parameters from the restricted model to the corresponding
    // positions in the full model.
    void expand_restricted_model_parameters();

    bool force_intercept() const {return force_intercept_;}

   private:
    // Indicates whether an intercept term is added to each of the subordinate
    // models.
    bool force_intercept_;

    // Something determines the actual subset of predictors that can be used.
    Selector predictor_candidates_;

    // Each subordinate model handles a chunk of the predictors.
    std::vector<Ptr<RegressionModel>> subordinate_models_;

    // The restricted model is created when the user calls
    // set_predictor_candidates.  It contains the candidate values selected by
    // the subordinate models.
    Ptr<RegressionModel> restricted_model_;

    void create_subordinate_models(uint xdim, int max_worker_dim, bool force_intercept);
  };


}  // namespace BOOM

#endif  // BOOM_REGRESSION_MODEL_H
