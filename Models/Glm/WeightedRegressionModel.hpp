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

#ifndef BOOM_WEIGHTED_REGRESSION_MODEL_HPP
#define BOOM_WEIGHTED_REGRESSION_MODEL_HPP

#include "Models/Glm/Glm.hpp"
#include "Models/Glm/RegressionModel.hpp"

namespace BOOM {

  //------------------------------------------------------------

  class WeightedRegSuf : public SufstatDetails<WeightedRegressionData> {
   public:
    typedef WeightedRegressionData data_type;
    typedef std::vector<Ptr<WeightedRegressionData> > dataset_type;

    // Args:
    //   xdim:  Number of predictor variables.
    explicit WeightedRegSuf(int xdim);

    // Args:
    //   X:  The matrix of predictors.
    //   y:  The vector of responses.
    //   w: The vector of weights.  If w is empty all weights are assumed to be
    //     1.0.
    WeightedRegSuf(const Matrix &X, const Vector &y, const Vector &w = Vector());

    // Args:
    //   data:  The data vector.  Each entry is an observation.
    explicit WeightedRegSuf(
        const std::vector<Ptr<WeightedRegressionData>> &data);

    WeightedRegSuf(const WeightedRegSuf &rhs) = default;
    WeightedRegSuf(WeightedRegSuf &&rhs) = default;

    WeightedRegSuf *clone() const override;

    // Recompute the sufficient statistics (from scratch).
    // Args:
    //   X:  The matrix of predictors.
    //   y:  The vector of responses.
    //   w:  The vector of weights.
    // w and y must have the same length, which must match the number of rows in
    //   X.
    virtual void recompute(const Matrix &X, const Vector &y, const Vector &w);
    virtual void recompute(
        const std::vector<Ptr<WeightedRegressionData>> &data);

    // Set the sufficient statistics to values computed externally.
    void reset(const SpdMatrix &xtwx, const Vector &xtwy, double ytwy,
               double sample_size, double sum_weights, double sum_log_weights);
    void set_xtwx(const SpdMatrix &xtwx);
    void set_xtwy(const Vector &xtwy);

    void Update(const WeightedRegressionData &) override;
    void add_data(const Vector &x, double y, double w);

    void clear() override;
    virtual uint size() const;                      // dimension of beta
    virtual double yty() const;                     // Y^t W Y
    virtual Vector xty() const;                     // X^T W Y
    virtual SpdMatrix xtx() const;                  // X^T W X
    virtual Vector xty(const Selector &) const;     // X^T W Y
    virtual SpdMatrix xtx(const Selector &) const;  // X^T W X
    virtual Vector beta_hat() const;                // WLS estimate
    double weighted_sum_of_squared_errors(const Vector &beta) const;
    virtual double SSE() const;   //
    virtual double SST() const;   // weighted sum of squares
    virtual double ybar() const;  // weighted average
    virtual double n() const;
    virtual double sumw() const;     // sum of weights
    virtual double sumlogw() const;  // sum of weights
    std::ostream &print(std::ostream &out) const override;
    void combine(const Ptr<WeightedRegSuf> &);
    void combine(const WeightedRegSuf &);
    WeightedRegSuf *abstract_combine(Sufstat *s) override;

    Vector vectorize(bool minimal = true) const override;
    Vector::const_iterator unvectorize(Vector::const_iterator &v,
                                       bool minimal = true) override;
    Vector::const_iterator unvectorize(const Vector &v,
                                       bool minimal = true) override;

   private:
    mutable SpdMatrix xtwx_;
    Vector xtwy_;
    double n_;  // xtx_(0,0) is the sum of the weights,
    double yt_w_y_;
    double sumw_;
    double sumlogw_;
    mutable bool sym_;

    void setup_mat(uint p);
    void make_symmetric() const;
  };

  inline std::ostream &operator<<(std::ostream &out, const WeightedRegSuf &s) {
    return s.print(out);
  }

  //------------------------------------------------------------

  class WeightedRegressionModel
      : public ParamPolicy_2<GlmCoefs, UnivParams>,
        public SufstatDataPolicy<WeightedRegressionData, WeightedRegSuf>,
        public PriorPolicy,
        public GlmModel,
        public NumOptModel {
   public:
    typedef WeightedRegressionData data_type;
    typedef WeightedRegSuf suf_type;

    explicit WeightedRegressionModel(uint p);
    WeightedRegressionModel(const Vector &b, double Sigma);
    WeightedRegressionModel(const WeightedRegressionModel &rhs);
    WeightedRegressionModel(const Matrix &X, const Vector &y);
    WeightedRegressionModel(const Matrix &X, const Vector &y, const Vector &w);
    explicit WeightedRegressionModel(const DatasetType &d, bool all = true);
    WeightedRegressionModel *clone() const override;

    GlmCoefs &coef() override;
    const GlmCoefs &coef() const override;
    Ptr<GlmCoefs> coef_prm() override;
    const Ptr<GlmCoefs> coef_prm() const override;
    Ptr<UnivParams> Sigsq_prm();
    const Ptr<UnivParams> Sigsq_prm() const;

    // beta() and Beta() inherited from GLM;
    //    void set_beta(const Vector &b);
    void set_sigsq(double s2);

    const double &sigsq() const;
    double sigma() const;

    void mle() override;
    // The argument is a vector with leading coefficients 'beta' and
    // final element sigsq.
    double Loglike(const Vector &beta_sigsq, Vector &g, Matrix &h,
                   uint nd) const override;
    double pdf(const Ptr<Data> &, bool) const;
    double pdf(const Ptr<data_type> &, bool) const;
  };

}  // namespace BOOM

#endif  // BOOM_WEIGHTED_REGRESSION_MODEL_HPP
