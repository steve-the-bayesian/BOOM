/*
  Copyright (C) 2005-2020 Steven L. Scott

  This library is free software; you can redistribute it and/or modify it under
  the terms of the GNU Lesser General Public License as published by the Free
  Software Foundation; either version 2.1 of the License, or (at your option)
  any later version.

  This library is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
  details.

  You should have received a copy of the GNU Lesser General Public License along
  with this library; if not, write to the Free Software Foundation, Inc., 51
  Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
*/

#include "Models/StateSpace/DynamicRegression.hpp"
#include "distributions.hpp"
#include "LinAlg/Cholesky.hpp"

namespace BOOM {
  namespace StateSpace {
    //=========================================================================
    namespace {
      using RDTP = RegressionDataTimePoint;
    }  // namespace

    RDTP::RegressionDataTimePoint(const RegressionDataTimePoint &rhs)
        : xdim_(rhs.xdim_),
          yty_(0.0),
          suf_(nullptr)
    {
      if (!!rhs.suf_) {
        suf_.reset(rhs.suf_->clone());
      } else {
        for (int i = 0; i < rhs.raw_data_.size(); ++i) {
          raw_data_.push_back(rhs.raw_data_[i]->clone());
        }
      }
    }

    RDTP::RegressionDataTimePoint(const Matrix &X, const Vector &y)
        : RegressionDataTimePoint(ncol(X)) {
      if (nrow(X) != y.size()) {
        report_error("Length of y must match the number of columns in X.");
      }
      if (nrow(X) >= ncol(X)) {
        suf_.reset(new NeRegSuf(X, y));
      } else {
        for (int i = 0; i < nrow(X); ++i) {
          NEW(RegressionData, dp)(y[i], X.row(i));
          add_data(dp);
        }
      }
    }

    std::ostream &RDTP::display(std::ostream &out) const {
      if (!!suf_) {
        out << "sufficient statistics for " << suf_->n() << " observations."
            << std::endl;
      } else {
        for (int i = 0; i < raw_data_.size(); ++i) {
          out << *raw_data_[i] << std::endl;
        }
      }
      return out;
    }

    void RDTP::add_data(const Ptr<RegressionData> &dp) {
      if (xdim_ == -1) {
        xdim_ = dp->xdim();
      } else {
        if (dp->xdim() != xdim_) {
          std::ostringstream err;
          err << "Attempt to add ata point of dimension " << dp->xdim()
              << " to RegressionDataTimePoint of dimension " << xdim_ << ".";
          report_error(err.str());
        }
      }
      if (suf_) {
        suf_->update(dp);
      } else {
        raw_data_.push_back(dp);
        yty_ += square(dp->y());
        if (raw_data_.size() >= dp->xdim()) {
          suf_.reset(new NeRegSuf(dp->xdim()));
          for (const auto &el : raw_data_) {
            suf_->update(el);
          }
          raw_data_.clear();
          yty_ = negative_infinity();
        }
      }
    }

    int RDTP::sample_size() const {
      if (!suf_) {
        return raw_data_.size();
      } else {
        return lround(suf_->n());
      }
    }

    std::pair<SpdMatrix, Vector> RDTP::xtx_xty(const Selector &inc) const {
      if (inc.nvars() == 0) {
        return std::make_pair(SpdMatrix(0), Vector(0));
      }

      if (!suf_) {
        SpdMatrix xtx(inc.nvars(), 0.0);
        Vector xty(inc.nvars(), 0.0);
        for (const auto &el : raw_data_) {
          Vector x = inc.select(el->x());
          xtx.add_outer(x, 1.0, false);
          xty.axpy(x, el->y());
        }
        xtx.reflect();
        return std::make_pair(xtx, xty);
      } else {
        return std::make_pair(suf_->xtx(inc), suf_->xty(inc));
      }
    }

    double RDTP::yty() const {
      if (!suf_) {
        return yty_;
      } else {
        return suf_->yty();
      }
    }

    double RDTP::SSE(const GlmCoefs &beta) const {
      double ans = 0;
      if (!suf_) {
        for (int i = 0; i < sample_size(); ++i) {
          double yhat = beta.predict(raw_data_[i]->x());
          ans += square(raw_data_[i]->y() - yhat);
        }
      } else {
        Vector b(beta.included_coefficients());
        ans = suf_->xtx(beta.inc()).Mdist(b)
            - 2 * b.dot(suf_->xty(beta.inc()))
            + suf_->yty();
      }
      return ans;
    }

    //===========================================================================
    namespace {
      using PSM = ProductSelectorMatrix;
    }
    Matrix PSM::dense() const {
      Matrix ans(inc2_.nvars(), inc1_.nvars(), 0.0);
      for (int i = 0; i < inc2_.nvars(); ++i) {
        int I = inc2_.indx(i);
        if (inc1_[I]) {
          ans(i, inc1_.INDX(I)) = 1.0;
        }
      }
      return ans;
    }

    Vector PSM::operator*(const Vector &v) const {
      return *this * (ConstVectorView(v));
    }

    Vector PSM::operator*(const ConstVectorView &v) const {
      Vector ans(inc2_.nvars(), 0.0);
      for (int i = 0; i < ans.size(); ++i) {
        int I = inc2_.indx(i);
        if (inc1_[I]) {
          ans[i] = v[inc1_.INDX(I)];
        }
      }
      return ans;
    }

    DiagonalMatrix PSM::sandwich(const DiagonalMatrix &d) const {
      return DiagonalMatrix(*this * d.diag());
    }

    SpdMatrix PSM::sandwich(const SpdMatrix &P) const {
      SpdMatrix ans(inc2_.nvars(), 0.0);
      for (int i = 0; i < ans.nrow(); ++i) {
        int  I = inc2_.indx(i);
        if (inc1_[I]) {
          for (int j = 0; j < ans.ncol(); ++j) {
            int J = inc2_.indx(j);
            if (inc1_[J]) {
              ans(i, j) = P(inc1_.INDX(I), inc1_.INDX(J));
            }
          }
        }
      }
      return ans;
    }


    //===========================================================================

    double DynamicRegressionKalmanFilterNode::initialize(
        const Selector &inc,
        const Vector &initial_mean,
        const SpdMatrix &unscaled_initial_precision,
        const RegressionDataTimePoint &data,
        double sigsq) {
      Vector prior_mean = inc.select(initial_mean);
      SpdMatrix unscaled_prior_precision =
          inc.select(unscaled_initial_precision);

      std::pair<SpdMatrix, Vector> suf = data.xtx_xty(inc);
      const SpdMatrix &xtx(suf.first);
      const Vector &xty(suf.second);

      state_variance_->set_ivar(xtx + unscaled_prior_precision);
      state_mean_ = unscaled_state_variance() * (
          xty + unscaled_prior_precision * prior_mean);

      return RegressionModel::marginal_log_likelihood(
          sigsq, xtx, xty, data.yty(), data.sample_size(),
          prior_mean, unscaled_prior_precision.chol(),
          state_mean_, state_variance_->ivar_chol());
    }

    // Set the mean and variance of this distribution (at time t) to match
    // p(beta[t, inc] | Y_t), where Y_t = y_0, ... y_t and 'inc' is the set of
    // all inclusion indicators at all times.
    //
    // Args:
    //   previous:  describes p(beta[t-1, inc] | Y_t-1)
    //   data: The data set at time t.
    //   model:  The model describing the data.
    //   time_index: t.
    //
    // Effects:
    //   The mean and variance are set to match the conditional mean and
    //   variance of the regression coefficients given data to time t.  The
    //   update is done conditional on the inclusion indicators stored in
    //   'model'.
    //
    // Returns:
    //   The increment to the marginal log likelihood from y_t.
    double DynamicRegressionKalmanFilterNode::update(
        const DynamicRegressionKalmanFilterNode &previous,
        const RegressionDataTimePoint &data,
        const DynamicRegressionModel &model,
        int time_index) {

      ProductSelectorMatrix transition_matrix(
          model.inclusion_indicators(time_index - 1),
          model.inclusion_indicators(time_index));

      // The marginal distribution of beta[t+1] = T * beta[t] + diagonal.x
      Vector prior_mean = transition_matrix * previous.state_mean();
      SpdMatrix prior_variance =
          transition_matrix.sandwich(previous.unscaled_state_variance());
      const Selector &inc(model.inclusion_indicators(time_index));
      prior_variance.diag() += inc.select(model.unscaled_innovation_variances());

      // Do a standard Bayes update here to get
      SpdMatrix unscaled_prior_precision = prior_variance.inv();
      std::pair<SpdMatrix, Vector> suf = data.xtx_xty(inc);
      const SpdMatrix &xtx(suf.first);
      const Vector &xty(suf.second);
      SpdMatrix unscaled_posterior_precision = unscaled_prior_precision + xtx;

      state_variance_->set_ivar(unscaled_posterior_precision);
      state_mean_ = state_variance_->var() *
          (xty + unscaled_prior_precision * prior_mean);

      double ans = RegressionModel::marginal_log_likelihood(
          model.residual_variance(), xtx, xty, data.yty(), data.sample_size(),
          prior_mean, unscaled_prior_precision.chol(),
          state_mean_, state_variance_->ivar_chol());

      return ans;
    }

    Vector DynamicRegressionKalmanFilterNode::simulate_coefficients(
        const DynamicRegressionModel &model, int time_index, RNG &rng) const {

      if (time_index < 0 || time_index >= model.time_dimension()) {
        std::ostringstream err;
        err << "time_index of " << time_index << " out of bounds for model with"
            << " time_dimension = " << model.time_dimension() << ".";
        report_error(err.str());
      }

      if (time_index + 1 == model.time_dimension()) {
        Selector fake_selector;
        Vector fake_vector;
        return simulate_coefficients_impl(
          std::sqrt(model.residual_variance()),
          time_index,
          model.time_dimension(),
          model.inclusion_indicators(time_index),
          fake_selector,
          fake_vector,
          model.unscaled_innovation_variances(),
          rng);
      } else {
        return simulate_coefficients_impl(
            std::sqrt(model.residual_variance()),
            time_index,
            model.time_dimension(),
            model.inclusion_indicators(time_index),
            model.inclusion_indicators(time_index + 1),  // check before calling.
            model.included_coefficients(time_index + 1),
            model.unscaled_innovation_variances(),
            rng);
      }
    }

    Vector DynamicRegressionKalmanFilterNode::simulate_coefficients_impl(
        double sigma,
        int time_index,
        int max_time_dimension,
        const Selector &inc_now,
        const Selector &inc_next,
        const Vector &beta_next,
        const Vector &unscaled_innovation_variances,
        RNG &rng) const {
      if (time_index < 0 || time_index >= max_time_dimension) {
        std::ostringstream err;
        err << "time_index of " << time_index << " out of bounds for model with"
            << " time_dimension = " << max_time_dimension << ".";
        report_error(err.str());
      } else if (time_index + 1 == max_time_dimension) {
        return rmvn_L_mt(rng, state_mean(),
                         sigma * state_variance_->var_chol());
      } else {
        // If x ~ N(mu, Omega) and y ~ N(Ax, Sigma) then
        // x | y ~ N(mu1, V1), with
        // V1^{-1} = omega^{-1} + A' Siginv A, and
        // mu = V1 * (Omega^{-1} mu + A' Siginv y)
        ProductSelectorMatrix A_transpose(inc_next, inc_now);

        DiagonalMatrix siginv(
            1.0 / inc_next.select(unscaled_innovation_variances));

        SpdMatrix unscaled_precision =
            unscaled_state_precision() + A_transpose.sandwich(siginv);
        Cholesky precision_chol(unscaled_precision);

        Vector mean = precision_chol.solve(
            unscaled_state_precision() * state_mean()
            + A_transpose * (siginv * beta_next));

        return rmvn_precision_upper_cholesky_mt(
            rng, mean, sigma * precision_chol.getLT());
      }
      return Vector(0);
    }

    //======================================================================
    double DynamicRegressionKalmanFilter::filter(
        const DynamicRegressionModel &model) {
      ensure_storage(model.time_dimension());
      double ans = nodes_[0].initialize(
          model.inclusion_indicators(0),
          model.initial_state_mean(),
          model.unscaled_initial_state_precision(),
          *model.data(0),
          model.residual_variance());
      for (int t = 1; t < model.time_dimension(); ++t) {
        ans += nodes_[t].update(nodes_[t-1], *model.data(t), model, t);
      }
      return ans;
    }

    void DynamicRegressionKalmanFilter::simulate_coefficients(
        DynamicRegressionModel &model, RNG &rng) {
      for (int t = model.time_dimension() - 1; t >= 0; --t) {
        Vector beta = nodes_[t].simulate_coefficients(model, t, rng);
        model.set_included_coefficients(t, beta);
      }
    }

    double DynamicRegressionKalmanFilter::impute_state(
        DynamicRegressionModel &model, RNG &rng) {
      double ans = filter(model);
      simulate_coefficients(model, rng);
      return ans;
    }

    void DynamicRegressionKalmanFilter::ensure_storage(
        int number_of_time_points) {
      if (nodes_.size() < number_of_time_points) {
        nodes_.resize(number_of_time_points);
      }
    }

  }  // namespace StateSpace


  //===========================================================================
  namespace {
    using TSRDP = TimeSeriesRegressionDataPolicy;
  }  // namespace

  TSRDP::TimeSeriesRegressionDataPolicy(int xdim)
      : xdim_(xdim) {}

  void TSRDP::add_data(const Ptr<Data> &dp) {
    Ptr<RegressionData> reg_ptr = dp.dcast<RegressionData>();
    if (!!reg_ptr) {
      add_data(reg_ptr);
      return;
    }

    Ptr<StateSpace::RegressionDataTimePoint> time_point_ptr =
        dp.dcast<StateSpace::RegressionDataTimePoint>();
    if (!!time_point_ptr) {
      add_data(time_point_ptr);
      return;
    }
    std::ostringstream err;
    err << "Data point " << *dp << " could not be converted to either "
        << "RegressionData or RegressionDataTimePoint.";
    report_error(err.str());
  }

  void TSRDP::add_data(const Ptr<RegressionData> &dp) {
    if (data_.empty()) {
      data_.push_back(new StateSpace::RegressionDataTimePoint(xdim_));
    }
    data_.back()->add_data(dp);
    ensure_time_dimension();
  }

  void TSRDP::add_data(
      const Ptr<RegressionData> &dp,
      int time) {
    while (time >= data_.size()) {
      data_.push_back(new StateSpace::RegressionDataTimePoint(xdim_));
    }
    data_[time]->add_data(dp);
    ensure_time_dimension();
  }

  void TSRDP::add_data(
      const Ptr<StateSpace::RegressionDataTimePoint> &dp) {
    data_.push_back(dp);
    ensure_time_dimension();
  }

  void TSRDP::clear_data() {
    data_.clear();
  }

  void TSRDP::combine_data(
      const Model &other_model, bool just_suf) {
    report_error("Not implemented.");
  }

  //===========================================================================

  namespace {
    using DRM = DynamicRegressionModel;
  }  // namespace

  DRM::DynamicRegressionModel(int xdim) :
      TSRDP(xdim),
      residual_variance_(new UnivParams(1.0)),
      initial_state_mean_(xdim),
      unscaled_initial_state_variance_(new SpdParams(xdim)),
      innovation_variances_current_(false),
      innovation_variances_(xdim)
  {
    if (xdim <= 0) {
      report_error("xdim must be positive in DynamicRegressionModel.");
    }
    for (int i = 0; i < xdim; ++i) {
      NEW(ZeroMeanGaussianModel, innovation_model)(1.0);
      innovation_model->Sigsq_prm()->add_observer(
          this,
          [this]() {this->observe_innovation_variances();});
      innovation_error_models_.push_back(innovation_model);
      ManyParamPolicy::add_params(innovation_model->Sigsq_prm());

      NEW(MarkovModel, transition)(2);
      inclusion_transition_models_.push_back(transition);
      ManyParamPolicy::add_params(transition->Q_prm());
    }
    innovation_variances_current_ = false;
  }

  DRM::DynamicRegressionModel(const DRM &rhs)
      : Model(rhs),
        ManyParamPolicy(rhs),
        TimeSeriesRegressionDataPolicy(rhs),
        PriorPolicy(rhs),
        residual_variance_(rhs.residual_variance_->clone())
  {
    ManyParamPolicy::clear();
    ManyParamPolicy::add_params(residual_variance_);
    for (int i = 0; i < rhs.coefficients_.size(); ++i) {
      coefficients_.push_back(rhs.coefficients_[i]->clone());
      ManyParamPolicy::add_params(coefficients_.back());
    }

    //////
    //////
    //////
    //////  TODO(finish this when the class is complete.
    //////
    //////
    //////
  }

  void DRM::set_initial_state_mean(const Vector &mean) {
    if (mean.size() != xdim()) {
      report_error("Wrong size mean passed to set_initial_state_mean.");
    }
    initial_state_mean_ = mean;
  }

  void DRM::set_unscaled_initial_state_variance(const SpdMatrix &variance) {
    if (variance.nrow() != xdim()) {
      report_error("Wrong size variance passed to set_initial_state_variance.");
    }
    unscaled_initial_state_variance_->set_var(variance);
  }

  const Vector &DRM::unscaled_innovation_variances() const {
    refresh_innovation_variances();
    return innovation_variances_;
  }

  void DRM::ensure_time_dimension() {
    while (coefficients_.size() < time_dimension()) {
      Vector beta(xdim(), 0.0);
      // Variables begin as all excluded.
      Selector inc(xdim(), false);
      NEW(GlmCoefs, coefs)(beta, inc);
      coefficients_.push_back(coefs);
    }
  }

  void DRM::refresh_innovation_variances() const {
    if (innovation_variances_current_) return;
    for (int i = 0; i < innovation_error_models_.size(); ++i) {
      innovation_variances_[i] = innovation_error_models_[i]->sigsq();
    }
    innovation_variances_current_ = true;
  }

}  // namespace BOOM
