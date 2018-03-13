/*
  Copyright (C) 2005-2012 Steven L. Scott

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

#include <Models/StateSpace/StateModels/DynamicRegressionStateModel.hpp>
#include <distributions.hpp>
#include <stats/moments.hpp>
#include <cpputil/math_utils.hpp>

namespace BOOM {
  namespace {
    typedef DynamicRegressionStateModel DRSM;
  }

  DRSM::DynamicRegressionStateModel(const Matrix &X)
      : xdim_(ncol(X)),
        initial_state_mean_(ncol(X), 0.0),
        initial_state_variance_(ncol(X), 1.0),
        transition_matrix_(new IdentityMatrix(xdim_))
  {
    std::vector<Ptr<UnivParams> > diagonal_variances_;
    diagonal_variances_.reserve(xdim_);
    for (int i = 0; i < xdim_; ++i) {
      coefficient_transition_model_.push_back(new ZeroMeanGaussianModel);
      ParamPolicy::add_model(coefficient_transition_model_.back());
      diagonal_variances_.push_back(
          coefficient_transition_model_.back()->Sigsq_prm());
    }

    X_.reserve(nrow(X));
    for(int i = 0; i < nrow(X); ++i){
      SparseVector x(xdim_);
      for(int j = 0; j < xdim_; ++j) x[j] = X(i, j);
      X_.push_back(x);
    }

    predictor_variance_.reserve(ncol(X));
    for(int i = 0; i < ncol(X); ++i){
      predictor_variance_.push_back(var(X.col(i)));
    }

    transition_variance_matrix_.reset(new UpperLeftDiagonalMatrix(
        diagonal_variances_,
        diagonal_variances_.size()));
  }

  DRSM::DynamicRegressionStateModel(
      const DynamicRegressionStateModel &rhs)
      : StateModel(rhs),
        CompositeParamPolicy(rhs),
        NullDataPolicy(rhs),
        PriorPolicy(rhs),
        xdim_(rhs.xdim_),
        initial_state_mean_(rhs.initial_state_mean_),
        initial_state_variance_(rhs.initial_state_variance_),
        X_(rhs.X_),
        transition_matrix_(rhs.transition_matrix_)
  {
    coefficient_transition_model_.reserve(xdim_);
    std::vector<Ptr<UnivParams> > diagonal_variances_;
    diagonal_variances_.reserve(xdim_);
    for(int i = 0; i < xdim_; ++i){
      coefficient_transition_model_.push_back(
          rhs.coefficient_transition_model_[i]->clone());
      add_model(coefficient_transition_model_.back());
      diagonal_variances_.push_back(
          coefficient_transition_model_.back()->Sigsq_prm());
    }

    transition_variance_matrix_.reset(new UpperLeftDiagonalMatrix(
        diagonal_variances_,
        diagonal_variances_.size()));
  }

  DynamicRegressionStateModel * DRSM::clone()const{
    return new DynamicRegressionStateModel(*this);}

  void DRSM::set_xnames(
      const std::vector<string> &xnames) {
    if (xnames.size() != state_dimension()) {
      std::ostringstream err;
      err << "Error in DRSM::set_xnames." << endl
          << "The size of xnames is " << xnames.size() << endl
          << "But ncol(X) is " << state_dimension() << endl;
      report_error(err.str());
    }
    xnames_ = xnames;
  }

  const std::vector<string> & DRSM::xnames()const{
    return xnames_;
  }

  void DRSM::clear_data() {
    for(int i = 0; i < coefficient_transition_model_.size(); ++i){
      coefficient_transition_model_[i]->clear_data();
    }
  }

  void DRSM::observe_state(
      const ConstVectorView then, const ConstVectorView now, int time_now){
    check_size(then.size());
    check_size(now.size());
    for(int i = 0; i < then.size(); ++i){
      double change = now[i] - then[i];
      coefficient_transition_model_[i]->suf()->update_raw(change);
    }
  }

  void DRSM::observe_initial_state(
      const ConstVectorView &state){}

  uint DRSM::state_dimension()const{return xdim_;}

  void DRSM::update_complete_data_sufficient_statistics(
      int t,
      const ConstVectorView &state_error_mean,
      const ConstSubMatrix &state_error_variance) {
    for (int s = 0; s < coefficient_transition_model_.size(); ++s){
      coefficient_transition_model_[s]->suf()->update_expected_value(
          1.0,
          state_error_mean[s],
          state_error_variance(s, s) + square(state_error_mean[s]));
    }
  }

  void DRSM::simulate_state_error(RNG &rng, VectorView eta, int t)const{
    check_size(eta.size());
    for (int i = 0; i < eta.size(); ++i) {
      eta[i] = rnorm_mt(rng, 0, coefficient_transition_model_[i]->sigma());
    }
  }

  Ptr<SparseMatrixBlock>
      DRSM::state_transition_matrix(int t)const{
    return transition_matrix_;
  }

  Ptr<SparseMatrixBlock>
      DRSM::state_variance_matrix(int t)const{
    return transition_variance_matrix_;
  }

  Ptr<SparseMatrixBlock>
      DRSM::state_error_expander(int t) const {
    return state_transition_matrix(t);
  }

  Ptr<SparseMatrixBlock>
      DRSM::state_error_variance(int t) const {
    return state_variance_matrix(t);
  }

  SparseVector DRSM::observation_matrix(int t)const{
    return X_[t];
  }

  Vector DRSM::initial_state_mean()const{
    return initial_state_mean_;
  }

  void DRSM::set_initial_state_mean(const Vector &mu){
    check_size(mu.size());
    initial_state_mean_ = mu;
  }

  SpdMatrix DRSM::initial_state_variance()const{
    return initial_state_variance_;
  }

  void DRSM::set_initial_state_variance(const SpdMatrix &V) {
    check_size(V.nrow());
    initial_state_variance_ = V;
  }

  const GaussianSuf * DRSM::suf(int i)const{
    return coefficient_transition_model_[i]->suf().get();
  }

  double DRSM::sigsq(int i)const{
    return coefficient_transition_model_[i]->sigsq();
  }

  void DRSM::set_sigsq(double sigsq, int i){
    coefficient_transition_model_[i]->set_sigsq(sigsq);
  }

  const Vector & DRSM::predictor_variance()const{
   return predictor_variance_;
  }

  Ptr<UnivParams> DRSM::Sigsq_prm(int i){
    return coefficient_transition_model_[i]->Sigsq_prm();
  }

  const Ptr<UnivParams> DRSM::Sigsq_prm(int i)const{
    return coefficient_transition_model_[i]->Sigsq_prm();
  }

  void DRSM::add_forecast_data(
      const Matrix &predictors) {
    if (ncol(predictors) != xdim_) {
      report_error("Forecast data has the wrong number of columns");
    }
    for (int i = 0; i < nrow(predictors); ++i){
      SparseVector forecast_observation(xdim_);
      for (int j = 0; j < xdim_; ++j) {
        forecast_observation[j] = predictors(i, j);
      }
      X_.push_back(forecast_observation);
    }
  }

  void DRSM::increment_expected_gradient(
      VectorView gradient,
      int t,
      const ConstVectorView &state_error_mean,
      const ConstSubMatrix &state_error_variance) {
    if (gradient.size() != xdim_
        || state_error_mean.size() != xdim_
        || state_error_variance.nrow() != xdim_
        || state_error_variance.ncol() != xdim_) {
      report_error(
          "Wrong size arguments passed to "
          "DynamicRegressionStateModel::increment_expected_gradient.");
    }
    for (int i = 0; i < xdim_; ++i) {
      double mean = state_error_mean[i];
      double var = state_error_variance(i, i);
      double sigsq = DynamicRegressionStateModel::sigsq(i);;
      double tmp = (var + mean * mean) / (sigsq * sigsq) - 1.0 / sigsq;
      gradient[i] += .5 * tmp;
    }
  }

  void DRSM::check_size(int n)const{
    if (n != xdim_) {
      report_error("Wrong sized vector or matrix argument in"
                   " DynamicRegressionStateModel");
    }
  }

}
