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
#include <algorithm>

#include "Models/DoubleModel.hpp"
#include "Models/ModelTypes.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "Models/VectorModel.hpp"
#include "TargetFun/Loglike.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"
#include "numopt.hpp"

namespace BOOM {

  Model::Model() {}

  Model::Model(const Model &) : RefCounted() {}

  Vector Model::vectorize_params(bool minimal) const {
    std::vector<Ptr<Params>> prm(parameter_vector());
    uint nprm = prm.size();
    uint N(0), nmax(0);
    for (uint i = 0; i < nprm; ++i) {
      uint n = prm[i]->size();
      N += n;
      nmax = std::max(nmax, n);
    }
    Vector ans(N);
    Vector workspace(nmax);
    Vector::iterator it = ans.begin();
    for (uint i = 0; i < nprm; ++i) {
      workspace = prm[i]->vectorize(minimal);
      it = std::copy(workspace.begin(), workspace.end(), it);
    }
    return ans;
  }

  void Model::unvectorize_params(const Vector &v, bool minimal) {
    std::vector<Ptr<Params>> prm(parameter_vector());
    Vector::const_iterator b = v.begin();
    for (uint i = 0; i < prm.size(); ++i) {
      b = prm[i]->unvectorize(b, minimal);
    }
  }

  //============================================================
  void PosteriorModeModel::find_posterior_mode(double epsilon) {
    if (number_of_sampling_methods() != 1) {
      report_error("find_posterior_mode requires a single posterior sampler.");
    }
    PosteriorSampler *posterior_sampler = sampler(0);
    if (!posterior_sampler->can_find_posterior_mode()) {
      report_error("Posterior sampler does not implement find_posterior_mode.");
    } else {
      posterior_sampler->find_posterior_mode(epsilon);
    }
  }

  bool PosteriorModeModel::can_find_posterior_mode() const {
    if (number_of_sampling_methods() != 1) {
      return false;
    }
    return sampler(0)->can_find_posterior_mode();
  }

  double PosteriorModeModel::log_prior_density(
      const ConstVectorView &parameters) const {
    if (number_of_sampling_methods() != 1) {
      report_error("log_prior_density requires a single posterior sampler.");
    }
    const PosteriorSampler *posterior_sampler = sampler(0);
    if (!posterior_sampler->can_evaluate_log_prior_density()) {
      report_error(
          "Posterior sampler does not implement "
          "log_prior_density.");
    }
    return posterior_sampler->log_prior_density(parameters);
  }

  bool PosteriorModeModel::can_evaluate_log_prior_density() const {
    if (number_of_sampling_methods() != 1) {
      return false;
    }
    return sampler(0)->can_evaluate_log_prior_density();
  }

  double PosteriorModeModel::increment_log_prior_gradient(
      const ConstVectorView &parameters, VectorView gradient) const {
    if (number_of_sampling_methods() != 1) {
      report_error(
          "increment_log_prior_gradient requires a "
          "single posterior sampler.");
    }
    const PosteriorSampler *posterior_sampler = sampler(0);
    if (!posterior_sampler->can_increment_log_prior_gradient()) {
      report_error(
          "Posterior sampler does not implement "
          "increment_log_prior_gradient.");
    }
    return posterior_sampler->increment_log_prior_gradient(parameters,
                                                           gradient);
  }

  bool PosteriorModeModel::can_increment_log_prior_gradient() const {
    if (number_of_sampling_methods() != 1) {
      return false;
    }
    return sampler(0)->can_increment_log_prior_gradient();
  }

  //============================================================
  void MLE_Model::initialize_params() { mle(); }

  MLE_Model::MLE_Model(MLE_Model &&rhs)
      : Model(rhs),
        status_(rhs.status_),
        error_message_(rhs.error_message_) {}

  MLE_Model & MLE_Model::operator=(MLE_Model &&rhs) {
    if (&rhs != this) {
      status_ = rhs.status_;
      error_message_ = rhs.error_message_;
      Model::operator=(rhs);
    }
    return *this;
  }

  //============================================================
  void LoglikeModel::mle() {
    LoglikeTF loglike(this);
    Vector prms = vectorize_params(true);
    max_nd0(prms, Target(loglike));
    unvectorize_params(prms, true);
  }

  void dLoglikeModel::mle() {
    dLoglikeTF loglike(this);
    Vector prms = vectorize_params(true);
    double logf;
    std::string error_message;
    bool ok = max_nd1_careful(prms, logf, Target(loglike), dTarget(loglike),
                              error_message, 1e-5);
    if (ok) {
      MLE_Model::set_status(SUCCESS, "");
      unvectorize_params(prms, true);
    } else {
      MLE_Model::set_status(FAILURE,
                            "MLE exceeded maximum number of iterations.");
    }
  }

  void d2LoglikeModel::mle() {
    Vector gradient;
    Matrix Hessian;
    mle_result(gradient, Hessian);
  }

  double d2LoglikeModel::mle_result(Vector &gradient, Matrix &Hessian) {
    d2LoglikeTF loglike(this);
    Vector parameters = vectorize_params(true);
    uint p = parameters.size();
    gradient.resize(p);
    Hessian.resize(p, p);
    std::string error_message;
    double logf;
    bool ok = max_nd2_careful(parameters, gradient, Hessian, logf,
                              Target(loglike), dTarget(loglike),
                              d2Target(loglike), 1e-5, error_message);
    if (ok) {
      unvectorize_params(parameters, true);
      MLE_Model::set_status(SUCCESS, error_message);
      return logf;
    } else {
      MLE_Model::set_status(FAILURE, error_message);
      return negative_infinity();
    }
  }

  double DoubleModel::pdf(const Ptr<Data> &dp, bool logscale) const {
    double x = dp.dcast<DoubleData>()->value();
    double ans = logp(x);
    return logscale ? ans : exp(ans);
  }

  double DoubleModel::pdf(const Data *dp, bool logscale) const {
    double x = dynamic_cast<const DoubleData *>(dp)->value();
    double ans = logp(x);
    return logscale ? ans : exp(ans);
  }

  //======================================================================
  double DiffDoubleModel::logp(double x) const {
    double g(0), h(0);
    return Logp(x, g, h, 0);
  }
  double DiffDoubleModel::dlogp(double x, double &g) const {
    double h(0);
    return Logp(x, g, h, 1);
  }
  double DiffDoubleModel::d2logp(double x, double &g, double &h) const {
    return Logp(x, g, h, 2);
  }
  //======================================================================
  double DiffVectorModel::logp(const Vector &x) const {
    Vector g;
    Matrix h;
    return Logp(x, g, h, 0);
  }
  double DiffVectorModel::dlogp(const Vector &x, Vector &g) const {
    Matrix h;
    return Logp(x, g, h, 1);
  }
  double DiffVectorModel::d2logp(const Vector &x, Vector &g, Matrix &h) const {
    return Logp(x, g, h, 2);
  }

}  // namespace BOOM
