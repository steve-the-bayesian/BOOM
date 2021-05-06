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

#include "Models/PosteriorSamplers/BetaBinomialPosteriorSampler.hpp"
#include <functional>

#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"
#include "numopt.hpp"
#include "numopt/NumericalDerivatives.hpp"
#include "stats/logit.hpp"

namespace BOOM {

  typedef BetaBinomialPosteriorSampler BBPS;

  //======================================================================
  // The log posterior of the beta binomial model on the (prob,
  // sample_size) scale.
  BetaBinomialLogPosterior::BetaBinomialLogPosterior(
      const BetaBinomialModel *model, const Ptr<BetaModel> &probability_prior,
      const Ptr<DiffDoubleModel> &sample_size_prior)
      : model_(model),
        probability_prior_(probability_prior),
        sample_size_prior_(sample_size_prior) {}

  double BetaBinomialLogPosterior::operator()(const Vector &prob_samplesize,
                                              Vector &gradient, Matrix &Hessian,
                                              uint nderiv) const {
    double prob = prob_samplesize[0];
    double sample_size = prob_samplesize[1];
    double a = prob * sample_size;
    double b = sample_size - a;
    Vector ab{a, b};
    double ans = model_->Loglike(ab, gradient, Hessian, nderiv);

    ProbSamplesizeJacobian jacobian;
    jacobian.evaluate_new_parameterization(prob_samplesize);
    if (nderiv > 0) {
      Vector original_gradient = gradient;
      // Transform the gradient from the (a,b) scale to the
      // (prob,size) scale.
      gradient = jacobian.transform_gradient(original_gradient, false);
      if (nderiv > 1) {
        Hessian = jacobian.transform_Hessian(original_gradient, Hessian, false);
      }
    }

    double prob_first_derivative, prob_second_derivative;
    ans += probability_prior_->Logp(prob, prob_first_derivative,
                                    prob_second_derivative, nderiv);
    double size_first_derivative, size_second_derivative;
    ans += sample_size_prior_->Logp(sample_size, size_first_derivative,
                                    size_second_derivative, nderiv);
    if (nderiv > 0) {
      gradient[0] += prob_first_derivative;
      gradient[1] += size_first_derivative;
      if (nderiv > 1) {
        Hessian(0, 0) += prob_second_derivative;
        Hessian(1, 1) += size_second_derivative;
      }
    }
    return ans;
  }

  namespace {
    struct LogitLogToProbSampleSize {
      // Args:
      //   eta_nu: A two-element vector with eta = logit(prob) in the
      //     first position and nu = log(sample_size) in the second.
      // Returns:
      //   A two component vector (prob, sample_size).
      Vector operator()(const Vector &eta_nu) const {
        Vector ans(2);
        ans[0] = plogis(eta_nu[0]);
        ans[1] = exp(eta_nu[1]);
        return ans;
      }
    };
  }  // namespace

  BBPS::BetaBinomialPosteriorSampler(
      BetaBinomialModel *model, const Ptr<BetaModel> &probability_prior,
      const Ptr<DiffDoubleModel> &sample_size_prior, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        probability_prior_(probability_prior),
        sample_size_prior_(sample_size_prior),
        probability_sampler_(
            [this](double prob) { return this->logp_prob(prob); }),
        sample_size_sampler_([this](double sample_size) {
          return this->logp_sample_size(sample_size);
        }),
        sampling_method_(DATA_AUGMENTATION),
        trouble_locating_mode_(false) {
    probability_sampler_.set_limits(0, 1);
    probability_sampler_.set_rng(&rng(), false);
    sample_size_sampler_.set_lower_limit(0);
    sample_size_sampler_.set_rng(&rng(), false);
    model_->add_observer([this] { this->observe_new_data(); });
  }

  double BBPS::logpri() const {
    double prob = model_->prior_mean();
    double sample_size = model_->prior_sample_size();
    return probability_prior_->logp(prob) +
           sample_size_prior_->logp(sample_size);
  }

  BBPS *BBPS::clone_to_new_host(Model *new_host) const {
    return new BBPS(dynamic_cast<BetaBinomialModel *>(new_host),
                    probability_prior_->clone(),
                    sample_size_prior_->clone(),
                    rng());
  }

  void BBPS::draw() {
    switch (sampling_method_) {
      case SLICE:
        draw_slice();
        return;

      case DATA_AUGMENTATION:
        draw_data_augmentation();
        return;

      case TIM:
        draw_tim();
        return;

      default:
        draw_data_augmentation();
        return;
    }
  }

  void BBPS::draw_data_augmentation() {
    double a = model_->a();
    double b = model_->b();
    complete_data_suf_.clear();

    const std::vector<Ptr<BinomialData> > &data(model_->dat());
    int nobs = data.size();
    for (int i = 0; i < nobs; ++i) {
      int y = data[i]->y();
      int n = data[i]->n();
      double theta;
      int failure_count = 0;
      do {
        // In obscure corner cases where either a or b is very close
        // to zero you can get theta == 0.0 or theta == 1.0
        // numerically.  In that case just keep trying.  If it takes
        // more than 100 tries then something is really wrong.
        theta = rbeta_mt(rng(), y + a, n - y + b);
        if (++failure_count > 100) {
          report_error(
              "Too many attempts at rbeta in "
              "BetaBinomialPosteriorSampler::draw_data_augmentation");
        }
      } while (theta == 0.0 || theta == 1.0 || !std::isfinite(theta));
      complete_data_suf_.update_raw(theta);
    }
    draw_slice();
  }

  void BBPS::draw_slice() {
    double prob = model_->prior_mean();
    prob = probability_sampler_.draw(prob);
    model_->set_prior_mean(prob);

    double sample_size = model_->prior_sample_size();
    sample_size = sample_size_sampler_.draw(sample_size);
    model_->set_prior_sample_size(sample_size);
  }

  double BBPS::logp(double prob, double sample_size) const {
    double a = prob * sample_size;
    double b = sample_size - a;
    double ans =
        probability_prior_->logp(prob) + sample_size_prior_->logp(sample_size);
    if (sampling_method_ == DATA_AUGMENTATION) {
      ans += beta_log_likelihood(a, b, complete_data_suf_);
    } else {
      ans += model_->loglike(a, b);
    }
    return ans;
  }

  double BBPS::logp_sample_size(double sample_size) const {
    double prob = model_->prior_mean();
    return logp(prob, sample_size);
  }

  double BBPS::logp_prob(double prob) const {
    double sample_size = model_->prior_sample_size();
    return logp(prob, sample_size);
  }

  void BBPS::set_prior_on_sample_size(
      const Ptr<DiffDoubleModel> &sample_size_prior) {
    sample_size_prior_ = sample_size_prior;
    if (!!tim_sampler_) {
      // Setting a new prior distribution invalidates the proxy object
      // in the MH sampler, so it needs to be re-initialized.  The
      // proxy objects in other samplers call *this, so they don't
      // need to be re-initialized.
      allocate_tim_sampler();
    }
  }

  BetaBinomialLogPosterior BBPS::prob_sample_size_log_posterior() {
    return BetaBinomialLogPosterior(model_, probability_prior_,
                                    sample_size_prior_);
  }

  Transformation BBPS::approximately_gaussian_log_posterior() {
    LogitLogToProbSampleSize inverse_transformation;
    return Transformation(prob_sample_size_log_posterior(),
                          inverse_transformation, new LogitLogJacobian);
  }

  void BBPS::find_posterior_mode(double epsilon) {
    Vector gaussian_approx_parameters(2);
    gaussian_approx_parameters[0] = qlogis(model_->prior_mean());
    gaussian_approx_parameters[1] = log(model_->prior_sample_size());
    Transformation log_posterior = approximately_gaussian_log_posterior();
    Vector gradient(2);
    Matrix hessian(2, 2);
    double max_value = negative_infinity();
    std::string error_message;
    bool ok = max_nd2_careful(gaussian_approx_parameters, gradient, hessian,
                              max_value, log_posterior, log_posterior,
                              log_posterior, epsilon, error_message);
    if (!ok) {
      report_error("Trouble finding posterior mode:\n" + error_message);
    }
    if (!tim_proposal_distribution_) {
      tim_proposal_distribution_.reset(
          new MvtIndepProposal(gaussian_approx_parameters, -hessian, 1.0));
    } else {
      tim_proposal_distribution_->set_mu(gaussian_approx_parameters);
      tim_proposal_distribution_->set_ivar(-hessian);
    }
    double prob = plogis(gaussian_approx_parameters[0]);
    double sample_size = exp(gaussian_approx_parameters[1]);
    double a = prob * sample_size;
    double b = sample_size - a;
    model_->set_a(a);
    model_->set_b(b);
  }

  void BBPS::draw_tim() {
    if (trouble_locating_mode_) {
      draw_data_augmentation();
    } else {
      try {
        if (!tim_sampler_) {
          allocate_tim_sampler();
        }
        Vector gaussian_approx_parameters(2);
        gaussian_approx_parameters[0] = qlogis(model_->prior_mean());
        gaussian_approx_parameters[1] = log(model_->prior_sample_size());
        gaussian_approx_parameters =
            tim_sampler_->draw(gaussian_approx_parameters);
        if (tim_sampler_->last_draw_was_accepted()) {
          model_->set_prior_mean(plogis(gaussian_approx_parameters[0]));
          model_->set_prior_sample_size(exp(gaussian_approx_parameters[1]));
        }
      } catch (...) {
        trouble_locating_mode_ = true;
        draw_data_augmentation();
      }
    }
  }

  void BBPS::allocate_tim_sampler() {
    if (!tim_proposal_distribution_) {
      // Calling find_posterior_mode allocates
      // tim_proposal_distribution_.
      find_posterior_mode();
    }
    tim_sampler_.reset(
        new MetropolisHastings(approximately_gaussian_log_posterior(),
                               tim_proposal_distribution_, &rng()));
  }

  namespace {
    typedef ProbSamplesizeJacobian PSJ;
  }

  PSJ::ProbSamplesizeJacobian() { prefer_new_parameterization(); }

  void PSJ::evaluate_original_parameterization(const Vector &ab) {
    sample_size_ = ab[0] + ab[1];
    prob_ = ab[0] / sample_size_;
    matrix_is_current_ = false;
  }

  void PSJ::evaluate_new_parameterization(const Vector &prob_samplesize) {
    prob_ = prob_samplesize[0];
    sample_size_ = prob_samplesize[1];
    matrix_is_current_ = false;
  }

  Matrix &PSJ::matrix() {
    if (matrix_is_current_) {
      return matrix_;
    }
    matrix_.resize(2, 2);
    matrix_(0, 0) = sample_size_;
    matrix_(0, 1) = -sample_size_;
    matrix_(1, 0) = prob_;
    matrix_(1, 1) = 1 - prob_;
    matrix_is_current_ = true;
    return matrix_;
  }

  void PSJ::transform_second_order_gradient(SpdMatrix &working_hessian,
                                            const Vector &original_gradient) {
    working_hessian(0, 1) +=
        second_order_element(0, 1, 0) * original_gradient[0] +
        second_order_element(0, 1, 1) * original_gradient[1];
    working_hessian(1, 0) +=
        second_order_element(1, 0, 0) * original_gradient[0] +
        second_order_element(1, 0, 1) * original_gradient[1];
  }

  void PSJ::add_logdet_gradient(Vector &gradient) {
    gradient[1] += 1.0 / sample_size_;
  }

  void PSJ::add_logdet_Hessian(Matrix &Hessian) {
    Hessian(1, 1) -= 1.0 / square(sample_size_);
  }
  //======================================================================
  namespace {
    typedef LogitLogJacobian LLJ;
  }

  LLJ::LogitLogJacobian() { prefer_original_parameterization(); }

  void LLJ::evaluate_original_parameterization(const Vector &prob_size) {
    prob_ = prob_size[0];
    sample_size_ = prob_size[1];
    matrix_is_current_ = false;
  }

  void LLJ::evaluate_new_parameterization(const Vector &eta_nu) {
    prob_ = plogis(eta_nu[0]);
    sample_size_ = exp(eta_nu[1]);
    matrix_is_current_ = false;
  }

  double LLJ::logdet() { return log(prob_ * (1 - prob_) * sample_size_); }

  Matrix &LLJ::matrix() {
    matrix_.resize(2, 2);
    matrix_(0, 0) = prob_ * (1 - prob_);
    matrix_(1, 1) = sample_size_;
    matrix_(1, 0) = matrix_(0, 1) = 0.0;
    matrix_is_current_ = true;
    return matrix_;
  }

  void LLJ::transform_second_order_gradient(SpdMatrix &working_hessian,
                                            const Vector &original_gradient) {
    working_hessian(0, 0) +=
        second_order_element(0, 0, 0) * original_gradient[0];
    working_hessian(1, 1) +=
        second_order_element(1, 1, 1) * original_gradient[1];
  }

  void LLJ::add_logdet_gradient(Vector &gradient) {
    // Derivative of log(p(1-p)) / d eta
    //   = (1/pq) * d(pq) / d eta
    //   = (1/pq) * (1 - 2p) * dp / d eta
    //   = (1/pq) * (1 - 2p) * pq
    //   = 1-2p
    gradient[0] += 1.0 - 2.0 * prob_;

    // Derivative of log(sample_size) / d nu
    //   = 1/sample_size * dsample_size / d nu
    //   = 1/sample_size * sample_size
    //   = 1.0;
    gradient[1] += 1.0;
  }

  void LLJ::add_logdet_Hessian(Matrix &hessian) {
    // Cross derivatives are zero, and the second component of the
    // gradient is a constant, so there is only one nonzero component
    // of the Hessian.
    //
    // d (1-2p) / d_eta = -2 d_p/d_eta = -2pq
    hessian(0, 0) -= 2.0 * prob_ * (1 - prob_);
  }

}  // namespace BOOM
