// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-20015 Steven L. Scott

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
#include "Models/PosteriorSamplers/DirichletPosteriorSampler.hpp"
#include <utility>
#include "Samplers/ScalarSliceSampler.hpp"
#include "cpputil/math_utils.hpp"
#include "distributions.hpp"

namespace BOOM {
  typedef DirichletPosteriorSampler DPS;

  // Args:
  //   model:  The model being managed.
  //   phi_prior: A prior for the discrete probability distribution
  //     parameter phi.  This should assign logp = negative_infinity
  //     if phi is not a discrete probability distribution.
  //   alpha_prior: A prior for the "sample size" parameter alpha.
  //     This should assign logp = negative_infinity if alpha <= 0.
  //   seeding_rng:  The random number generator used to seed the sampler.
  DPS::DirichletPosteriorSampler(DirichletModel *model,
                                 const Ptr<VectorModel> &phi_prior,
                                 const Ptr<DoubleModel> &alpha_prior,
                                 RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        phi_prior_(phi_prior),
        alpha_prior_(alpha_prior) {
    set_method(std::shared_ptr<DirichletSampler::DirichletSamplerImpl>(
                   new DirichletSampler::MlogitSliceImpl(
                       model_, phi_prior_.dcast<DiffVectorModel>(),
                       alpha_prior_.dcast<DiffDoubleModel>(), &rng())),
               1.0);
  }

  DPS *DPS::clone_to_new_host(Model *new_host) const {
    return new DPS(dynamic_cast<DirichletModel *>(new_host),
                   phi_prior_->clone(),
                   alpha_prior_->clone(),
                   rng());
  }

  double DPS::logpri() const {
    const Vector &nu(model_->nu());
    double alpha = sum(nu);
    double ans = alpha_prior_->logp(alpha);
    ans += phi_prior_->logp(nu / alpha);
    // Add in the Jacobian term to make the prior with respect to nu.
    ans -= (dim() - 1) * log(alpha);
    return ans;
  }

  void DPS::set_method(
      const std::shared_ptr<DirichletSampler::DirichletSamplerImpl> &method,
      double weight) {
    sampler_implementations_.clear();
    weights_.clear();
    add_method(method, weight);
  }

  void DPS::add_method(
      const std::shared_ptr<DirichletSampler::DirichletSamplerImpl> &method,
      double weight) {
    if (weight <= 0) {
      report_error("Argument 'weight' must be positive.");
    }
    sampler_implementations_.push_back(method);
    weights_.push_back(weight);
  }

  void DPS::draw_impl(
      const std::vector<std::shared_ptr<DirichletSampler::DirichletSamplerImpl>>
          &impl,
      const Vector &weights) {
    int n = impl.size();
    if (n == 0) {
      report_error("Either no sampling methods were set, or all failed.");
    }
    int which_sampler = 0;
    if (n > 1) {
      which_sampler = rmulti_mt(rng(), weights);
    }
    try {
      impl[which_sampler]->draw();
    } catch (std::exception &e) {
      if (n > 1) {
        std::vector<std::shared_ptr<DirichletSampler::DirichletSamplerImpl>>
            other_implementations(impl);
        other_implementations.erase(other_implementations.begin() +
                                    which_sampler);
        Vector other_weights = weights;
        other_weights.erase(other_weights.begin() + which_sampler);
        draw_impl(other_implementations, other_weights);
      } else {
        throw;
      }
    }
  }

  void DPS::draw() { draw_impl(sampler_implementations_, weights_); }

  uint DPS::dim() const { return model_->nu().size(); }

  //======================================================================
  namespace DirichletSampler {
    namespace {
      // Reflect the upper triangle of a square matrix to the lower
      // triangle.  The diagonal is not accessed.
      void reflect_upper_triangle(Matrix &m) {
        if (m.nrow() != m.ncol()) {
          report_error("Matrix must be square.");
        }
        for (int i = 0; i < m.nrow(); ++i) {
          for (int j = i + 1; j < m.ncol(); ++j) {
            m(j, i) = m(i, j);
          }
        }
      }
    }  // namespace

    typedef DirichletLogp DLP;
    DLP::DirichletLogp(uint pos, const Vector &nu, const Vector &sumlogpi,
                       double nobs, const VectorModel *phi_prior,
                       const DoubleModel *alpha_prior, double min_nu)
        : sumlogpi_(sumlogpi),
          nobs_(nobs),
          pos_(pos),
          nu_(nu),
          min_nu_(min_nu),
          alpha_prior_(alpha_prior),
          phi_prior_(phi_prior) {}

    //----------------------------------------------------------------------
    double DLP::operator()(double nu) const {
      if (nu < min_nu_) return BOOM::negative_infinity();
      nu_[pos_] = nu;
      return logp();
    }

    //----------------------------------------------------------------------
    double DLP::logp() const {
      double alpha = sum(nu_);
      if (alpha <= 0) return BOOM::negative_infinity();
      uint d = nu_.size();
      double ans = alpha_prior_->logp(alpha);  // alpha prior
      if (!std::isfinite(ans)) return ans;
      ans += phi_prior_->logp(nu_ / alpha);  // phi prior
      if (!std::isfinite(ans)) return ans;
      ans -= (d - 1) * log(alpha);  // jacobian
      ans += dirichlet_loglike(nu_, 0, 0, sumlogpi_, nobs_);
      return ans;
    }
    //----------------------------------------------------------------------
    DirichletSamplerImpl::DirichletSamplerImpl(
        DirichletModel *model, const Ptr<VectorModel> &phi_prior,
        const Ptr<DoubleModel> &alpha_prior, RNG *rng)
        : model_(model),
          phi_prior_(phi_prior),
          alpha_prior_(alpha_prior),
          rng_(rng) {}

    NuSliceImpl::NuSliceImpl(DirichletModel *model,
                             const Ptr<VectorModel> &phi_prior,
                             const Ptr<DoubleModel> &alpha_prior, RNG *rng)
        : DirichletSamplerImpl(model, phi_prior, alpha_prior, rng) {}

    void NuSliceImpl::draw() {
      Vector nu = model()->nu();
      uint dim = nu.size();
      for (uint i = 0; i < dim; ++i) {
        // We need to use the local variable nu here so that each
        // draw will be conditional on the most recent value of the
        // other nu elements.
        DirichletLogp logp(i, nu, model()->suf()->sumlog(), model()->suf()->n(),
                           phi_prior(), alpha_prior());
        ScalarSliceSampler sam(logp, true, 1.0, rng());
        sam.set_lower_limit(0);
        nu[i] = sam.draw(nu[i]);
      }
      model()->set_nu(nu);
    }

    //----------------------------------------------------------------------
    MlogitSliceImpl::MlogitSliceImpl(DirichletModel *model,
                                     const Ptr<DiffVectorModel> &phi_prior,
                                     const Ptr<DiffDoubleModel> &alpha_prior,
                                     RNG *rng)
        : DirichletSamplerImpl(model, phi_prior, alpha_prior, rng),
          phi_logpost_(MultinomialLogitLogPosterior(model, phi_prior)),
          phi_sampler_(phi_logpost_, 1.0, true, rng),
          alpha_logpost_(model, alpha_prior),
          log_alpha_sampler_(alpha_logpost_, true, 1.0, rng) {}

    void MlogitSliceImpl::draw() {
      Vector nu = model()->nu();
      double alpha = sum(nu);
      Vector phi = nu / alpha;
      Vector eta = phi_sampler_.draw(phi_logpost_.to_eta(phi));
      phi = phi_logpost_.to_full_phi(eta);
      model()->set_nu(alpha * phi);
      double log_alpha = log_alpha_sampler_.draw(log(alpha));
      model()->set_nu(exp(log_alpha) * phi);
    }

    //----------------------------------------------------------------------
    TimImpl::TimImpl(DirichletModel *model,
                     const Ptr<DiffVectorModel> &phi_prior,
                     const Ptr<DiffDoubleModel> &alpha_prior, RNG *rng)
        : DirichletSamplerImpl(model, phi_prior, alpha_prior, rng),
          phi_logpost_(MultinomialLogitLogPosterior(
              model, phi_prior.dcast<DiffVectorModel>())),
          phi_sampler_(phi_logpost_, 3, rng),
          alpha_logpost_(model, alpha_prior),
          log_alpha_sampler_(alpha_logpost_, true, 1.0, rng) {}

    void TimImpl::draw() {
      draw_alpha_given_phi();
      draw_phi_given_alpha();
    }

    void TimImpl::draw_alpha_given_phi() {
      Vector nu = model()->nu();
      double alpha = sum(nu);
      Vector phi = nu / alpha;
      if (min(phi) < 0) {
        report_error("All elements of nu must be non-negative.");
      }
      double log_alpha = log_alpha_sampler_.draw(log(alpha));
      nu = exp(log_alpha) * phi;
      model()->set_nu(nu);
    }

    void TimImpl::draw_phi_given_alpha() {
      Vector nu = model()->nu();
      double alpha = sum(nu);
      Vector eta = log(nu / nu[0]);
      eta.erase(eta.begin());
      eta = phi_sampler_.draw(eta);
      model()->set_nu(alpha * phi_logpost_.to_full_phi(eta));
    }

    //----------------------------------------------------------------------
    DirichletPhiLogPosterior::DirichletPhiLogPosterior(
        DirichletModel *model, const Ptr<DiffVectorModel> &phi_prior)
        : model_(model), phi_prior_(phi_prior) {}

    // Note the phi argument here has element 0 omitted because it is
    // a function of the other elements.
    double DirichletPhiLogPosterior::operator()(const Vector &truncated_phi,
                                                Vector &gradient,
                                                Matrix &Hessian,
                                                uint nderiv) const {
      if (truncated_phi.size() != model_->dim() - 1) {
        report_error("truncated_phi is the wrong size.");
      }
      double phi0 = 1 - sum(truncated_phi);
      if (phi0 <= 0 || phi0 >= 1.0) {
        return negative_infinity();
      }
      if (nderiv > 0) {
        gradient.resize(truncated_phi.size());
        gradient = 0;
        if (nderiv > 1) {
          Hessian.resize(truncated_phi.size(), truncated_phi.size());
          Hessian = 0;
        }
      }
      const Vector &sumlog(model_->suf()->sumlog());
      double n = model_->suf()->n();
      double alpha = sum(model_->nu());
      double ans = phi_prior_->Logp(truncated_phi, gradient, Hessian, nderiv);
      if (ans <= negative_infinity()) {
        return ans;
      }

      double nu0 = alpha * phi0;
      ans += (nu0 - 1) * sumlog[0] - n * lgamma(nu0);
      for (int s = 0; s < truncated_phi.size(); ++s) {
        double nu = truncated_phi[s] * alpha;
        ans += (nu - 1) * sumlog[s + 1] - n * lgamma(nu);
        if (nderiv > 0) {
          gradient[s] += alpha * (sumlog[s + 1] - sumlog[0]) -
                         n * alpha * (digamma(nu) - digamma(nu0));
          if (nderiv > 1) {
            for (int r = 0; r < truncated_phi.size(); ++r) {
              Hessian(r, s) -= n * square(alpha) * trigamma(nu0);
              if (r == s) {
                Hessian(s, s) -= n * square(alpha) * trigamma(nu);
              }
            }
          }
        }
      }
      return ans;
    }

    //----------------------------------------------------------------------

    typedef MultinomialLogitLogPosterior Mlogit;

    Mlogit::MultinomialLogitLogPosterior(DirichletModel *model,
                                         const Ptr<DiffVectorModel> &phi_prior)
        : model_(model), phi_prior_(phi_prior) {}

    Vector Mlogit::to_eta(const Vector &phi) const {
      Vector ans = log(phi / phi[0]);
      ans.erase(ans.begin());
      return ans;
    }

    Vector Mlogit::to_full_phi(const Vector &eta) const {
      Vector ans(eta.size() + 1);
      ans[0] = 0;
      VectorView(ans, 1) = eta;
      // By subtracting off the maximal value, the largest element of
      // exp(ans) will be 1, which will prevent overflow.
      ans -= max(ans);
      ans = exp(ans);
      ans /= sum(ans);
      return ans;
    }

    double Mlogit::scalar_derivative(const Vector &eta, double &derivative,
                                     int position) const {
      Vector gradient;
      Matrix hessian;
      double ans = operator()(eta, gradient, hessian, 1);
      derivative = gradient[position];
      return ans;
    }

    double Mlogit::operator()(const Vector &eta, Vector &gradient,
                              Matrix &Hessian, uint nderiv) const {
      Vector truncated_phi = to_full_phi(eta);
      truncated_phi.erase(truncated_phi.begin());
      DirichletPhiLogPosterior logp_phi(model_, phi_prior_);
      // Gradient and Hessian of logp(truncated_phi) with respect to
      // phi.
      Vector phi_gradient;
      Matrix phi_hessian;
      double ans = logp_phi(truncated_phi, phi_gradient, phi_hessian, nderiv);
      MultinomialLogitJacobian jacobian;
      ans += jacobian.logdet(truncated_phi);

      if (nderiv > 0) {
        SpdMatrix jacobian_matrix = jacobian.matrix(truncated_phi);
        gradient = jacobian_matrix * phi_gradient;
        jacobian.add_logits_gradient(truncated_phi, gradient, jacobian_matrix, true);

        if (nderiv > 1) {
          Hessian = sandwich(jacobian_matrix.transpose(), phi_hessian);
          for (int r = 0; r < phi_gradient.size(); ++r) {
            for (int s = 0; s < phi_gradient.size(); ++s) {
              for (int t = 0; t < phi_gradient.size(); ++t) {
                Hessian(r, s) += jacobian.second_order_element(
                    r, s, t, truncated_phi);
              }
            }
          }
          reflect_upper_triangle(Hessian);
          jacobian.add_logits_hessian(truncated_phi, Hessian, jacobian_matrix, true);
        }
      }
      return ans;
    }

    //----------------------------------------------------------------------
    LogAlphaLogPosterior::LogAlphaLogPosterior(
        DirichletModel *model, const Ptr<DiffDoubleModel> &alpha_prior)
        : model_(model), alpha_prior_(alpha_prior) {}

    double LogAlphaLogPosterior::operator()(double log_alpha, double &d1,
                                            double &d2, uint nderiv) const {
      double alpha = exp(log_alpha);
      if (!std::isfinite(alpha)) {
        return negative_infinity();
      }
      Vector phi = model_->nu() / sum(model_->nu());
      const DirichletSuf &suf(*model_->suf());
      const Vector &sumlog(suf.sumlog());
      double n = suf.n();
      if (nderiv > 0) {
        d1 = 0;
        if (nderiv > 1) {
          d2 = 0;
        }
      }
      double ans = alpha_prior_->Logp(alpha, d1, d2, nderiv);

      ans += n * lgamma(alpha);
      if (nderiv > 0) {
        d1 = n * digamma(alpha);
        if (nderiv > 1) {
          d2 = n * trigamma(alpha);
        }
      }

      for (int s = 0; s < phi.size(); ++s) {
        ans += (alpha * phi[s] - 1) * sumlog[s] - n * lgamma(alpha * phi[s]);
        if (nderiv > 0) {
          d1 += phi[s] * (sumlog[s] - n * digamma(alpha * phi[s]));
          if (nderiv > 1) {
            d2 += -n * trigamma(alpha * phi[s]) * square(phi[s]);
          }
        }
      }
      // Now d1 and d2 are derivatives with respect to alpha.  Use the
      // jacobian to transform them to derivatives with respect to
      // log_alpha.
      if (nderiv > 0) {
        // Let lambda = log_alpha.  The jacobian is J = d_alpha /
        // d_lambda = alpha.
        d1 *= alpha;
        if (nderiv > 1) {
          // The second derivative with respect to lambda comes from
          // the product rule:
          // d_d1 / d_lambda
          //     = (d_alpha / d_lamda) * alpha_gradient
          //       + alpha * d_alpha_gradient / d_lambda
          //     = d1 + alpha * d_alpha_gradient / d_alpha * d_alpha / d_lambda
          //     = d1 + alpha * alpha_hessian * alpha
          d2 = d1 + square(alpha) * d2;
        }
      }

      // Now we need to add in the log jacobian and its derivatives.
      ans += log_alpha;
      if (nderiv > 0) {
        d1 += 1;  // This is the derivative with respect to log_alpha.
        // The second derivative with respect to log alpha is zero, so
        // we don't need to do anything else even if nderiv > 1.
      }
      return ans;
    }

    LangevinImpl::LangevinImpl(DirichletModel *model,
                               const Ptr<DiffVectorModel> &phi_prior,
                               const Ptr<DiffDoubleModel> &alpha_prior,
                               RNG *rng)
        : DirichletSamplerImpl(model, phi_prior, alpha_prior, rng),
          phi_logpost_(new MultinomialLogitLogPosterior(model, phi_prior)),
          phi_sampler_(phi_logpost_, model->nu().size() - 1, .05, rng),
          alpha_logpost_(new LogAlphaLogPosterior(model, alpha_prior)),
          log_alpha_sampler_(alpha_logpost_, .05, rng) {
      phi_sampler_.allow_adaptation(false);
      log_alpha_sampler_.allow_adaptation(false);
    }

    void LangevinImpl::draw() {
      draw_alpha_given_phi();
      draw_phi_given_alpha();
    }

    void LangevinImpl::draw_alpha_given_phi() {
      Vector nu = model()->nu();
      double old_alpha = sum(nu);
      double log_alpha = log_alpha_sampler_.draw(log(old_alpha));
      nu *= exp(log_alpha) / old_alpha;
      model()->set_nu(nu);
    }

    void LangevinImpl::draw_phi_given_alpha() {
      Vector nu = model()->nu();
      double alpha = sum(nu);
      Vector eta = phi_sampler_.draw(phi_logpost_->to_eta(nu / alpha));
      model()->set_nu(alpha * phi_logpost_->to_full_phi(eta));
    }

  }  // namespace DirichletSampler

}  // namespace BOOM
