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

#include "Models/Glm/PosteriorSamplers/BinomialLogitCompositeSpikeSlabSampler.hpp"
#include "Samplers/TIM.hpp"
#include "cpputil/math_utils.hpp"
#include "distributions.hpp"

#include <ctime>

namespace BOOM {
  double BinomialLogitLogPostChunk::operator()(const Vector &beta_chunk) const {
    Vector g;
    Matrix h;
    return (*this)(beta_chunk, g, h, 0);
  }
  //----------------------------------------------------------------------
  double BinomialLogitLogPostChunk::operator()(const Vector &beta_chunk,
                                               Vector &grad, Matrix &hess,
                                               int nd) const {
    Vector nonzero_beta = m_->included_coefficients();
    VectorView nonzero_beta_chunk(nonzero_beta, start_, chunk_size_);
    nonzero_beta_chunk = beta_chunk;

    const std::vector<Ptr<BinomialRegressionData> > &data(m_->dat());
    const Selector &inc(m_->coef().inc());
    const SpdMatrix siginv(inc.select(pri_->siginv()));
    const Vector mu(inc.select(pri_->mu()));

    double ans = dmvn(nonzero_beta, mu, siginv, 0.0, true);
    if (nd > 0) {
      Selector chunk_selector(nonzero_beta.size(), false);
      for (int i = start_; i < start_ + chunk_size_; ++i) chunk_selector.add(i);
      grad = -1 * chunk_selector.select(siginv * (nonzero_beta - mu));
      if (nd > 1) {
        hess = chunk_selector.select(siginv);
        hess *= -1;
      }
    }

    int nobs = data.size();
    for (int i = 0; i < nobs; ++i) {
      double yi = data[i]->y();
      double ni = data[i]->n();
      Vector x = inc.select(data[i]->x());
      double eta = nonzero_beta.dot(x);
      double prob = plogis(eta);
      ans += dbinom(yi, ni, prob, true);
      if (nd > 0) {
        const ConstVectorView x_chunk(x, start_, chunk_size_);
        grad.axpy(x_chunk, yi - ni * prob);
        if (nd > 1) {
          hess.add_outer(x_chunk, x_chunk, -ni * prob * (1 - prob));
        }
      }
    }
    return ans;
  }
  //----------------------------------------------------------------------
  typedef BinomialLogitCompositeSpikeSlabSampler BLCSSS;
  BLCSSS::BinomialLogitCompositeSpikeSlabSampler(
      BinomialLogitModel *model, const Ptr<MvnBase> &prior,
      const Ptr<VariableSelectionPrior> &vpri, int clt_threshold, double tdf,
      int max_tim_chunk_size, int max_rwm_chunk_size,
      double rwm_variance_scale_factor, RNG &seeding_rng)
      : BinomialLogitSpikeSlabSampler(model, prior, vpri, clt_threshold,
                                      seeding_rng),
        m_(model),
        pri_(prior),
        tdf_(tdf),
        max_tim_chunk_size_(max_tim_chunk_size),
        max_rwm_chunk_size_(max_rwm_chunk_size),
        rwm_variance_scale_factor_(rwm_variance_scale_factor) {
    set_sampler_weights(1.0, 1.0, 1.0);
  }
  //----------------------------------------------------------------------
  void BLCSSS::draw() {
    enum SamplingMethod { DATA_AUGMENTATION = 0, RWM = 1, TIM = 2 };
    SamplingMethod method = SamplingMethod(rmulti_mt(rng(), sampler_weights_));
    switch (method) {
      case DATA_AUGMENTATION: {
        MoveTimer timer = move_accounting_.start_time("auxmix");
        BinomialLogitSpikeSlabSampler::draw();
        move_accounting_.record_acceptance("auxmix");
        break;
      }
      case RWM: {
        MoveTimer timer = move_accounting_.start_time("rwm (total time)");
        rwm_draw();
        break;
      }
      case TIM: {
        MoveTimer timer = move_accounting_.start_time("TIM (total time)");
        tim_draw();
        break;
      }
      default:
        report_error("Unknown method in BinomialLogitSpikeSlabSampler::draw.");
    }  // switch
  }
  //----------------------------------------------------------------------
  void BLCSSS::rwm_draw() {
    if (m_->coef().nvars() == 0) return;
    int total_number_of_chunks = compute_number_of_chunks(max_rwm_chunk_size_);
    for (int chunk = 0; chunk < total_number_of_chunks; ++chunk) {
      rwm_draw_chunk(chunk);
    }
  }
  //----------------------------------------------------------------------
  void BLCSSS::rwm_draw_chunk(int chunk) {
    const Selector &inc(m_->coef().inc());
    int nvars = inc.nvars();
    Vector full_nonzero_beta = m_->included_coefficients();
    // Compute information matrix for proposal distribution.  For
    // efficiency, also compute the log-posterior of the current beta.
    Vector mu(inc.select(pri_->mu()));
    SpdMatrix siginv(inc.select(pri_->siginv()));
    double original_logpost = dmvn(full_nonzero_beta, mu, siginv, 0, true);

    const std::vector<Ptr<BinomialRegressionData> > &data(m_->dat());
    int nobs = data.size();

    int full_chunk_size = compute_chunk_size(max_rwm_chunk_size_);
    int chunk_start = chunk * full_chunk_size;
    int elements_remaining = nvars - chunk_start;
    int this_chunk_size = std::min(elements_remaining, full_chunk_size);
    Selector chunk_selector(nvars, false);
    for (int i = chunk_start; i < chunk_start + this_chunk_size; ++i) {
      chunk_selector.add(i);
    }

    SpdMatrix proposal_ivar = chunk_selector.select(siginv);

    for (int i = 0; i < nobs; ++i) {
      Vector x = inc.select(data[i]->x());
      double eta = x.dot(full_nonzero_beta);
      double prob = plogis(eta);
      double weight = prob * (1 - prob);
      VectorView x_chunk(x, chunk_start, this_chunk_size);
      // Only upper triangle is accessed.  Need to reflect at end of loop.
      proposal_ivar.add_outer(x_chunk, weight, false);
      original_logpost += dbinom(data[i]->y(), data[i]->n(), prob, true);
    }
    proposal_ivar.reflect();
    VectorView beta_chunk(full_nonzero_beta, chunk_start, this_chunk_size);
    if (tdf_ > 0) {
      beta_chunk = rmvt_ivar_mt(
          rng(), beta_chunk, proposal_ivar / rwm_variance_scale_factor_, tdf_);
    } else {
      beta_chunk = rmvn_ivar_mt(rng(), beta_chunk,
                                proposal_ivar / rwm_variance_scale_factor_);
    }

    double logpost = dmvn(full_nonzero_beta, mu, siginv, 0, true);
    Vector full_beta(inc.expand(full_nonzero_beta));
    logpost += m_->log_likelihood(full_beta, 0, 0, false);
    double log_alpha = logpost - original_logpost;
    double logu = log(runif_mt(rng()));
    if (logu < log_alpha) {
      m_->set_included_coefficients(full_nonzero_beta);
      move_accounting_.record_acceptance("rwm_chunk");
    } else {
      move_accounting_.record_rejection("rwm_chunk");
    }
  }
  //----------------------------------------------------------------------
  // TODO:  This code currently discards the tim_sampler
  void BLCSSS::tim_draw() {
    int nvars = m_->coef().nvars();
    if (nvars == 0) return;
    int chunk_size = compute_chunk_size(max_tim_chunk_size_);
    int number_of_chunks = compute_number_of_chunks(max_tim_chunk_size_);
    assert(number_of_chunks * chunk_size >= nvars);

    for (int chunk = 0; chunk < number_of_chunks; ++chunk) {
      clock_t mode_start = clock();
      TIM tim_sampler(log_posterior(chunk, max_tim_chunk_size_), tdf_, &rng());
      Vector beta = m_->included_coefficients();
      int start = chunk_size * chunk;
      int elements_remaining = nvars - start;
      VectorView beta_chunk(beta, start,
                            std::min(elements_remaining, chunk_size));
      bool ok = tim_sampler.locate_mode(beta_chunk);
      move_accounting_.stop_time("tim mode finding", mode_start);
      if (ok) {
        move_accounting_.record_acceptance("tim mode finding");
        tim_sampler.fix_mode(true);
        MoveTimer timer = move_accounting_.start_time("TIM chunk");
        beta_chunk = tim_sampler.draw(beta_chunk);
        m_->set_included_coefficients(beta);
        if (tim_sampler.last_draw_was_accepted()) {
          move_accounting_.record_acceptance("TIM chunk");
        } else {
          move_accounting_.record_rejection("TIM chunk");
        }
      } else {
        move_accounting_.record_rejection("tim mode finding");
        rwm_draw_chunk(chunk);
      }
    }
  }
  //----------------------------------------------------------------------
  BinomialLogitLogPostChunk BLCSSS::log_posterior(int chunk,
                                                  int max_chunk_size) const {
    return BinomialLogitLogPostChunk(m_, pri_.get(),
                                     compute_chunk_size(max_chunk_size), chunk);
  }
  //----------------------------------------------------------------------
  void BLCSSS::set_sampler_weights(double da_weight, double rwm_weight,
                                   double tim_weight) {
    if (da_weight < 0 || rwm_weight < 0 || tim_weight < 0) {
      report_error("All three weights must be non-negative.");
    }
    if (da_weight <= 0 && rwm_weight <= 0 && tim_weight <= 0) {
      report_error("At least one weight must be positive.");
    }
    sampler_weights_.resize(3);
    sampler_weights_[0] = da_weight;
    sampler_weights_[1] = rwm_weight;
    sampler_weights_[2] = tim_weight;
    sampler_weights_ /= sum(sampler_weights_);
  }
  //----------------------------------------------------------------------
  int BLCSSS::compute_chunk_size(int max_chunk_size) const {
    int nvars = m_->coef().nvars();
    if (max_chunk_size <= 0) return nvars;
    int number_of_full_chunks = nvars / max_chunk_size;
    bool has_partial_chunk = number_of_full_chunks * max_chunk_size < nvars;
    int total_chunks = number_of_full_chunks + has_partial_chunk;
    int full_chunk_size = divide_rounding_up(nvars, total_chunks);
    return full_chunk_size;
  }
  //----------------------------------------------------------------------
  int BLCSSS::compute_number_of_chunks(int max_chunk_size) const {
    if (max_chunk_size <= 0) return 1;
    int nvars = m_->coef().nvars();
    int number_of_full_chunks = nvars / max_chunk_size;
    bool has_partial_chunk = number_of_full_chunks * max_chunk_size < nvars;
    return number_of_full_chunks + has_partial_chunk;
  }

  std::ostream &BLCSSS::time_report(std::ostream &out) const {
    out << move_accounting_.to_matrix();
    return out;
  }
}  // namespace BOOM
