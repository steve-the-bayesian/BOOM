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

#include "Models/Glm/PosteriorSamplers/MultinomialLogitCompositeSpikeSlabSampler.hpp"
#include "Samplers/TIM.hpp"
#include "cpputil/math_utils.hpp"
#include "distributions.hpp"

namespace BOOM {
  namespace {
    typedef MultinomialLogitCompositeSpikeSlabSampler MLCS3;

    // A class for representing a chunk of the MultinomialLogit log posterior.
    // Usage:
    //   MultinomialLogitLogPosteriorChunk chunk(model, prior, 10, 7);
    //   Vector beta(10);
    //   double log_posterior = chunk(beta);
    class MultinomialLogitLogPosteriorChunk {
     public:
      // Args:
      //   model:  The model for which this is a chunk of the log posterior.
      //   prior:  The prior.
      //   max_chunk_size: The chunk size that will be used prior to
      //     the last chunk, which will use the remainder of beta.
      //   chunk_number:  The number of this chunk, counting from 0.
      MultinomialLogitLogPosteriorChunk(const MultinomialLogitModel *model,
                                        const MvnBase *prior,
                                        int max_chunk_size, int chunk_number)
          : model_(model),
            prior_(prior),
            chunk_size_(max_chunk_size),
            start_(max_chunk_size * chunk_number) {
        int beta_dim = model_->coef().inc().nvars();
        if (start_ >= beta_dim) {
          report_error(
              "Too large a chunk_number passed to "
              "MultinomialLogitLogPosteriorChunk constructor.");
        }
        if (beta_dim - start_ < chunk_size_) {
          chunk_size_ = beta_dim - start_;
        }
      }

      // Args:
      //   beta: The values at which to evaluate log_posterior.  The
      //     specified chunk evaluated at beta_chunk, and the rest of
      //     beta evaluated at its current value as stored in the
      //     model.
      // Returns: log posterior of beta_chunk.
      double operator()(const Vector &beta_chunk) const {
        Vector gradient;
        Matrix hessian;
        return (*this)(beta_chunk, gradient, hessian, 0);
      }

      // As above, but compute the gradient and Hessian with respect
      // to the chunk of beta.
      // Args:
      //   beta_chunk:  The values at which to evaluate log posterior.
      //   gradient: If nd > 0 then gradient will be filled with the
      //     gradient of log posterior.  Otherwise it is left unused.
      //   Hessian: If nd > 1 then Hessian will be filled with the
      //     matrix of second derivatives of log posterior.  Otherwise
      //     it is left unused.
      // Returns:
      //   Log posterior evaluated at beta_chunk.
      double operator()(const Vector &beta_chunk, Vector &gradient,
                        Matrix &Hessian, int nd) const {
        Vector beta = model_->coef().included_coefficients();
        VectorView beta_chunk_view(beta, start_, chunk_size_);
        beta_chunk_view = beta_chunk;

        // Indicates which elements of beta are being handled in this
        // chunk.
        Selector chunk_mask(beta.size(), false);
        for (int i = 0; i < chunk_size_; ++i) {
          int pos = start_ + i;
          chunk_mask.add(pos);
        }

        // The call to log_likelihood computes g and h with respect to
        // beta.  Afterwards, they need to be subset using chunk_mask.
        Vector g;
        Matrix h;
        double ans = model_->log_likelihood(beta, g, h, nd);

        Vector *gradient_pointer = nd > 0 ? &g : nullptr;
        Matrix *Hessian_pointer = nd > 1 ? &h : nullptr;
        const Selector &inclusion(model_->coef().inc());
        ans += prior_->logp_given_inclusion(beta, gradient_pointer,
                                            Hessian_pointer, inclusion, false);
        if (nd > 0) {
          gradient = chunk_mask.select(g);
          if (nd > 1) {
            Hessian = chunk_mask.select_square(h);
          }
        }
        return ans;
      }

     private:
      const MultinomialLogitModel *model_;
      const MvnBase *prior_;
      int chunk_size_;
      int start_;
    };

  }  // namespace

  //----------------------------------------------------------------------
  MLCS3::MultinomialLogitCompositeSpikeSlabSampler(
      MultinomialLogitModel *model, const Ptr<MvnBase> &prior,
      const Ptr<VariableSelectionPrior> &inclusion_prior, double tdf,
      double rwm_variance_scale_factor, uint nthreads, int max_chunk_size,
      bool check_initial_condition, RNG &seeding_rng)
      : MLVS(model, prior, inclusion_prior, nthreads, check_initial_condition,
             seeding_rng),
        model_(model),
        prior_(prior),
        inclusion_prior_(inclusion_prior),
        max_chunk_size_(max_chunk_size),
        tdf_(tdf),
        rwm_variance_scale_factor_(rwm_variance_scale_factor),
        move_probs_(".45 .45 .10") {
    if (max_chunk_size_ <= 0) {
      max_chunk_size_ = model_->beta().size();
    }
  }

  //----------------------------------------------------------------------
  void MLCS3::draw() {
    MoveType move(MoveType(rmulti_mt(rng(), move_probs_)));
    switch (move) {
      case DATA_AUGMENTATION_MOVE: {
        MoveTimer timer = accounting_.start_time("DA");
        MLVS::draw();
        accounting_.record_acceptance("DA");
      } break;

      case RWM_MOVE:
        rwm_draw();
        break;

      case TIM_MOVE:
        tim_draw();
        break;

      default:
        report_error(
            "Unknown move type sampled in "
            "MultinomialLogitCompositeSpikeSlabSampler::draw().");
    }
  }

  //----------------------------------------------------------------------
  void MLCS3::tim_draw() {
    int number_of_chunks = compute_number_of_chunks();
    if (number_of_chunks == 0) {
      return;
    }
    Vector beta = model_->coef().included_coefficients();
    int full_chunk_size = compute_chunk_size();
    for (int chunk = 0; chunk < number_of_chunks; ++chunk) {
      MoveTimer move_timer = accounting_.start_time("TIMchunk");
      MultinomialLogitLogPosteriorChunk logpost(model_, prior_.get(),
                                                full_chunk_size, chunk);
      TIM tim_sampler(logpost, tdf_);
      int start = full_chunk_size * chunk;
      int beta_dim = beta.size();  // type coercsion uint -> int
      int chunk_size = std::min<int>(full_chunk_size, beta_dim - start);
      VectorView beta_chunk(beta, start, chunk_size);
      bool found_mode = tim_sampler.locate_mode(beta_chunk);
      if (found_mode) {
        tim_sampler.fix_mode(true);
        beta_chunk = tim_sampler.draw(beta_chunk);
        if (tim_sampler.last_draw_was_accepted()) {
          accounting_.record_acceptance("TIMchunk");
          model_->coef().set_included_coefficients(beta);
        } else {
          accounting_.record_rejection("TIMchunk");
        }
      } else {
        accounting_.record_special("TIMchunk", "failed.to.find.mode");
        move_timer.stop();
        rwm_draw_chunk(chunk);
      }
    }
  }

  //----------------------------------------------------------------------
  void MLCS3::rwm_draw() {
    int number_of_chunks = compute_number_of_chunks();
    for (int chunk = 0; chunk < number_of_chunks; ++chunk) {
      rwm_draw_chunk(chunk);
    }
  }

  //----------------------------------------------------------------------
  void MLCS3::rwm_draw_chunk(int chunk) {
    MoveTimer move_timer = accounting_.start_time("RWMchunk");
    int chunk_size = compute_chunk_size();
    MultinomialLogitLogPosteriorChunk logpost(model_, prior_.get(), chunk_size,
                                              chunk);
    int chunk_begin = chunk_size * chunk;
    Vector beta = model_->coef().included_coefficients();
    int beta_dim = beta.size();  // type coercion
    chunk_size = std::min<int>(chunk_size, beta_dim - chunk_begin);
    VectorView beta_chunk(beta, chunk_begin, chunk_size);

    Vector gradient;
    Matrix Hessian;
    double original_logpost = logpost(beta_chunk, gradient, Hessian, 2);
    Vector candidate;
    if (tdf_ > 0) {
      candidate = rmvt_ivar_mt(rng(), beta_chunk,
                               -Hessian / rwm_variance_scale_factor_, tdf_);
    } else {
      candidate = rmvn_ivar_mt(rng(), beta_chunk,
                               -Hessian / rwm_variance_scale_factor_);
    }
    double candidate_logpost = logpost(candidate);
    double log_alpha = candidate_logpost - original_logpost;
    double logu = log(runif_mt(rng()));
    if (logu < log_alpha) {
      beta_chunk = candidate;
      model_->coef().set_included_coefficients(beta);
      accounting_.record_acceptance("RWMchunk");
    } else {
      accounting_.record_rejection("RWMchunk");
    }
  }

  //----------------------------------------------------------------------
  LabeledMatrix MLCS3::timing_report() const { return accounting_.to_matrix(); }

  void MLCS3::set_move_probabilities(double data_augmentation, double rwm,
                                     double tim) {
    if (data_augmentation < 0 || rwm < 0 || tim < 0) {
      report_error(
          "All probabilities must be non-negative in "
          "MultinomialLogitCompositeSpikeSlabSampler::"
          "set_move_probabilities().");
    }
    move_probs_[DATA_AUGMENTATION_MOVE] = data_augmentation;
    move_probs_[RWM_MOVE] = rwm;
    move_probs_[TIM_MOVE] = tim;
    double total = sum(move_probs_);
    if (total == 0.0) {
      report_error("At least one move probability must be positive.");
    }
    move_probs_ /= total;
  }

  //----------------------------------------------------------------------
  int MLCS3::compute_chunk_size() const {
    int nvars = model_->coef().nvars();
    if (max_chunk_size_ <= 0 || nvars == 0) return nvars;
    int number_of_full_chunks = nvars / max_chunk_size_;
    bool has_partial_chunk = number_of_full_chunks * max_chunk_size_ < nvars;
    int total_chunks = number_of_full_chunks + has_partial_chunk;
    int full_chunk_size = divide_rounding_up(nvars, total_chunks);
    return full_chunk_size;
  }

  //----------------------------------------------------------------------
  int MLCS3::compute_number_of_chunks() const {
    int beta_dim = model_->coef().nvars();
    int chunk_size = compute_chunk_size();
    if (chunk_size == 0) {
      return 0;
    }
    int number_of_chunks = beta_dim / chunk_size;
    if (chunk_size * number_of_chunks < beta_dim) {
      ++number_of_chunks;
    }
    return number_of_chunks;
  }

}  // namespace BOOM
