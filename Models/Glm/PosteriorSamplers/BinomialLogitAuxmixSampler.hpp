// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2013 Steven L. Scott

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

#ifndef BOOM_BINOMIAL_LOGIT_AUXMIX_SAMPLER_HPP_
#define BOOM_BINOMIAL_LOGIT_AUXMIX_SAMPLER_HPP_

#include "Models/PosteriorSamplers/Imputer.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"

#include "Models/Glm/BinomialLogitModel.hpp"
#include "Models/Glm/PosteriorSamplers/BinomialLogitDataImputer.hpp"
#include "Models/MvnBase.hpp"
#include "cpputil/RefCounted.hpp"

namespace BOOM {
  namespace BinomialLogit {
    // A sufficient statistics class to hold the complete data
    // sufficient statistics from the auxiliary mixture sampling
    // algorithm for binomial logit models.
    //
    // This class was designed to work with the SufstatImputeWorker
    // class defined in Imputer.hpp.
    class SufficientStatistics : private RefCounted {
     public:
      // Args:
      //   dim: The dimension of the coefficient vector in the model
      //     being sampled.
      explicit SufficientStatistics(int dim);

      SufficientStatistics *clone() const;
      void clear();
      void combine(const SufficientStatistics &rhs);

      void update(const Vector &x, double weighted_value, double weight);
      const SpdMatrix &xtx() const;
      const Vector &xty() const;
      int sample_size() const { return sample_size_; }

     private:
      mutable SpdMatrix xtx_;
      Vector xty_;
      mutable bool sym_;
      int sample_size_;
      friend void intrusive_ptr_add_ref(SufficientStatistics *w) {
        w->up_count();
      }
      friend void intrusive_ptr_release(SufficientStatistics *w) {
        w->down_count();
        if (w->ref_count() == 0) delete w;
      }
    };

    // An worker class for drawing latent data in the auxiliary
    // mixture sampling algorithm for binomial logit models.
    class ImputeWorker : public SufstatImputeWorker<BinomialRegressionData,
                                                    SufficientStatistics> {
     public:
      // Args:
      //   global_suf: A reference to the global sufficient statistics
      //     object held by the primary sampler.
      //   global_suf_mutex:  A reference to a mutex protecting global_suf.
      //   clt_threshold: The number of iterations 'n' at which one
      //    should trust the central limit theorem.  The complete data
      //    sufficient statistics are sums of conditionally normal
      //    random variables.  If the number of trials for an
      //    observation is less than 'clt_threshold' then a separate
      //    latent variable will be imputed for each trial.  If it is
      //    larger than clt_threshold then the moments of the sum will
      //    be computed and a normal approximation will be used instead.
      //    A small integer like 5 seems to work well here.
      //   coef: A pointer to a set of logisitic regression
      //     coefficients, owned by the model for which this imputer
      //     is responsible.
      //   rng:  A random number generator or nullptr.
      //   seeding_rng: A RNG used to initialize a new RNG in the case
      //     that rng==nullptr.
      ImputeWorker(SufficientStatistics &global_suf,
                   std::mutex &global_suf_mutex, int clt_threshold,
                   const GlmCoefs *coef, RNG *rng = nullptr,
                   RNG &seeding_rng = GlobalRng::rng);

      void impute_latent_data_point(const BinomialRegressionData &data,
                                    SufficientStatistics *suf,
                                    RNG &rng) override;

     private:
      BinomialLogitCltDataImputer binomial_data_imputer_;
      const GlmCoefs *coefficients_;
    };
  }  // namespace BinomialLogit

  //======================================================================
  class BinomialLogitAuxmixSampler
      : public PosteriorSampler,
        public LatentDataSampler<BinomialLogit::ImputeWorker> {
   public:
    BinomialLogitAuxmixSampler(BinomialLogitModel *model,
                               const Ptr<MvnBase> &prior,
                               int clt_threshold = 10,
                               RNG &seeding_rng = GlobalRng::rng);
    double logpri() const override;
    void draw() override;
    void draw_params();

    Ptr<BinomialLogit::ImputeWorker> create_worker(std::mutex &m) override;
    void clear_latent_data() override;

    void assign_data_to_workers() override;

    // TODO: remove calls to this function and replace
    // them with calls to clear_latent_data().
    //
    // Clear the complete data sufficient statistics.  This is
    // normally unnecessary.  This function is primarily intended for
    // nonstandard situations where the complete data sufficient
    // statistics need to be manipulated by an outside actor.
    void clear_complete_data_sufficient_statistics();

    // Increment the complete data sufficient statistics with the
    // given quantities.  This is normally unnecessary.  This function
    // is primarily intended for nonstandard situations where the
    // complete data sufficient statistics need to be manipulated by
    // an outside actor.
    void update_complete_data_sufficient_statistics(
        double precision_weighted_sum, double total_precision, const Vector &x);

    const BinomialLogit::SufficientStatistics &suf() const { return suf_; }

    int clt_threshold() const { return clt_threshold_; }

   private:
    BinomialLogitModel *model_;
    Ptr<MvnBase> prior_;
    BinomialLogit::SufficientStatistics suf_;
    int clt_threshold_;
  };

}  // namespace BOOM

#endif  // BOOM_BINOMIAL_LOGIT_AUXMIX_SAMPLER_HPP_
