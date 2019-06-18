// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2010 Steven L. Scott

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
#ifndef BOOM_BINOMIAL_MIXTURE_SAMPLER_TIM_HPP_
#define BOOM_BINOMIAL_MIXTURE_SAMPLER_TIM_HPP_

#include "Models/Glm/BinomialLogitModel.hpp"
#include "Models/MvnBase.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"

#include "Samplers/TIM.hpp"

namespace BOOM {

  // A posterior sampler for the BinomialLogitModel based on tailored
  // independence Metropolis (TIM).  The sampler approximates the
  // posterior distribution with a T distribution centered on the
  // mode, with scatter determined by the Fisher information matrix,
  // and with pre-specified degrees of freedom.  The approximation is
  // used as a proposal distribution for a independence
  // Metropolis-Hastings sampler.
  class BinomialLogitSamplerTim : public PosteriorSampler {
   public:
    // Args:
    //   model:  The model for which posterior samples are desired.
    //   prior: The prior distribution on the set of logistic
    //     regression coefficients.  If some elements of the
    //     coefficient vector are excluded using the GlmCoefs include
    //     / exclude mechanism then only the included components of
    //     the prior mean and prior information matrix are used.
    //   mode_is_stable: If true then the posterior mode will not
    //     change once it is found.  If the predictor matrix or
    //     response vector for the model will change (e.g. if the
    //     model contains random effects) then set this to false so
    //     the posterior mode will be located each time.  Otherwise
    //     set it to true, so the mode will only be found once.
    //   nu: The degrees of freedom parameter for the proposal
    //     distribution.  Positive close to zero correspond to heavy
    //     tails, with nu = 1 corresponding to the Cauchy
    //     distribution.  If nu <= 0 then a Gaussian proposal will be
    //     used.
    //   seeding_rng: The random number generator used to set the seed
    //     for the RNG owned by this sampler.
    BinomialLogitSamplerTim(BinomialLogitModel *model,
                            const Ptr<MvnBase> &prior,
                            bool mode_is_stable = true, double nu = 3,
                            RNG &seeding_rng = GlobalRng::rng);

    void draw() override;
    double logpri() const override;

    double logp(const Vector &beta) const;
    double dlogp(const Vector &beta, Vector &g) const;
    double d2logp(const Vector &beta, Vector &g, Matrix &H) const;
    double Logp(const Vector &beta, Vector &g, Matrix &h, int nd) const;

   private:
    BinomialLogitModel *m_;
    Ptr<MvnBase> pri_;
    TIM sam_;
    bool save_modes_;

    struct Mode {
      Vector location;
      SpdMatrix precision;
      bool empty() const { return location.size() == 0; }
    };
    std::map<Selector, Mode> modes_;

    const Mode &locate_mode(const Selector &included_coefficients);
  };

}  // namespace BOOM

#endif  //  BOOM_BINOMIAL_MIXTURE_SAMPLER_TIM_HPP_
