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
#ifndef BOOM_SEP_STRAT_SAMPLER_HPP_
#define BOOM_SEP_STRAT_SAMPLER_HPP_
#include "LinAlg/Cholesky.hpp"
#include "Models/GammaModel.hpp"
#include "Models/MvnModel.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"

namespace BOOM {

  class SepStratSampler : public PosteriorSampler {
   public:
    SepStratSampler(MvnModel *mod,
                    const std::vector<Ptr<GammaModel>> &ivar,
                    RNG &seeding_rng = GlobalRng::rng);
    SepStratSampler(MvnModel *mod,
                    const Ptr<CorrelationModel> &Rprior,
                    const std::vector<Ptr<GammaModel>> &ivar,
                    RNG &seeding_rng = GlobalRng::rng);
    SepStratSampler *clone_to_new_host(Model *new_host) const override;
    void draw() override;
    double logpri() const override;

    void set_max_tries(int);
    void set_polar_frac(double);
    void set_alpha(double);

    int nfast() const { return fast_count_; }
    int nstable() const { return stable_count_; }
    int npolar() const { return polar_count_; }

    double fast_seconds() const { return fast_time_; }
    double stable_seconds() const { return stable_time_; }
    double wasted_seconds() const { return wasted_time_; }
    double polar_seconds() const { return polar_time_; }
    SpdMatrix Sigma();

   private:
    friend class SigmaTarget;
    friend class SigmaPolarTarget;
    void setup();
    bool fast_draw();
    void stable_draw();
    void polar_draw();
    void draw_sigsq(int i);
    void draw_R(int i, int j);
    void set_R(double r);
    void fill_siginv(bool have_Rinv);  // given sd and Rinv_
    void fill_sigma();
    double logp_slice_R(double r);
    double logp_slice_ivar(double ivar);
    void find_limits();
    double logp0(const SpdMatrix &Sigma, double alpha) const;
    double logprior(const SpdMatrix &Sigma) const;
    double detR(double r);

    // fundamental data
    MvnModel *mod_;
    Ptr<CorrelationModel> Rpri_;
    std::vector<Ptr<GammaModel> > sinv_pri_;

    // data used jointly by fast_draw and stable_draw
    SpdMatrix cand_;
    double n_;
    SpdMatrix sumsq_;
    Matrix sumsq_upper_chol_;
    mutable CorrelationMatrix R_;  // workspace for fast_draw.  state for stable_draw
    mutable Vector sd_;    // workspace for fast_draw.  state for stable_draw

    int fast_count_;
    int stable_count_;
    int polar_count_;
    double fast_time_;
    double stable_time_;
    double wasted_time_;
    double polar_time_;

    // data specific to fast_draw
    double alpha_;
    int count_;
    int max_tries_;
    double polar_frac_;

    // data specific to stable_draw
    int i_, j_;
    double lo_;
    double hi_;
    SpdMatrix Rinv_;
  };

}  // namespace BOOM
#endif  // BOOM_SEP_STRAT_SAMPLER_HPP_
