/*
  Copyright (C) 2007 Steven L. Scott

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
#ifndef BOOM_MVREG_SAMPLER_HPP
#define BOOM_MVREG_SAMPLER_HPP
#include <Models/Glm/MvReg2.hpp>

namespace BOOM{

  class MvRegSampler
    : public PosteriorSampler{
  public:
    // assumes Beta|Sigma ~ N(B, Sigma \otimes I/kappa)
    // and Sigma^{-1} ~ Wishart(prior_df/2, SS/2);

    MvRegSampler(MvReg *m, const Matrix &B, double kappa, double prior_df,
                 const SpdMatrix & Sigma_guess, RNG &seeding_rng = GlobalRng::rng);

    double logpri()const override;
    void draw() override;
    void draw_Beta();
    void draw_Sigma();

  private:
    MvReg *mod;
    SpdMatrix SS;
    double prior_df;
    SpdMatrix Ominv;
    double ldoi;
    Matrix B;
  };

}
#endif// BOOM_MVREG_SAMPLER_HPP
