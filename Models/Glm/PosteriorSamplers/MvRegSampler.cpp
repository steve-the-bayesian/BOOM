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
#include <Models/Glm/PosteriorSamplers/MvRegSampler.hpp>
#include <distributions.hpp>

namespace BOOM{

  typedef MvRegSampler MRS;

  MRS::MvRegSampler(MvReg *m, const Matrix &Beta_guess, double prior_beta_nobs,
            double Prior_df, const SpdMatrix & Sigma_guess, RNG &seeding_rng)
    : PosteriorSampler(seeding_rng),
      mod(m),
      SS(Sigma_guess * Prior_df),
      prior_df(Prior_df),
      Ominv(m->xdim()),
      B(Beta_guess)
  {
    double kappa = prior_beta_nobs;
    Ominv.set_diag(kappa);
    ldoi = m->ydim() * log(kappa);
  }

  double MRS::logpri()const{
    const SpdMatrix & Siginv(mod->Siginv());
    double ldsi  = mod->ldsi();
    const Matrix & Beta(mod->Beta());
    double ans = dWish(Siginv, SS, prior_df, true);
    ans += dmatrix_normal_ivar(Beta, B, Siginv, ldsi, Ominv, ldoi,true);
    return ans;
  }

  void MRS::draw(){
    draw_Sigma();
    draw_Beta();
  }

  void MRS::draw_Beta(){
    Ptr<NeMvRegSuf> s(mod->suf().dcast<NeMvRegSuf>());

    SpdMatrix ivar = Ominv + s->xtx();
    Matrix Mu = s->xty() + Ominv*B;
    Mu = ivar.solve(Mu);
    Matrix ans = rmatrix_normal_ivar(Mu, mod->Siginv(), ivar);
    mod->set_Beta(ans);
  }

  void MRS::draw_Sigma(){
    Ptr<MvRegSuf> s(mod->suf());
    SpdMatrix sumsq = SS + s->SSE(mod->Beta());
    double df = prior_df + s->n();
    SpdMatrix ans = rWish(df, sumsq.inv());
    mod->set_Siginv(ans);
  }
}
