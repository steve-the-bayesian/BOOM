/*
  Copyright (C) 2005-2009 Steven L. Scott

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
#include "Models/Glm/PosteriorSamplers/ProbitRegressionSampler.hpp"
#include "distributions.hpp"

namespace BOOM{

  typedef ProbitRegressionSampler PRS;

  PRS::ProbitRegressionSampler(ProbitRegressionModel *model,
                               const Ptr<MvnBase> &prior,
                               RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        mod_(model),
        pri_(prior),
        xtx_(mod_->xdim()),
        xtz_(mod_->xdim()),
        beta_(mod_->xdim())
  {
    refresh_xtx();
  }

  double PRS::logpri()const{
    return pri_->logp(mod_->Beta());
  }

  void PRS::draw(){
    impute_latent_data();
    draw_beta();
  }

  void PRS::draw_beta(){
    const SpdMatrix & siginv(pri_->siginv());
    beta_ = rmvn_suf_mt(rng(),
                        xtx_ + siginv,
                        xtz_ + siginv * pri_->mu());
    mod_->set_Beta(beta_);
  }

  void PRS::impute_latent_data(){
    const ProbitRegressionModel::DatasetType & data(mod_->dat());
    int n = data.size();
    const Vector & beta(mod_->Beta());
    xtz_ = 0;
    for(int i = 0; i < n; ++i){
      const Vector & x(data[i]->x());
      double eta = x.dot(beta);
      bool y = data[i]->y();
      double z = rtrun_norm_mt(rng(), eta, 1, 0, y);
      xtz_.axpy(x,z);
    }
  }

  const Vector & PRS::xtz()const{ return xtz_; }
  const SpdMatrix & PRS::xtx()const{ return xtx_; }

  void PRS::refresh_xtx(){
    int p = mod_->xdim();
    xtx_.resize(p);
    xtx_ = 0;
    const ProbitRegressionModel::DatasetType & data(mod_->dat());
    int n = data.size();
    for(int i = 0; i < n; ++i){
      const Vector & x(data[i]->x());
      xtx_.add_outer(x, 1, false);
    }
    xtx_.reflect();
  }

}
