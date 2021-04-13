// Copyright 2018 Google LLC. All Rights Reserved.
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
#include "Models/PosteriorSamplers/MultinomialDirichletSampler.hpp"
#include "Models/DirichletModel.hpp"
#include "Models/MultinomialModel.hpp"
#include "distributions.hpp"

#include "cpputil/report_error.hpp"

namespace BOOM {
  typedef MultinomialDirichletSampler MDS;
  typedef MultinomialModel MM;
  typedef DirichletModel DM;

  MDS::MultinomialDirichletSampler(MM *Mod, const Vector &Nu, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng), mod_(Mod), pri_(new DM(Nu)) {}

  MDS::MultinomialDirichletSampler(MM *Mod, const Ptr<DM> &prior,
                                   RNG &seeding_rng)
      : PosteriorSampler(seeding_rng), mod_(Mod), pri_(prior) {}

  MDS::MultinomialDirichletSampler(const MDS &rhs)
      : PosteriorSampler(rhs),
        mod_(rhs.mod_->clone()),
        pri_(rhs.pri_->clone()) {}


  MDS *MDS::clone_to_new_host(Model *new_host) const {
    return new MDS(
        dynamic_cast<MultinomialModel *>(new_host),
        pri_->clone(),
        rng());
  }

  void MDS::draw() {
    Vector counts = pri_->nu() + mod_->suf()->n();
    Vector pi = rdirichlet_mt(rng(), counts);
    mod_->set_pi(pi);
  }

  void MDS::find_posterior_mode(double) {
    Vector counts = pri_->nu() + mod_->suf()->n();
    Vector pi = mdirichlet(counts);
    mod_->set_pi(pi);
  }

  double MDS::logpri() const { return pri_->logp(mod_->pi()); }

  namespace {
    using CMDS = ConstrainedMultinomialDirichletSampler;
  }

  CMDS::ConstrainedMultinomialDirichletSampler(
      MultinomialModel *model,
      const Vector &prior_counts,
      RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        prior_counts_(prior_counts) {
    if (prior_counts_.size() != model->dim()) {
      std::ostringstream err;
      err << "Dimension of model (" << model->dim()
          << ") does not match dimension of prior counts ("
          << prior_counts_.size() << ").";
      report_error(err.str());
    }
    check_at_least_one_positive(prior_counts_);
  }

  CMDS *CMDS::clone_to_new_host(Model *new_host) const {
    return new CMDS(dynamic_cast<MultinomialModel *>(new_host),
                    prior_counts_,
                    rng());
  }

  void CMDS::draw() {
    Vector ans(prior_counts_.size(), 0.0);
    double total = 0;
    for (int i = 0; i < ans.size(); ++i) {
      if (prior_counts_[i] > 0) {
        ans[i] = rgamma_mt(rng(), prior_counts_[i] + model_->suf()->n()[i], 1.0);
        total += ans[i];
      }
    }
    if (total > 0) {
      ans /= total;
    } else {
      report_error("Total was not positive.");
    }
    model_->set_pi(ans);
  }

  double CMDS::logpri() const {
    Vector prob;
    Vector nu;
    for(int i = 0; i < model_->dim(); ++i) {
      if (prior_counts_[i] <= 0) {
        if (model_->pi()[i] > 0) {
          return negative_infinity();
        }
      } else {
        nu.push_back(prior_counts_[i]);
        prob.push_back(model_->pi()[i]);
      }
    }
    return ddirichlet(prob, nu, true);
  }

  void CMDS::check_at_least_one_positive(const Vector &counts) {
    for (const auto &el : counts) {
      if (el > 0) return;
    }
    report_error("At least one element must be positive.");
  }
}  // namespace BOOM
