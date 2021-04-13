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
#include "Models/PosteriorSamplers/BetaBinomialSampler.hpp"
#include "cpputil/math_utils.hpp"
#include "distributions.hpp"

namespace BOOM {

  typedef BetaBinomialSampler BBS;

  BBS::BetaBinomialSampler(BinomialModel *model, const Ptr<BetaModel> &prior,
                           RNG &seeding_rng)
      : ConjugateHierarchicalPosteriorSampler(seeding_rng),
        model_(model),
        prior_(prior) {}

  BBS *BBS::clone_to_new_host(Model *new_host) const {
    return new BBS(dynamic_cast<BinomialModel *>(new_host),
                   prior_->clone(),
                   rng());
  }

  void BBS::draw() { draw_model_parameters(*model_); }

  double BBS::logpri() const {
    double p = model_->prob();
    return prior_->logp(p);
  }

  void BBS::draw_model_parameters(Model &model) {
    draw_model_parameters(dynamic_cast<BinomialModel &>(model));
  }

  void BBS::draw_model_parameters(BinomialModel &model) {
    double a = prior_->a();
    double b = prior_->b();
    double nyes = model.suf()->sum();
    double n = model.suf()->nobs();
    double nno = n - nyes;
    double p;
    int ntries = 0;
    do {
      // In most cases this do loop will finish without repeating.  It
      // exists to guard against cases where rbeta returns values of p
      // on the boundary, or nan.
      p = rbeta_mt(rng(), a + nyes, b + nno);
      if (++ntries > 500) {
        const double epsilon = std::numeric_limits<double>::epsilon();
        if (p >= 1.0 || ((nyes > nno) && (b + nno < 1))) {
          // One way for p to be nan is for nyes to be substantially larger than
          // nno, with a small prior value of b.  This is a signal that p should
          // be 1.
          p = 1.0 - epsilon;
        } else if (p <= 0.0 || ((nno > nyes) && (a + nyes < 1))) {
          // By symmetry to the case above.
          p = epsilon;
        } else if (!std::isfinite(p)) {
          ostringstream err;
          err << "Too many attempts in BetaBinomialSampler::draw()." << endl
              << "a = " << a << endl
              << "b = " << b << endl
              << "a + nyes = " << a + nyes << endl
              << "b + nno  = " << b + nno << endl
              << "last simulated value of p: " << p << endl;
          report_error(err.str());
        }
      }
    } while (p <= 0 || p >= 1 || !std::isfinite(p));
    model.set_prob(p);
  }

  double BBS::log_prior_density(const ConstVectorView &parameters) const {
    if (!parameters.empty()) {
      report_error("Wrong size parameters in log_prior_density.");
    }
    return prior_->logp(parameters[0]);
  }

  double BBS::log_prior_density(const Model &model) const {
    return log_prior_density(dynamic_cast<const BinomialModel &>(model));
  }

  double BBS::log_prior_density(const BinomialModel &model) const {
    return prior_->logp(model.prob());
  }

  void BBS::find_posterior_mode(double) {
    double a = prior_->a();
    double b = prior_->b();
    double y = model_->suf()->sum() + a;
    double n = model_->suf()->nobs() + a + b;
    model_->set_prob((y - 1) / (n - 2));
  }

  double BBS::log_marginal_density(const Ptr<Data> &dp,
                                   const ConjugateModel *abstract_model) const {
    const BinomialModel *model(
        dynamic_cast<const BinomialModel *>(abstract_model));
    if (!model) {
      report_error(
          "The BetaBinomialSampler is only conjugate with "
          "BinomialModel objects.");
    }
    return log_marginal_density(*model->DAT(dp), model);
  }

  double BBS::log_marginal_density(const BinomialData &data,
                                   const BinomialModel *model) const {
    double n = data.n() + model->suf()->nobs();
    double y = data.y() + model->suf()->sum();
    double a = prior_->a();
    double b = prior_->b();
    if (n <= 0 || y < 0 || n - y < 0 || a <= 0 || b <= 0) {
      return negative_infinity();
    }
    return lgamma(a + b) + lgamma(n + 1) + lgamma(a + y) + lgamma(b + n - y) -
           lgamma(a) - lgamma(b) - lgamma(y + 1) - lgamma(n - y + 1) -
           lgamma(a + b + n);
  }

}  // namespace BOOM
