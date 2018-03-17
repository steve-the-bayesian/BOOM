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

#include "Models/Glm/PosteriorSamplers/BinomialLogitAuxmixSampler.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {
  namespace {
    typedef BinomialLogitAuxmixSampler BLAMS;
  }  // namespace

  namespace BinomialLogit {
    SufficientStatistics::SufficientStatistics(int dim)
        : xtx_(dim), xty_(dim), sym_(false), sample_size_(0) {}

    SufficientStatistics *SufficientStatistics::clone() const {
      return new SufficientStatistics(*this);
    }

    void SufficientStatistics::clear() {
      xtx_ = 0;
      xty_ = 0;
      sym_ = false;
      sample_size_ = 0;
    }

    void SufficientStatistics::combine(const SufficientStatistics &rhs) {
      xtx_ += rhs.xtx_;
      xty_ += rhs.xty_;
      sym_ = sym_ && rhs.sym_;
      sample_size_ += rhs.sample_size_;
    }

    const SpdMatrix &SufficientStatistics::xtx() const {
      if (!sym_) {
        xtx_.reflect();
        sym_ = true;
      }
      return xtx_;
    }

    const Vector &SufficientStatistics::xty() const { return xty_; }

    void SufficientStatistics::update(const Vector &x, double weighted_value,
                                      double weight) {
      sym_ = false;
      xtx_.add_outer(x, weight, false);
      xty_.axpy(x, weighted_value);
      ++sample_size_;
    }

    ImputeWorker::ImputeWorker(SufficientStatistics &global_suf,
                               std::mutex &global_suf_mutex, int clt_threshold,
                               const GlmCoefs *coef, RNG *rng, RNG &seeding_rng)
        : SufstatImputeWorker<BinomialRegressionData, SufficientStatistics>(
              global_suf, global_suf_mutex, rng, seeding_rng),
          binomial_data_imputer_(clt_threshold),
          coefficients_(coef) {}

    void ImputeWorker::impute_latent_data_point(
        const BinomialRegressionData &observation, SufficientStatistics *suf,
        RNG &rng) {
      const Vector &x(observation.x());
      double eta = coefficients_->predict(x);
      try {
        std::pair<double, double> imputed = binomial_data_imputer_.impute(
            rng, observation.n(), observation.y(), eta);
        double sum = imputed.first;
        double weight = imputed.second;
        suf->update(x, sum, weight);
      } catch (std::exception &e) {
        ostringstream err;
        err << "caught an exception "
            << "with the following message:" << e.what() << endl
            << "n   = " << observation.n() << endl
            << "y   = " << observation.y() << endl
            << "eta = " << eta << endl;
        report_error(err.str());
      }
    }
  }  // namespace BinomialLogit

  using namespace BinomialLogit;

  BLAMS::BinomialLogitAuxmixSampler(BinomialLogitModel *model,
                                    const Ptr<MvnBase> &prior,
                                    int clt_threshold, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        prior_(prior),
        suf_(model->xdim()),
        clt_threshold_(clt_threshold) {
    set_number_of_workers(1);
  }

  double BLAMS::logpri() const { return prior_->logp(model_->Beta()); }

  void BLAMS::draw() {
    impute_latent_data();
    draw_params();
  }

  Ptr<ImputeWorker> BLAMS::create_worker(std::mutex &suf_mutex) {
    return new ImputeWorker(suf_, suf_mutex, clt_threshold_,
                            model_->coef_prm().get(), nullptr, rng());
  }

  void BLAMS::draw_params() {
    SpdMatrix ivar = prior_->siginv() + suf_.xtx();
    Vector ivar_mu = suf_.xty() + prior_->siginv() * prior_->mu();
    Vector draw = rmvn_suf_mt(rng(), ivar, ivar_mu);
    model_->set_Beta(draw);
  }

  void BLAMS::clear_latent_data() { suf_.clear(); }
  void BLAMS::clear_complete_data_sufficient_statistics() { suf_.clear(); }

  void BLAMS::update_complete_data_sufficient_statistics(
      double precision_weighted_sum, double total_precision, const Vector &x) {
    suf_.update(x, precision_weighted_sum, total_precision);
  }

  void BLAMS::assign_data_to_workers() {
    BOOM::assign_data_to_workers(model_->dat(), workers());
  }

}  // namespace BOOM
