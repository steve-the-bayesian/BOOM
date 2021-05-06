// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2012 Steven L. Scott

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

#include "Models/PosteriorSamplers/BetaPosteriorSampler.hpp"
#include "cpputil/math_utils.hpp"
#include "distributions.hpp"

namespace BOOM {

  namespace {
    //----------------------------------------------------------------------
    // A functor for evaluating the un-normalized log posterior for the
    // mean parameter in the mean/sample_size parameterization.
    class BetaMeanLogPosterior {
     public:
      BetaMeanLogPosterior(const BetaModel *model,
                           const DoubleModel *mean_prior)
          : model_(model), mean_prior_(mean_prior) {}

      double operator()(double mean) const {
        if (mean <= 0 || mean >= 1.0) {
          return negative_infinity();
        }
        double ans = mean_prior_->logp(mean);
        double sample_size = model_->sample_size();
        double a = sample_size * mean;
        double b = sample_size * (1 - mean);
        ans += model_->log_likelihood(a, b);
        return ans;
      }

     private:
      const BetaModel *model_;
      const DoubleModel *mean_prior_;
    };

    //----------------------------------------------------------------------
    // A functor for evaluating the un-normalized log posterior for the
    // sample_size parameter in the mean/sample_size parameterization.
    class BetaSampleSizeLogPosterior {
     public:
      BetaSampleSizeLogPosterior(const BetaModel *model,
                                 const DoubleModel *sample_size_prior)
          : model_(model), sample_size_prior_(sample_size_prior) {}

      double operator()(double sample_size) const {
        if (sample_size <= 0) {
          return negative_infinity();
        }
        double ans = sample_size_prior_->logp(sample_size);
        double mean = model_->mean();
        double a = sample_size * mean;
        double b = sample_size * (1 - mean);
        ans += model_->log_likelihood(a, b);
        return ans;
      }

     private:
      const BetaModel *model_;
      const DoubleModel *sample_size_prior_;
    };
  }  // namespace

  //======================================================================
  BetaPosteriorSampler::BetaPosteriorSampler(
      BetaModel *model, const Ptr<DoubleModel> &mean_prior,
      const Ptr<DoubleModel> &sample_size_prior, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        mean_prior_(mean_prior),
        sample_size_prior_(sample_size_prior),
        mean_sampler_(BetaMeanLogPosterior(model_, mean_prior_.get()), false,
                      1.0, &rng()),
        sample_size_sampler_(
            BetaSampleSizeLogPosterior(model_, sample_size_prior_.get()), false,
            1.0, &rng()) {
    mean_sampler_.set_limits(0.0, 1.0);
    sample_size_sampler_.set_lower_limit(0.0);
  }

  BetaPosteriorSampler *BetaPosteriorSampler::clone_to_new_host(
      Model *new_host) const {
    return new BetaPosteriorSampler(dynamic_cast<BetaModel *>(new_host),
                                    mean_prior_->clone(),
                                    sample_size_prior_->clone(),
                                    rng());
  }

  void BetaPosteriorSampler::draw() {
    try {
      double sample_size = sample_size_sampler_.draw(model_->sample_size());
      model_->set_sample_size(sample_size);
    } catch (const std::exception &e) {
      report_error(error_message("the sample size parameter", &e));
    } catch (...) {
      report_error(error_message("the sample size parameter", NULL));
    }

    try {
      double mean = mean_sampler_.draw(model_->mean());
      model_->set_mean(mean);
    } catch (std::exception &e) {
      report_error(error_message("the mean_parameter", &e));
    } catch (...) {
      report_error(error_message("the mean parameter", NULL));
    }
  }

  double BetaPosteriorSampler::logpri() const {
    double mean = model_->mean();
    double sample_size = model_->sample_size();
    if (mean <= 0 || sample_size <= 0) {
      return negative_infinity();
    }
    return mean_prior_->logp(mean) + sample_size_prior_->logp(sample_size);
  }

  std::string BetaPosteriorSampler::error_message(
      const char *thing_being_drawn,
      const std::exception *e) const {
    std::ostringstream err;
    err << "The slice sampler generated an exception when drawing "
        << thing_being_drawn << " for the beta distribution.  " << endl
        << "Current parameter values are:  " << endl
        << "      a = " << model_->a() << endl
        << "      b = " << model_->b() << endl
        << "  a/a+b = " << model_->mean() << endl
        << "    a+b = " << model_->sample_size() << endl
        << "    sufficient statistics: " << endl
        << "              n  = " << model_->suf()->n() << endl
        << "     sum(log(p)) = " << model_->suf()->sumlog() << endl
        << " sum(log(1 - p)) = " << model_->suf()->sumlogc() << endl;
    if (e) {
      err << "The exception message was: " << endl << e->what() << endl;
    }
    return err.str();
  }

}  // namespace BOOM
