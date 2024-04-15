/*
  Copyright (C) 2005-2024 Steven L. Scott

  This library is free software; you can redistribute it and/or modify it under
  the terms of the GNU Lesser General Public License as published by the Free
  Software Foundation; either version 2.1 of the License, or (at your option)
  any later version.

  This library is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
  details.

  You should have received a copy of the GNU Lesser General Public License along
  with this library; if not, write to the Free Software Foundation, Inc., 51
  Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
*/

#include "Models/FactorModels/PosteriorSamplers/PoissonFactorHierarchicalSampler.hpp"
#include "TargetFun/SumMultinomialLogitTransform.hpp"
#include "distributions.hpp"
#include "Samplers/UnivariateSliceSampler.hpp"
#include "Models/PosteriorSamplers/MvnConjSampler.hpp"

namespace BOOM {

  namespace {
    using Site = FactorModels::PoissonSite;
    using Visitor = FactorModels::PoissonVisitor;
  }  // namespace

  PoissonFactorHierarchicalSampler::PoissonFactorHierarchicalSampler(
      PoissonFactorModel *model,
      const Vector &default_prior_class_probabilities,
      const Vector &prior_mean,
      double kappa,
      const SpdMatrix &Sigma_guess,
      double prior_df,
      int MH_threshold,
      RNG &seeding_rng)
      : PoissonFactorPosteriorSamplerBase(
            model,
            default_prior_class_probabilities,
            seeding_rng),
        profile_hyperprior_(new MvnModel(model->number_of_classes() - 1)),
        MH_acceptance_(0),
        MH_failure_(0),
        slice_sample_draws_(0),
        MH_threshold_(MH_threshold)
  {
    hyperprior_sampler_.reset(new MvnConjSampler(
        profile_hyperprior_.get(),
        prior_mean,
        kappa,
        Sigma_guess,
        prior_df,
        rng()));
    profile_hyperprior_->set_method(hyperprior_sampler_);
  }

  double PoissonFactorHierarchicalSampler::logpri() const {
    return negative_infinity();
  }

  void PoissonFactorHierarchicalSampler::draw() {
    PoissonFactorPosteriorSamplerBase::impute_visitors();
    draw_site_parameters();
    draw_hyperparameters();
  }

  void PoissonFactorHierarchicalSampler::draw_hyperparameters() {
    profile_hyperprior_->sample_posterior();
  }

  void PoissonFactorHierarchicalSampler::draw_site_parameters() {
    SumMultinomialLogitTransform transform;
    profile_hyperprior_->clear_data();

     for (auto &site_it : model()->sites()) {
       Ptr<Site> &site(site_it.second);
       Matrix visitor_counts = site->visitor_counts();
       
       // column 0 of visitor_counts is the number of imputed visits in each category.
       // column 1 is the number of imputed visitors in each category.
       if (visitor_counts.col(0).min() >= MH_threshold_) {
         draw_site_parameters_MH(site);
       } else {
         draw_site_parameters_slice(site);
       }
       Vector eta = transform.to_sum_logits(site->lambda());
       // The first element of eta is the sum of the lambdas.  The remaining
       // elements are the multinomial logit transform of the lambda profile
       // (i.e. of lambda divided by its sum).  Only the logits are described
       // by the profile_hyperprior.
       profile_hyperprior_->suf()->update_raw(ConstVectorView(eta, 1));
     }
  }

  // A functor for evaluating the conditional log posterior of a site's
  // intensity parameters.  This class is intended to be ephemeral.  It stores
  // information about a site's visitors that should be recomputed as soon as
  // those visitors' latent classes are re-imputed.
  //
  // The log posterior can be evaluated either on the raw scale (i.e. density
  // with respect to lambda) or on the transformed scale (i.e. density with
  // respect to the sum and multinomial logits).
  class SiteParameterLogPosterior {
   public:

    // Indicates whether the density function should be evaluated on the raw
    // scale or the transformed scale.
    enum Scale {RAW, TRANSFORMED};

    // Args:
    //   site:  The site whose parameters are to be evaluated.
    //   mlogit_profile_prior: A multivariate normal prior on the multinomial
    //     logit transformation of the profile of a site's intensity parameters.
    //     The 'profile' is the set of intensity parameters divided by their
    //     sum.
    //   exposures: The total number of users of each demographic category in
    //     the complete data set.
    //   scale: If RAW then construct a density function with respect to the
    //     lambdas.  If TRANSFORMED construct a density function with respect to
    //     sum and logits.
    SiteParameterLogPosterior(
        const Ptr<Site> &site,
        const Ptr<MvnModel> &mlogit_profile_prior,
        const Vector &exposures,
        Scale scale = RAW)
        : site_(site),
          mlogit_profile_prior_(mlogit_profile_prior),
          exposures_(exposures),
          scale_(scale)
    {
      visit_counts_ = site_->visitor_counts().col(0);
    }

    // The argument to operator() can be either lambda, or sum_and_logits,
    // depending on how scale_ was set during construction.
    double operator()(const Vector &lambda) const {
      return logp(lambda);
    }

    double logp(const Vector &y) const {
      SumMultinomialLogitTransform transform;
      double ans = 0;
      Vector lambda, eta;
      if (scale_ == RAW) {
        lambda = y;
        eta = transform.to_sum_logits(lambda);
      } else {
        eta = y;
        lambda = transform.from_sum_logits(eta);
      }

      // All lambdas must be positive.
      if (lambda.min() <= 0.0) {
        return negative_infinity();
      }

      for (int k = 0; k < lambda.size(); ++k) {
        ans += visit_counts_[k] * log(lambda[k]) - exposures_[k] * lambda[k];
      }
      
      // Assume a flat prior for sum(lambda).
      const ConstVectorView logits(eta, 1);

      ans += mlogit_profile_prior_->logp(logits);
      if (scale_ == RAW) {
        SumMultinomialLogitJacobian jacobian;
        ans -= jacobian.logdet(lambda);
      }
      return ans;
    }

   private:
    Ptr<Site> site_;
    Ptr<MvnModel> mlogit_profile_prior_;
    Vector visit_counts_;
    Vector exposures_;
    Scale scale_;
  };


  // A Metropolis-Hastings update that jointly draws the (a[j,k], b[j, k])
  // parameters from independent gamma distributions.  This update is
  // appropriate for high traffic sites with sufficient information for the
  // likelihood to dominate the prior.
  //
  // Args:
  //   site:  The Site to be updated.
  //
  // Effects:
  //   The parameters of the supplied site are updated with MH draws from their
  //   full conditional distribution.
  void PoissonFactorHierarchicalSampler::draw_site_parameters_MH(
      Ptr<Site> &site) {
    Vector lambda = site->lambda();
    Matrix counts = site->visitor_counts();
    const ConstVectorView visit_counts(counts.col(0));

    SiteParameterLogPosterior logpost(
        site,
        profile_hyperprior_,
        exposure_counts(),
        SiteParameterLogPosterior::Scale::RAW);

    for (size_t i = 0; i < lambda.size(); ++i) {
      // Find alpha and beta for the proposal distribution.
      double alpha = visit_counts[i] + 1;
      double beta = exposure_counts()[i];
      double log_denominator =
          logpost(lambda) - dgamma(lambda[i], alpha, beta, true);

      double original_lambda = lambda[i];
      lambda[i] = rgamma_mt(rng(), alpha, beta);
      double log_numerator =
          logpost(lambda) - dgamma(lambda[i], alpha, beta, true);

      double logu = negative_infinity();
      while (!std::isfinite(logu)) {
        logu = log(runif_mt(rng()));
      }
      if (logu < log_numerator - log_denominator) {
        // Accept the draw by doing nothing.
        ++MH_acceptance_;
      } else {
        // Reject the draw by reverting to the original lambda value.
        lambda[i] = original_lambda;
        ++MH_failure_;
      }
    }
    site->set_lambda(lambda);
  }

  void PoissonFactorHierarchicalSampler::draw_site_parameters_slice(
      Ptr<Site> &site) {
    ++slice_sample_draws_;
    SumMultinomialLogitTransform transformation;
    Vector eta = transformation.to_sum_logits(site->lambda());
    SiteParameterLogPosterior logpost(
        site, profile_hyperprior_, exposure_counts(),
        SiteParameterLogPosterior::Scale::TRANSFORMED);

    UnivariateSliceSampler sampler(logpost);
    Vector lower_limit(eta.size(), negative_infinity());
    Vector upper_limit(eta.size(), infinity());
    lower_limit[0] = 0.0;
    sampler.set_limits(lower_limit, upper_limit);

    try {
      eta = sampler.draw(eta);
    } catch (std::exception &ex) {
      std::ostringstream err;
      err << "Slice sampler failed with the following message while sampling site "
          << site->id() << " with lambda parameters " << site->lambda() << ".\n"
          << "Falling back to Metropolis-Hastings algorithm.\n\n"
          << ex.what();
      report_warning(err.str());
      draw_site_parameters_MH(site);
      return;
    }
    site->set_lambda(transformation.from_sum_logits(eta));
  }

  void PoissonFactorHierarchicalSampler::set_MH_threshold(int threshold) {
    MH_threshold_ = threshold;
  }
  
  std::string PoissonFactorHierarchicalSampler::sampling_report() const {
    Int total_samples = MH_failure_ + MH_acceptance_ + slice_sample_draws_;
    std::ostringstream report;
    double MH_samples = MH_failure_ + MH_acceptance_;
    double MH_fraction = MH_samples / total_samples;
    double MH_accept_fraction = MH_acceptance_ / MH_samples;

    report << "Sampler made " << total_samples << " draws.\n"
           << std::setw(8) << MH_acceptance_ << " MH acceptances\n"
           << std::setw(8) << MH_failure_ << " MH rejections\n"
           << std::setw(8) << slice_sample_draws_
           << " slice sampling draws.\n"
           << std::setw(8) << std::fixed << 100 * MH_fraction 
           << "% of draws were MH draws.\n"
           << std::setw(8) << std::fixed << 100 * MH_accept_fraction
           << "% of MH draws were accepted.\n";
        ;
    return report.str();
  }
  
}  // namespace BOOM
