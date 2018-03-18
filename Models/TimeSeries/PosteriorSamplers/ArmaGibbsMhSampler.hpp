#ifndef BOOM_ARMA_GIBBS_MH_SAMPLER_
#define BOOM_ARMA_GIBBS_MH_SAMPLER_

#include "Models/DoubleModel.hpp"
#include "Models/MvnBase.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "Models/TimeSeries/ArmaModel.hpp"
#include "Models/VectorModel.hpp"

namespace BOOM {

  // A PosteriorSampler for ARMA models based on a separation between the AR and
  // MA coefficients.  The AR coefficients are drawn conditional on the MA
  // coefficients, using a Gibbs sampler.  The MA coefficients are drawn
  // conditional on the AR coefficients using an adaptive Metropolis-Hastings
  // scheme.

  class ArmaGibbsMhSampler : public PosteriorSampler {
   public:
    ArmaGibbsMhSampler(ArmaModel *model,
                       const Ptr<MvnBase> &ar_prior,
                       const Ptr<VectorModel> &ma_prior,
                       const Ptr<DoubleModel> &precision_prior);

    void draw() override;
    double logpri() const override;

    void draw_ar_given_ma();
    void draw_ma_given_ar();
    double log_posterior(const Vector &ar_coefficients,
                         const Vector &ma_coefficients, double precision) const;

   private:
    ArmaModel *model_;

    Ptr<MvnBase> ar_prior_;
    Ptr<VectorModel> ma_prior_;
    Ptr<DoubleModel> precision_prior_;
  };

}  // namespace BOOM

#endif  //  BOOM_ARMA_GIBBS_MH_SAMPLER_
