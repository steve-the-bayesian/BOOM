#include "gtest/gtest.h"
#include "distributions.hpp"

#include "Models/MvnGivenSigma.hpp"
#include "Models/WishartModel.hpp"
#include "Models/Mixtures/DirichletProcessMvnModel.hpp"
#include "Models/Mixtures/PosteriorSamplers/DirichletProcessMvnCollapsedGibbsSampler.hpp"
#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;

  class DpMvnTest : public ::testing::Test {
   protected:
    DpMvnTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(DpMvnTest, SmallExample) {
    Vector mu1{3.0, -1.2, 2.7};
    Vector mu2{5.0, 8.1, -2.7};

    int dim = mu1.size();

    SpdMatrix Sigma1(Vector{
        1, .8, .3,
        .8, 1.5, .4,
        .3, .4, 2.0});
    SpdMatrix Sigma2(Vector{
        1, .8, .3,
        .8, 1.5, .4,
        .3, .4, 2.0});
    int n1 = 100;
    int n2 = 50;

    NEW(DirichletProcessMvnModel, model)(dim, 1.0);
    for (int i = 0; i < n1; ++i) {
      NEW(VectorData, dp)(rmvn(mu1, Sigma1));
      model->add_data(dp);
    }
    for (int i = 0; i < n2; ++i){
      NEW(VectorData, dp)(rmvn(mu2, Sigma2));
      model->add_data(dp);
    }

    NEW(MvnGivenSigma, mean_base_measure)(.5 * (mu1 + mu2), 1.0);
    NEW(WishartModel, precision_base_measure)(dim + 1, .5 * (Sigma1 + Sigma2));
    NEW(DirichletProcessMvnCollapsedGibbsSampler, sampler)(
        model.get(),
        mean_base_measure,
        precision_base_measure);
    model->set_method(sampler);

    for (int i = 0; i < 1000; ++i) {
      model->sample_posterior();
    }
  }

}  // namespace
