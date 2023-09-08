#include "gtest/gtest.h"

#include "Models/GP/HierarchicalGpRegressionModel.hpp"
#include "Models/GP/GaussianProcessRegressionModel.hpp"
#include "Models/GP/PosteriorSamplers/MahalanobisKernelSampler.hpp"
#include "Models/GP/PosteriorSamplers/LinearMeanFunctionSampler.hpp"
#include "Models/GP/PosteriorSamplers/GaussianProcessRegressionPosteriorSampler.hpp"
#include "Models/GP/PosteriorSamplers/HierarchicalGpPosteriorSampler.hpp"
#include "Models/GP/GpMeanFunction.hpp"
#include "Models/GP/kernels.hpp"
#include "Models/ChisqModel.hpp"
#include "distributions.hpp"

#include "test_utils/test_utils.hpp"
#include "stats/moments.hpp"
#include <fstream>

namespace {
  using namespace BOOM;

  class HierarchicalGpTest : public ::testing::Test {
   protected:
    HierarchicalGpTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  // The smoke test just tests that the model can be built.
  TEST_F(HierarchicalGpTest, SmokeTest) {
    NEW(ZeroFunction, zero)();
    NEW(RadialBasisFunction, rbf)();
    NEW(GaussianProcessRegressionModel, prior_model)(
        zero, rbf, new UnivParams(1.0));

    HierarchicalGpRegressionModel model(prior_model);
  }

  // Tests that the sampler runs without crashing on an HGP with a linear
  // hyperprior.
  TEST_F(HierarchicalGpTest, LinearMcmcTest) {

    int xdim = 3;
    int num_groups = 5;
    Vector true_beta(xdim);
    true_beta.randomize_gaussian();

    NEW(GlmCoefs, coefficients)(true_beta);
    NEW(LinearMeanFunction, hyperprior_mean_function)(coefficients);

    NEW(ZeroFunction, zero)();
    NEW(RadialBasisFunction, rbf)();
    NEW(GP::NullSampler, null_sampler)();

    NEW(GaussianProcessRegressionModel, prior)(
        hyperprior_mean_function,
        rbf,
        new UnivParams(1.0));
    NEW(LinearMeanFunctionSampler, linear_mean_sampler)(
        hyperprior_mean_function.get(),
        prior.get(),
        new MvnModel(Vector(xdim, 0.0), SpdMatrix(xdim, 1.0)));
    NEW(GaussianProcessRegressionPosteriorSampler, prior_sampler)(
        prior.get(),
        linear_mean_sampler,
        null_sampler,
        new ChisqModel(1.0, 1.0));
    prior->set_method(prior_sampler);

    NEW(HierarchicalGpRegressionModel, model)(prior);

    Matrix experiment_betas(num_groups, xdim);
    std::vector<std::string> group_names;

    for (int i = 0; i < num_groups; ++i) {
      std::string group_name = std::to_string(i);
      group_names.push_back(group_name);
      experiment_betas.row(i) = true_beta + rnorm_vector(xdim, 0, .1);

      int group_sample_size = rpois(10);

      Matrix experiment_predictors(group_sample_size, xdim);
      experiment_predictors.randomize_gaussian(0, 1);

      Vector experiment_responses =
          experiment_predictors * experiment_betas.row(i) +
          rnorm_vector(group_sample_size, 0, 1);

      NEW(GaussianProcessRegressionModel, data_model)(
          zero, rbf, new UnivParams(1.0));
      model->add_model(data_model, group_name);

      NEW(GaussianProcessRegressionPosteriorSampler, data_model_sampler)(
          data_model.get(), null_sampler, null_sampler, new ChisqModel(1.0, 1.0));
      data_model->set_method(data_model_sampler);

      for (int j = 0; j < group_sample_size; ++j) {
        NEW(HierarchicalRegressionData, data_point)(
            experiment_responses[j],
            experiment_predictors.row(j),
            group_name);
        model->add_data(data_point);
      }  // loop over group level data (j)
    }  // loop over groups (i)

    NEW(HierarchicalGpPosteriorSampler, sampler)(model.get());
    model->set_method(sampler);


    int niter = 10;
    for (int i = 0; i < niter; ++i) {
      model->sample_posterior();
    }
  }


}  // namespace
