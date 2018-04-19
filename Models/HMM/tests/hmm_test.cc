#include "gtest/gtest.h"
#include "Models/HMM/HMM2.hpp"
#include "Models/PoissonModel.hpp"
#include "Models/MarkovModel.hpp"
#include "Models/ProductDirichletModel.hpp"
#include "Models/DirichletModel.hpp"
#include "Models/PosteriorSamplers/PoissonGammaSampler.hpp"
#include "Models/PosteriorSamplers/MarkovConjSampler.hpp"
#include "Models/HMM/PosteriorSamplers/HmmPosteriorSampler.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  Vector to_rows(const Matrix &mat) {
    int dim = mat.nrow() * mat.ncol();
    int start = 0;
    Vector ans(dim);
    for (int i = 0; i < mat.nrow(); ++i) {
      VectorView view(ans, start, mat.ncol());
      view = mat.row(i);
      start += mat.ncol();
    }
    return ans;
  }
  
  class HmmTest : public ::testing::Test {
   protected:
    HmmTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(HmmTest, Basics) {
  }

  TEST_F(HmmTest, Poisson) {
    std::vector<Ptr<PoissonModel>> mixture_components;
    mixture_components.push_back(new PoissonModel(1.0));
    mixture_components.push_back(new PoissonModel(2.0));
    mixture_components.push_back(new PoissonModel(4.0));
    NEW(GammaModel, poisson_prior_0)(.1, 1);
    NEW(PoissonGammaSampler, component_sampler_0)(
        mixture_components[0].get(), poisson_prior_0);
    mixture_components[0]->set_method(component_sampler_0);

    NEW(GammaModel, poisson_prior_1)(2, 1);
    NEW(PoissonGammaSampler, component_sampler_1)(
        mixture_components[1].get(), poisson_prior_1);
    mixture_components[1]->set_method(component_sampler_1);

    NEW(GammaModel, poisson_prior_2)(4, 1);
    NEW(PoissonGammaSampler, component_sampler_2)(
        mixture_components[2].get(), poisson_prior_2);
    mixture_components[2]->set_method(component_sampler_2);

    Ptr<MarkovModel> mark(new MarkovModel(3));
    NEW(ProductDirichletModel, transition_prior)(3);
    NEW(DirichletModel, initial_state_prior)(3);
    NEW(MarkovConjSampler, markov_sampler)(
        mark.get(),
        transition_prior,
        initial_state_prior);
    mark->set_method(markov_sampler);

    NEW(HiddenMarkovModel, model)(
        std::vector<Ptr<MixtureComponent>>(mixture_components.begin(),
                                           mixture_components.end()),
        mark);
    NEW(HmmPosteriorSampler, sampler)(model.get());
    model->set_method(sampler);
    
    std::ifstream lamb("./Models/HMM/tests/fetal.lamb.data");
    EXPECT_TRUE(lamb) << "Could not open fetal.lamb.data file.";
    std::vector<int> lamb_data;
    int measurement;
    while(lamb >> measurement) {
      lamb_data.push_back(measurement);
    }
    EXPECT_EQ(240, lamb_data.size());

    for (int i = 0; i < lamb_data.size(); ++i) {
      NEW(IntData, dp)(lamb_data[i]);
      model->add_data(dp);
    }

    int niter = 1000;
    Matrix lambda_draws(niter, 3);
    Matrix transition_probablity_draws(niter, 3 * 3);
    for (int i = 0; i < niter; ++i) {
      model->sample_posterior();
      lambda_draws(i, 0) = mixture_components[0]->lam();
      lambda_draws(i, 1) = mixture_components[1]->lam();
      lambda_draws(i, 2) = mixture_components[2]->lam();
      transition_probablity_draws.row(i) = to_rows(mark->Q());
    }

    // The prior distributions that we used identify the model so there is no
    // issue with label switching.  The values 
    auto status = check_mcmc_matrix(lambda_draws, Vector{.02, .46, 3.13});
    EXPECT_TRUE(status.ok) << "Lambda draws failed to cover" << status;

    status = check_mcmc_matrix(transition_probablity_draws,
                           Vector {0.881, 0.095, 0.024,
                                 0.083, 0.898, 0.019,
                                 0.221, 0.198, 0.580});
    EXPECT_TRUE(status.ok) << "Transition_Probablity_Draws failed to cover"
                           << status;
  }
  
}  // namespace
