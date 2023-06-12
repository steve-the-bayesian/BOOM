#include "gtest/gtest.h"

#include "stats/DataTable.hpp"
#include "distributions.hpp"

#include "Models/MvnModel.hpp"
#include "Models/Glm/VariableSelectionPrior.hpp"
#include "Models/Glm/MultinomialLogitModel.hpp"
#include "Models/Glm/PosteriorSamplers/MultinomialLogitCompositeSpikeSlabSampler.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>


namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  class MultinomialLogitTest : public ::testing::Test {
   protected:
    MultinomialLogitTest() {
      GlobalRng::rng.seed(8675309);
      std::string raw_path = "Models/Glm/tests/autopref.txt";
      autopref_.read_file(raw_path, false, "\t");
      autopref_.set_vnames(std::vector<std::string>{
          "country", "age", "sex", "married", "size", "type", "is25"});
      // American	34	Male	Married	Large	Family	No
      // Japanese	36	Male	Single	Small	Sporty	No
      // Japanese	23	Male	Married	Small	Family	No
      // American	29	Male	Single	Large	Family	No
      // American	39	Male	Married	Medium	Family	No
    }

    DataTable autopref_;
  };

  TEST_F(MultinomialLogitTest, FindMle) {
    Vector age = autopref_.getvar(1);
    CategoricalVariable sex = autopref_.get_nominal(2);
    CategoricalVariable vehicle_type = autopref_.get_nominal(5);
    int subject_xdim = 3;
    int choice_xdim = 0;
    int sample_size = autopref_.nobs();
    int nchoices = vehicle_type[0]->nlevels();

    NEW(MultinomialLogitModel, model)(nchoices, subject_xdim, choice_xdim);
    for (int i = 0; i < sample_size; ++i) {
      NEW(VectorData, subject_predictors)(
          Vector{1.0, age[i], double(sex.label(i) == "Male")});
      std::vector<Ptr<VectorData>> empty_choice_predictors;
      NEW(ChoiceData, data_point)(*vehicle_type[i],
                                  subject_predictors,
                                  empty_choice_predictors);
      model->add_data(data_point);
    }
    model->mle();
  }

  TEST_F(MultinomialLogitTest, MCMC) {
    Vector age = autopref_.getvar(1);
    CategoricalVariable sex = autopref_.get_nominal(2);
    CategoricalVariable vehicle_type = autopref_.get_nominal(5);
    int subject_xdim = 3;
    int choice_xdim = 0;
    int sample_size = autopref_.nobs();
    int nchoices = vehicle_type[0]->nlevels();

    NEW(MultinomialLogitModel, model)(nchoices, subject_xdim, choice_xdim);
    for (int i = 0; i < sample_size; ++i) {
      NEW(VectorData, subject_predictors)(
          Vector{1.0, age[i], double(sex.label(i) == "Male")});
      std::vector<Ptr<VectorData>> empty_choice_predictors;
      NEW(ChoiceData, data_point)(*vehicle_type[i],
                                  subject_predictors,
                                  empty_choice_predictors);
      model->add_data(data_point);
    }

    int xdim = model->beta_size();
    NEW(MvnModel, coefficient_prior)(xdim);
    Vector prior_inclusion_probabilities(xdim, 1.0 / xdim);
    NEW(VariableSelectionPrior, inclusion_prior)(prior_inclusion_probabilities);
    NEW(MultinomialLogitCompositeSpikeSlabSampler, sampler)(
        model.get(),
        coefficient_prior,
        inclusion_prior);
    model->set_method(sampler);

    int niter = 100;
    for (int i = 0; i < niter; ++i) {
      model->sample_posterior();
    }
  }

}  // namespace
