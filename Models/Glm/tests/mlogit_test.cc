#include "gtest/gtest.h"

#include "stats/DataTable.hpp"
#include "distributions.hpp"

#include "Models/MvnModel.hpp"
#include "Models/Glm/VariableSelectionPrior.hpp"
#include "Models/Glm/MultinomialLogitModel.hpp"
#include "Models/Glm/PosteriorSamplers/MultinomialLogitCompositeSpikeSlabSampler.hpp"

#include "TargetFun/MultinomialLogitTransform.hpp"

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
      // std::string raw_path = "Models/Glm/tests/autopref.txt";
      // autopref_.read_file(raw_path, false, "\t");
      // autopref_.set_vnames(std::vector<std::string>{
      //          "country", "age", "sex", "married", "size", "type", "is25"});
      // American	34	Male	Married	Large	Family	No
      // Japanese	36	Male	Single	Small	Sporty	No
      // Japanese	23	Male	Married	Small	Family	No
      // American	29	Male	Single	Large	Family	No
      // American	39	Male	Married	Medium	Family	No
    }

    DataTable autopref_;
  };

  // TEST_F(MultinomialLogitTest, FindMle) {
  //   Vector age = autopref_.getvar(1);
  //   CategoricalVariable sex = autopref_.get_nominal(2);
  //   CategoricalVariable vehicle_type = autopref_.get_nominal(5);
  //   int subject_xdim = 3;
  //   int choice_xdim = 0;
  //   int sample_size = autopref_.nobs();
  //   int nchoices = vehicle_type[0]->nlevels();

  //   NEW(MultinomialLogitModel, model)(nchoices, subject_xdim, choice_xdim);
  //   for (int i = 0; i < sample_size; ++i) {
  //     NEW(VectorData, subject_predictors)(
  //         Vector{1.0, age[i], double(sex.label(i) == "Male")});
  //     std::vector<Ptr<VectorData>> empty_choice_predictors;
  //     NEW(ChoiceData, data_point)(*vehicle_type[i],
  //                                 subject_predictors,
  //                                 empty_choice_predictors);
  //     model->add_data(data_point);
  //   }
  //   model->mle();
  // }

  TEST_F(MultinomialLogitTest, model_works_when_choice_dimension_is_zero) {
    int nobs = 5000;
    int Nchoices = 4;
    int xdim = 20;

    // Simulate some fake data.
    Matrix Xsubject(nobs, xdim);
    Xsubject.randomize();
    Xsubject *= 5;
    Xsubject.col(0) = 1.0;

    Matrix beta_subject(Nchoices, xdim, 0.0);
    beta_subject.col(0).randomize();
    beta_subject.col(1).randomize();
    beta_subject.row(0) = 0.0;

    Matrix logits = Xsubject * beta_subject.transpose();
    MultinomialLogitTransform transform;
    Matrix probs = logits;
    std::vector<int> choices(nobs);
    for (int i = 0; i < nobs; ++i) {
      // std::cout << transform.to_probs_full(logits.row(i)) << "\n";
      probs.row(i) = transform.to_probs_full(logits.row(i));
      choices[i] = rmulti(probs.row(i));
    }
    std::vector<Ptr<CategoricalData>> responses =
        create_categorical_data(choices);

    NEW(MultinomialLogitModel, model)(responses, Xsubject);
    for (int m = 1; m < Nchoices; ++m) {
      model->set_beta_subject(beta_subject.row(m), m);
    }

    // Check that the predict method works.
    NEW(ChoiceData, first_observation)(*responses[0], new VectorData(Xsubject.row(0)));
    Vector pred = model->predict(first_observation);
    const Vector &first_xsubject(first_observation->Xsubject());
    Vector logits_vector(Nchoices);
    for (int m = 0; m < Nchoices; ++m) {
      logits_vector[m] = model->beta_subject(m).dot(first_xsubject);
    }
    Vector probs_vector = transform.to_probs_full(logits_vector);

    EXPECT_TRUE(VectorEquals(probs_vector, pred))
        << "\n"
        << "probs_vector: " << probs_vector << "\n"
        << "pred        : " << pred << "\n";

    int beta_dim = Xsubject.ncol() * (model->Nchoices() - 1);
    NEW(MvnModel, slab)(beta_dim);
    NEW(VariableSelectionPrior, spike)(Vector(beta_dim, 1.0 / beta_dim));
    NEW(MultinomialLogitCompositeSpikeSlabSampler, sampler)(
        model.get(), slab, spike);
    model->set_method(sampler);

    model->mle();

    // Create space to hold the MCMC draws.
    int niter = 1000;
    std::vector<Matrix> beta_draws;
    for (int m = 0; m < model->Nchoices(); ++m) {
      beta_draws.push_back(Matrix(niter, Xsubject.ncol(), 0.0));
    }

    // Run the MCMC algorithm.
    for (int i = 0; i < niter; ++i) {
      model->sample_posterior();
      for (int m = 0; m < model->Nchoices(); ++m) {
        beta_draws[m].row(i) = model->beta_subject(m);
      }
    }

    // Compute the posterior probability that each coefficient is nonzero.
    // There should be an intercept term and a single coefficient in position 1.
    Matrix prob_nonzero(model->Nchoices(), Xsubject.ncol(), 0.0);
    for (int i = 0; i < niter; ++i) {
      for (int m = 0; m < model->Nchoices(); ++m) {
        for (int j = 0; j < beta_draws[m].ncol(); ++j) {
          if (fabs(beta_draws[m](i, j)) > 1e-8) {
            ++prob_nonzero(m, j);
          }
        }
      }
    }
    prob_nonzero /= niter;
    for (int m = 0; m < model->Nchoices(); ++m) {
      for (int j = 0; j <= 1; ++j) {
        if (m > 0) {
          EXPECT_GT(prob_nonzero(m, j), .5)
              << "\n-----------------\n"
              << "At choice level " << m << " predictor " << j
              << " and Nchoices = " << model->Nchoices() << ", "
              << "prob_nonzero(" << m << ", " << j << ") = "
              << prob_nonzero(m, j);

        } else {
          EXPECT_EQ(prob_nonzero(m, j), 0.0);
        }
      }
      for (int j = 2; j < Xsubject.ncol(); ++j) {
        EXPECT_LT(prob_nonzero(m, j), .30);
      }
    }

  }

  // TEST_F(MultinomialLogitTest, MCMC) {
  //   Vector age = autopref_.getvar(1);
  //   CategoricalVariable sex = autopref_.get_nominal(2);
  //   CategoricalVariable vehicle_type = autopref_.get_nominal(5);
  //   int subject_xdim = 3;
  //   int choice_xdim = 0;
  //   int sample_size = autopref_.nobs();
  //   int nchoices = vehicle_type[0]->nlevels();

  //   NEW(MultinomialLogitModel, model)(nchoices, subject_xdim, choice_xdim);
  //   for (int i = 0; i < sample_size; ++i) {
  //     NEW(VectorData, subject_predictors)(
  //         Vector{1.0, age[i], double(sex.label(i) == "Male")});
  //     std::vector<Ptr<VectorData>> empty_choice_predictors;
  //     NEW(ChoiceData, data_point)(*vehicle_type[i],
  //                                 subject_predictors,
  //                                 empty_choice_predictors);
  //     model->add_data(data_point);
  //   }

  //   int xdim = model->beta_size();
  //   NEW(MvnModel, coefficient_prior)(xdim);
  //   Vector prior_inclusion_probabilities(xdim, 1.0 / xdim);
  //   NEW(VariableSelectionPrior, inclusion_prior)(prior_inclusion_probabilities);
  //   NEW(MultinomialLogitCompositeSpikeSlabSampler, sampler)(
  //       model.get(),
  //       coefficient_prior,
  //       inclusion_prior);
  //   model->set_method(sampler);

  //   int niter = 100;
  //   for (int i = 0; i < niter; ++i) {
  //     model->sample_posterior();
  //   }
  // }

  TEST_F(MultinomialLogitTest, SpikeSlabTest) {

  }

}  // namespace
