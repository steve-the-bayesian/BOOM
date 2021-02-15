#include "gtest/gtest.h"
#include "Models/Impute/MvRegCopulaDataImputer.hpp"
#include "Models/Glm/PosteriorSamplers/MultivariateRegressionSampler.hpp"
#include "Models/MvnModel.hpp"
#include "distributions.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  struct SimulatedData {
    Matrix y_obs;
    Matrix y_true;
    Matrix y_nu;
    Matrix predictors;
  };

  // Args:
  //   sample_size: The number of observations to simulate.
  //   Sigma:  The residual variance matrix.  ydim x ydim
  //   coefficients:  The regression coefficients.  xdim x ydim.
  //   atoms:  A set of discrete values that might replace the regression.
  //   atom_probs: The probability of each atom.  Each entry is a discrete
  //     probability distribution with
  SimulatedData SimulateData(
      int sample_size,
      const SpdMatrix &Sigma,
      const Matrix &coefficients,
      const std::vector<Vector> &atoms,
      const std::vector<Vector> &atom_probs,
      const std::vector<Matrix> &observed_data_probs) {

    int xdim = coefficients.nrow();
    int ydim = Sigma.nrow();
    if (coefficients.ncol() != ydim) {
      report_error("The number of coefficient rows must match the dimension"
                   " of Sigma.");
    }
    Matrix X(sample_size, xdim);
    X.randomize();
    X.col(0) = 1.0;
    const double NA = std::numeric_limits<double>::quiet_NaN();

    Matrix yhat_true = X * coefficients;
    Matrix y_nu = yhat_true + rmvn_repeated(sample_size, Sigma);
    Matrix y_true = y_nu;
    Matrix y_obs = y_true;

    for (int i = 0; i < sample_size; ++i) {
      for (int j = 0; j < ydim; ++j) {
        int truth_atom = rmulti(atom_probs[j]);
        if (truth_atom < atoms[j].size()) {
          y_true(i, j) = atoms[j][truth_atom];
        }
        int obs_atom = rmulti(observed_data_probs[j].row(truth_atom));
        if (obs_atom < atoms[j].size()) {
          y_obs(i, j) = atoms[j][obs_atom];
        } else if (obs_atom == atoms[j].size()) {
          y_obs(i, j) = y_nu(i, j);
        } else {
          y_obs(i, j) = NA;
        }
      }
    }
    SimulatedData ans;
    ans.y_obs = y_obs;
    ans.y_true = y_true;
    ans.y_nu = y_nu;
    ans.predictors = X;
    return ans;
  }

  //===========================================================================
  class ErrorCorrectionModelTest : public ::testing::Test {
   protected:
    ErrorCorrectionModelTest() {
      GlobalRng::rng.seed(8675309);
      atoms_ = {0.0, 9999.0};
      atom_probs_ = {.05, 0.0, .95};
      atom_error_probs_ = Matrix(
        {
            Vector{.05, 0.0, .85, .10},
            Vector{.05, 0.0, .85, .10},
            Vector{.00, .45, .50, .50}
        }, true);
    }

    Vector atoms_;
    Vector atom_probs_;
    Matrix atom_error_probs_;
  };

    // Check that workspace updates when a model parameter changes.
  TEST_F(ErrorCorrectionModelTest, WorkspaceUpdates) {
    ErrorCorrectionModel model(atoms_);
    model.set_atom_probs(atom_probs_);
    model.set_atom_error_probs(atom_error_probs_);
    double logp1 = model.logp(1.0);
    model.set_atom_probs(Vector{.3, .2, .5});
    double logp2 = model.logp(1.0);
    EXPECT_GT(fabs(logp1- logp2), .1);
  }

  TEST_F(ErrorCorrectionModelTest, CopyParameters) {
    ErrorCorrectionModel model(atoms_);
    model.set_atom_probs(atom_probs_);
    EXPECT_TRUE(VectorEquals(atom_probs_, model.atom_probs()));

    model.set_atom_error_probs(atom_error_probs_);
    EXPECT_TRUE(MatrixEquals(atom_error_probs_, model.atom_error_probs()));

    ErrorCorrectionModel model2(atoms_);
    model2.copy_parameters(model);
    EXPECT_TRUE(VectorEquals(model2.atom_probs(), atom_probs_))
        << "atom_probs:           " << atom_probs_
        << "\n"
        << "model2.atom_probs:    " << model2.atom_probs();

    EXPECT_TRUE(MatrixEquals(model2.atom_error_probs(), atom_error_probs_))
        << "atom_error_probs:\n"
        << atom_error_probs_
        << "model2.atom_error_probs:\n"
        << model2.atom_error_probs();
  }

  TEST_F(ErrorCorrectionModelTest, CombineSuf) {
    ErrorCorrectionModel model(atoms_);
    model.set_atom_probs(atoms_);
    model.set_atom_error_probs(atom_error_probs_);

    ErrorCorrectionModel worker1(atoms_);
    worker1.copy_parameters(model);

    ErrorCorrectionModel worker2(atoms_);
    worker2.copy_parameters(model);

    worker1.impute_atom(0.0, GlobalRng::rng, true);
    worker1.impute_atom(0.0, GlobalRng::rng, true);
    worker1.impute_atom(0.0, GlobalRng::rng, true);
    worker1.impute_atom(1.0, GlobalRng::rng, true);
    worker1.impute_atom(atoms_.back(), GlobalRng::rng, true);

    worker2.impute_atom(0.0, GlobalRng::rng, true);
    worker2.impute_atom(42.0, GlobalRng::rng, true);
    worker2.impute_atom(42.0, GlobalRng::rng, true);
    worker2.impute_atom(42.0, GlobalRng::rng, true);
    worker2.impute_atom(42.0, GlobalRng::rng, true);
    worker2.impute_atom(8.0, GlobalRng::rng, true);
    worker2.impute_atom(1.0, GlobalRng::rng, true);
    worker2.impute_atom(atoms_.back(), GlobalRng::rng, true);

    EXPECT_DOUBLE_EQ(worker2.atom_prob_model().suf()->n().sum(), 8.0);
    double total =
        worker2.atom_error_prob_model(0).suf()->n().sum()
        + worker2.atom_error_prob_model(1).suf()->n().sum()
        + worker2.atom_error_prob_model(2).suf()->n().sum();
    EXPECT_DOUBLE_EQ(total, 8.0);

    model.clear_data();
    model.combine_sufficient_statistics(worker1);
    model.combine_sufficient_statistics(worker2);

    EXPECT_TRUE(VectorEquals(
        model.atom_prob_model().suf()->n(),
        worker1.atom_prob_model().suf()->n()
        + worker2.atom_prob_model().suf()->n()));

    for (int i = 0; i < atoms_.size() + 1; ++i) {
      EXPECT_TRUE(VectorEquals(
          model.atom_error_prob_model(i).suf()->n(),
          worker1.atom_error_prob_model(i).suf()->n()
          + worker2.atom_error_prob_model(i).suf()->n()));
    }
  }

  TEST_F(ErrorCorrectionModelTest, ImputeAtomTest) {
    Vector atoms = {0.0, 99999.0};
    ErrorCorrectionModel model(atoms);

    EXPECT_TRUE(std::isnan(model.true_value(2, 0.0)));
    EXPECT_TRUE(std::isnan(model.true_value(2, 99999.0)));
    EXPECT_DOUBLE_EQ(37.2, model.true_value(2, 37.2));

    EXPECT_DOUBLE_EQ(37.2, model.numeric_value(2, 37.2));
    EXPECT_TRUE(std::isnan(model.numeric_value(2, 0.0)));
    EXPECT_TRUE(std::isnan(model.numeric_value(2, 99999.0)));
  }

  //===========================================================================
  class MvRegCopulaDataImputerTest : public ::testing::Test {
   protected:
    MvRegCopulaDataImputerTest() {
      GlobalRng::rng.seed(8675309);

      sample_size_ = 100;
      xdim_ = 4;
      ydim_ = 3;
      coefficients_ = Matrix(xdim_, ydim_);
      coefficients_.randomize();

      Sigma_ = SpdMatrix(ydim_);
      Sigma_.randomize();

      atoms_.push_back({0});
      atoms_.push_back({0, 99999});
      atoms_.push_back({8675309});

      atom_probs_.push_back(Vector{.05, .95});
      atom_probs_.push_back(Vector{.05, 0, .95});
      atom_probs_.push_back(Vector{.01, .99});

      observed_data_probs_.push_back(Matrix(
          "0.95 0.00 0.05 | 0.25 0.5 0.25"));
      observed_data_probs_.push_back(Matrix(
          "0.80 0.00 0.10 0.05 | 0.25 0.50 0.20 0.05 | 0.25 0.25 0.25 0.25"));
      observed_data_probs_.push_back(Matrix(
          "0.95 0.00 0.05 | 0.25 0.5 0.25"));
      sim_ = SimulateData(sample_size_, Sigma_, coefficients_, atoms_,
                          atom_probs_, observed_data_probs_);
    }

    int sample_size_;
    int xdim_;
    int ydim_;
    Matrix coefficients_;
    SpdMatrix Sigma_;
    std::vector<Vector> atoms_;
    std::vector<Vector> atom_probs_;
    std::vector<Matrix> observed_data_probs_;
    SimulatedData sim_;
  };

  TEST_F(MvRegCopulaDataImputerTest, Construction) {
    int num_clusters = 4;
    MvRegCopulaDataImputer model(num_clusters, atoms_, xdim_);
  }

  TEST_F(MvRegCopulaDataImputerTest, ImputationTest) {
    int num_clusters = 4;
    MvRegCopulaDataImputer imputer(num_clusters, atoms_, xdim_);
    for (int i = 0; i < sample_size_; ++i) {
      NEW(MvRegData, data_point)(sim_.y_obs.row(i), sim_.predictors.row(i));
      imputer.add_data(data_point);
    }

    imputer.setup_worker_pool(4);

  }

  TEST_F(MvRegCopulaDataImputerTest, McmcTest) {
    int num_clusters = 4;
    MvRegCopulaDataImputer imputer(num_clusters, atoms_, xdim_);
    for (int i = 0; i < sample_size_; ++i) {
      NEW(MvRegData, data_point)(sim_.y_obs.row(i), sim_.predictors.row(i));
      imputer.add_data(data_point);
    }

    // Now set the prior.
    Ptr<MultivariateRegressionModel> reg = imputer.regression();
    NEW(MultivariateRegressionSampler, regression_sampler)(
        reg.get(),
        Matrix(xdim_, ydim_, 0.0),
        1.0,
        ydim_ + 1,
        SpdMatrix(ydim_, 1.0));
    reg->set_method(regression_sampler);

    int niter = 100;
    imputer.setup_worker_pool(4);
    for (int i = 0; i < niter; ++i) {
      imputer.sample_posterior();
    }

    // Check that all the parameters are moving.

    // Check that the imputed values are moving.

    // Check that the imputed values cover the true values.
  }

}  // namespace
