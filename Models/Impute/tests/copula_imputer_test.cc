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

  class MvRegCopulaDataImputerTest : public ::testing::Test {
   protected:
    MvRegCopulaDataImputerTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  class ErrorCorrectionModelTest : public ::testing::Test {
   protected:
    ErrorCorrectionModelTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  struct SimulatedParameters {
  };

  SimulatedParameters SimulateParameters(
                                         );

  struct SimulatedData {
    Matrix y_obs;
    Matrix y_true;
    Matrix chi;
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
    Matrix chi = yhat_true + rmvn_repeated(sample_size, Sigma);
    Matrix y_true = chi;
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
          y_obs(i, j) = chi(i, j);
        } else {
          y_obs(i, j) = NA;
        }
      }
    }
    SimulatedData ans;
    ans.y_obs = y_obs;
    ans.y_true = y_true;
    ans.chi = chi;
    ans.predictors = X;
    return ans;
  }

  TEST_F(ErrorCorrectionModelTest, WorkspaceUpdates) {
    // Check that workspace updates when a model parameter changes.
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

  TEST_F(MvRegCopulaDataImputerTest, Construction) {
    std::vector<Vector> atoms;
    atoms.push_back({0});
    atoms.push_back({0, 99999});
    atoms.push_back({8675309});

    int num_clusters = 4;
    int xdim = 3;
    MvRegCopulaDataImputer model(num_clusters, atoms, xdim);
  }

  TEST_F(MvRegCopulaDataImputerTest, McmcTest) {
    int sample_size = 100;
    int xdim = 4;
    int ydim = 3;
    Matrix coefficients(xdim, ydim);
    coefficients.randomize();

    SpdMatrix Sigma(ydim);
    Sigma.randomize();

    std::vector<Vector> atoms;
    atoms.push_back({0});
    atoms.push_back({0, 99999});
    atoms.push_back({8675309});

    std::vector<Vector> atom_probs;
    atom_probs.push_back(Vector{.05, .95});
    atom_probs.push_back(Vector{.05, 0, .95});
    atom_probs.push_back(Vector{.01, .99});

    std::vector<Matrix> observed_data_probs;
    observed_data_probs.push_back(Matrix(
        "0.95 0.00 0.05 | 0.25 0.5 0.25"));
    observed_data_probs.push_back(Matrix(
        "0.80 0.00 0.10 0.05 | 0.25 0.50 0.20 0.05 | 0.25 0.25 0.25 0.25"));
    observed_data_probs.push_back(Matrix(
        "0.95 0.00 0.05 | 0.25 0.5 0.25"));

    SimulatedData sim = SimulateData(sample_size,
                                     Sigma,
                                     coefficients,
                                     atoms,
                                     atom_probs,
                                     observed_data_probs);

    int num_clusters = 4;
    MvRegCopulaDataImputer imputer(num_clusters, atoms, xdim);

    for (int i = 0; i < sample_size; ++i) {
      NEW(MvRegData, data_point)(sim.y_obs.row(i), sim.predictors.row(i));
      imputer.add_data(data_point);
    }

    // Now set the prior.
    Ptr<MultivariateRegressionModel> reg = imputer.regression();
    NEW(MultivariateRegressionSampler, regression_sampler)(
        reg.get(),
        Matrix(xdim, ydim, 0.0),
        1.0,
        ydim + 1,
        SpdMatrix(ydim, 1.0));
    reg->set_method(regression_sampler);

    int niter = 100;
    imputer.setup_worker_pool(16);
    for (int i = 0; i < niter; ++i) {
      imputer.sample_posterior();
    }

    // Check that all the parameters are moving.

    // Check that the imputed values are moving.

    // Check that the imputed values cover the true values.
  }

}  // namespace
