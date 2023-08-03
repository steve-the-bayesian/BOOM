#include "gtest/gtest.h"

#include "Models/GP/GaussianProcessRegressionModel.hpp"
#include "distributions.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;

  class GpTest : public ::testing::Test {
   protected:
    GpTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(GpTest, MeanPredictionTest) {
    GaussianProcessRegressionModel model(
        new ZeroFunction,
        new RadialBasisFunction(1.7));

    int nobs = 20;

    Matrix X(nobs, 1);
    X.randomize();
    Vector y = 3 * X.col(0) + rnorm_vector(nobs, 4, 7);

    std::cout << cbind(y, X);

    for (int i = 0; i < nobs; ++i) {
      NEW(RegressionData, data_point)(y[i], X.row(i));
      model.add_data(data_point);
    }

    int nnew = 5;
    Matrix Xnew(nnew, 1);
    Vector ynew = 3 * Xnew.col(0) + rnorm_vector(nnew, 4, 7);

    Ptr<MvnModel> predictive_distribution = model.predict_distribution(Xnew);



  }

}  // namespace
