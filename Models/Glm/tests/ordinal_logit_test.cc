#include "gtest/gtest.h"

#include "Models/MvnModel.hpp"
#include "Models/ExponentialIncrementModel.hpp"

#include "Models/Glm/OrdinalCutpointModel.hpp"
#include "Models/Glm/PosteriorSamplers/OrdinalLogitImputer.hpp"
#include "Models/Glm/PosteriorSamplers/OrdinalLogitPosteriorSampler.hpp"

#include "distributions.hpp"
#include "stats/FreqDist.hpp"
#include "test_utils/test_utils.hpp"
#include "test_utils/check_derivatives.hpp"

#include <fstream>
#include <functional>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  class OrdinalLogitTest : public ::testing::Test {
   protected:
    OrdinalLogitTest()
        : response_distribution_(std::vector<int>()) {
      GlobalRng::rng.seed(8675309);
    }

    // Populate the data_ member with a data set.  The number of values for y is
    // cutpoints.size() + 2.  There is an implicit zero on low end, and an
    // implicit infinity on the high end.
    void SimulateData(int sample_size, int xdim, const Vector &cutpoints) {
      NEW(FixedSizeIntCatKey, response_key)(cutpoints.size() + 2);
    
      Matrix predictors(sample_size, xdim);
      predictors.randomize();
      
      coefficients_ = rnorm_vector(xdim, 0, 1);

      Vector latent_response(sample_size);
      Vector yhat = predictors * coefficients_;
      std::vector<int> response_vector;
      for (int i = 0; i < sample_size; ++i) {
        latent_response[i] = rlogis() + yhat[i];

        int response = -1;
        if (latent_response[i] < 0) {
          response = 0;
        } else {
          for (int j = 0; j < cutpoints.size(); ++j) {
            if (latent_response[i] < cutpoints[j]) {
              response = j + 1;
              break;
            }
          }
        }
        if (response == -1) {
          response = cutpoints.size() + 1;
        }
        response_vector.push_back(response);
        NEW(OrdinalData, response_ptr)(response, response_key);
        NEW(VectorData, predictor_ptr)(predictors.row(i));
        NEW(OrdinalRegressionData, data_point)(response_ptr, predictor_ptr);
        data_.push_back(data_point);
      }
      response_distribution_ = FrequencyDistribution(response_vector, true);
    }

    std::vector<Ptr<OrdinalRegressionData>> data_;
    Vector coefficients_;
    FrequencyDistribution response_distribution_;
  };

  void TestDataImputer(double eta, double lower_cutpoint, double upper_cutpoint,
                       int niter) {
    OrdinalLogitImputer imputer;
    Vector draws(niter);
    for (int i = 0; i < niter; ++i){
      draws[i] = imputer.impute(GlobalRng::rng, eta, lower_cutpoint,
                                upper_cutpoint);
      // As we draw, check that the draws are within the lower and upper
      // cutpoints.
      EXPECT_GE(draws[i], lower_cutpoint);
      EXPECT_LE(draws[i], upper_cutpoint);
    }

    // Use a KS test to determine if the distribution of draws matches the
    // theoretical CDF.
    bool ok = DistributionsMatch(
        draws,
        [eta, lower_cutpoint, upper_cutpoint](double x) {
          return (plogis(x - eta) - plogis(lower_cutpoint - eta)) /
                  (plogis(upper_cutpoint - eta)
                   - plogis(lower_cutpoint - eta));} ); 
    if (!ok) {
      std::ostringstream filename;
      filename << "draws." << lower_cutpoint << "." << upper_cutpoint;
      std::ofstream out(filename.str());
      out << draws;
    }
    
    EXPECT_TRUE(ok) << "\n"
                    << "eta   = " << eta << "\n"
                    << "lower = " << lower_cutpoint << "\n"
                    << "upper = " << upper_cutpoint << std::endl;
  }
  
  TEST_F(OrdinalLogitTest, DataImputer) {
    TestDataImputer(-7.4, negative_infinity(), 0, 1000);
    TestDataImputer(-7.4, 0, 1.2, 1000);
    TestDataImputer(-7.4, 1.2, infinity(), 1000);
  }

  TEST_F(OrdinalLogitTest, Init) {
    int xdim = 3;
    int nlevels = 4;
    OrdinalLogitModel model(xdim, nlevels);
    EXPECT_EQ(nlevels, model.nlevels());
    EXPECT_EQ(model.xdim(), xdim);
  }

  TEST_F(OrdinalLogitTest, Cutpoints) {
    Vector coefficients(2);
    coefficients.randomize();
    Vector cutpoints = {.3, .8, 1.2};
    OrdinalLogitModel model(coefficients, cutpoints);

    EXPECT_EQ(negative_infinity(), model.lower_cutpoint(0));
    EXPECT_EQ(0, model.upper_cutpoint(0));

    EXPECT_EQ(0, model.lower_cutpoint(1));
    EXPECT_EQ(cutpoints[0], model.upper_cutpoint(1));

    EXPECT_EQ(cutpoints[0], model.lower_cutpoint(2));
    EXPECT_EQ(cutpoints[1], model.upper_cutpoint(2));
    
    EXPECT_EQ(cutpoints[1], model.lower_cutpoint(3));
    EXPECT_EQ(cutpoints[2], model.upper_cutpoint(3));

    EXPECT_EQ(cutpoints[2], model.lower_cutpoint(4));
    EXPECT_EQ(infinity(), model.upper_cutpoint(4));
  }

  
  
  TEST_F(OrdinalLogitTest, LinkFunctionDerivatives) {
    int xdim = 3;
    Vector cutpoints = {.25, .83, 1.6};
    OrdinalLogitModel model(xdim, cutpoints.size() + 2);

    auto scalar_target = [&model](double x, double &d1, double &d2, int nd) {
      double ans = model.link_inv(x);
      if (nd > 0) {
        d1 = model.dlink_inv(x);
        if (nd > 1) {
          d2 = model.ddlink_inv(x);
        }
      }
      return ans;
    };
    
    EXPECT_EQ("", CheckDerivatives(scalar_target, .7))
        << "Scalar derivatives failed.";

    // link_inv is a CDF, it should rise from 0 to 1.
    EXPECT_DOUBLE_EQ(0.0, model.link_inv(negative_infinity()));
    EXPECT_DOUBLE_EQ(1.0, model.link_inv(infinity()));

    // dlink_inv is a PDF.  It should be zero at the extremes.
    EXPECT_DOUBLE_EQ(0.0, model.dlink_inv(negative_infinity()));
    EXPECT_DOUBLE_EQ(0.0, model.dlink_inv(infinity()));

    // ddlink is the derivative of the PDF.  It should be 'positive zero' at
    // negative_infinity and 'negative_zero' at positive infinity.
    EXPECT_DOUBLE_EQ(0.0, model.ddlink_inv(negative_infinity()));
    EXPECT_DOUBLE_EQ(0.0, model.ddlink_inv(infinity()));
  }

  TEST_F(OrdinalLogitTest, LoglikeDerivatives) {
    int sample_size = 20;
    int xdim = 3;
    Vector cutpoints = {.25, .83, 1.6};
    SimulateData(sample_size, xdim, cutpoints);
    OrdinalLogitModel model(xdim, cutpoints.size() + 2);
    for (const auto &dp : data_) { model.add_data(dp); }
    
    Vector beta(xdim);
    beta.randomize();
    Vector theta = concat(beta, Vector{1, 2, 3});
    auto loglike_target = [&model](
        const Vector &x, Vector &g, Matrix &h, int nd) {
      return model.Loglike(x, g, h, nd);
    };
    EXPECT_EQ("", CheckDerivatives(loglike_target, theta, .1))
        << "log likelihood derivatives failed. " << endl
        << "response distribution: " << endl << response_distribution_;

  }
  
  TEST_F(OrdinalLogitTest, Mcmc) {
    int sample_size = 200;
    int xdim = 2;
    Vector cutpoints{0.23, .78};
    int nlevels = cutpoints.size() + 2;
    SimulateData(sample_size, xdim, cutpoints);
    NEW(OrdinalLogitModel, model)(xdim, nlevels);
    for (const auto &data_point : data_) {
      model->add_data(data_point);
    }
    
    NEW(MvnModel, beta_prior)(xdim);
    NEW(ExponentialIncrementModel, cutpoint_prior)(
        Vector(cutpoints.size(), .10));
    NEW(OrdinalLogitPosteriorSampler, sampler)(
        model.get(), beta_prior, cutpoint_prior);
    model->set_method(sampler);

    int niter = 10000;
    Matrix beta_draws(niter, xdim);
    Matrix cutpoint_draws(niter, cutpoints.size());
    for (int i = 0; i < niter; ++i) {
      model->sample_posterior();
      beta_draws.row(i) =  model->Beta();
      cutpoint_draws.row(i) = model->cutpoint_vector();
    }

    auto beta_status = CheckMcmcMatrix(beta_draws, coefficients_);
    EXPECT_TRUE(beta_status.ok) << beta_status;

    auto cutpoint_status = CheckMcmcMatrix(
        cutpoint_draws, cutpoints, .95, true,
        "cutpoint_draws.txt");
    EXPECT_TRUE(cutpoint_status.ok) << cutpoint_status;
  }
  
}  // namespace
