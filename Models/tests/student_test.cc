#include "gtest/gtest.h"

#include "Models/PosteriorSamplers/GenericStudentSampler.hpp"
#include "distributions.hpp"
#include "cpputil/Constants.hpp"

#include "test_utils/check_derivatives.hpp"
#include "test_utils/test_utils.hpp"

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  class StudentTest : public ::testing::Test {
   protected:
    StudentTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  class StudentLoglikeDirect {
   public:
    StudentLoglikeDirect(double sigma, double nu)
        : sigma_(sigma),
          nu_(nu)
    {}

    double operator()(const Vector &y) const {
      double ans = 0;
      for (int i = 0; i < y.size(); ++i) {
        ans += BOOM::lgamma((nu_ + 1) / 2) - BOOM::lgamma(nu_ / 2)
            -.5 * log(nu_ * Constants::pi) - log(sigma_)
            - .5 * (nu_ + 1) * log(1 + square(y[i]) / (nu_ * square(sigma_)));
      }
      return ans;
    }

   private:
    double sigma_;
    double nu_;
  };

  TEST_F(StudentTest, LogLikelihood) {
    Vector data = rnorm_vector(10, 0, 1);
    ZeroMeanStudentLogLikelihood loglike(data);
    d2TargetFunPointerAdapter target(loglike);
    Vector theta = {1.0, 3.0};

    StudentLoglikeDirect loglike_test(theta[0], theta[1]);

    double loglike_value = loglike(theta, nullptr, nullptr, true);
    EXPECT_NEAR(loglike_test(data), loglike_value, 1e-5);
    double loglike_direct = 0.0;
    for (int i = 0; i < data.size(); ++i) {
      loglike_direct += dstudent(data[i], 0, theta[0], theta[1], true);
    }
    EXPECT_NEAR(loglike_direct, loglike_value, 1e-8);

    std::string msg = CheckDerivatives(target, Vector{1.0, 3.0});
    if (msg != "") {
      std::cerr << msg;
    }
    EXPECT_EQ("", msg);
  }


}  // namespace
