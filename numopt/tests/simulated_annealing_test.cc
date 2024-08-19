#include "gtest/gtest.h"
#include "numopt.hpp"
#include "numopt/SimulatedAnnealingOptimizer.hpp"
#include "LinAlg/Matrix.hpp"
#include "cpputil/math_utils.hpp"
#include "test_utils/test_utils.hpp"

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  template <class T>
  std::string print_vector(const std::vector<T> &stuff) {
    std::ostringstream out;
    for (size_t i = 0; i < stuff.size(); ++i) {
      out << stuff[i];
      if (i + 1  < stuff.size()) {
        out << ", ";
      }
    }
    return out.str();
  }


  class SimulatedAnnealingTest : public ::testing::Test {
   protected:
    SimulatedAnnealingTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(SimulatedAnnealingTest, SmallExample) {
    auto target = [](const Vector &x) {
      double ans = 17;
      for (size_t i = 0; i < x.size(); ++i) {
        ans += square(x[i] - i);
      }
      return ans;
    };

    Vector x(4);
    x.randomize();

    SimulatedAnnealingOptimizer opt(target);
    opt.set_max_fun_count(100000);
    opt.set_max_fun_count_per_temperatue(1000);
    opt.set_initial_temp(0.001);
    double min_value = opt.minimize(x, GlobalRng::rng);

    // std::cout << "fun count:  "  << opt.function_count() << "\n";

    EXPECT_GE(min_value, 17.0);
    EXPECT_LE(min_value, 17.1);

    double eps = 1e-2;

    EXPECT_NEAR(x[0], 0.0, eps);
    EXPECT_NEAR(x[1], 1.0, eps);
    EXPECT_NEAR(x[2], 2.0, eps);
    EXPECT_NEAR(x[3], 3.0, eps);
  }

}  // namespace
