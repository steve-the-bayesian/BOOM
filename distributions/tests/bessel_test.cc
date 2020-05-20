#include <functional>
#include "gtest/gtest.h"
#include "distributions.hpp"
#include "test_utils/test_utils.hpp"

namespace {

  using namespace BOOM;
  using std::cout;
  using std::endl;

  TEST(BesselKTest, MatchesStdLib) {
    // Order of inputs is reversed.
    EXPECT_DOUBLE_EQ(bessel_k(3.0, 4.0, 1),
                     std::cyl_bessel_k(4.0, 3.0));
  }

}  // namespace
