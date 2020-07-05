#include "gtest/gtest.h"
#include "cpputil/seq.hpp"
#include "LinAlg/Vector.hpp"
#include "stats/Resampler.hpp"
#include "test_utils/test_utils.hpp"
#include "stats/ChiSquareTest.hpp"
#include "stats/FreqDist.hpp"
#include "stats/DataTable.hpp"

namespace {
  using namespace BOOM;
  using std::endl;

  class MixedMultivariateDataTest : public ::testing::Test {
   protected:
    MixedMultivariateDataTest()
        : color_key_(new CatKey({"red", "blue", "green"})),
          shape_key_(new CatKey({"circle", "square", "triangle", "rhombus"}))
    {
      GlobalRng::rng.seed(8675309);
    }

    Ptr<CatKey> color_key_;
    Ptr<CatKey> shape_key_;
  };

  TEST_F(MixedMultivariateDataTest, DefaultConstructor) {
    MixedMultivariateData data;
  }

}  // namespace
