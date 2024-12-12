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

  TEST_F(MixedMultivariateDataTest, Blah) {
    bool header = false;
    std::string path = "stats/tests/autopref.txt";
    DataTable autopref(path, header, "\t");
    /*
      American	34	Male	Married	Large	Family	No
      Japanese	36	Male	Single	Small	Sporty	No
      Japanese	23	Male	Married	Small	Family	No
    */
    EXPECT_EQ(autopref.nobs(), 263);
    EXPECT_EQ(autopref.nvars(), 7);
    EXPECT_EQ(autopref.variable_type(0), VariableType::categorical);
    EXPECT_EQ(autopref.variable_type(1), VariableType::numeric);
    EXPECT_EQ(autopref.variable_type(2), VariableType::categorical);
    EXPECT_EQ(autopref.variable_type(3), VariableType::categorical);
    EXPECT_EQ(autopref.variable_type(4), VariableType::categorical);
    EXPECT_EQ(autopref.variable_type(5), VariableType::categorical);
    EXPECT_EQ(autopref.variable_type(6), VariableType::categorical);
    EXPECT_EQ(autopref.vnames()[0], "V.0");
    EXPECT_EQ(autopref.vnames()[1], "V.1");

    header=true;
    path = "stats/tests/CarsClean.csv";
    DataTable cars(path, header, ",");
    EXPECT_EQ(cars.nobs(), 94);
    EXPECT_EQ(cars.nvars(), 22);
    EXPECT_EQ(cars.vnames()[0], "Make/Model");
    EXPECT_EQ(cars.vnames()[1], "MPGCity");
    EXPECT_EQ(cars.vnames()[21], "GP1000MCity");
  }
}  // namespace
