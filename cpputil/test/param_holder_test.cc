#include "gtest/gtest.h"
#include "cpputil/ParamHolder.hpp"
#include "Models/ParamTypes.hpp"
#include "test_utils/test_utils.hpp"

namespace {
  using namespace BOOM;
  using std::endl;
  
  TEST(ParamHolder, restores_params) {
    NEW(UnivParams, uprm)(1.0);
    {
      Vector storage;
      ParamHolder holder(uprm, storage);
      uprm->set(2.0);
      EXPECT_DOUBLE_EQ(uprm->value(), 2.0);
    }
    EXPECT_DOUBLE_EQ(uprm->value(), 1.0);


    NEW(VectorParams, vprm)({1.2, 3, 4});
    {
      Vector storage;
      ParamHolder holder(vprm, storage);
      vprm->set(Vector{ 3, 4, 5});
      EXPECT_TRUE(VectorEquals(vprm->value(), Vector{3, 4, 5}));
    }
    EXPECT_TRUE(VectorEquals(vprm->value(), Vector{1.2, 3, 4}));
  }

  TEST(RealValueHolder, restores_value) {
    double x = 1.2;
    {
      RealValueHolder holder(x);
      x = 3.14;
    }
    EXPECT_DOUBLE_EQ(x, 1.2);
  }
  
}  // namespace
