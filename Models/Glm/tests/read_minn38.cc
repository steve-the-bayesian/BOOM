#include "gtest/gtest.h"

#include "stats/DataTable.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  class ReadMinn38Test : public ::testing::Test {
   protected:
    ReadMinn38Test() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(ReadMinn38Test, ReadFile) {
    DataTable table("Models/Glm/tests/minn38.csv", true, ",");
    EXPECT_EQ(table.nobs(), 168);
  }

}  // namespace
