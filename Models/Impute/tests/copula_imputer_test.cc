#include "gtest/gtest.h"
#include "Models/Impute/MvRegCopulaDataImputer.hpp"
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

  TEST_F(MvRegCopulaDataImputerTest, Construction) {
    std::vector<Vector> atoms;
    atoms.push_back({0});
    atoms.push_back({0, 99999});
    atoms.push_back({8675309});
  }

}  // namespace
