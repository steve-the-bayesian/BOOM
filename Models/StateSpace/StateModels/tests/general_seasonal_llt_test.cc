#include "gtest/gtest.h"
#include "Models/StateSpace/StateModels/GeneralSeasonalStateModel.hpp"

namespace {

  using namespace BOOM;
  using std::endl;
  using std::cout;

  class GeneralSeasonalLLTTest : public ::testing::Test {
   protected:
    GeneralSeasonalLLTTest()
    {}
  };


  //===========================================================================
  TEST_F(GeneralSeasonalLLTTest, SmokeTest) {
    NEW(GeneralSeasonalLLT, model)(4);
  }

  TEST_F(GeneralSeasonalLLTTest, ModelMatrices) {
    NEW(GeneralSeasonalLLT, model)(3);

    Ptr<SparseMatrixBlock> transition = model->state_transition_matrix(0);
    EXPECT_EQ(transition->nrow(), 6);
    EXPECT_EQ(transition->ncol(), 6);
    Matrix base_dense_transition(6, 6, 0.0);
    Matrix LLT("1 1|0 1");
  }

}  // namespace
