#include "gtest/gtest.h"
#include "Models/ChisqModel.hpp"
#include "Models/PosteriorSamplers/IndependentMvnVarSampler.hpp"
#include "Models/StateSpace/StateModels/TrigStateModel.hpp"
#include "Models/StateSpace/StateModels/LocalLevelStateModel.hpp"
#include "Models/StateSpace/StateSpaceModel.hpp"
#include "cpputil/Date.hpp"
#include "cpputil/seq.hpp"
#include "cpputil/math_utils.hpp"
#include "distributions.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;
  
  template <class V1, class V2>
  bool VectorEquals(const V1 &v1, const V2 &v2) {
    Vector v = v1 - v2;
    return v.max_abs() < 1e-8;
  }

  template <class M1, class M2>
  bool MatrixEquals(const M1 &m1, const M2 &m2) {
    Matrix m = m1 - m2;
    return m.max_abs() < 1e-8;
  }

  class TrigStateModelTest : public ::testing::Test {
   protected:
    TrigStateModelTest()
        : period_(365.25),
          frequencies_(seq<double>(1.0, 12.0)),
          coefficient_innovation_precision_prior_(new ChisqModel(10.0, 3.0))
    {
      GlobalRng::rng.seed(8675309);
    }

    double period_;
    Vector frequencies_;
    Ptr<ChisqModel> coefficient_innovation_precision_prior_;
    std::vector<Ptr<GammaModelBase>> specific_coefficient_precision_priors_;
    Ptr<IndependentMvnVarSampler> coefficient_precision_sampler_;
  };

  inline double dsquare(double x) {return x * x;}
  
  TEST_F(TrigStateModelTest, ModelMatrices) {
    TrigStateModel trig(period_, frequencies_);
    EXPECT_EQ(trig.state_dimension(), 2 * frequencies_.size());

    SparseVector Z = trig.observation_matrix(17.0);
    // Successive elements of Z are pairs of sines and cosines.
    EXPECT_EQ(Z.size(), 2 * frequencies_.size());
    EXPECT_DOUBLE_EQ(dsquare(Z[0]) + dsquare(Z[1]), 1.0);
    EXPECT_DOUBLE_EQ(dsquare(Z[2]) + dsquare(Z[3]), 1.0);
    EXPECT_DOUBLE_EQ(dsquare(Z[4]) + dsquare(Z[5]), 1.0);

    EXPECT_DOUBLE_EQ(0.0, trig.suf()->n());
    for (int i = 0; i < trig.state_dimension(); ++i) {
      EXPECT_DOUBLE_EQ(0.0, trig.suf()->sum(i));
      EXPECT_DOUBLE_EQ(0.0, trig.suf()->sumsq(i));
    }
    Vector then = seq<double>(1.0, trig.state_dimension());
    Vector now = then + 2;
    trig.observe_state(then, now, 3, nullptr);
    EXPECT_DOUBLE_EQ(1.0, trig.suf()->n());
    for (int i = 0; i < trig.state_dimension(); ++i) {
      EXPECT_DOUBLE_EQ(2, trig.suf()->sum(i));
      EXPECT_DOUBLE_EQ(4, trig.suf()->sumsq(i));
    }
  }
  
}  // namespace
