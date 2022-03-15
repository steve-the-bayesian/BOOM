#include "gtest/gtest.h"

#include "Models/Glm/GlmCoefs.hpp"
#include "distributions.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  class GlmCoefsTest : public ::testing::Test {
   protected:
    GlmCoefsTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(GlmCoefsTest, Constructors) {
    GlmCoefs c1(Vector{1.0, 0.0, -3.2}, false);
    EXPECT_EQ(c1.nvars(), 3);
    EXPECT_EQ(c1.nvars_possible(), 3);

    GlmCoefs c2(Vector{1.0, 0.0, -3.2}, true);
    EXPECT_EQ(c2.nvars(), 2);
    EXPECT_EQ(c2.nvars_possible(), 3);

    GlmCoefs c3(Vector{1.0, 0.0, -3.2}, Selector("001"));
    EXPECT_TRUE(VectorEquals(c3.Beta(), Vector{0.0, 0.0, -3.2}));
  }

  TEST_F(GlmCoefsTest, AddDrop) {
    GlmCoefs  c1(Vector{1.0, 0.0, -3.2}, true);

    EXPECT_TRUE(c1.inc(0));
    EXPECT_FALSE(c1.inc(1));
    EXPECT_TRUE(c1.inc(2));
    EXPECT_EQ(2, c1.nvars());
    EXPECT_EQ(1, c1.nvars_excluded());
    EXPECT_EQ(3, c1.nvars_possible());

    c1.drop(0);
    EXPECT_FALSE(c1.inc(0));
    EXPECT_FALSE(c1.inc(1));
    EXPECT_TRUE(c1.inc(2));
    EXPECT_EQ(1, c1.nvars());
    EXPECT_EQ(2, c1.nvars_excluded());
    EXPECT_EQ(3, c1.nvars_possible());

    c1.add(1);
    EXPECT_FALSE(c1.inc(0));
    EXPECT_TRUE(c1.inc(1));
    EXPECT_TRUE(c1.inc(2));
    EXPECT_EQ(2, c1.nvars());
    EXPECT_EQ(1, c1.nvars_excluded());
    EXPECT_EQ(3, c1.nvars_possible());

    c1.flip(2);
    EXPECT_FALSE(c1.inc(0));
    EXPECT_TRUE(c1.inc(1));
    EXPECT_FALSE(c1.inc(2));
    EXPECT_EQ(1, c1.nvars());
    EXPECT_EQ(2, c1.nvars_excluded());
    EXPECT_EQ(3, c1.nvars_possible());

    c1.flip(2);
    EXPECT_FALSE(c1.inc(0));
    EXPECT_TRUE(c1.inc(1));
    EXPECT_TRUE(c1.inc(2));
    EXPECT_EQ(2, c1.nvars());
    EXPECT_EQ(1, c1.nvars_excluded());
    EXPECT_EQ(3, c1.nvars_possible());

    c1.drop_all();
    EXPECT_FALSE(c1.inc(0));
    EXPECT_FALSE(c1.inc(1));
    EXPECT_FALSE(c1.inc(2));
    EXPECT_EQ(0, c1.nvars());
    EXPECT_EQ(3, c1.nvars_excluded());
    EXPECT_EQ(3, c1.nvars_possible());

    c1.add_all();
    EXPECT_TRUE(c1.inc(0));
    EXPECT_TRUE(c1.inc(1));
    EXPECT_TRUE(c1.inc(2));
    EXPECT_EQ(3, c1.nvars());
    EXPECT_EQ(0, c1.nvars_excluded());
    EXPECT_EQ(3, c1.nvars_possible());
  }

  // Check that the predict method corresponds to the correct dot product with
  // either the full or sub-selected vector or matrix.
  TEST_F(GlmCoefsTest, PredictTest) {
    GlmCoefs c1(Vector{2.3, 1.8, 0, 0, -6.4}, true);
    Vector beta = c1.included_coefficients();
    EXPECT_TRUE(VectorEquals(beta, Vector{2.3, 1.8, -6.4}));
    EXPECT_TRUE(VectorEquals(c1.Beta(), Vector{2.3, 1.8, 0, 0, -6.4}));

    Vector sparse_x(3);
    sparse_x.randomize();
    double yhat = c1.predict(sparse_x);
    EXPECT_DOUBLE_EQ(yhat, beta.dot(sparse_x));

    VectorView sparse_view(sparse_x);
    EXPECT_DOUBLE_EQ(yhat, beta.dot(sparse_view));

    ConstVectorView const_sparse_view(sparse_x);
    EXPECT_DOUBLE_EQ(yhat, beta.dot(const_sparse_view));

    Vector dense_x(5);
    dense_x.randomize();
    yhat = c1.predict(dense_x);
    sparse_x = c1.inc().select(dense_x);
    EXPECT_DOUBLE_EQ(yhat, beta.dot(sparse_x));

    Matrix X(10, 3);
    X.randomize();
    Vector pred = X * beta;
    EXPECT_TRUE(VectorEquals(pred, c1.predict(X)));

    Matrix bigX(10, 5);
    bigX.randomize();
    pred = bigX * c1.Beta();
    EXPECT_TRUE(VectorEquals(pred, c1.predict(bigX)))
        << "     pred = " << pred << "\n"
        << "predict() = " << c1.predict(X) << "\n";
  }

  TEST_F(GlmCoefsTest, GetSetTest) {
    GlmCoefs c1(Vector{2.3, 1.8, 0, 0, -6.4}, true);
    EXPECT_TRUE(VectorEquals(
        c1.included_coefficients(), Vector{2.3, 1.8, -6.4}));
    EXPECT_TRUE(VectorEquals(
        c1.Beta(), Vector{2.3, 1.8, 0, 0, -6.4}));
    EXPECT_DOUBLE_EQ(c1.Beta(0), 2.3);
    EXPECT_DOUBLE_EQ(c1.Beta(1), 1.8);
    EXPECT_DOUBLE_EQ(c1.Beta(2), 0.0);
    EXPECT_DOUBLE_EQ(c1.Beta(3), 0.0);
    EXPECT_DOUBLE_EQ(c1.Beta(4), -6.4);

    EXPECT_EQ(c1.nvars(), 3);
    // When setting a dense vector, nonzero values get coerced to zero.
    c1.set_Beta(Vector{8, 5, 3, 0, 9});
    EXPECT_TRUE(VectorEquals(
        c1.Beta(), Vector{8, 5, 0, 0, 9}));

    // When setting a subset, make sure the right elements are set to zero.
    c1.set_inc(Selector("11011"));
    c1.set_Beta(Vector{1, 2, 3, 4, 5});
    c1.set_subset(Vector{10, 9, 8}, 1);
    EXPECT_TRUE(VectorEquals(c1.Beta(), Vector{1, 10, 0, 8, 5}));

    //
    c1.set_sparse_coefficients(Vector{8.6, 7.5, 3.09},
                               std::vector<BOOM::uint>{1, 2, 4});

    EXPECT_EQ(c1.size(false), 5);
    EXPECT_EQ(c1.size(true), 3);
    EXPECT_EQ(c1.nvars(), 3);
    EXPECT_EQ(c1.nvars_possible(), 5);

    EXPECT_FALSE(c1.inc(0));
    EXPECT_TRUE(c1.inc(1));
    EXPECT_TRUE(c1.inc(2));
    EXPECT_FALSE(c1.inc(3));
    EXPECT_TRUE(c1.inc(4));

    EXPECT_DOUBLE_EQ(c1.Beta(0), 0.0);
    EXPECT_DOUBLE_EQ(c1.Beta(1), 8.6);
    EXPECT_DOUBLE_EQ(c1.Beta(2), 7.5);
    EXPECT_DOUBLE_EQ(c1.Beta(3), 0.0);
    EXPECT_DOUBLE_EQ(c1.Beta(4), 3.09);

  }


}  // namespace
