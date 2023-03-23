#include "gtest/gtest.h"

#include "Models/StateSpace/Filters/SparseVector.hpp"
#include "distributions.hpp"
#include "LinAlg/Vector.hpp"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/SpdMatrix.hpp"

#include "test_utils/test_utils.hpp"

namespace {

  using namespace BOOM;
  using std::endl;

  template <class T>
  std::string to_string(const std::vector<T> &v) {
    std::ostringstream out;
    for (int i = 0; i < v.size(); ++i) {
      out << v[i] << " ";
    }
    out << "\n";
    return out.str();
  }

  class SparseVectorTest : public ::testing::Test {
   protected:
    SparseVectorTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(SparseVectorTest, EmptyVector) {
    SparseVector empty(3);
    EXPECT_EQ(empty.size(), 3);
  }


  // Check that the size of the vector does not change as elements are added.
  TEST_F(SparseVectorTest, Sizes) {
    SparseVector sparse(7);
    EXPECT_EQ(sparse.size(), 7);

    sparse[3] = 18.0;
    EXPECT_EQ(sparse.size(), 7);

    sparse[5] = -12.0;
    EXPECT_EQ(sparse.size(), 7);
  }

  TEST_F(SparseVectorTest, SimpleMath) {
    SparseVector sparse(7);
    sparse[2] = 1.8;
    EXPECT_DOUBLE_EQ(1.8, sparse.sum());
    sparse[5] = 7.2;
    EXPECT_DOUBLE_EQ(1.8 + 7.2, sparse.sum());

    sparse *= 4.2;
    EXPECT_DOUBLE_EQ(sparse[2], 1.8 * 4.2);
    EXPECT_DOUBLE_EQ(sparse[5], 7.2 * 4.2);
    EXPECT_DOUBLE_EQ(sparse.sum(), 4.2 * (1.8 + 7.2));

    sparse /= 4.2;
    EXPECT_DOUBLE_EQ(sparse[2], 1.8);
    EXPECT_DOUBLE_EQ(sparse[5], 7.2);
    EXPECT_DOUBLE_EQ(sparse.sum(), 1.8 + 7.2);
  }

  TEST_F(SparseVectorTest, DotProducts) {
    SparseVector sparse(7);
    sparse[2] = 1.8;
    sparse[5] = 7.2;

    Vector v(7);
    v.randomize();
    EXPECT_DOUBLE_EQ(v[2] * 1.8 + v[5] * 7.2, sparse.dot(v));
    EXPECT_DOUBLE_EQ(v[2] * 1.8 + v[5] * 7.2, sparse.dot(VectorView(v)));
    EXPECT_DOUBLE_EQ(v[2] * 1.8 + v[5] * 7.2, sparse.dot(ConstVectorView(v)));
  }

  TEST_F(SparseVectorTest, AddThisTo) {
    SparseVector sparse(7);
    sparse[2] = 1.8;
    sparse[5] = 7.2;

    Vector v(7);
    v.randomize();
    Vector original = v;

    sparse.add_this_to(v, 2.3);
    EXPECT_DOUBLE_EQ(v[2], original[2] + 2.3 * 1.8);
    EXPECT_DOUBLE_EQ(v[5], original[5] + 2.3 * 7.2);

    v = original;
    VectorView view(v);
    sparse.add_this_to(view, 2.3);
    EXPECT_DOUBLE_EQ(v[2], original[2] + 2.3 * 1.8);
    EXPECT_DOUBLE_EQ(v[5], original[5] + 2.3 * 7.2);
  }

  TEST_F(SparseVectorTest, LinearAlgebra) {
    SparseVector sparse(7);
    sparse[2] = 1.8;
    sparse[5] = 7.2;

    Vector dense = sparse.dense();
    EXPECT_TRUE(VectorEquals(dense, Vector{0, 0, 1.8, 0, 0, 7.2, 0}));

    SpdMatrix P(7);
    P.randomize();

    EXPECT_DOUBLE_EQ(sparse.sandwich(P), dense.dot(P * dense));

    Vector x(4);
    x.randomize();
    EXPECT_TRUE(MatrixEquals(
        sparse.outer_product_transpose(x, 2.3),
        x.outer(dense, 2.3)));

    SpdMatrix newP = P;
    sparse.add_outer_product(newP, 1.4);
    EXPECT_TRUE(MatrixEquals(
        newP,
        P + 1.4 * dense.outer()));
  }

  TEST_F(SparseVectorTest, ViewReadsAndWrites) {
    SparseVector base(20);
    base[3] = 1.8;
    base[5] = 2.4;
    base[6] = -0.7;
    base[7] = 19.4;

    SparseVectorView view(&base, 1, 4, 2);
    EXPECT_EQ(4, view.size());
    EXPECT_DOUBLE_EQ(view[0], base[1]);
    EXPECT_DOUBLE_EQ(view[1], base[3]);
    EXPECT_DOUBLE_EQ(view[2], base[5]);
    EXPECT_DOUBLE_EQ(view[3], base[7]);

    view[2] *= 3.0;
    EXPECT_DOUBLE_EQ(base[5], 2.4 * 3.0);
    EXPECT_DOUBLE_EQ(view[2], 2.4 * 3.0);

    view *= 2.0;
    auto it = view.begin();
    EXPECT_EQ(it->first, 3);
    EXPECT_EQ(it->second, 1.8 * 2.0);

    ++it;
    EXPECT_EQ(it->first, 5);
    EXPECT_EQ(it->second, 2.4 * 3.0 * 2.0);

    EXPECT_DOUBLE_EQ(base[1], 0.0);
    EXPECT_DOUBLE_EQ(base[3], 1.8 * 2.0);
    EXPECT_DOUBLE_EQ(base[5], 2.4 * 3.0 * 2.0);
    EXPECT_DOUBLE_EQ(base[7], 19.4 * 2.0);

    view /= 2.0;
    EXPECT_DOUBLE_EQ(base[1], 0.0);
    EXPECT_DOUBLE_EQ(base[3], 1.8 );
    EXPECT_DOUBLE_EQ(base[5], 2.4 * 3.0);
    EXPECT_DOUBLE_EQ(base[7], 19.4);

    std::vector<double> dense;
    std::vector<int> pos;
    for (auto it = view.begin(); it != view.end(); ++it) {
      pos.push_back(it->first);
      dense.push_back(it->second);
    }
    EXPECT_EQ(pos.size(), 3) << "\n" << to_string(pos) << to_string(dense);

  }
}  // namespace
