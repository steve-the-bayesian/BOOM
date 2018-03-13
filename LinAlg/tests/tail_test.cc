#include "gtest/gtest.h"
#include "LinAlg/Vector.hpp"
#include "LinAlg/VectorView.hpp"

namespace {
  using namespace BOOM;
  using std::endl;
  
  template <class V1, class V2>
  bool VectorEquals(const V1 &v1, const V2 &v2) {
    Vector v = v1 - v2;
    return v.max_abs() < 1e-8;
  }
  
  TEST(tail, works_as_intended) {
    Vector jenny = {8, 6, 7, 5, 3, 0, 9};
    VectorView jenny_view(jenny);
    ConstVectorView const_jenny_view(jenny);
    EXPECT_TRUE(VectorEquals(
        tail(jenny, 3),
        Vector({3, 0, 9})));
    EXPECT_TRUE(VectorEquals(
        tail(jenny_view, 3),
        Vector({3, 0, 9})));
    EXPECT_TRUE(VectorEquals(
        const_tail(jenny, 3),
        Vector({3, 0, 9})));
    EXPECT_TRUE(VectorEquals(
        const_tail(const_jenny_view, 3),
        Vector({3, 0, 9})));
    EXPECT_TRUE(VectorEquals(
        const_tail(jenny_view, 3),
        Vector({3, 0, 9})))
        << "jenny_view is " << jenny_view << endl
        << "const_tail(jenny_view, 3) is " << const_tail(jenny_view, 3) << endl;
  }
  
}  // namespace
