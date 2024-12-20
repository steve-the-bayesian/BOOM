#include "gtest/gtest.h"

#include "cpputil/Ptr.hpp"
#include "distributions.hpp"
#include "Models/Glm/Glm.hpp"
#include "stats/FreqDist.hpp"
#include "stats/ChiSquareTest.hpp"

#include "test_utils/test_utils.hpp"

namespace {
  using namespace BOOM;
  using std::endl;

  TEST(Ptr, works_as_intended) {
    GlobalRng::rng.seed(8675309);

    int sample_size = 1000000;
    std::vector<Ptr<RegressionData>> data_vector;

    for (int i = 0; i < sample_size; ++i) {
      Vector x(4);
      x.randomize();
      NEW(RegressionData, data_point)(rnorm(), x);
      data_vector.push_back(data_point);
    }

    // Try to trigger ASAN.
    std::vector<Ptr<RegressionData>> other_data_vector(data_vector);

    // Try to trigger ASAN.
    std::vector<Ptr<RegressionData>> moved_data_vector(
        std::move(data_vector));
  }

  class Base;
  void intrusive_ptr_add_ref(Base *object);
  void intrusive_ptr_release(Base *object);
  class Base : public RefCounted {
   public:
    friend void intrusive_ptr_add_ref(Base *object);
    friend void intrusive_ptr_release(Base *object);
    Base(int x)
        : x_(x)
    {}

   private:
    int x_;
  };

  void intrusive_ptr_add_ref(Base *object) {
    object->up_count();
  }

  void intrusive_ptr_release(Base *object) {
    object->down_count();
    if (object->ref_count() == 0) {
      delete object;
    }
  }

  class Derived : public Base {
   public:
    Derived(int x, int y)
        : Base(x),
          y_(y)
    {}
   private:
    int y_;
  };

  // Check that Ptr objects of different concrete classes, but which point to
  // the same object, evaluate to 'true' under the '==' operator.
  TEST(Ptr, BaseEqualsDerived) {
    Ptr<Derived> derived = new Derived(1, 2);
    Ptr<Base> base = derived;
    EXPECT_TRUE(base == derived);
  }

}  // namespace
