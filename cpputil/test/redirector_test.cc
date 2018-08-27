#include "gtest/gtest.h"
#include "cpputil/Redirector.hpp"
#include "test_utils/test_utils.hpp"

namespace {
  using namespace BOOM;
  using std::endl;
  
  TEST(RedirectorTest, StealCout) {
   std::ostringstream out;
   {
      Redirector redirect(std::cout, out);
      std::cout << "foo";
   }
   EXPECT_EQ("foo", out.str());

   std::cout << " bar";
   EXPECT_EQ("foo", out.str());

   out << " baz";
   EXPECT_EQ("foo baz", out.str());
  }
  
}  // namespace
