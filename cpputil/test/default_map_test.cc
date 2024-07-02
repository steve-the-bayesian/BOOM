#include "gtest/gtest.h"
#include "cpputil/DefaultMap.hpp"
// #include "test_utils/test_utils.hpp"

namespace {
  using namespace BOOM;
  using std::endl;
  
  TEST(DefaultMap, works_as_intended) {

    std::map<std::string, int> map;
    map["foo"] = 0;
    map["bar"] = 1;
    map["baz"] = 2;
    map["default"] = -1;

    DefaultMap<std::string, int> dmap(&map, "default");
    EXPECT_EQ(dmap["foo"], 0);
    EXPECT_EQ(dmap["bar"], 1);
    EXPECT_EQ(dmap["default"], -1);
    EXPECT_EQ(dmap["blah"], -1);
    
  }
  
}  // namespace
