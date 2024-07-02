#include "gtest/gtest.h"
#include "cpputil/timer.hpp"
#include <thread>
#include <chrono>

namespace {
  using namespace BOOM;
  using std::endl;

  void sleep(int milliseconds) {
    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
  }

  // Shufle the same vector many times, and check that the values in each vector
  // position are uniformly distributed.
  TEST(timer, works_as_intended) {

    TimeRecorder recorder;
    for (int i = 0; i < 3; ++i) {
      ScopedTimer timer(&recorder, "for_loop");
      sleep(100);
      int counter = 0;
      while(counter++ < 4) {
        ScopedTimer while_loop_timer(&recorder, "while_loop");
        sleep(10);
      }
    }
    EXPECT_NEAR(recorder["while_loop"], .12, .02);
    EXPECT_NEAR(recorder["for_loop"], .12 + .3, .02);
    std::cout << recorder;
  }

}  // namespace
