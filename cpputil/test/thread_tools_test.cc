#include "gtest/gtest.h"
#include "cpputil/ThreadTools.hpp"
#include "test_utils/test_utils.hpp"

namespace {
  using namespace BOOM;
  using std::endl;

  TEST(threading, record_integers) {
    ThreadWorkerPool pool;
    int num_threads = 3;
    int num_tasks = 7;
    pool.add_threads(num_threads);
    std::vector<std::future<void>> futures;
    std::vector<int> answers(num_tasks);
    for (size_t i = 0; i < num_tasks; ++i) {
      futures.emplace_back(pool.submit(
          [i, &answers]() {
            answers[i] = i;
          }));
    }
    for (auto &f : futures) {
      f.get();
    }
    for (size_t i = 0; i < answers.size(); ++i) {
      EXPECT_EQ(answers[i], i);
    }
  }

}  // namespace
