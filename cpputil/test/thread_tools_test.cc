#include "gtest/gtest.h"
#include "cpputil/ThreadTools.hpp"
#include "test_utils/test_utils.hpp"

namespace {
  using namespace BOOM;
  using std::endl;

  TEST(ThreadVectorTest, joins_threads) {
    ThreadVector v;
    for (int i = 0; i < 8; ++i) {
      v.push_back(std::thread());
    }
  }

  TEST(MoveOnlyTaskWrapperTest, blah) {
  }

  TEST(ThreadSafeTaskQueueTest, blah) {
    ThreadSafeTaskQueue q;
    EXPECT_TRUE(q.empty());

    // Check that a task can be added.
    q.push(MoveOnlyTaskWrapper([]() {std::cout << "Hello, ";}));
    q.push(MoveOnlyTaskWrapper([]() {std::cout << "World!\n";}));
    EXPECT_FALSE(q.empty());

    // Check that a task can be popped.
    MoveOnlyTaskWrapper hello;
    MoveOnlyTaskWrapper world;
    EXPECT_TRUE(q.wait_and_pop(hello));
    EXPECT_TRUE(q.wait_and_pop(world));
    hello();
    world();

    EXPECT_TRUE(q.empty());
  }

  TEST(ThreadWorkerPoolTest, blah) {
    ThreadWorkerPool pool;
    EXPECT_TRUE(pool.no_threads());
    EXPECT_EQ(0, pool.number_of_threads());
    EXPECT_EQ(0, pool.number_of_joinable_threads());

    pool.add_threads(1);
    EXPECT_FALSE(pool.no_threads());
  }

  TEST(threading, record_integers) {
    ThreadWorkerPool pool;
    // int num_threads = 3;
    // int num_tasks = 7;
    // pool.add_threads(num_threads);
    // std::vector<std::future<void>> futures;
    // futures.reserve(num_tasks);

    // std::vector<int> answers(num_tasks);
    // for (size_t i = 0; i < num_tasks; ++i) {
    //   futures.emplace_back(pool.submit(
    //       [i, &answers]() {
    //         answers[i] = i;
    //       }));
    // }
    // for (auto &f : futures) {
    //   f.get();
    // }
    // for (size_t i = 0; i < answers.size(); ++i) {
    //   EXPECT_EQ(answers[i], i);
    // }
  }

}  // namespace
