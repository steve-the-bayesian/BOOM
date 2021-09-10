#ifndef BOOM_CPPUTIL_THREAD_TOOLS_HPP_
#define BOOM_CPPUTIL_THREAD_TOOLS_HPP_

// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2016 Steven L. Scott

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2.1 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA
*/

#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>

// The main object defined here is the ThreadWorkerPool.  Before defining that
// object, we must first define some building blocks.

namespace BOOM {

  // A std::vector<std::thread> that calls join() on all joinable
  // threads when it goes out of scope.
  class ThreadVector : public std::vector<std::thread> {
    typedef std::vector<std::thread> ParentType;

   public:
    ThreadVector() = default;
    ~ThreadVector() { join_threads(); }

    void clear() {
      join_threads();
      ParentType::clear();
    }

    void join_threads() {
      ParentType &threads(*this);
      for (size_t i = 0; i < threads.size(); ++i) {
        if (threads[i].joinable()) {
          threads[i].join();
        }
      }
    }
  };

  //======================================================================
  // A move-only function wrapper.
  class MoveOnlyTaskWrapper {
   public:
    // An empty task.
    MoveOnlyTaskWrapper() = default;

    // Construct a task from a function like object with a void(void)
    // signature.
    //
    // cppcheck-suppress noExplicitConstructor
    template <typename F> MoveOnlyTaskWrapper(F &&f)  // NOLINT
        : impl_(new ConcreteFunctor<F>(std::move(f))) {}

    // Move constructor
    MoveOnlyTaskWrapper(MoveOnlyTaskWrapper &&other)
        : impl_(std::move(other.impl_)) {}

    // Move-only assignment operator.
    MoveOnlyTaskWrapper &operator=(MoveOnlyTaskWrapper &&other) {
      impl_ = std::move(other.impl_);
      return *this;
    }

    // Delete copy constructors and the traditional assignment
    // operator.
    MoveOnlyTaskWrapper(const MoveOnlyTaskWrapper &rhs) = delete;
    MoveOnlyTaskWrapper(MoveOnlyTaskWrapper &rhs) = delete;
    MoveOnlyTaskWrapper &operator=(const MoveOnlyTaskWrapper &rhs) = delete;

    // Invoke the wrapped function.
    void operator()() { impl_->call(); }

   private:
    // A base class that can be stored in a pointer, supplying an
    // interface to call().
    struct FunctorInterface {
      virtual void call() = 0;
      virtual ~FunctorInterface() {}
    };

    // A concrete derived class that can hold functors of various
    // types.
    template <typename F>
    struct ConcreteFunctor : public FunctorInterface {
      F f;
      // cppcheck-suppress noExplicitConstructor
      ConcreteFunctor(F &&f_) : f(std::move(f_)) {}  // NOLINT
      void call() override { f(); }
    };

    // A pointer to store the functor.
    std::unique_ptr<FunctorInterface> impl_;
  };

  //======================================================================
  // A queue for passing objects between threads.  All operations are
  // thread safe.
  class ThreadSafeTaskQueue {
   public:
    // Pushes a task onto the queue.
    void push(MoveOnlyTaskWrapper &&task);

    // Try to pop the front of the queue into the first argument.
    // Args:
    //   task: If there is work to do in the queue, the waiting task
    //     is moved into the 'task' argument.
    //   timeout: The maximum amount of time a thread will spend
    //     waiting for new work before yielding to the next thread.
    //     If there is no work to do for a long time threads will keep
    //     yielding to one another.  This argument is here to keep
    //     threads from waiting forever, so that a global 'done' flag
    //     can be set, allowing threads to exit gracefully.
    //
    // Returns:
    //   The return value is true if a task was successfully placed in
    //   the first argument (i.e. work was found on the queue).  It is
    //   false if there was no available task in the alotted time.
    bool wait_and_pop(
        MoveOnlyTaskWrapper &task,
        std::chrono::milliseconds timeout = std::chrono::milliseconds(100));

    // Returns true if the queue is empty, false otherwise.
    bool empty() const;

   private:
    mutable std::mutex task_queue_mutex_;
    std::condition_variable new_work_;
    std::queue<MoveOnlyTaskWrapper> task_queue_;
  };

  //======================================================================
  // Manages a collection of threads and a ThreadSafeTaskQueue for
  // passing work to them.
  //
  // The idiom for using this is:
  //
  // ThreadWorkerPool pool;
  // pool.add_threads(10);  // consider std::hardware_concurrency()
  // std::vector<future<void>> futures;
  // for (int i = 0; i < 7; ++i) {
  //   futures.emplace_back(pool.submit([](){do_some_work();}));
  // }
  // for (int i = 0; i < futures.size(); ++i) {
  //   futures[i].get();
  // }
  //
  // Note that the call to futures[i].get() passes any exceptions
  // encountered by worker threads back to the calling thread.
  class ThreadWorkerPool {
   public:
    // Start a worker pool with the given number of threads.
    explicit ThreadWorkerPool(int number_of_threads = 0);

    // Shuts down waiting threads.
    ~ThreadWorkerPool();

    // Add the specified number of threads to the pool.
    void add_threads(int number_of_additional_threads);

    // Sets the number of threads in the pool to the given value.  If
    // the pool currently contains this many or more joinable threads
    // then nothing is done.  If (number_of_threads <= 0) then all
    // threads are joined and destroyed.  If more threads are needed
    // then they will be added.
    void set_number_of_threads(int number_of_threads);

    // Submit a job to the pool.
    // Args:
    //   task: A function-like object with signature void(void),
    //     representing an item of work to be done by a waiting
    //     thread.
    //
    // Returns:
    //   The return value is a future.  Calling wait() on the return
    //   value will pause the current thread until the job on the
    //   remote thread completes, or an exception is thrown.  If an
    //   exception is thrown by the remote thread then wait() passes
    //   it to the current thread.
    template <typename FunctionType>
    std::future<void> submit(FunctionType work) {
      std::packaged_task<void()> task(std::move(work));
      std::future<void> res(task.get_future());
      work_queue_.push(std::move(task));
      return res;
    }

    // Returns true() if there are currently no threads available to
    // do work.  Worker threads can be added by calling add_threads().
    bool no_threads() const { return threads_.empty(); }

    int number_of_threads() const { return threads_.size(); }

    int number_of_joinable_threads() const {
      int ans = 0;
      for (int i = 0; i < threads_.size(); ++i) {
        ans += threads_[i].joinable();
      }
      return ans;
    }

   private:
    // A flag indicating that worker threads should shut down.
    std::atomic_bool done_;

    // A queue for passing work to worker threads.
    ThreadSafeTaskQueue work_queue_;

    // The collection of worker threads.
    ThreadVector threads_;

    // A thread to run in the background.  Continually checks to see
    // if there is work in the queue.  If it finds a task then do it,
    // otherwise yield to the next thread.
    void worker_thread();
  };

}  // namespace BOOM

#endif  //  BOOM_CPPUTIL_THREAD_TOOLS_HPP_
