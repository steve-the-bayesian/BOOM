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

#include "cpputil/ThreadTools.hpp"

namespace BOOM {

  void ThreadSafeTaskQueue::push(MoveOnlyTaskWrapper &&task) {
    std::lock_guard<std::mutex> task_queue_lock(task_queue_mutex_);
    new_work_.notify_one();
    task_queue_.emplace(std::move(task));
  }

  bool ThreadSafeTaskQueue::wait_and_pop(MoveOnlyTaskWrapper &task,
                                         std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(task_queue_mutex_);
    new_work_.wait_for(lock, timeout,
                       [this]() { return !task_queue_.empty(); });
    if (!task_queue_.empty()) {
      task = std::move(task_queue_.front());
      task_queue_.pop();
      return true;
    } else {
      return false;
    }
  }

  bool ThreadSafeTaskQueue::empty() const {
    std::lock_guard<std::mutex> lock(task_queue_mutex_);
    return task_queue_.empty();
  }

  ThreadWorkerPool::ThreadWorkerPool(int number_of_threads) : done_(false) {
    if (number_of_threads > 0) {
      add_threads(number_of_threads);
    }
  }

  ThreadWorkerPool::~ThreadWorkerPool() { done_ = true; }

  void ThreadWorkerPool::add_threads(int number_of_threads) {
    try {
      for (int i = 0; i < number_of_threads; ++i) {
        threads_.push_back(std::thread(&ThreadWorkerPool::worker_thread, this));
      }
    } catch (...) {
      done_ = true;
      throw;
    }
  }

  void ThreadWorkerPool::set_number_of_threads(int n) {
    if (n <= 0) {
      done_ = true;
      threads_.clear();
      return;
    } else {
      int current_number_of_joinable_threads = 0;
      done_ = false;
      for (int i = 0; i < threads_.size(); ++i) {
        current_number_of_joinable_threads += threads_[i].joinable();
      }
      if (current_number_of_joinable_threads < n) {
        add_threads(n - current_number_of_joinable_threads);
      }
    }
  }

  void ThreadWorkerPool::worker_thread() {
    while (!done_) {
      MoveOnlyTaskWrapper task;
      if (work_queue_.wait_and_pop(task)) {
        task();
      } else {
        std::this_thread::yield();
      }
    }
  }

}  // namespace BOOM
