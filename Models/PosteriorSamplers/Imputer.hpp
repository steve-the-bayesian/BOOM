// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2014-2017 Steven L. Scott

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

#ifndef BOOM_LATENT_DATA_IMPUTER_HPP
#define BOOM_LATENT_DATA_IMPUTER_HPP

#include <cstddef>
#include <future>
#include <memory>

#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "cpputil/report_error.hpp"

#include "cpputil/RefCounted.hpp"
#include "cpputil/ThreadTools.hpp"

// The main class implemented in this file is ParallelLatentDataImputer.
//
// To use the imputer, a concrete instance of a LatentDataImputerWorker must
// first be defined.  The imputer work in conjunction with a mutex that guards
// the "global" complete data repository (e.g. complete data sufficient
// statistics, or complete data model).
//
// The idiom is...
//
// class MyConcreteImputeWorker : public LatentDataImputeWorker {};
// class MyPosteriorSampler : public LatentDataSampler {};
// MyPosteriorSampler sampler;
// sampler.set_number_of_workers(12);
//
// In the common case where the latent data is stored in complete data
// sufficient statistics, then MyConcreteImputeWorker can inherit from
// SufstatImputeWorker instead of LatentDataImputeWorker.
namespace BOOM {

  // A base class implementing the interface needed by imputation workers who
  // want to participate in a ParallelLatentDataImputer worker pool.
  //
  // Child classes must implement impute_latent_data() and
  // combine_complete_data().
  class LatentDataImputerWorker : private RefCounted {
   public:
    // Args:
    //   shared_resource_mutex: The mutex guarding the shared data repository.
    //     To avoid race conditions, this class should be the only one capable
    //     of locking the mutex.  Child classes should not lock this mutex.
    explicit LatentDataImputerWorker(std::mutex &shared_resource_mutex)
        : shared_resource_mutex_(shared_resource_mutex) {}

    // Clear the local repository holding the complete data, and refill it with
    // newly imputed values.  Overrides of this function should not write to
    // resources shared by other threads.
    virtual void impute_latent_data() = 0;

    // To be called when impute_latent_data is finished.  Combine the completed
    // data owned by this object with the shared set of complete data.  The code
    // implementing this function need not be thread safe.  When called through
    // data_imputation_callback() the data combination step will be
    // protected shared_resource_mutex_.
    virtual void combine_complete_data() = 0;

    // Return the number of observations managed by the worker.  That is, the
    // number of observed data points for which latent data is to be imputed.
    // This is a useful quantity to check for a signal that the observed data
    // may have changed.
    virtual int number_of_observations_managed() const = 0;

    // Wraps this object in a callback which imputes the latent data and stores
    // it in a local repository.  The local repository is then combined with the
    // global repository in a thread-safe way.
    std::function<void(void)> data_imputation_callback() {
      return [this]() {
        this->impute_latent_data();
        std::unique_lock<std::mutex> lock(shared_resource_mutex_);
        this->combine_complete_data();
      };
    }

   private:
    std::mutex &shared_resource_mutex_;

    // A LatentDataImputerWorker can be held by a Ptr.
    friend void intrusive_ptr_add_ref(LatentDataImputerWorker *w) {
      w->up_count();
    }
    friend void intrusive_ptr_release(LatentDataImputerWorker *w) {
      w->down_count();
      if (w->ref_count() == 0) delete w;
    }
  };

  //======================================================================
  // A concrete instantiation of LatentDataImputerWorker, where the complete
  // data are stored in a Sufstat object.  The latent data are to be imputed
  // point-by-point using the pure virtual function impute_latent_data_point.
  //
  // Template Types:
  //   OBSERVED_DATA:  The type of the observed data held by the model.
  //   SUFFICIENT_STATISTICS: The type of the object used to hold the
  //     complete data sufficient statistics.
  template <class OBSERVED_DATA, class SUFFICIENT_STATISTICS>
  class SufstatImputeWorker : public LatentDataImputerWorker {
   public:
    typedef typename std::vector<Ptr<OBSERVED_DATA>>::const_iterator Iterator;

    // Build a SufstatImputeWorker that uses the given set of sufficient
    // statistics and the specified imputation generator.
    // Args:
    //   global_suf: A reference to the global object representing the complete
    //     data sufficient statistics.  It must have methods combine(), clear(),
    //     and clone(), and be storable in a Ptr.
    //   global_suf_mutex:  A mutex protecting the global_suf object.
    //   rng: A random number generator, or nullptr.  If a random number
    //     generator is provided it will be used as the source of randomness.
    //     Otherwise a new RNG will be created.
    //   seeding_rng: If a new random number generator must be created, then
    //     this RNG will be used to seed it with an initial value.
    SufstatImputeWorker(SUFFICIENT_STATISTICS &global_suf,
                        std::mutex &global_suf_mutex, RNG *rng = nullptr,
                        RNG &seeding_rng = GlobalRng::rng)
        : LatentDataImputerWorker(global_suf_mutex),
          suf_(global_suf.clone()),
          global_suf_(global_suf) {
      if (!rng) {
        rng_storage_.reset(new RNG(seed_rng(seeding_rng)));
        rng_ = rng_storage_.get();
      } else {
        rng_ = rng;
      }
      std::vector<Ptr<OBSERVED_DATA>> empty_vector;
      observed_data_begin_ = empty_vector.end();
      observed_data_end_ = empty_vector.end();
    }

    // Assign this object a range of data over which to impute.
    void set_data(Iterator begin, Iterator end) {
      observed_data_begin_ = begin;
      observed_data_end_ = end;
    }

    // The number of data points managed by this worker.
    int number_of_observations_managed() const override {
      return std::distance(observed_data_begin_, observed_data_end_);
    }

    // Args:
    //   data_point: An individual piece of observed data to be augmented with
    //     missing data.
    //   suf: The local sufficient statistics that store the augmented data.
    //   rng: The random number generator used to impute the missing data.
    //
    // Details:
    //   Imputes the missing data corresponding to data_point, and adds the
    //   augmented data to suf.
    virtual void impute_latent_data_point(const OBSERVED_DATA &data_point,
                                          SUFFICIENT_STATISTICS *suf,
                                          RNG &rng) = 0;

    void impute_latent_data() override {
      suf_->clear();
      for (Iterator it = observed_data_begin_; it != observed_data_end_; ++it) {
        impute_latent_data_point(**it, suf_.get(), *rng_);
      }
    };

    void combine_complete_data() override { global_suf_.combine(*suf_); }

   private:
    Ptr<SUFFICIENT_STATISTICS> suf_;
    SUFFICIENT_STATISTICS &global_suf_;
    Iterator observed_data_begin_;
    Iterator observed_data_end_;
    RNG *rng_;
    std::unique_ptr<RNG> rng_storage_;
  };

  //======================================================================
  // An object that manages the thread pool and the vector of workers
  // responsible for imputing latent data.  Clients will typically not deal with
  // this class directly.  It is part of the implementation for
  // LatentDataSampler.
  class ParallelLatentDataImputer {
   public:
    ParallelLatentDataImputer() {}

    // Set the number of background threads to use for data augmentation.  If n
    // <= 0 then no background threads are created.
    void set_number_of_threads(int n) { pool_.set_number_of_threads(n); }

    // Add a worker.  The number of workers need not be the same as the number
    // of threads.
    void add_worker(const Ptr<LatentDataImputerWorker> &worker) {
      workers_.push_back(worker);
    }

    // Removes all elements from the collection of workers.
    void clear_workers() { workers_.clear(); }

    // The total number of data points seen by all the workers.
    int number_of_observations_managed() const {
      int ans = 0;
      for (int i = 0; i < workers_.size(); ++i) {
        ans += workers_[i]->number_of_observations_managed();
      }
      return ans;
    }

    // Impute the latent data.  If the pool contains any threads then the work
    // is done by the worker pool.  Otherwise it is done by the workers,
    // sequentially.
    void impute_latent_data();

   private:
    ThreadWorkerPool pool_;
    std::vector<Ptr<LatentDataImputerWorker>> workers_;
  };

  //======================================================================
  // A mix-in class for PosteriorSampler classes that want to sample latent data
  // in parallel.  This class owns and manages the latent data imputer object
  // and the mutex for the protecting the complete data.
  //
  // Template arguments:
  //   WORKER:  A class inheriting from LatentDataImputerWorker.
  template <class WORKER>
  class LatentDataSampler {
   public:
    LatentDataSampler()
        : latent_data_fixed_(false), reassign_data_each_time_(false) {}

    // Create a new worker, which has access to the global repository of
    // complete data, protected by mutex.
    virtual Ptr<WORKER> create_worker(std::mutex &mutex) = 0;

    // Allocate the vector of observed data among the workers.
    virtual void assign_data_to_workers() = 0;

    // Empty the complete data sufficient statistics or other object
    // being used to hold the complete data.
    virtual void clear_latent_data() = 0;

    // Use up to 'n' logical threads to simulate the latent data.  If
    // n <= 1 then run single threaded.
    virtual void set_number_of_workers(int n) {
      if (n < 1) {
        n = 1;
      }
      imputer_.clear_workers();
      workers_.clear();
      for (int i = 0; i < n; ++i) {
        Ptr<WORKER> worker = create_worker(global_complete_data_mutex_);
        imputer_.add_worker(worker);
        workers_.push_back(worker);
      }
      imputer_.set_number_of_threads(n == 1 ? 0 : n);
      assign_data_to_workers();
    }

    // By default, this class updates its own latent data through a call to
    // impute_latent_data().  Calling this function with a 'true' argument (the
    // default), sets a flag that turns impute_latent_data into a no-op.  The
    // latent data can still be manipulated through calls to
    // clear_sufficient_statistics() and update_sufficient_statistics(), but
    // implicit data augmentation is turned off.  Calling this function with a
    // 'false' argument turns data augmentation back on.
    void fix_latent_data(bool fixed = true) { latent_data_fixed_ = fixed; }

    // In the typical use case, a model has a set of data that does not change.
    // When set_number_of_workers() is called, subsets of that data are assigned
    // to workers, and that assignment will not change.  In special cases where
    // the model's data changes from one iteration to the next (meaning that the
    // model has different objects in its data set, same objects with different
    // values are okay) call reassign_data_each_time() so that the workers
    // continue to have access to valid data.  This might be necessary if the
    // model is used as a mixture component in a finite mixture model, for
    // example.
    //
    // Args:
    //   reassign: If true then call assign_data_to_workers each time
    //     impute_latent_data() is called.
    void reassign_data_each_time(bool reassign = true) {
      reassign_data_each_time_ = reassign;
    }

    // This is the main method of this class.  Workers, which may be running in
    // background threads, each sample the latent data for their chunk of the
    // data set.  Once each worker's draw is complete it is combined with the
    // global complete data set.
    virtual void impute_latent_data() {
      if (!latent_data_fixed_) {
        clear_latent_data();
        if (reassign_data_each_time_ ||
            imputer_.number_of_observations_managed() == 0) {
          assign_data_to_workers();
        }
        imputer_.impute_latent_data();
      }
    }

   protected:
    std::vector<Ptr<WORKER>> &workers() { return workers_; }

   private:
    // If this flag is set then latent data will not be changed from its current
    // values.
    bool latent_data_fixed_;

    // In the typical case the model that owns the data has a fixed set of data
    // and it can be assigned to workers in the worker pool one time.  However,
    // if the model is being used as a mixture component in a mixture model then
    // it will have different observations assigned to it each iteration.  In
    // that case (or in any other case where the data is expected to change
    // across MCMC iterations), this flag should be set so that workers are
    // drawing from the correct data set.
    bool reassign_data_each_time_;

    // A mutex protecting the global sufficient statistics held by a child
    // object.
    std::mutex global_complete_data_mutex_;

    // This vector is kept in parallel to the vector of workers in imputer_ so
    // that data can be assigned to workers after construction.
    std::vector<Ptr<WORKER>> workers_;

    // The latent data imputer does the actual drawing.
    ParallelLatentDataImputer imputer_;
  };

  //======================================================================
  // A default implementation of the the "assign_data_to_workers" member
  // function declared in LatentDataSampler.
  template <class OBSERVED_DATA, class WORKER>
  void assign_data_to_workers(const std::vector<Ptr<OBSERVED_DATA>> &data,
                              std::vector<Ptr<WORKER>> &workers) {
    size_t number_of_workers = workers.size();
    if (number_of_workers == 0) return;
    size_t nobs = data.size();
    if (nobs == 0) return;
    size_t chunk_size = nobs / number_of_workers;
    typedef typename std::vector<Ptr<OBSERVED_DATA>>::const_iterator Iterator;
    Iterator it = data.begin();
    Iterator end = data.end();
    if (chunk_size == 0) {
      for (int i = 0; i < nobs; ++i) {
        workers[i]->set_data(it, it + 1);
        ++it;
      }
      for (int i = nobs; i < number_of_workers; ++i) {
        workers[i]->set_data(end, end);
      }
    } else {
      for (int i = 0; i < number_of_workers; ++i) {
        Iterator e = it + chunk_size;
        if (e > end || (i + 1) == number_of_workers) e = end;
        workers[i]->set_data(it, e);
        it = e;
      }
    }
  }

}  // namespace BOOM

#endif  // BOOM_LATENT_DATA_IMPUTER_HPP
