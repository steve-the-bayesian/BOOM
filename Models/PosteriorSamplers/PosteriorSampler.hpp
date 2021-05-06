// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005 Steven L. Scott

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

#ifndef BOOM_SAMPLING_METHOD_HPP
#define BOOM_SAMPLING_METHOD_HPP

#include "LinAlg/Vector.hpp"
#include "LinAlg/VectorView.hpp"
#include "cpputil/Ptr.hpp"
#include "cpputil/RefCounted.hpp"
#include "distributions/rng.hpp"


namespace BOOM {

  class Model;

  // The job of a PosteriorSampler is primarily to simulate a set of
  // model parameters from their posterior distribution.  Concrete
  // instances of a PosteriorSampler should contain a "dumb" pointer
  // to a specific concrete Model to be managed (because the model
  // owns the Sampler, not the other way around), as well as a Ptr to
  // one or more other model objects constituting the prior.
  //
  // Some PosteriorSamplers also allow you to find the posterior mode
  // of the model that they manage.  If so, they should override the
  // can_find_posterior_mode method to return true.
  class PosteriorSampler : private RefCounted {
   public:
    // Args:
    //   seeding_rng: The random number generator used to set the seed
    //     for the RNG owned by this sampler.
    explicit PosteriorSampler(RNG &seeding_rng);
    PosteriorSampler(const PosteriorSampler &);

    // The base implementation of this function throws an error using
    // 'report_error'.  Specific instances will implement an overload that does
    // the cloning.
    virtual PosteriorSampler *clone_to_new_host(Model *host) const;

    virtual void draw() = 0;
    virtual double logpri() const = 0;
    ~PosteriorSampler() override {}
    RNG &rng() const { return rng_; }
    void set_seed(unsigned long);

    // Returns true if the child class implements
    // find_posterior_mode().  Returns false otherwise.
    virtual bool can_find_posterior_mode() const { return false; }
    virtual bool can_evaluate_log_prior_density() const { return false; }
    virtual bool can_increment_log_prior_gradient() const { return false; }

    // The default implementations of the following three functions
    // throw an exception through report_error().
    virtual void find_posterior_mode(double epsilon = 1e-5);

    // Args:
    //   parameters: A vector of model parameters.  The parameter
    //     ordering is the same as the result of
    //     model->vectorize_parameters().
    // Returns:
    //   The value of the log prior_density at the specified model parameters.
    // NOTE:
    //   The default implementation throws an exception that this
    //   function is not implemented.  Each child class should
    //   override this function and can_evaluate_log_prior_density().
    virtual double log_prior_density(const ConstVectorView &parameters) const;

    // Args:
    //   parameters: A vector of model parameters.  The parameter
    //     ordering is the same as the result of
    //     model->vectorize_parameters().
    //   gradient: The elements of gradient will be incremented by the
    //     gradient of the log prior density at the specified
    //     parameters.
    // Returns:
    //   The value of the log prior_density at the specified model parameters.
    // NOTE:
    //   The default implementation throws an exception that this
    //   function is not implemented.  Each child class should
    //   override this function and
    //   can_increment_log_prior_gradient().
    virtual double increment_log_prior_gradient(
        const ConstVectorView &parameters, VectorView gradient) const;

    friend void intrusive_ptr_add_ref(PosteriorSampler *m);
    friend void intrusive_ptr_release(PosteriorSampler *m);

   private:
    mutable RNG rng_;
  };

}  // namespace BOOM
#endif  // BOOM_SAMPLING_METHOD_HPP
