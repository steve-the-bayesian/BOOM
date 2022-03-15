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
#ifndef BOOM_MODEL_TYPES_HPP
#define BOOM_MODEL_TYPES_HPP

#include <set>

#include "uint.hpp"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/Vector.hpp"
#include "Models/DataTypes.hpp"
#include "Models/ParamTypes.hpp"
#include "distributions/rng.hpp"

namespace BOOM {

  class PosteriorSampler;

  // A Model is the basic unit of operation in statistical learning.
  // In BOOM, each Model manages Params, Data, and learning methods.
  // a Model should also inherit from a ParamPolicy, a DataPolicy, and
  // a PriorPolicy.
  //
  // If the recommended set of policies are used, to inherit from
  // Model a class must provide:
  //   * clone()   (covariant return)
  //
  // Any class inheriting from model should do so virtually, because
  // Model contains a reference count that should not be duplicated.
  class Model : private RefCounted {
   public:
    friend void intrusive_ptr_add_ref(Model *d) { d->up_count(); }
    friend void intrusive_ptr_release(Model *d) {
      d->down_count();
      if (d->ref_count() == 0) delete d;
    }

    //------ constructors, destructors, operator=/== -----------
    Model();

    Model(const Model &rhs);  // ref count is not copied

    // The result of clone() (and copies generally) should have
    // identical parameters in distinct memory.  It should not have
    // any data assigned.  Nor should it include the same priors and
    // sampling methods.
    virtual Model *clone() const = 0;

    virtual ~Model() {}

    //----------- parameter interface  ---------------------
    // implemented in ParmPolicy
    virtual std::vector<Ptr<Params>> parameter_vector() = 0;
    virtual const std::vector<Ptr<Params>> parameter_vector() const = 0;

    virtual Vector vectorize_params(bool minimal = true) const;
    virtual void unvectorize_params(const Vector &v, bool minimal = true);

    //------------ functions implemented in DataPolicy -----

    // add_data adds 'dp' to the set of Data objects managed by the
    // model.  It is assumed that dp points to a Data object of the
    // type produced by the concrete model.  The Data type is made
    // concrete in the model's DataPolicy.
    virtual void add_data(const Ptr<Data> &dp) = 0;

    // Discard all the data that has been added using add_data().
    virtual void clear_data() = 0;

    // Combine the data managed by other_model with the data managed
    // by *this.  If just_suf is true and the model has sufficient
    // statistics, the actual data from 'other_model' is not copied.
    virtual void combine_data(const Model &other_model,
                              bool just_suf = true) = 0;

    //------------ functions over-ridden in PriorPolicy ----
    virtual void sample_posterior() = 0;
    virtual double logpri() const = 0;  // evaluates current params
    //    virtual void set_method(const Ptr<PosteriorSampler> &) = 0;
    virtual int number_of_sampling_methods() const = 0;
    virtual void clear_methods() {}
    virtual void set_method(const Ptr<PosteriorSampler> &sampler) {}
    virtual PosteriorSampler *sampler(int i) = 0;
    virtual PosteriorSampler const *const sampler(int i) const = 0;
  };

  // The result of deepclone will have identical parameters in distinct
  // memory.  It will contain pointers to the same data.  It will contain
  // equivalent posterior samplers (but pointing to the new data structures).
  //
  // A collection of 'deepclone' models is suitable for a thread worker pool.
  template <class MODEL>
  Ptr<MODEL> deepclone(const MODEL &model) {
    Ptr<MODEL> ans = model.clone();
    for (int s = 0; s < model.number_of_sampling_methods(); ++s) {
      ans->set_method(model.sampler(s)->clone_to_new_host(ans.get()));
    }
    return ans;
  }

  //============= mix-in classes =========================================

  class IntModel : virtual public Model {
   public:
    // Evaluate the log probability mass function at x.  Return
    // negative_infinity() For values of x outside the support of the
    // model.
    virtual double logp(int x) const = 0;
  };

  //======================================================================
  // A MLE_Model has parameters that can be estimated by maximum likelihood.
  class MLE_Model : virtual public Model {
   public:
    MLE_Model() : status_(NOT_CALLED) {}
    MLE_Model(const MLE_Model &rhs) = default;
    MLE_Model(MLE_Model &&rhs);
    MLE_Model & operator=(const MLE_Model &rhs) = default;
    MLE_Model & operator=(MLE_Model &&rhs);

    // Set the paramters to their maximum likelihood estimates.
    virtual void mle() = 0;

    // Initialize an MCMC run by setting parameters to their maximum likelihood
    // estimates.
    virtual void initialize_params();

    enum MleStatus { NOT_CALLED = -1, FAILURE = 0, SUCCESS = 1 };
    MleStatus mle_status() const { return status_; }
    const std::string &mle_error_message() const { return error_message_; }
    bool mle_success() const { return status_ == SUCCESS; }

   private:
    MleStatus status_;
    std::string error_message_;

   protected:
    void set_status(MleStatus status, const std::string &error_message) {
      status_ = status;
      error_message_ = error_message;
    }
  };
  //======================================================================
  // A PosteriorModeModel has parameters that can be estimated by their
  // posterior mode.
  class PosteriorModeModel : virtual public Model {
   public:
    // Set parameters to their MAP (posterior mode maximizing) values.
    // In most cases, the work for this call will have to be delegated
    // to a posterior sampler that knows how to find the posterior
    // mode.  The default implementation for this method is
    // 1) check that a posterior sampler has been set.  Throw if not.
    // 2) check that the posterior sampler implements
    //    find_posterior_mode.  Throw if not.
    // 3) call the posterior sampler's find_posterior_mode method.
    //
    // Args:
    //   epsilon: If the mode finding algorithm is iterative, use
    //     epsilon as its convergence criterion.
    virtual void find_posterior_mode(double epsilon = 1e-5);

    // A model can only find a posterior mode if it has been assigned
    // a PosteriorSampler that can help out.  This function returns
    // 'true' if an appropriate sampler has been assigned, and 'false'
    // otherwise.
    bool can_find_posterior_mode() const;

    // Pass the parameters on to the PosteriorSampler with a request
    // to evaluate the log prior density.  This can result in an
    // exception being thrown if no sampler was assigned (or if
    // multiple samplers were assigned), or if the assigned sampler
    // can't do the evaluation.  To check that the operation is safe,
    // check can_evaluate_log_prior_density() before calling this
    // function.
    double log_prior_density(const ConstVectorView &parameters) const;

    // Returns 'true' if the model was assigned a single
    // PosteriorSampler that is capable of evaluating
    // log_prior_density.  Returns 'false' otherwise.
    bool can_evaluate_log_prior_density() const;

    // Pass the parameters on to the PosteriorSampler with a request
    // to evaluate the gradient of the log prior density.  This can
    // result in an exception being thrown if no sampler was assigned
    // (or if multiple samplers were assigned), or if the assigned
    // sampler can't do the evaluation.  To check that the operation
    // is safe, check can_increment_log_prior_gradient() before
    // calling this function.
    //
    // Args:
    //   parameters: The model parameters where the log prior density
    //     is to be evaluated.
    //   gradient: The gradient of log_prior_density will be added to
    //     the elements already in gradient.  This is to facilitate
    //     computations like log_posterior = log_prior +
    //     log_likelihood, where the likelihood gradient may already
    //     have been computed.
    //
    // Returns:
    //   The value of log prior density at parameters.
    double increment_log_prior_gradient(const ConstVectorView &parameters,
                                        VectorView gradient) const;

    // Returns 'true' if the model has been assigned a single
    // PosteriorSampler capable of computing the gradient of the log
    // prior density.  Returns false otherwise.
    bool can_increment_log_prior_gradient() const;
  };
  //======================================================================
  class LoglikeModel : public MLE_Model {
   public:
    // Evaluate log likelihood at the given parameter vector.
    virtual double loglike(const Vector &theta) const = 0;

    // Evaluate log likelihood with the current set of model parameters.
    virtual double log_likelihood() const {
      return loglike(vectorize_params(true));
    }

    // Set model parameters to their maximum likelihood estimates.
    void mle() override;
  };

  class dLoglikeModel : public LoglikeModel {
   public:
    virtual double dloglike(const Vector &x, Vector &g) const = 0;
    void mle() override;
  };

  class d2LoglikeModel : public dLoglikeModel {
   public:
    virtual double d2loglike(const Vector &x, Vector &g, Matrix &H) const = 0;
    void mle() override;
    virtual double mle_result(Vector &gradient, Matrix &hessian);
  };

  class NumOptModel : public d2LoglikeModel {
   public:
    virtual double Loglike(const Vector &x, Vector &g, Matrix &H,
                           uint nd) const = 0;
    double loglike(const Vector &x) const override {
      Vector g;
      Matrix h;
      return Loglike(x, g, h, 0);
    }
    double dloglike(const Vector &x, Vector &g) const override {
      Matrix h;
      return Loglike(x, g, h, 1);
    }
    double d2loglike(const Vector &x, Vector &g, Matrix &h) const override {
      return Loglike(x, g, h, 2);
    }
  };
  //======================================================================
  class LatentVariableModel : virtual public Model {
   public:
    virtual void impute_latent_data(RNG &rng) = 0;
  };
  //======================================================================
  class CorrelationModel : virtual public Model {
   public:
    virtual double logp(const CorrelationMatrix &) const = 0;
    CorrelationModel *clone() const override = 0;
  };
  //======================================================================
  class MixtureComponent : virtual public Model {
   public:
    MixtureComponent() : component_(-1) {}

    virtual double pdf(const Data *, bool logscale) const = 0;

    // The number of data points that have been allocated to this model.  This
    // might have been called "sample_size", but that sometimes refers to
    // certain model parameters, such as the beta distribution.
    virtual int number_of_observations() const = 0;

    MixtureComponent *clone() const override = 0;

    //----------------------------------------------------------------------
    // Tools for keeping track of a mixture component's position in a finite
    // mixture model or the like.
    int mixture_component_index() const { return component_; }

    // Values less than zero indicate that a component is purposefully unused.
    void set_mixture_component_index(int i) { component_ = i; }

    void decrement_mixture_component_index() { --component_; }

    void increment_mixture_component_index() { ++component_; }

   private:
    int component_;
  };

  //======================================================================
  class ConjugateModel : virtual public Model {
   public:
    virtual ConjugateModel *clone() const override = 0;
  };
  //======================================================================

  // A mixture component that can also call remove_data, which is an important
  // ability for components assigned to a Dirichlet process.
  //
  // TODO: Remove this class once enough MixtureComponent child classes have had
  // this ability added,
  class DirichletProcessMixtureComponent : virtual public MixtureComponent {
   public:
    DirichletProcessMixtureComponent *clone() const override = 0;

    // Returns the vector of data stored by the model.
    virtual std::set<Ptr<Data>> abstract_data_set() const = 0;

    virtual void remove_data(const Ptr<Data> &dp) = 0;

    virtual double log_likelihood() const = 0;
  };

  //======================================================================
  class ConjugateDirichletProcessMixtureComponent
      : virtual public ConjugateModel,
        virtual public DirichletProcessMixtureComponent {
   public:
    ConjugateDirichletProcessMixtureComponent *clone() const override = 0;
  };

  //===========================================================================
  // Some complex computations for certain models (e.g. log likelihood and
  // derivatives) can really only be done using the current set of model
  // parameters.  To enable likelihood evaluation at parameter other than their
  // current values, one idiom is to replace the parameters, evalute the
  // function, and restore the old set of parameters when the computation
  // concludes.
  //
  // This class holds the "old" set of model parameters while a computation
  // takes place, and restores them when it goes out of scope.
  //
  // Mutable values held by the model (e.g. the kalman filter for state space
  // models, or the log likelihood for HMM's and mixture models) will no longer
  // be current after the swap is undone.  It is the model's responsibility to
  // keep track of these quantities.
  class ParameterHolder {
   public:
    ParameterHolder(Model *model, const Vector &parameters)
        : original_parameters_(model->vectorize_params()), model_(model) {
      model_->unvectorize_params(parameters);
    }

    ~ParameterHolder() { model_->unvectorize_params(original_parameters_); }

   private:
    Vector original_parameters_;
    Model *model_;
  };




}  // namespace BOOM
#endif  // BOOM_MODEL_TYPES_HPP
