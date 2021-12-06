// Copyright 2018 Google Inc. All Rights Reserved.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA

#ifndef BSTS_SRC_MODEL_MANAGER_H_
#define BSTS_SRC_MODEL_MANAGER_H_

#include "r_interface/boom_r_tools.hpp"
#include "r_interface/list_io.hpp"
#include "Models/StateSpace/StateSpaceModelBase.hpp"
#include "Models/StateSpace/Multivariate/MultivariateStateSpaceModelBase.hpp"

#include "timestamp_info.h"

namespace BOOM {
  namespace bsts {

    class ScalarModelManager;

    //===========================================================================
    // The code that computes out of sample one-step prediction errors is
    // designed for multi-threading.  This base class provides the interface for
    // computing the prediction errors.
    class HoldoutErrorSamplerImpl {
     public:
      virtual ~HoldoutErrorSamplerImpl() {}

      // Simulate from the distribution of one-step prediction errors given data
      // up to some cutpoint.  Child classes must be equipped with enough state
      // to carry out this operation and store the results in an appropriate
      // data structure.
      virtual void sample_holdout_prediction_errors() = 0;
    };

    // A null class that can be used by model families that do not support one
    // step prediction errors (e.g. logit and Poisson).
    class NullErrorSampler : public HoldoutErrorSamplerImpl {
     public:
      void sample_holdout_prediction_errors() override {}
    };

    // A pimpl-based functor for computing out of sample prediction errors, with
    // the appropriate interface for submitting to a ThreadPool.
    class HoldoutErrorSampler {
     public:
      explicit HoldoutErrorSampler(HoldoutErrorSamplerImpl *impl)
          : impl_(impl) {}
      void operator()() {impl_->sample_holdout_prediction_errors();}

     private:
      std::unique_ptr<HoldoutErrorSamplerImpl> impl_;
    };

    //===========================================================================
    // The job of a ModelManager is to construct the BOOM models that bsts uses
    // for inference, and to provide any additional data that the models need
    // for tasks other than statistical inference.  In this way a ModelManager
    // is much like a factory, but the ModelManager is expected to persist and
    // own global state, whereas a factory traditionally does not.
    //
    // The point of the ModelManager is to be an intermediary between the calls
    // in bsts.cc and the underlying BOOM code, so that R can pass lists of
    // data, priors, and model options formatted as expected by the child
    // ModelManager classes for specific model types.
    //
    // The class hierarchy for Model manager splits first on the type of model
    // being created (e.g. Scalar vs Multivariate), and then on the error
    // distribution within that model family.
    class ModelManager {
     public:
      ModelManager();

      virtual ~ModelManager() {}

      // Time stamps are considered trivial if either (a) no time stamp
      // information was provided by the user, or (b) each time stamp contains
      // one observation and there are no gaps in the  observation series.
      bool TimestampsAreTrivial() const {
        return timestamp_info_.trivial();
      }

      // Because of missing data, or multiplexed observations, the number of
      // time points might be different than the sample size.
      int NumberOfTimePoints() const {
        return timestamp_info_.number_of_time_points();
      }

      // Returns the timestamp number (index) of observation i.  The index is
      // given in C's 0-based counting system.
      int TimestampMapping(int observation_number) const {
        return timestamp_info_.mapping(observation_number);
      }

      RNG & rng() {return rng_;}

      const std::vector<int> &ForecastTimestamps() {
        return timestamp_info_.forecast_timestamps();
      }

     protected:
      // Checks to see if r_data_list has a field named timestamp.info, and use
      // it to populate the follwoing fields: number_of_time_points_,
      // timestamps_are_trivial_, and timestamp_mapping_.
      void UnpackTimestampInfo(SEXP r_data_list) {
        timestamp_info_.Unpack(r_data_list);
      }

      // Checks to see if r_prediction_data (which is an R list) contains an
      // element with the name 'timestamps', which is a vector of integers
      // giving the number of observations after the end of the training data
      // for each data point where a prediction is desired.
      //
      // Timestamps must be non-decreasing, and their length must be the same
      // dimension as the number of rows in the covariate matrix used for
      // predictions.
      void UnpackForecastTimestamps(SEXP r_prediction_data) {
        timestamp_info_.UnpackForecastTimestamps(r_prediction_data);
      }

      // Add data to the model object managed by the child classes.  The data
      // can come either from a previous bsts object, or from an R list
      // containing appropriately formatted data.
      virtual void AddDataFromBstsObject(SEXP r_bsts_object) = 0;
      virtual void AddDataFromList(SEXP r_data_list) = 0;

      // Allocates and fills the appropriate data structures needed for
      // forecasting, held by the child classes.
      // Args:
      //    r_prediction_data: An R list containing data needed for prediction.
      //
      // Returns:
      //    The number of periods to be forecast.
      virtual int UnpackForecastData(SEXP r_prediction_data) = 0;

      Vector &final_state() {return final_state_;}

     private:
      RNG rng_;
      Vector final_state_;

      TimestampInfo timestamp_info_;
    };

    //===========================================================================
    class ScalarModelManager : public ModelManager {
     public:
      // Create a ModelManager instance suitable for working with the specified
      // family.
      // Args:
      //   family: A text string identifying the model family.  "gaussian",
      //     "logit", "poisson", or "student".
      //   xdim: Dimension of the predictors in the observation model
      //     regression.  This can be zero if there are no regressors.
      static ScalarModelManager * Create(
          const std::string &family_name, int xdim);

      // Create a model manager by reinstantiating a previously constructed bsts
      // model.
      // Args:
      //   r_bsts_object:  An object previously created by a call to bsts.
      static ScalarModelManager * Create(SEXP r_bsts_object);

      // Creates a BOOM state space model suitable for learning with MCMC.
      // Args:
      //   r_data_list: An R list containing the data to be modeled in the
      //     format expected by the requested model family.  This list generally
      //     contains an object called 'response' and a logical vector named
      //     'response.is.observed'.  If the model is a (generalized) regression
      //     model then it will contain a 'predictors' object as well, otherwise
      //     'predictors' will be NULL.  For logit or Poisson models an
      //     additional component should be present giving the number of trials
      //     or the exposure.
      //   r_state_specification: The R list created by the state configuration
      //     functions (e.g. AddLocalLinearTrend, AddSeasonal, etc).
      //   r_prior: The prior distribution for the observation model.  If the
      //     model is a regression model (determined by whether r_data_list
      //     contains a non-NULL 'predictors' element) then this must be some
      //     form of spike and slab prior.  Otherwise it is a prior for the
      //     error in the observation equation.  For single parameter error
      //     distributions like binomial or Poisson this can be NULL.
      //   r_options: Model or family specific options such as the technique to
      //     use for model averaging (ODA vs SVSS).
      //   io_manager: The io_manager responsible for writing MCMC output to an
      //     R object, or streaming it from an existing object.
      //
      // Returns:
      //  A pointer to the created model.  The pointer is owned by a Ptr
      //  in the model manager, and should be caught by a Ptr in the caller.
      //
      // Side Effects:
      //   The returned pointer is also held in a smart pointer owned by
      //   the child class.
      virtual ScalarStateSpaceModelBase * CreateModel(
          SEXP r_data_list,
          SEXP r_state_specification,
          SEXP r_prior,
          SEXP r_options,
          RListIoManager *io_manager);

      // Returns a HoldoutErrorSampler that holds a family specific
      // implementation pointer that samples one-step prediction errors for data
      // in r_bsts_object beyond observation number 'cutpoint'.  This object can
      // be submitted to a ThreadPool for parallel processing.
      //
      // Args:
      //   r_bsts_object: A bsts object fit to full data, for which one-step
      //     prediction errors are desired.
      //   cutpoint: An integer giving the index of the last data point in
      //     r_bsts_object to be considered training data.  Observations after
      //     'cutpoint' are considered holdout data.
      //   standardize: Logical indicating whether the prediction errors should
      //     be standardized by dividing by the square root of the one step
      //     ahead forecast variance.
      //   prediction_error_output: A reference to a Matrix, with rows
      //     corresponding to MCMC iterations, and columns to observations in
      //     the holdout data set.  The matrix will be resized to appropriate
      //     dimensions by this call.
      //
      // Note that one step prediction errors are only supported for Gaussian
      // models.
      virtual HoldoutErrorSampler CreateHoldoutSampler(
          SEXP r_bsts_object, int cutpoint, bool standardize,
          BOOM::Matrix *prediction_error_output) = 0;

      // Returns a set of draws from the posterior predictive distribution.
      // Args:
      //   r_bsts_object:  The R object created from a previous call to bsts().
      //   r_prediction_data: Data needed to make the prediction.  This might be
      //     a data frame for models that have a regression component, or a
      //     vector of exposures or trials for binomial or Poisson data.
      //   r_options: If any special options need to be passed in order to do
      //     the prediction, they should be included here.
      //   r_observed_data: In most cases, the prediction takes place starting
      //     with the time period immediately following the last observation in
      //     the training data.  If so then r_observed_data should be
      //     R_NilValue, and the observed data will be taken from r_bsts_object.
      //     However, if more data have been added (or if some data should be
      //     omitted) from the training data, a new set of training data can be
      //     passed here.
      //
      // Returns:
      //   An R matrix, with rows corresponding to MCMC draws and columns to
      //   time, containing posterior predictive draws for the forecast.
      virtual Matrix Forecast(SEXP r_bsts_object, SEXP r_prediction_data,
                              SEXP r_burn, SEXP r_observed_data);

     private:
      // If the model contains a dynamic regression component then unpack the
      // predictors and tack them on the end of the dynamic regression state
      // model object.
      //
      // Args:
      //   r_prediction_data: An R list containing data needed for prediction.
      //     The signal that a dynamic regression model is present is that
      //     r_prediction_data contains an element named
      //     dynamic.regression.predictors.
      //   model:  The model containing a dynamic regression state model.
      //
      // Effects:
      //   dynamic.regression.predictors is extracted from r_prediction_data,
      //   and converted to a matrix.  The matrix is appended to the dynamic
      //   regression component of model.
      //
      //   This function assumes that only one dynamic regression component
      //   exists.
      void UnpackDynamicRegressionForecastData(
          SEXP r_prediction_data, ScalarStateSpaceModelBase *model);

      // Create the specific StateSpaceModel suitable for the given model
      // family.  The posterior sampler for the model is set, and entries for
      // its model parameters are created in io_manager.  This function does not
      // add state to the the model.  It is primarily intended to aid the
      // implementation of CreateModel.
      //
      // The arguments are documented in the comment to CreateModel.
      //
      // Returns:
      //   A pointer to the created model.  The pointer is owned by a Ptr in the
      //   the child class, so working with the raw pointer privately is
      //   exception safe.
      virtual ScalarStateSpaceModelBase * CreateBareModel(
          SEXP r_data_list,
          SEXP r_prior,
          SEXP r_options,
          RListIoManager *io_manager) = 0;

      // This function must not be called before UnpackForecastData.  It takes
      // the current state of the model held by the child classes, along with
      // the data obtained by UnpackForecastData(), and simulates one draw from
      // the posterior predictive forecast distribution.
      virtual Vector SimulateForecast(const Vector &final_state) = 0;

    };

    //=========================================================================
    // A base class for model managers handling models describing multiple time
    // series.  The number of time series is assumed known and fixed.
    class MultivariateModelManagerBase : public ModelManager {
     public:

      // Create a MultivariateModelManager instance suitable for working with a
      // specified model family.
      //
      // Args:
      //   family: A string indicating the familiy of the error distribution.
      //     Currently only "gaussian" is supported.
      //   nseries: Dimension of the response being modeled.  The number of time
      //     series.
      //   xdim: The dimension (number of columns) of the predictor matrix.
      //     This can be zero if there are no regressors.
      static MultivariateModelManagerBase * Create(
          const std::string &family, int nseries, int xdim);

      // Create a MultivariateModelManager by reinstantiating a previously
      // constructed bsts model.
      // Args:
      //   r_bsts_object:  An mbsts model object.
      static MultivariateModelManagerBase * Create(SEXP r_bsts_object);

      // Creates a BOOM state space model suitable for learning with MCMC.
      // Args:
      //   r_data_list: An R list containing the data to be modeled in the
      //     format expected by the requested model family.  This list generally
      //     contains an object called 'response' and a logical vector named
      //     'response.is.observed'.  If the model is a (generalized) regression
      //     model then it will contain a 'predictors' object as well, otherwise
      //     'predictors' will be NULL.  For logit or Poisson models an
      //     additional component should be present giving the number of trials
      //     or the exposure.
      //   r_shared_state_specification: An R list specifying the state models
      //     shared across multiple series.
      //   r_series_state_specification: A list of lists, containing the state
      //     specification for the series-specific portion of state.  The outer
      //     list may be NULL (R_NilValue), to indicate that no series-specic
      //     state component exists.  However, if non-NULL it must have length
      //     equal to the number of series being modeled.
      //   r_prior: The prior distribution for the observation model.  If the
      //     model is a regression model (determined by whether r_data_list
      //     contains a non-NULL 'predictors' element) then this must be a list
      //     of spike and slab priors.  Otherwise it must be a list of SdPriors.
      //   r_options: Model or family specific options such as the technique to
      //     use for model averaging (ODA vs SVSS).
      //   io_manager: The io_manager responsible for writing MCMC output to an
      //     R object, or streaming it from an existing object.
      //
      // Returns:
      //  A pointer to the created model.  The pointer is owned by a Ptr
      //  in the model manager, and should be caught by a Ptr in the caller.
      //
      // Side Effects:
      //   The returned pointer is also held in a smart pointer owned by
      //   the child class.
      virtual MultivariateStateSpaceModelBase * CreateModel(
          SEXP r_data_list,
          SEXP r_shared_state_specification,
          SEXP r_series_state_specification,
          SEXP r_prior,
          SEXP r_options,
          RListIoManager *io_manager) = 0;

      // Returns a set of draws from the posterior predictive distribution of
      // the multivariate time series.
      //
      // Args:
      //   r_bsts_object:  The R object created from a previous call to bsts().
      //   r_prediction_data: Data needed to make the prediction.  This might be
      //     a data frame for models that have a regression component, or a
      //     vector of exposures or trials for binomial or Poisson data.
      //   r_options: If any special options need to be passed in order to do
      //     the prediction, they should be included here.
      //
      // Returns:
      //   An array with dimension [iterations, time, nseries] containing draws
      //   from the posterior predictive distribution.
      virtual Array Forecast(SEXP r_mbsts_object,
                             SEXP r_prediction_data,
                             SEXP r_burn) = 0;

     private:
      // Create the specific StateSpaceModel suitable for the given model
      // family.  The posterior sampler for the model is set, and entries for
      // its model parameters are created in io_manager.  This function does not
      // add state to the the model.  It is primarily intended to aid the
      // implementation of CreateModel.
      //
      // The arguments are documented in the comment to CreateModel.
      //
      // Returns:
      //   A pointer to the created model.  The pointer is owned by a Ptr in the
      //   the child class, so working with the raw pointer privately is
      //   exception safe.
      virtual MultivariateStateSpaceModelBase * CreateBareModel(
          SEXP r_data_list,
          SEXP r_prior,
          SEXP r_options,
          RListIoManager *io_manager) = 0;

    };

  }  // namespace bsts
}  // namespace BOOM

#endif  // BSTS_SRC_MODEL_MANAGER_H_
