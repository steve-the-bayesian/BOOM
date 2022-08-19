#ifndef BOOM_STATE_SPACE_MULTIVARIATE_KALMAN_FILTER_BASE_HPP_
#define BOOM_STATE_SPACE_MULTIVARIATE_KALMAN_FILTER_BASE_HPP_

/*
  Copyright (C) 2005-2022 Steven L. Scott

  This library is free software; you can redistribute it and/or modify it under
  the terms of the GNU Lesser General Public License as published by the Free
  Software Foundation; either version 2.1 of the License, or (at your option)
  any later version.

  This library is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
  details.

  You should have received a copy of the GNU Lesser General Public License along
  with this library; if not, write to the Free Software Foundation, Inc., 51
  Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
*/

#include "LinAlg/Vector.hpp"
#include "LinAlg/SpdMatrix.hpp"
#include "Models/StateSpace/Filters/KalmanFilterBase.hpp"
#include "Models/StateSpace/Filters/SparseMatrix.hpp"
#include "cpputil/math_utils.hpp"

namespace BOOM {

  class MultivariateStateSpaceModelBase;
  class SparseKalmanMatrix;

  namespace Kalman {
    // The marginal distribution of state when the observed data is
    // multivariate.  Note that the dimension of the observed data need not be
    // the same at each time point.
    //
    // Examples of concrete marginal distributions include:
    // - The response variables are conditionally IID given state.  (e.g. a
    //   dynamic regression model).
    // - The response variables are conditionally independent given state, but
    //   not identically distributed.
    // - The response variables have an unspecified correlation matrix.
    // - Others are possible.
    //
    // Each marginal distribution describes a time point, and there is one
    // marginal distribution for each time point in the time series.  The time
    // indices of the marginal distributions are sequential integers: 0, 1, 2,
    // ... .
    class MultivariateMarginalDistributionBase
        : public MarginalDistributionBase {
     public:
      // Args:
      //   state_dim:  The dimension of the latent state vector.
      //   time_index: The index of the time point described by this marginal
      //     distribution.  I.e. time 0, time 1, time 2...
      MultivariateMarginalDistributionBase(int state_dim, int time_index)
          : MarginalDistributionBase(state_dim, time_index)
      {}

      // Update this marginal distribution to reflect the observed data at this
      // time point, and the marginal information from the preceding time point.
      //
      // Args:
      //    observation: The vector of responses at this time point.  Missing
      //      values may either be excluded or filled with dummy values, so the
      //      vector may be smaller than 'nseries' in size.
      //    observed: Indicates which elements of observation are actually
      //      observed.
      //
      // Preconditions:
      //   state_mean and state_variance have been set to the conditional mean
      //   and variance of this period's state given preceding data.
      //
      // Postconditions:
      //   state_mean and state_variance are updated to give the conditional
      //   mean and variance of the NEXT time period given data to THIS time
      //   period.  The other data members of this class are also populated.
      //
      // Returns:
      //   The log likelihood log p(y_t | Y_{t-1}).
      double update(const Vector &observation, const Selector &observed);

      // The difference between the observed data at this time point and its
      // expected value given past data.  If any data elements are missing, they
      // are omitted from the prediction error.  To get a prediction error of
      // the same dimension as the original observation, call
      // observed.expand(prediction_error) where observed is the Selector
      // associated with the observation.
      const Vector &prediction_error() const {return prediction_error_;}
      void set_prediction_error(const Vector &err) {prediction_error_ = err;}

      // The precision matrix (inverse of the variance matrix) for the
      // prediction_error, conditional on all past data.
      virtual Ptr<SparseKalmanMatrix> sparse_forecast_precision() const = 0;

      // The log of the determinant of forecast_precision().
      virtual double forecast_precision_log_determinant() const = 0;

      // The set of regression coefficients used to adjust the expected value of
      // the state given the prediction error.  The dimension is state_dim x
      // ydim.
      virtual Ptr<SparseMatrixProduct> sparse_kalman_gain(
          const Selector &observed,
          const Ptr<SparseKalmanMatrix> &forecast_precision) const;

      // After the call to update(), state_mean() and state_variance() refer to
      // the predictive mean and variance of the state at time_dimension() + 1
      // given data to time_dimension().
      //
      // contemporaneous_state_XXX refers to the moments at the current time,
      // given data to the current time.
      Vector contemporaneous_state_mean() const override;
      SpdMatrix contemporaneous_state_variance(
          const Ptr<SparseKalmanMatrix> &forecast_precision) const;

      // The marginal distribution of the state at the preceding time point.
      // Return 'nullptr' if there is no previous time point.
      virtual MultivariateMarginalDistributionBase *previous() = 0;
      virtual const MultivariateMarginalDistributionBase *previous() const = 0;

      // The model describing the time series containing this marginal
      // distribution.  Access to the model is necessary because it carries the
      // structural matrices defining the state space model.
      virtual const MultivariateStateSpaceModelBase *model() const = 0;

     private:
      // Store a minimial set of information to allow sparse_forecast_precision
      // to be quickly computed.
      virtual void update_sparse_forecast_precision(
          const Selector &observed) = 0;

      // Implement update() in the case where y[t] is fully missing (i.e. no
      // part of it is observed.
      double fully_missing_update();

      // y[t] - E(y[t] | Y[t-1]).  The dimension matches y[t], which might vary
      // across t.
      Vector prediction_error_;
    };
  }  // namespace Kalman

  //===========================================================================
  // An intermediate base class for handling the parts of the multivariate
  // Kalman filter that don't depend on the observation variance.
  class MultivariateKalmanFilterBase : public KalmanFilterBase {
   public:
    // Run the Kalman filter algorithm with the current data.
    void update() override;

    // Run the Kalman state smoother.  This replaces state_mean() and
    // state_variance() at each time point with the contemporaneous state mean
    // and variance given all observed data.  It invalidates other filter
    // computations, such as the forecast precision and the Kalman gain
    // coefficients.
    void smooth();

    // Update the marginal distribution at a single time point.  The simulation
    // filter calls this method based on simulated data, so we can't rely on the
    // stored model object to supply the data in all cases.
    //
    // Args:
    //   observation: The observed data at time t.  Missing values may either be
    //     excluded or filled with dummy values.
    //   observed:  Indicates which elements of observation were actually observed.
    //   time: The time point at which observation was observed.
    void update_single_observation(
        const Vector &observation, const Selector &observed, int time);

    // Run Durbin and Koopman's fast disturbance smoother.
    void fast_disturbance_smooth() override;

    // The prediction error at time t.
    Vector prediction_error(int t, bool standardize = false) const;

    Kalman::MultivariateMarginalDistributionBase & operator[](size_t pos)
        override = 0;
    const Kalman::MultivariateMarginalDistributionBase & operator[](size_t pos)
        const override = 0;

    // Add nodes to the collection of marginal distributions until it is large
    // enough to hold 'time' elements.
    virtual void ensure_size(int time) = 0;

   protected:
    virtual Kalman::MultivariateMarginalDistributionBase &node(size_t time) = 0;
    virtual const Kalman::MultivariateMarginalDistributionBase &node(
        size_t time) const = 0;

    // The model describing the data being filtered.
    virtual const MultivariateStateSpaceModelBase *model() const = 0;
    virtual MultivariateStateSpaceModelBase *model() = 0;
  };

  //===========================================================================
  // The various multivariate Kalman filters are parameterized by the types of
  // the marginal distributions comprising them.  Each marginal distribution
  // defines a type 'ModelType', that inherits from
  // MultivariateStateSpaceModelBase.
  template <class MARGINAL>
  class MultivariateKalmanFilter
      : public MultivariateKalmanFilterBase {
   public:
    typedef typename MARGINAL::ModelType ModelType;
    typedef MARGINAL MarginalType;

    // Args:
    //   model: A pointer to the model that owns this filter.  The model
    //     provides the model matrices to the nodes of the filter, and controls
    //     the observation status at each time point.
    explicit MultivariateKalmanFilter(ModelType *model)
        : model_(model) {}

    // Return filter node 't'.
    MarginalType &operator[](size_t t) override {
      return nodes_[t];
    }
    const MarginalType &operator[](size_t t) const override {
      return nodes_[t];
    }

    // The number of time points described by the filter.
    int size() const override {return nodes_.size();}

    // Add nodes (marginal distributions) to the filter until its size is at
    // least 't'.
    void ensure_size(int time) override {
      while(nodes_.size() <=  time) {
        nodes_.push_back(MarginalType(model_, this, nodes_.size()));
      }
    }

    // The marginal distribution managed by the filter.
    const MarginalType &back() const {return nodes_.back();}

   protected:
    MarginalType &node(size_t pos) override {
      return nodes_[pos];
    }
    const MarginalType &node(size_t pos) const override {
      return nodes_[pos];
    }

    ModelType * model() override {return model_;}
    const ModelType * model() const override {return model_;}

   private:
    ModelType *model_;
    std::vector<MarginalType> nodes_;
  };

}  // namespace BOOM

#endif  // BOOM_STATE_SPACE_MULTIVARIATE_KALMAN_FILTER_BASE_HPP_
