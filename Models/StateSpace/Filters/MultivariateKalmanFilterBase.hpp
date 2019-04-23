#ifndef BOOM_STATE_SPACE_MULTIVARIATE_KALMAN_FILTER_BASE_HPP_
#define BOOM_STATE_SPACE_MULTIVARIATE_KALMAN_FILTER_BASE_HPP_

/*
  Copyright (C) 2005-2018 Steven L. Scott

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
    class MultivariateMarginalDistributionBase
        : public MarginalDistributionBase {
     public:
      // Args:
      //   state_dim:  The dimension of the latent state vector.
      //   time_index: The index of the time point described by this marginal
      //     distribution.
      MultivariateMarginalDistributionBase(int state_dim, int time_index)
          : MarginalDistributionBase(state_dim, time_index),
            forecast_precision_log_determinant_(negative_infinity()) {}

      // The difference between the observed data at this time point and its
      // expected value given past data.  If any data elements are missing, they
      // are omitted from the prediction error.  To get a prediction error of
      // the same dimension as the original observation call
      // observed.expand(prediction_error) where observed is the Selector
      // associated with the observation.
      const Vector &prediction_error() const {return prediction_error_;}
      void set_prediction_error(const Vector &err) {prediction_error_ = err;}

      // The precision matrix (inverse of the variance matrix) for the
      // prediction_error, conditional on all past data.  Calling
      // forecast_precision() can be expensive for high dimensional data.
      //
      // The forecast precision shows up in the Kalman filter in two forms.
      //   (1) It multiplies the state error in some formulas, and
      //   (2) its log determinant is needed in order to compute log likelihood.
      //
      // Thus, most problems can be reworked so that forecast_precision() is not
      // called, with scaled_prediction_error() (for case 1) and
      // forecast_precision_log_determinant() (for case 2) being called instead.
      virtual SpdMatrix forecast_precision() const = 0;

      // The prediction error is y[t] - E(y[t] | Y[t-1]), where y[t] is the
      // observation at time t, and Y[t-1] is the set of all preceding
      // observations. The scaled prediction error is forecast_precision() *
      // prediction_error().
      const Vector &scaled_prediction_error() const {
        return scaled_prediction_error_;
      }
      void set_scaled_prediction_error(const Vector &err) {
        scaled_prediction_error_ = err;
      }

      // The log of the determinant of forecast_precision().
      double forecast_precision_log_determinant() const {
        return forecast_precision_log_determinant_;
      }
      void set_forecast_precision_log_determinant(double logdet);

      // The set of regression coefficients used to adjust the expected value of
      // the state given the prediction error.
      const Matrix &kalman_gain() const {return kalman_gain_;}
      void set_kalman_gain(const Matrix &gain) {kalman_gain_ = gain;}

      // Update this marginal distribution to reflect the observed data at this
      // time point, and the marginal information from the preceding time point.
      //
      // Args:
      //    observation: The full vector of responses at this time point,
      //      including dummy values for those which are unobserved.
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
      virtual double update(const Vector &observation,
                            const Selector &observed);

      // An observation is considered high dimensional if the number of observed
      // series at a given time point exceeds the state dimension by a specified
      // factor.  Different child classes can have different thresholds for what
      // it means to be high dimensional.
      virtual bool high_dimensional(const Selector &observed) const;
      virtual double high_dimensional_threshold_factor() const = 0;
      
      // The marginal distribution of the state at the preceding time point.
      virtual MultivariateMarginalDistributionBase *previous() = 0;
      virtual const MultivariateMarginalDistributionBase *previous() const = 0;

      // The model describing the time series containing this marginal
      // distribution.
      virtual const MultivariateStateSpaceModelBase *model() const = 0;

      // After the call to update(), state_mean() and state_variance() refer to
      // the predictive mean and variance of the state at time_dimension() + 1
      // given data to time_dimension().
      //
      // contemporaneous_state_XXX refers to the moments at the current time,
      // given data to the current time.
      Vector contemporaneous_state_mean() const override;
      SpdMatrix contemporaneous_state_variance() const override;

     private:
      // Update the prediction error, scaled prediction error, forecast variance
      // (if computed) and kalman gain, while trying to take advantage of
      // sparsity using tricks like the binomial inverse theorem and the
      // woodbury formula.
      virtual void high_dimensional_update(
          const Vector &observation,
          const Selector &observed,
          const SparseKalmanMatrix &transition,
          const SparseKalmanMatrix &observation_coefficients) = 0;

      // Update the prediction error, scaled prediction error, forecast variance
      // (if computed) and Kalman gain, using the textbook formulas for the
      // Kalman filter updates.
      virtual void low_dimensional_update(
          const Vector &observation,
          const Selector &observed,
          const SparseKalmanMatrix &transition,
          const SparseKalmanMatrix &observation_coefficients) = 0;

      // Implement update() in the case where y[t] is fully missing (i.e. no
      // part of it is observed.
      double fully_missing_update();
      
      // y[t] - E(y[t] | Y[t-1]).  The dimension matches y[t], which might vary
      // across t.
      Vector prediction_error_;

      // The scaled prediction_error_ is Finv * prediction_error_, where Finv is
      // the forecast error precision matrix: the inverse of Var(y[t] | data to
      // t-1).
      Vector scaled_prediction_error_;

      // The log determinant of the forecast precision matrix Finv (see above).
      double forecast_precision_log_determinant_;
      
      // The Kalman gain K[t] shows up in the updating equation:
      //       a[t+1] = T[t] * a[t] + K[t] * v[t].
      // Rows correspond to states and columns to observation elements, so the
      // dimension is S x m.
      Matrix kalman_gain_;
    };
  }  // namespace Kalman

  //===========================================================================
  // An intermediate base class for handling the parts of the multivariate
  // Kalman filter that don't depend on the observation variance.  
  class MultivariateKalmanFilterBase : public KalmanFilterBase {
   public:
    // Args:
    //   model:  The model to be filtered.
    MultivariateKalmanFilterBase(MultivariateStateSpaceModelBase *model);
    
    void update() override;

    // Update the marginal distribution at a single time point.  The simulation
    // filter calls this method based on simulated data, so we can't rely on the
    // stored model object to supply the data in all cases.
    //
    // Args:
    //   observation:  The observed data at time t.
    //   observed:  Indicates which elements of observation were actually observed.
    //   t: The time point at which observation was observed.
    void update_single_observation(
        const Vector &observation, const Selector &observed, int t);

    // Run Durbin and Koopman's fast disturbance smoother.  
    void fast_disturbance_smooth() override;

    // The prediction error at time t.
    Vector prediction_error(int t, bool standardize = false) const;

    Kalman::MultivariateMarginalDistributionBase & operator[](size_t pos)
        override = 0;
    const Kalman::MultivariateMarginalDistributionBase & operator[](size_t pos)
        const override = 0;

    // Add nodes to the collection of marginal distributions until it is large
    // enough to hold t elements.
    virtual void ensure_size(int t) = 0;

    // The model describing the data being filtered.
    const MultivariateStateSpaceModelBase *model() const {
      return model_;
    }
    
   protected:
    virtual Kalman::MultivariateMarginalDistributionBase &node(size_t t) = 0;
    virtual const Kalman::MultivariateMarginalDistributionBase &node(
        size_t t) const = 0;
    
   private:
    MultivariateStateSpaceModelBase *model_;
  };

  //===========================================================================
  // The various multivariate Kalman filters are parameterized by the types of
  // the marginal distributions comprising them.  Each marginal distribution
  // must define a type ModelType.
  template <class MARGINAL>
  class MultivariateKalmanFilter
      : public MultivariateKalmanFilterBase {
   public:
    typedef typename MARGINAL::ModelType ModelType;
    typedef MARGINAL MarginalType;

    explicit MultivariateKalmanFilter(ModelType *model)
        : MultivariateKalmanFilterBase(model),
          model_(model) {}

    MarginalType &operator[](size_t pos) override {
      return nodes_[pos];
    }
    const MarginalType &operator[](size_t pos) const override {
      return nodes_[pos];
    }

    // The number of time points described by the filter.
    int size() const override {return nodes_.size();}

    // Add nodes (marginal distributions) to the filter until its size is at
    // least 't'.
    void ensure_size(int t) override {
      while(nodes_.size() <=  t) {
        MarginalType *previous = nodes_.empty() ? nullptr : &nodes_.back();
        nodes_.push_back(MarginalType(model_, previous, nodes_.size()));
      }
    }

    // The marginal distribution managed by the filter.
    const MarginalType &back() const {return nodes_.back();}

   protected:
    MarginalType &node(size_t pos) override {return nodes_[pos];}
    const MarginalType &node(size_t pos) const override {return nodes_[pos];}
    
   private:
    ModelType *model_;
    std::vector<MarginalType> nodes_;
  };
  
}  // namespace BOOM

#endif  // BOOM_STATE_SPACE_MULTIVARIATE_KALMAN_FILTER_BASE_HPP_
