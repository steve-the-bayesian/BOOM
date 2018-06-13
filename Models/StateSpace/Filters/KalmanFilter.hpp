#ifndef BOOM_KALMAN_FILTER_HPP_
#define BOOM_KALMAN_FILTER_HPP_

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

#include "LinAlg/SpdMatrix.hpp"
#include "LinAlg/Vector.hpp"
#include "LinAlg/VectorView.hpp"
#include "LinAlg/Selector.hpp"

namespace BOOM {
  class StateSpaceModelBase;
  class ScalarStateSpaceModelBase;
  class MultivariateStateSpaceModelBase;
  
  namespace Kalman {

    //---------------------------------------------------------------------------
    class MarginalDistributionBase {
     public:
      MarginalDistributionBase(int dim);
      const Vector &state_mean() const {return state_mean_;}
      void set_state_mean(const Vector &state_mean) {
        state_mean_ = state_mean;
      }

      void increment_state_mean(const Vector &v) {
        state_mean_ += v;
      }
      
      const SpdMatrix &state_variance() const {return state_variance_;}
      void set_state_variance(const SpdMatrix &state_variance) {
        state_variance_ = state_variance;
      }

      void increment_state_variance(const SpdMatrix &m) {
        state_variance_ += m;
      }

     protected:
      SpdMatrix & mutable_state_variance() {return state_variance_;}
      
     private:
      Vector state_mean_;
      SpdMatrix state_variance_;
    };

    //---------------------------------------------------------------------------
    class ScalarMarginalDistribution
        : public MarginalDistributionBase {
     public:
      ScalarMarginalDistribution(int state_dimension)
          : MarginalDistributionBase(state_dimension), prediction_error_(0),
            prediction_variance_(0), kalman_gain_(state_dimension, 0) {}

      double update(double y,
                    bool missing,
                    int t,
                    const ScalarStateSpaceModelBase *model,
                    double observation_variance_scale_factor = 1.0);
      
      double prediction_error() const {return prediction_error_;}
      void set_prediction_error(double err) {prediction_error_ = err;}

      double prediction_variance() const {return prediction_variance_;}
      void set_prediction_variance(double var) {prediction_variance_ = var;}

      const Vector &kalman_gain() const {return kalman_gain_;}
      void set_kalman_gain(const Vector &gain) {kalman_gain_ = gain;}
      
     private:
      double prediction_error_;
      double prediction_variance_;
      Vector kalman_gain_;
    };

    //---------------------------------------------------------------------------
    class MultivariateMarginalDistributionBase
        : public MarginalDistributionBase {
     public:
      MultivariateMarginalDistributionBase(int state_dim, int observation_dim)
          : MarginalDistributionBase(state_dim),
            forecast_error_(observation_dim),
            kalman_gain_(state_dim, observation_dim),
            r_(state_dim)
      {}
      
      // Inverse of Var(y[t] | Y[t-1]).
      virtual SpdMatrix forecast_precision() const = 0;

      // The log determinant of forecast_precision().
      virtual double forecast_precision_log_determinant() const = 0;

     private:
      // y[t] - E(y[t] | Y[t-1]).  The dimension matches y[t], which might vary
      // across t.
      Vector forecast_error_;

      // The Kalman gain K[t] shows up in the updating equation:
      //       a[t+1] = T[t] * a[t] + K[t] * v[t].
      // Rows correspond to states and columns to observation elements.
      Matrix kalman_gain_;

      // Computed from the Durbin-Koopman disturbance smoother.  DK do a poor job
      // of explaining what r_ is, but it is a scaled version of the state
      // disturbance error (or something...).  It is produced by
      // smooth_disturbances_fast() and used by propagate_disturbances().
      Vector r_;
    };

    //---------------------------------------------------------------------------
    // Storage class to use when the error variance for the observation equation
    // is a constant times the identity matrix
    class ConstantIndependentMarginalDistribution
        : public MultivariateMarginalDistributionBase {
     public:
      SpdMatrix forecast_precision() const;
      double forecast_precision_log_determinant() const;
     private:
    };

    //---------------------------------------------------------------------------
    // Storage class to use when the error variance for the observation equation
    // is a diagonal matrix.
    class ConditionallyIndependentMarginalDistribution
        : public MultivariateMarginalDistributionBase {
     public:
     private:
    };

    //---------------------------------------------------------------------------
    // Storage class to use when the error variance for the observation equation
    // is a generic SpdMatrix.
    class GeneralMultivariateMultivariateMarginalDistribution
        : public MultivariateMarginalDistributionBase {
     public:
     private:
    };
  }  // namespace Kalman
  //======================================================================

  class KalmanFilterBase {
   public:
    // The status of the Kalman filter.
    // Values:
    //   NOT_CURRENT: The filter must be re-run before its entries can be used.
    //   MCMC_CURRENT: neither parameter nor data have changed since
    //     impute_state() was last called.  state posterior means and variances
    //     are not available.
    //   CURRENT: Neither parameters nor data have changed since
    //     full_kalman_filter() was last called.
    enum KalmanFilterStatus { NOT_CURRENT, MCMC_CURRENT, CURRENT };

    double log_likelihood() const {
      return log_likelihood_;
    }

    double compute_log_likelihood() {
      if (status_ == NOT_CURRENT) {
        clear();
        update();
      }
      return log_likelihood_;
    }

    void clear();
    virtual void update() = 0;
    virtual Vector fast_disturbance_smooth() = 0;

    void mark_not_current() {
      status_ = NOT_CURRENT;
    }

    virtual Kalman::MarginalDistributionBase & operator[](size_t pos) = 0;
    virtual const Kalman::MarginalDistributionBase & operator[](
        size_t pos) const = 0;

    void set_status(const KalmanFilterStatus &status) {
      status_ = status;
    }
    
   protected:
    void increment_log_likelihood(double loglike) {
      log_likelihood_ += loglike;
    }

    void observe_model_parameters(StateSpaceModelBase *model);

   private:
    KalmanFilterStatus status_;
    double log_likelihood_;
  };

  //---------------------------------------------------------------------------

  class ScalarKalmanFilter : public KalmanFilterBase {
   public:
    ScalarKalmanFilter(ScalarStateSpaceModelBase *model = nullptr);

    void set_model(ScalarStateSpaceModelBase *model);
    
    // Fun the full Kalman filter over all the data held by the model.
    void update() override;

    Vector fast_disturbance_smooth() override;
      
    // Update the Kalman filter at time t given observation y, which might be
    // different than y[t] held by the model (e.g. when doing posterior
    // simulation).
    void update(double y, int t, bool missing = false);

    // Return the one-step prediction error held by the filter at time t.  If
    // 'standardize' is true then divide the prediction error by the square
    // root of the prediction variance.
    double prediction_error(int t, bool standardize = false) const;

    Kalman::ScalarMarginalDistribution &operator[](size_t pos) override {
      return nodes_[pos];
    }

    const Kalman::ScalarMarginalDistribution &operator[](
        size_t pos) const override {
      return nodes_[pos];
    }
      
    const Kalman::ScalarMarginalDistribution &back() const;
      
   private:
    ScalarStateSpaceModelBase *model_;
    std::vector<Kalman::ScalarMarginalDistribution> nodes_;
  };

  
  }  // namespace BOOM


#endif //  BOOM_KALMAN_FILTER_HPP_
