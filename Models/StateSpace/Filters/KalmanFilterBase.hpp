#ifndef BOOM_STATE_SPACE_KALMAN_FILTER_BASE_HPP_
#define BOOM_STATE_SPACE_KALMAN_FILTER_BASE_HPP_

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
  namespace Kalman {
    //---------------------------------------------------------------------------
    // The Kalman filter sequentially updates a set of marginal distributions
    // conditional on prior data.  The marginal distributions describe the
    // latent state variables, as well as the error terms of the model and the
    // forecast distribution of the next observation.
    //
    // Several base classes are needed for marginal distributions to account for
    // the fact that univariate and multivariate data must be handled
    // differently, and that there are several potential simplifying assumptions
    // in the multivariate case.
    //---------------------------------------------------------------------------

    // A base class to handle quantities common to all marginal distributions.
    class MarginalDistributionBase {
     public:
      // Args:
      //   state_dimension: The dimension of the state variable described by
      //     this marginal distribution.
      //   time_index: The index of the time point described by this
      //     distribution.
      explicit MarginalDistributionBase(int state_dimension, int time_index);

      virtual ~MarginalDistributionBase() {}

      // The time index for the time point described by this marginal
      // distribution.
      int time_index() const { return time_index_; }

      // The marginal distribution at time t should be initialized with the
      // state mean from distribution t-1.  After updating, the state_mean()
      // refers to the mean of the state at time t+1 given data to time t.
      const Vector &state_mean() const {return state_mean_;}
      void set_state_mean(const Vector &state_mean) {state_mean_ = state_mean;}
      void increment_state_mean(const Vector &mean_increment) {
        state_mean_ += mean_increment;
      }

      // The marginal distribution at time t should be initialized with the
      // state variance from distribution t-1.  After updating, the
      // state_variance() refers to the variance of the state at time t+1 given
      // data to time t.
      const SpdMatrix &state_variance() const {return state_variance_;}
      void set_state_variance(const SpdMatrix &var);
      void increment_state_variance(const SpdMatrix &variance_increment);

      // Convert the state mean and variance from forward-looking moments
      // (e.g. E(state[t+1] | Data to t)) to contemporaneous moments
      // (e.g. E(state[t] | Data to t)).
      //
      // If a[t] = E(state[t] | Data to t-1) then the contemporaneous state mean is
      // a[t] + P[t] * Z[t].transpose() * Finv * v;
      virtual Vector contemporaneous_state_mean() const = 0;
      //      virtual SpdMatrix contemporaneous_state_variance() const = 0;

      // Durbin and Koopman's r[t].  Recall that eta[t] is the error term for
      // moving from state t to state t+1.  The conditional mean of eta[t] given
      // all observed data is
      //
      // hat(eta[t]) = Q[t] * R[t]' * r[t],
      //
      // where Q[t] is the error variance at time t (a model paramter), and R[t]
      // is the error expander.  In this equation R[t]' is a contractor (moving
      // from the state dimension to the error dimension).
      const Vector &scaled_state_error() const { return scaled_state_error_; }
      void set_scaled_state_error(const Vector &scaled_error) {
        scaled_state_error_ = scaled_error;
      }

     protected:
      SpdMatrix & mutable_state_variance() {return state_variance_;}
      void check_variance(const SpdMatrix &v) const;

     private:
      // The time point that this marginal distribution describes.
      int time_index_;

      // After updating, these describe the mean and variance of the state at
      // time_index_ + 1 given data to time_index_.
      Vector state_mean_;
      SpdMatrix state_variance_;

      // The r[t] parameter computed from the Durbin-Koopman disturbance
      // smoother.  DK do a poor job of explaining what r is, but it is a scaled
      // version of the state error (see note above).  It is produced by
      // smooth_disturbances_fast() and used by propagate_disturbances().
      Vector scaled_state_error_;
    };

  }  // namespace Kalman

  //===========================================================================
  // A base class for Kalman filter objects.  This class keeps track of the log
  // likelihood, the status of the filter, and takes responsibility for setting
  // observers on parameters and data.
  class KalmanFilterBase {
   public:
    KalmanFilterBase();
    virtual ~KalmanFilterBase() {}

    //--------------------------------------------------------------------------
    // Accessors and basic status.
    //--------------------------------------------------------------------------

    // The status of the Kalman filter object.
    // Values:
    //   NOT_CURRENT: The filter must be re-run before its entries can be used.
    //   MCMC_CURRENT: parameters and data are unchanged since impute_state()
    //     was last called.  state posterior means and variances are not
    //     available.
    //   CURRENT: Parameters and data are unchanged since full_kalman_filter()
    //     was last called.
    enum KalmanFilterStatus { NOT_CURRENT, MCMC_CURRENT, CURRENT };

    void set_status(const KalmanFilterStatus &status) { status_ = status; }

    // Print the state mean of each marginal distribution.
    virtual std::ostream & print(std::ostream &out) const;
    std::string to_string() const;

    // The marginal distribution at time t.
    virtual Kalman::MarginalDistributionBase & operator[](size_t pos) = 0;
    virtual const Kalman::MarginalDistributionBase & operator[](
        size_t pos) const = 0;

    // The number of nodes (time points) managed by the filter.
    virtual int size() const = 0;

    // Return the last computed value of log likelihood.
    double log_likelihood() const {
      return log_likelihood_;
    }

    const Vector &initial_scaled_state_error() const {
      return initial_scaled_state_error_;
    }

    // The matrix of state means.  Each column of the matrix represents a time
    // point.
    Matrix state_mean() const;

    //--------------------------------------------------------------------------
    // Filtering operations
    //--------------------------------------------------------------------------

    // If the filter status is current then return the pre-computed log
    // likelihood.  If not current the compute and store the log likelihood and
    // return the saved value.
    double compute_log_likelihood();

    // Concrete classes hold a pointer to a model object.  Calling update() runs
    // the kalman filter over all the data contained in *model_.
    virtual void update() = 0;

    // Run the Durbin and Koopman fast disturbance smoother.
    virtual void fast_disturbance_smooth() = 0;

   protected:
    // Set log likelihood to zero and status to NOT_CURRENT.
    void clear_loglikelihood();

    void increment_log_likelihood(double loglike) {
      log_likelihood_ += loglike;
    }

    void set_initial_scaled_state_error(const Vector &err) {
      initial_scaled_state_error_ = err;
    }

   private:
    // For explanation, please see the comments for the enum definition ofr
    // KalmanFilterStatus.
    KalmanFilterStatus status_;

    // The log likelihood of the data as computed by the last forward update.
    double log_likelihood_;

    // Durbin and Koopman's r0 from the fast disturbance smoother (see equation
    // (5) in Durbin and Koopman (2002, Biometrika), or equation 4.32 in Durbin
    // and Koopman (2001, first edition)).
    Vector initial_scaled_state_error_;
  };

  inline std::ostream &operator<<(std::ostream &out,
                                  const KalmanFilterBase &filter) {
    return filter.print(out);
  }

}  // namespace BOOM

#endif  // BOOM_STATE_SPACE_KALMAN_FILTER_BASE_HPP_
