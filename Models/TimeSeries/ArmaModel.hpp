/*
  Copyright (C) 2005-2018 Steven L. Scott

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

#ifndef BOOM_ARMA_MODEL_HPP_
#define BOOM_ARMA_MODEL_HPP_

#include "Models/Glm/GlmCoefs.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/ParamPolicy_3.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/StateSpace/Filters/SparseMatrix.hpp"
#include "Models/TimeSeries/TimeSeries.hpp"

namespace BOOM {

  // The state transition matrix used when putting an ARMA model in state space
  // form.
  //
  // phi_1   1 0 0 ... 0
  // phi_2   0 1 0 ... 0
  // ...
  // phi_p-1 0 0 0 ... 1
  // phi_p   0 0 0 ... 0
  //
  // This matrix is the transpose of the AutoRegressionTransitionMatrix found in
  // Models/StateSpace/Filters/SparseMatrix.hpp.
  class ArmaStateSpaceTransitionMatrix : public SparseMatrixBlock {
   public:
    explicit ArmaStateSpaceTransitionMatrix(const Vector &expanded_phi);
    ArmaStateSpaceTransitionMatrix *clone() const override {
      return new ArmaStateSpaceTransitionMatrix(*this);
    }

    int nrow() const override { return expanded_phi_.size(); }
    int ncol() const override { return expanded_phi_.size(); }
    void multiply(VectorView lhs, const ConstVectorView &rhs) const override;
    void multiply_and_add(VectorView lhs,
                          const ConstVectorView &rhs) const override;

    void Tmult(VectorView lhs, const ConstVectorView &rhs) const override;
    void multiply_inplace(VectorView x) const override;
    SpdMatrix inner() const override;
    SpdMatrix inner(const ConstVectorView &weights) const override;
    void add_to_block(SubMatrix block) const override;
    Matrix dense() const override;

   private:
    // Vector of AR coefficients, expanded by including trailing zeros up to the
    // desired state dimension.
    Vector expanded_phi_;
  };

  // The variance of the state innovation errors for an ARMA model in state
  // space form.  The matrix has the structure RQR^T,  where
  // R^T = 1 theta_1 theta_2 ... theta_{r-1}
  class ArmaStateSpaceVarianceMatrix : public SparseMatrixBlock {
   public:
    // Args:
    //   expanded_theta: The vector of MA coefficients, including the leading 1,
    //     and with zero-padding at the end to match the length of the AR
    //     coefficients in the event that the AR coefficients are longer.
    //   sigsq:  The residual variance.
    ArmaStateSpaceVarianceMatrix(const Vector &expanded_theta, double sigsq);
    ArmaStateSpaceVarianceMatrix *clone() const override {
      return new ArmaStateSpaceVarianceMatrix(*this);
    }

    int nrow() const override { return theta_.size(); }
    int ncol() const override { return theta_.size(); }
    void multiply(VectorView lhs, const ConstVectorView &rhs) const override;
    // Because RQR is a symmetric matrix.
    void multiply_and_add(VectorView lhs,
                          const ConstVectorView &rhs) const override;
    void Tmult(VectorView lhs, const ConstVectorView &rhs) const override {
      multiply(lhs, rhs);
    }
    void multiply_inplace(VectorView x) const override;
    SpdMatrix inner() const override;
    SpdMatrix inner(const ConstVectorView &weights) const override;
    void add_to_block(SubMatrix block) const override;
    Matrix dense() const override;

   private:
    Vector theta_;
    double sigsq_;
  };

  // An ARMA(p, q) model describes a time series
  //   y[t+1] = phi[0] * y[t] + phi[1] * y[t-1] + ... + phi[p-1] * y[t - p +1]
  //            + theta[0] * epsilon[t] + ... + theta[q-1] * epsilon[t - q + 1]
  //            + epsilon[t+1]
  //
  // The phi's are the autoregression (AR) coefficients, and the theta's are the
  // "moving average" (MA) coefficients.  Epsilon[t] ~IID N(0, sigma^2).
  //
  // If p == 0 then the model is said to be MA(q).  If q == 0 then the model is
  // AR(p) (but in that case consider an ArModel instead).
  class ArmaModel : public ParamPolicy_3<GlmCoefs, VectorParams, UnivParams>,
                    public IID_DataPolicy<DoubleData>,
                    public PriorPolicy {
   public:
    // Args:
    //   p:  The number of AR lags.  p >= 0.
    //   q:  The number of MA lags.  q >= 0.
    // p + q > 0.
    ArmaModel(int p, int q);

    // Args:
    //   ar_params:  The autoregression coefficients (phi).
    //   ma_params:  The moving average coefficients (theta).
    //   residual_variance: The variance of the epsilon[t] white noise process
    //     (sigma^2).
    ArmaModel(const Ptr<GlmCoefs> &ar_params,
              const Ptr<VectorParams> &ma_params,
              const Ptr<UnivParams> &residual_variance);

    ArmaModel *clone() const override { return new ArmaModel(*this); }

    // phi[0], phi[1], ... phi[p-1].  Might be empty.
    const Vector &ar_coefficients() const;
    void set_ar_coefficients(const Vector &ar_coefficients);

    // theta[0], theta[1], ... theta[q-1].  Might be empty.
    const Vector &ma_coefficients() const;
    void set_ma_coefficients(const Vector &ma_coefficients);

    // Variance and SD of the white noise process.
    double sigsq() const;
    double sigma() const { return sqrt(sigsq()); }
    void set_sigsq(double sigsq);
    void set_sigma(double sigma) { set_sigsq(sigma * sigma); }

    // The number of AR coefficients.
    int ar_dimension() const { return ar_coefficients().size(); }

    // The number of MA coefficients.  This does not include the leading '1'
    // associated with the current error term.
    int ma_dimension() const { return ma_coefficients().size(); }

    // An ARMA model is 'invertible' (i.e. it can be written as a weighted sum
    // of white noise: MA(infinity)), if all the roots of the AR polynomial
    // phi(z) = 1 - phi[0] * z - phi[1] * z^2 - ... - phi[p-1] * z^p lie outside
    // the unit circle.
    static bool is_invertible(const Vector &ar_coefficients);
    bool is_invertible() const { return is_invertible(ar_coefficients()); }

    // An ARMA model is 'causal' if it can be written as an infinite AR process.
    // This can be done iff all the roots of the MA polynomial
    //    theta(z) = 1 + theta[0] * z + theta[1] * z^2 + ... + theta[q-1] * z^q.
    // lie outside the unit circle.
    //
    // Note the word 'causal' is used in a mathematical sense (see
    // e.g. Brockwell and Davis 1986), and has nothing to do with causality in
    // the sense of causal inference, the Rubin Causal Model, etc.
    static bool is_causal(const Vector &ma_coefficients);
    bool is_causal() const { return is_causal(ma_coefficients()); }

    bool is_stationary() const { return is_causal() && is_invertible(); }

    // The first nlags of the autocovariance function.
    // Args:
    //   nlags:  The desired number of lags.  Must be non-negative.
    // Returns:
    //   A vector of size nlags + 1.  The first element is the variance of the
    //   stationary distribution.  Element k > 0 is the stationary covariance
    //   between Y[t] and Y[t + k].
    Vector autocovariance(int nlags) const;

    // The first 'nlags' lags of the autocorrelation function.  The return
    // vector is of size nlags+1.  Element 0 is always 1.0.
    Vector acf(int nlags) const {
      Vector acvf = autocovariance(nlags);
      return acvf / acvf[0];
    }

    // Args:
    //   ar_coefficients: The vector of autoregression coefficients.  The
    //     dimension must match the corresponding parameter in the model.
    //   ma_coefficients: The vector of moving average coefficients.  The
    //     dimension must match the corresponding parameter in the model.
    //   sigsq: The variance of the white noise process.  Must be strictly
    //     positive.
    //
    // Returns:
    //   The log likelihood of the data.
    double log_likelihood(const Vector &ar_coefficients,
                          const Vector &ma_coefficients, double sigsq) const;

    // Simulate an ARMA process of the specified length.
    // Args:
    //   length:  The desired number of observations in the simulated series.
    //   rng:  A random number generator used in the simulation.
    // Returns:
    //   A time series of the desired length, simulated from the model.  The
    //   initial value of the series is drawn from the stationary distribution.
    Vector simulate(int length, RNG &rng) const;

    // The expanded AR coefficients are [phi[0], phi[1], ..., phi[dimension-1]],
    // where elements of phi after the phi[p-1] are zero.
    Vector expand_ar_coefficients(const Vector &ar_coefficients,
                                  int dimension) const;
    Vector expanded_ar_coefficients(int dimension) const {
      return expand_ar_coefficients(ar_coefficients(), dimension);
    }

    // The expanded MA coefficients are [theta[0], theta[1], ...,
    // theta[dimension-1]], where element 0 is 1.0, and elements theta[q, ...]
    // are zero.
    Vector expand_ma_coefficients(const Vector &ma_coefficients,
                                  int dimension) const;
    Vector expanded_ma_coefficients(int dimension) const {
      return expand_ma_coefficients(ma_coefficients(), dimension);
    }

    // A causal ARMA model can be written in terms of a pure, infinite MA
    // process, which is a "filter" of white noise.  The ARMA equation is
    // phi(B)y = theta(B) epsilon, where
    //
    //    phi(z) = 1 - phi[1]z - phi[2]z^2 - ... - phi[p]z^p
    //
    // and
    //
    //    theta(z) + 1 + theta[1]z + theta[2]z^2 + ... + theta[q]z^q.
    //
    // We can write the ARMA equation as y = (theta(B) / phi(B)) epsilon.  Here
    // psi(B) = (theta(B) / phi(B)) is the polynomial corresponding to the
    // infinite MA process.  The length of psi is not easy to determine a
    // priori, but the algorithm for finding it can stop when several
    // coefficients are small.
    //
    // The coefficients of psi can be found directly by equating coefficient in
    // psi(z) * phi(z) = theta(z).  This looks like the following...
    //
    // (psi[0] + psi[1] z + psi[2] z^2 + ...) *
    // (1 - phi[1] z - phi[2] z^2 - ... - phi[p]z^p  =
    // 1 + theta[1] z + theta[2] z^2 + ... + theta[q] z^q
    //
    // Multiplying them out gives:
    //   psi[0] = 1
    //  -phi[1] z + psi[1] z = theta[1] z
    //  -phi[2] z^2 -phi[1]*psi[1] z^2 + psi[2] = theta[2] z^2
    //  -phi[3] z^3 - phi[2]*psi[1] z^3 - phi[1]*psi[2] z^3 + psi[3] z^3 =
    //  theta[3] z^3
    // ...
    //
    // This is a triangular system, with each line having only a single unknown.
    Vector filter_coefficients() const;

   private:
    // Returns the dot product of psi and reverse(phi), where psi is the current
    // set of filter coefficients, and phi is the set of AR coefficients.
    double filter_ar_dot_product(const Vector &filter_coefficients) const;

    // Convenience functions for accessing the AR and MA coefficients in the
    // unit-offset framework in which they're written about mathematically.  If
    // n is larger than the index of the largest coefficient then 0 is returned.

    // theta(n) is the n'th MA coefficient.
    double theta(int n) const;

    // phi(n) is the n'th AR coefficient.
    double phi(int n) const;
  };

}  // namespace BOOM

#endif  //  BOOM_ARMA_MODEL_HPP_
