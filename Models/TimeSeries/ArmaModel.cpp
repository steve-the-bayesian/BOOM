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

#include "Models/TimeSeries/ArmaModel.hpp"
#include "Models/StateSpace/Filters/SparseKalmanTools.hpp"
#include "Models/TimeSeries/ArModel.hpp"
#include "cpputil/math_utils.hpp"
#include "distributions.hpp"

namespace BOOM {

  namespace {
    using ASSTM = ArmaStateSpaceTransitionMatrix;
    using ASSVM = ArmaStateSpaceVarianceMatrix;
  }  // namespace

  ASSTM::ArmaStateSpaceTransitionMatrix(const Vector &expanded_ar_coefficients)
      : expanded_phi_(expanded_ar_coefficients) {}

  void ASSTM::multiply(VectorView lhs, const ConstVectorView &rhs) const {
    if (lhs.size() != ncol()) {
      report_error("Wrong sized 'lhs' argument.");
    }
    if (rhs.size() != nrow()) {
      report_error("Wrong sized 'rhs' argument.");
    }
    int dim = expanded_phi_.size();
    for (int i = 0; i < dim; ++i) {
      lhs[i] = expanded_phi_[i] * rhs[0] + (i + 1 < dim ? rhs[i + 1] : 0);
    }
  }

  // lhs += this * rhs
  void ASSTM::multiply_and_add(VectorView lhs,
                               const ConstVectorView &rhs) const {
    if (lhs.size() != ncol()) {
      report_error("Wrong sized 'lhs' argument.");
    }
    if (rhs.size() != nrow()) {
      report_error("Wrong sized 'rhs' argument.");
    }
    int dim = expanded_phi_.size();
    for (int i = 0; i < dim; ++i) {
      lhs[i] += expanded_phi_[i] * rhs[0] + (i + 1 < dim ? rhs[i + 1] : 0);
    }
  }

  void ASSTM::Tmult(VectorView lhs, const ConstVectorView &rhs) const {
    if (lhs.size() != ncol()) {
      report_error("Wrong sized 'lhs' argument.");
    }
    if (rhs.size() != nrow()) {
      report_error("Wrong sized 'rhs' argument.");
    }
    lhs[0] = expanded_phi_.dot(rhs);
    VectorView(lhs, 1, ncol() - 1) = ConstVectorView(rhs, 0, ncol() - 1);
  }

  void ASSTM::multiply_inplace(VectorView x) const {
    if (x.size() != nrow()) {
      report_error("Wrong sized argument.");
    }
    double first_element = x[0];
    int dim = x.size();
    for (int i = 0; i < dim; ++i) {
      x[i] = expanded_phi_[i] * first_element + (i + 1 < dim ? x[i + 1] : 0);
    }
  }

  SpdMatrix ASSTM::inner() const {
    SpdMatrix ans(ncol(), 0.0);
    ans.diag() = 1.0;
    ConstVectorView shifted_coefficients(
        expanded_phi_, 0, expanded_phi_.size() - 1);
    VectorView(ans.row(0), 1) = shifted_coefficients;
    VectorView(ans.col(0), 1) = shifted_coefficients;
    ans(0, 0) = expanded_phi_.dot(expanded_phi_);
    return ans;
  }

  // This matrix is the transpose of the 
  SpdMatrix ASSTM::inner(const ConstVectorView &weights) const {
    SpdMatrix ans(ncol(), 0.0);
    const ConstVectorView truncated_weights(weights, 0, weights.size() - 1);
    const ConstVectorView truncated_coefficients(
        expanded_phi_, 0, expanded_phi_.size() - 1);
    VectorView(ans.diag(), 1) = truncated_weights;
    VectorView(ans.row(0), 1) = truncated_weights * truncated_coefficients;
    double qform = 0;
    for (int i = 0; i < expanded_phi_.size(); ++i) {
      qform += square(expanded_phi_[i]) * weights[i];
    }
    ans(0, 0) = qform;
    ans.col(0) = ans.row(0);
    return ans;
  }
  
  void ASSTM::add_to_block(SubMatrix block) const {
    if (block.nrow() != nrow() || block.ncol() != ncol()) {
      report_error("Wrong sized argument.");
    }
    block.col(0) += expanded_phi_;
    block.superdiag(1) += 1.0;
  }

  Matrix ASSTM::dense() const {
    Matrix ans(nrow(), ncol(), 0.0);
    ans.col(0) = expanded_phi_;
    ans.superdiag(1) = 1.0;
    return ans;
  }
  //======================================================================

  ASSVM::ArmaStateSpaceVarianceMatrix(const Vector &expanded_ma_coefficients,
                                      double sigsq)
      : theta_(expanded_ma_coefficients), sigsq_(sigsq) {}

  void ASSVM::multiply(VectorView lhs, const ConstVectorView &rhs) const {
    double scale_factor = sigsq_ * theta_.dot(rhs);
    lhs = theta_;
    lhs *= scale_factor;
  }

  void ASSVM::multiply_and_add(VectorView lhs,
                               const ConstVectorView &rhs) const {
    double scale_factor = sigsq_ * theta_.dot(rhs);
    lhs += theta_ * scale_factor;
  }

  void ASSVM::multiply_inplace(VectorView x) const {
    x = theta_ * (theta_.dot(x) * sigsq_);
  }

  SpdMatrix ASSVM::inner() const {
    SpdMatrix ans(ncol(), 1.0);
    matrix_multiply_inplace(SubMatrix(ans));
    matrix_transpose_premultiply_inplace(SubMatrix(ans));
    return ans;
  }

  // The matrix is theta * theta' * sigsq, so the inner product is
  // theta theta' weights theta theta' * sigsq * sigsq.
  //
  // The theta' * weights * 
  SpdMatrix ASSVM::inner(const ConstVectorView &weights) const {
    SpdMatrix ans(nrow(), 0.0);
    double inner = 0.0;
    for (int i = 0; i < weights.size(); ++i) {
      inner += weights[i] * square(theta_[i]);
    }
    inner *= square(sigsq_);
    ans.add_outer(theta_, inner);
    return ans;
  }
  
  void ASSVM::add_to_block(SubMatrix block) const {
    block += dense();
  }

  Matrix ASSVM::dense() const {
    SpdMatrix ans(nrow(), 0.0);
    ans.add_outer(theta_, sigsq_);
    return std::move(ans);
  }

  //======================================================================
  ArmaModel::ArmaModel(int p, int q) {
    if (p < 0 || q < 0) {
      report_error("ARMA models do not admit negative indices.");
    }
    if (p + q == 0) {
      report_error("At least one of p or q must be positive.");
    }
    NEW(GlmCoefs, ar_coefficients)(p);
    NEW(VectorParams, ma_coefficients)(q);
    NEW(UnivParams, residual_variance)(1.0);
    set_params(ar_coefficients, ma_coefficients, residual_variance);
  }

  ArmaModel::ArmaModel(const Ptr<GlmCoefs> &ar_coefficients,
                       const Ptr<VectorParams> &ma_coefficients,
                       const Ptr<UnivParams> &residual_variance)
      : ParamPolicy(ar_coefficients, ma_coefficients, residual_variance) {}

  const Vector &ArmaModel::ar_coefficients() const {
    return prm1_ref().value();
  }
  const Vector &ArmaModel::ma_coefficients() const {
    return prm2_ref().value();
  }
  double ArmaModel::sigsq() const { return prm3_ref().value(); }

  bool ArmaModel::is_invertible(const Vector &ar_coefficients) {
    return ArModel::check_stationary(ar_coefficients);
  }

  bool ArmaModel::is_causal(const Vector &theta) {
    // The MA polynomial is 1 + theta1 * z + ... + thetaq * z^q, whereas the AR
    // polynomial is 1 -phi1 * z - ... - phi_p * z^p.  The conditions on
    // stationarity are the same for both polynomials: all roots must be outside
    // the unit circle.  The MA polynomial is the same as the AR polynomial
    // evaluated at -theta.
    return ArModel::check_stationary(-theta);
  }

  Vector ArmaModel::autocovariance(int number_of_lags) const {
    Vector filter_coefficients = this->filter_coefficients();
    Vector ans(number_of_lags + 1);
    for (int lag = 0; lag <= number_of_lags; ++lag) {
      int n = filter_coefficients.size() - lag;
      if (n < 0) {
        VectorView(ans, lag) = 0.0;
        break;
      } else {
        const ConstVectorView psi(filter_coefficients, 0, n);
        const ConstVectorView lag_psi(filter_coefficients, lag, n);
        ans[lag] = psi.dot(lag_psi);
      }
    }
    return ans * sigsq();
  }

  double ArmaModel::log_likelihood(const Vector &ar_coefficients,
                                   const Vector &ma_coefficients,
                                   double sigsq) const {
    if (ar_coefficients.size() != ar_dimension()) {
      report_error("ar_coefficients are the wrong size.");
    }
    if (ma_coefficients.size() != ma_dimension()) {
      report_error("ma_coefficients are the wrong size.");
    }
    if (sigsq <= 0) {
      return negative_infinity();
    }
    double ans = 0;

    int state_dimension = std::max(ar_dimension(), ma_dimension() + 1);

    SparseVector Z(state_dimension);
    Z[0] = 1.0;

    BlockDiagonalMatrix transition_matrix;
    transition_matrix.add_block(new ArmaStateSpaceTransitionMatrix(
        expand_ar_coefficients(ar_coefficients, state_dimension)));

    BlockDiagonalMatrix state_variance_matrix;
    state_variance_matrix.add_block(new ArmaStateSpaceVarianceMatrix(
        expand_ma_coefficients(ma_coefficients, state_dimension), sigsq));

    // Initial distribution of the state at time 0.  It might be better to use
    // the stationary or prior distribution here.
    Vector a(state_dimension, 0.0);
    SpdMatrix P(state_dimension, 0.0);
    P.diag() = 10 * sigsq;

    // Define some working variables needed by the Kalman filter.
    Vector kalman_gain(state_dimension);
    double forecast_error_variance = 0;
    double forecast_error = 0;
    bool missing = false;

    const std::vector<Ptr<DoubleData>> &data(dat());
    int time_dimension = data.size();
    for (int t = 1; t < time_dimension; ++t) {
      ans += sparse_scalar_kalman_update(
          data[t]->value(), a, P, kalman_gain, forecast_error_variance,
          forecast_error, missing, Z, 0, transition_matrix,
          state_variance_matrix);
    }
    return ans;
  }

  Vector ArmaModel::expand_ar_coefficients(const Vector &ar_coefficients,
                                           int dimension) const {
    if (dimension < ar_coefficients.size()) {
      report_error("Dimension must be larger than the vector being expanded.");
    }
    Vector ans(dimension, 0.0);
    VectorView(ans, 0, ar_dimension()) = ar_coefficients;
    return ans;
  }

  Vector ArmaModel::expand_ma_coefficients(const Vector &ma_coefficients,
                                           int dimension) const {
    if (dimension < ma_coefficients.size() + 1) {
      report_error(
          "Dimension must be at least one more than the size of the "
          "MA coefficients");
    }
    Vector ans(dimension, 0.0);
    ans[0] = 1.0;
    VectorView(ans, 1, ma_dimension()) = ma_coefficients;
    return ans;
  }

  Vector ArmaModel::simulate(int length, RNG &rng) const {
    if (length < 0) {
      report_error("Length must be non-negative.");
    } else if (length == 0) {
      return Vector(0);
    }
    // TODO: replace this with a draw from the stationary distribution once
    // you've got the ACF worked out.
    int burn = 50;
    Vector ans(length + burn);

    int state_dimension = std::max(ar_dimension(), ma_dimension() + 1);
    Vector state(state_dimension);
    double white_noise_sd = sigma();
    for (int i = 0; i < state_dimension; ++i) {
      state[i] = rnorm_mt(rng, 0, white_noise_sd);
    }

    BlockDiagonalMatrix transition_matrix;
    transition_matrix.add_block(new ArmaStateSpaceTransitionMatrix(
        expanded_ar_coefficients(state_dimension)));

    BlockDiagonalMatrix state_variance_matrix;
    state_variance_matrix.add_block(new ArmaStateSpaceVarianceMatrix(
        expanded_ma_coefficients(state_dimension), sigsq()));

    Vector R = expanded_ma_coefficients(state_dimension);
    for (int i = 0; i < ans.size(); ++i) {
      state = transition_matrix * state + rnorm_mt(rng, 0, white_noise_sd) * R;
      ans[i] = state[0];
    }
    return Vector(VectorView(ans, burn, length));
  }

  Vector ArmaModel::filter_coefficients() const {
    if (!is_invertible()) {
      report_error(
          "Filter coefficients are not meaningful because the model is "
          "not invertible.  A root of the AR polynomial lies inside the "
          "unit circle.");
    }
    Vector filter_coefficients(2);
    filter_coefficients[0] = 1.0;
    filter_coefficients[1] = theta(1) + phi(1);
    bool done = false;
    while (!done) {
      double coefficient = theta(filter_coefficients.size()) +
                           filter_ar_dot_product(filter_coefficients);
      filter_coefficients.push_back(coefficient);
      done = filter_coefficients.size() > ar_dimension() &&
             const_tail(filter_coefficients, ar_dimension()).abs_norm() < 1e-6;
    }
    return filter_coefficients;
  }

  // Args:
  //   filter_coefficients: The current set of filter coefficients for the
  //     infinite "polynomial" describing the process as an MA(infinity) model.
  //     The coefficients are psi[0], psi[1], ... psi[n], with psi[0] = 1.
  //
  // Returns:
  //   Let phi = phi[1], ..., phi[p], with phi[0] = 1.  The returned dot product
  //   is phi[n] * psi[0] + phi[n-1] * psi[1] + ... + phi[0] * psi[n].
  //
  //   The value of n is the power of z in the polynomial.....
  double ArmaModel::filter_ar_dot_product(
      const Vector &filter_coefficients) const {
    if (filter_coefficients.size() == 0) {
      return 0;
    }
    if (filter_coefficients.size() < ar_dimension()) {
      ConstVectorView phi(ar_coefficients(), 0, filter_coefficients.size() - 1);
      return filter_coefficients.dot(rev(phi));
    } else {
      return const_tail(filter_coefficients, ar_dimension())
          .dot(rev(ar_coefficients()));
    }
  }

  double ArmaModel::theta(int n) const {
    double ans = negative_infinity();
    if (n < 0) {
      report_error("Negative MA index is not allowed.");
    } else if (n == 0) {
      ans = 1;
    } else if (n > ma_dimension()) {
      ans = 0;
    } else {
      ans = ma_coefficients()[n - 1];
    }
    return ans;
  }

  double ArmaModel::phi(int n) const {
    double ans = negative_infinity();
    if (n <= 0) {
      report_error("AR index must be positive.");
    } else if (n > ar_dimension()) {
      ans = 0;
    } else {
      ans = ar_coefficients()[n - 1];
    }
    return ans;
  }

}  // namespace BOOM
