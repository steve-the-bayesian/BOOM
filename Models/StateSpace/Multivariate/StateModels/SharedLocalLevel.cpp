/*
  Copyright (C) 2005-2021 Steven L. Scott

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

#include "Models/StateSpace/Multivariate/StateModels/SharedLocalLevel.hpp"
#include "distributions.hpp"

namespace BOOM {
  namespace {
    using SLLSMB = SharedLocalLevelStateModelBase;
  }

  SLLSMB::SharedLocalLevelStateModelBase(
      int number_of_factors, int nseries)
      : initial_state_mean_(0),
        initial_state_variance_(0),
        initial_state_variance_cholesky_(0, 0)
  {
    for (int i = 0; i < number_of_factors; ++i) {
      innovation_models_.push_back(new ZeroMeanGaussianModel);
    }
    initialize_model_matrices();
  }

  SLLSMB::SharedLocalLevelStateModelBase(const SLLSMB &rhs) {
    operator=(rhs);
  }

  SLLSMB & SLLSMB::operator=(const SLLSMB &rhs) {
    if (&rhs != this) {
      initial_state_mean_ = rhs.initial_state_mean_;
      initial_state_variance_ = rhs.initial_state_variance_;
      initial_state_variance_cholesky_ = rhs.initial_state_variance_cholesky_;
      innovation_models_.clear();
      for (int i = 0; i < rhs.innovation_models_.size(); ++i) {
        innovation_models_.push_back(rhs.innovation_models_[i]->clone());
      }
      initialize_model_matrices();
    }
    return *this;
  }

  SLLSMB::SharedLocalLevelStateModelBase(SLLSMB &&rhs)
      : innovation_models_(std::move(rhs.innovation_models_)),
        state_transition_matrix_(std::move(rhs.state_transition_matrix_)),
        state_variance_matrix_(std::move(rhs.state_variance_matrix_)),
        initial_state_mean_(std::move(rhs.initial_state_mean_)),
        initial_state_variance_(std::move(rhs.initial_state_variance_)),
        initial_state_variance_cholesky_(std::move(
            rhs.initial_state_variance_cholesky_))
  {}

  SLLSMB & SLLSMB::operator=(SLLSMB &&rhs) {
    if (&rhs != this) {
      innovation_models_ = std::move(rhs.innovation_models_);
      state_transition_matrix_ = std::move(rhs.state_transition_matrix_);
      state_variance_matrix_ = std::move(rhs.state_variance_matrix_);
      initial_state_mean_ = std::move(rhs.initial_state_mean_);
      initial_state_variance_ = std::move(rhs.initial_state_variance_);
      initial_state_variance_cholesky_ = std::move(rhs.initial_state_variance_cholesky_);
    }
    return *this;
  }

  void SLLSMB::clear_state_transition_data() {
    for (int i = 0; i < innovation_models_.size(); ++i) {
      innovation_models_[i]->clear_data();
    }
  }

  void SLLSMB::simulate_state_error(RNG &rng, VectorView eta, int t) const {
    for (int i = 0; i < number_of_factors(); ++i) {
      eta[i] = rnorm_mt(rng, 0, innovation_models_[i]->sd());
    }
  }

  void SLLSMB::simulate_initial_state(RNG &rng, VectorView eta) const {
    if (initial_state_mean_.size() != state_dimension()) {
      report_error("You need to set the mean and variance for "
                   "the initial state.");
    }
    eta = rmvn_mt(rng, initial_state_mean_, initial_state_variance_);
  }

  Vector SLLSMB::initial_state_mean() const {
    if (initial_state_mean_.size() != state_dimension()) {
      report_error("Initial state mean has not been set in "
                   "SharedLocalLevelStateModelBase.");
    }
    return initial_state_mean_;
  }

  void SLLSMB::set_initial_state_mean(const Vector &mean) {
    if (mean.size() != state_dimension()) {
      std::ostringstream err;
      err << "Wrong size argument in set_initial_state_mean. \n"
          << "State dimension is " << state_dimension()
          << " but the proposed mean is " << mean;
      report_error(err.str());
    }
    initial_state_mean_ = mean;
  }

  SpdMatrix SLLSMB::initial_state_variance() const {
    if (initial_state_variance_.nrow() != state_dimension()) {
      report_error("Initial state variance has not been set in "
                   "SharedLocalLevelStateModel.");
    }
    return initial_state_variance_;
  }

  void SLLSMB::set_initial_state_variance(const SpdMatrix &variance) {
    if (variance.nrow() != state_dimension()) {
      report_error("Wrong size argument in set_initial_state_variance.");
    }
    initial_state_variance_ = variance;
    bool ok = true;
    initial_state_variance_cholesky_ = variance.chol(ok);
    if (!ok) {
      report_error("Variance is not positive definite in "
                   "set_initial_state_variance.");
    }
  }

  void SLLSMB::initialize_model_matrices() {

    state_transition_matrix_.reset(new IdentityMatrix(state_dimension()));

    state_variance_matrix_.reset(new DiagonalMatrixParamView);
    for (int i = 0; i < innovation_models_.size(); ++i) {
      state_variance_matrix_->add_variance(innovation_models_[i]->Sigsq_prm());
    }
  }


  void SLLSMB::update_complete_data_sufficient_statistics(
      int t, const ConstVectorView &state_error_mean,
      const ConstSubMatrix &state_error_variance) {
    report_error("update_complete_data_sufficient_statistics "
                 "is not implemented.");
  }

  void SLLSMB::increment_expected_gradient(
      VectorView gradient, int t, const ConstVectorView &state_error_mean,
      const ConstSubMatrix &state_error_variance) {
    report_error("increment_expected_gradient is not implemented.");
  }


  // The logic here is :
  // Y = Z * alpha
  //   = Beta.tranpose() * alpha
  //   = (QR).transpose() * alpha
  //   = R.transpose() * Q.transpose * alpha
  // Thus, if we set Beta = R and pre_multiply alpha by Q.transpose then the
  // constraints will be satisfied.
  //
  // NOTE:  still need to scale by diag(R)
  void SLLSMB::impose_identifiability_constraint() {
    // Matrix Beta = coefficient_model_->Beta();
    // QR BetaQr(Beta);
    // Matrix R = BetaQr.getR();
    // DiagonalMatrix Rdiag(R.diag());
    // const Matrix &Q(BetaQr.getQ());

    // coefficient_model_->set_Beta(BetaQr.getR());
    // SubMatrix state = host_->mutable_full_state_subcomponent(index());
    // Vector workspace(state.nrow());
    // for (int i = 0; i < state.ncol(); ++i) {
    //   workspace = Q.Tmult(state.col(i));
    //   Rdiag.multiply_inplace(workspace);
    //   state.col(i) = workspace;
    // }
    // coefficient_model_->set_Beta(Rdiag.solve(R));
  }

  // Args:
  //   then: The portion of the state vector associated with this object at time
  //     point time_now - 1.
  //   now: The portion of the state vector associated with this object at time
  //     point time_now.
  //   time_now:  The index of the current time point.
  void SLLSMB::observe_state(const ConstVectorView &then,
                            const ConstVectorView &now,
                            int time_now) {
    for (int i = 0; i < innovation_models_.size(); ++i) {
      double diff = now[i] - then[i];
      innovation_models_[i]->suf()->update_raw(diff);
    }

    // Residual y is the residual remaining after the other state components
    // have made their contributions.
    //
    // This logic assumes that
    //   (1) the state of the model has been set,
    //   (2) any missing values have been imputed, and
    //   (3) any other additive effects (e.g. regression effects) have been
    //       subtracted off.
    //     Selector fully_observed(host()->state_dimension(), true);
    const Selector &observed(host()->observed_status(time_now));

    // Subtract off the effect of other state models (or regression models), and
    // add in the effect of this one, so that the only effect present is from
    // this state model and random error.
    //
    // The first "state" calculation below uses the full state vector.  The
    // second uses 'now' which is a subset.
    Vector residual_y =
        host()->adjusted_observation(time_now)
        - (*host()->observation_coefficients(time_now, observed)
           * host()->shared_state(time_now))
        + (*observation_coefficients(time_now, observed)) * now;

    // The concrete child classes will record residual_y in the manner specific
    // to their coefficient models.
    record_observed_data_given_state(residual_y, now, time_now);
  }

  //===========================================================================

  namespace {
    using GSLLSM = GeneralSharedLocalLevelStateModel;
  }  // namespace

  GSLLSM::GeneralSharedLocalLevelStateModel(
      MultivariateStateSpaceModelBase *host,
      int number_of_factors,
      int nseries)
      : SLLSMB(number_of_factors, nseries),
        host_(host) {
    set_param_policy();
    initialize_observation_coefficient_matrix();
    set_observation_coefficients_observer();
  }

  GSLLSM::GeneralSharedLocalLevelStateModel(const GSLLSM &rhs)
      : SharedLocalLevelStateModelBase(rhs) {
    operator=(rhs);
  }

  GSLLSM & GSLLSM::operator=(const GSLLSM &rhs) {
    if (&rhs != this) {
      SLLSMB::operator=(rhs);
      coefficient_model_.reset(rhs.coefficient_model_->clone());
      initialize_observation_coefficient_matrix();
      set_observation_coefficients_observer();
    }
    return *this;
  }

  GSLLSM::GeneralSharedLocalLevelStateModel(GSLLSM &&rhs)
      : SLLSMB(rhs),
        host_(rhs.host_),
        coefficient_model_(std::move(rhs.coefficient_model_)),
        observation_coefficients_(std::move(rhs.observation_coefficients_)),
        empty_(std::move(rhs.empty_))
  {}

  GSLLSM * GSLLSM::clone() const { return new GSLLSM(*this); }

  void GSLLSM::record_observed_data_given_state(const Vector &residual_y,
                                                const ConstVectorView &now,
                                                int time_now) {
    coefficient_model_->suf()->update_raw_data(residual_y, now, 1.0);
  }

  Ptr<SparseMatrixBlock> GSLLSM::observation_coefficients(
      int t, const Selector &observed) const {
    if (observed.nvars() == observed.nvars_possible()) {
      return observation_coefficients_;
    } else if (observed.nvars() == 0) {
      return empty_;
    } else {
      return new DenseMatrix(observed.select_rows(
          observation_coefficients_->dense()));
    }
  }

  void GSLLSM::set_observation_coefficients_observer() {
    std::function<void(void)> observer = [this]() {
      this->sync_observation_coefficients();
    };
    coefficient_model_->Beta_prm()->add_observer(this, observer);
  }

  void GSLLSM::sync_observation_coefficients() {
    observation_coefficients_->set(coefficient_model_->Beta().transpose());
  }

  void GSLLSM::set_param_policy() {
    ParamPolicy::add_model(coefficient_model());
    for (int i = 0; i < state_dimension(); ++i) {
      ParamPolicy::add_model(innovation_model(i));
    }
  }

  void GSLLSM::initialize_observation_coefficient_matrix() {
    // The multivariate regression model is organized as (xdim, ydim).  The 'X'
    // in our case is the state, where we want y = Z * state, so we need the
    // transpose of the coefficient matrix from the regression.
    Matrix Beta = coefficient_model_->Beta() * 0.0;
    Beta.diag() = 1.0;
    observation_coefficients_.reset(new DenseMatrix(Beta.transpose()));
    if (!empty_) {
      empty_.reset(new EmptyMatrix);
    }
  }


  //===========================================================================
  namespace {
    using CindSLLM = ConditionallyIndependentSharedLocalLevelStateModel;
  }  // namespace

  CindSLLM::ConditionallyIndependentSharedLocalLevelStateModel(
      ConditionallyIndependentMultivariateStateSpaceModelBase *host,
      int number_of_factors,
      int nseries)
      : SharedLocalLevelStateModelBase(number_of_factors, nseries),
        host_(host),
        observation_coefficients_(new DenseMatrix(Matrix(
            nseries, number_of_factors))),
        observation_coefficients_current_(false)
  {
    Vector all_ones(number_of_factors, 1.0);
    for (int i = 0; i < nseries; ++i) {
      // The observation coefficents for series i.
      NEW(GlmCoefs, series_observation_coefficients)(all_ones, true);
      raw_observation_coefficients_.push_back(series_observation_coefficients);
      sufficient_statistics_.push_back(new WeightedRegSuf(number_of_factors));
    }
    set_observation_coefficients_observer();
  }

  CindSLLM::ConditionallyIndependentSharedLocalLevelStateModel(
      const CindSLLM &rhs):
      SharedLocalLevelStateModelBase(rhs),
      host_(rhs.host_),
      observation_coefficients_(new DenseMatrix(
          rhs.observation_coefficients_->matrix())),
      observation_coefficients_current_(false)
  {
    for (const auto &el : rhs.raw_observation_coefficients_) {
      raw_observation_coefficients_.push_back(el->clone());
    }

    for (const auto &el : rhs.sufficient_statistics_) {
      sufficient_statistics_.push_back(el->clone());
    }
    set_observation_coefficients_observer();
  }


  CindSLLM * CindSLLM::clone() const {return new CindSLLM(*this);}

  void CindSLLM::clear_data() {
    clear_state_transition_data();
    for (auto &el : sufficient_statistics_) {
      el->clear();
    }
  }

  int CindSLLM::nseries() const {
    return host_->nseries();
  }

  Ptr<SparseMatrixBlock> CindSLLM::observation_coefficients(
      int time, const Selector &observed) const {
    ensure_observation_coefficients_current();
    if (observed.nvars_excluded() == 0) {
      return observation_coefficients_;
    } else {
      return new DenseMatrix(observed.select_rows(
          observation_coefficients_->matrix()));
    }
  }

  void CindSLLM::ensure_observation_coefficients_current() const {
    if (!observation_coefficients_current_) {
      Matrix coefficients(nseries(), number_of_factors());
      for (int i = 0; i < nseries(); ++i) {
        coefficients.row(i) = raw_observation_coefficients_[i]->Beta();
      }
      observation_coefficients_->set(coefficients);
      observation_coefficients_current_ = true;
    }
  }

  void CindSLLM::record_observed_data_given_state(const Vector &residual_y,
                                                  const ConstVectorView &now,
                                                  int time_now) {
    const Selector &observed(host_->observed_status(time_now));
    for (int i = 0; i < residual_y.size(); ++i) {
      int I = observed.dense_index(i);
      sufficient_statistics_[I]->add_data(now, residual_y[i], 1.0);
    }
  }

  void CindSLLM::set_observation_coefficients_observer() {
    for (int i = 0; i < raw_observation_coefficients_.size(); ++i) {
      raw_observation_coefficients_[i]->add_observer(
          this,
          [this]() {
            this->observation_coefficients_current_ = false;
          });
    }
  }

}  // namespace BOOM
