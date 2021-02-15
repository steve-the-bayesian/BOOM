/*
  Copyright (C) 2005-2020 Steven L. Scott

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

#include <future>

#include "Models/Impute/MvRegCopulaDataImputer.hpp"
#include "Models/PosteriorSamplers/MultinomialDirichletSampler.hpp"
#include "Models/Glm/PosteriorSamplers/MultivariateRegressionSampler.hpp"
#include "distributions.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  namespace {
    using CICM = ConditionallyIndependentCategoryModel;

    void check_for_nan(const Vector &v) {
      for (int i = 0; i < v.size(); ++i) {
        if (std::isnan(v[i])) {
          report_error("Found a NaN where it shouldn't exist.");
        }
      }
    }

  }  // namespace
  //===========================================================================
  namespace Imputer {
    CompleteData::CompleteData(const Ptr<MvRegData> &observed)
        : observed_data_(observed),
          y_true_(observed->y()),
          y_numeric_(observed->y())
    {}

    CompleteData *CompleteData::clone() const {
      return new CompleteData(*this);
    }

    std::ostream &CompleteData::display(std::ostream &out) const {
      out << *observed_data_ << "\n"
          << "true: " << y_true_ << "\n"
          << "numeric: " << y_numeric_ << std::endl;
      return out;
    }

  }  // namespace Imputer
  //===========================================================================
  ErrorCorrectionModel::ErrorCorrectionModel(const Vector &atoms)
      : atoms_(atoms),
        marginal_of_true_data_(new MultinomialModel(atoms.size() + 1)),
        joint_distribution_(atoms.size() + 1, atoms.size() + 2),
        observed_log_probability_table_(atoms.size() + 2),
        workspace_is_current_(false)
  {
    for (int i = 0; i <= atoms.size(); ++i) {
      NEW(MultinomialModel, conditional_model)(atoms.size() + 2);
      conditional_observed_given_true_.push_back(conditional_model);
    }
    set_observers();
  }

  ErrorCorrectionModel::ErrorCorrectionModel(const ErrorCorrectionModel &rhs)
      : atoms_(rhs.atoms_),
        marginal_of_true_data_(rhs.marginal_of_true_data_->clone()),
        joint_distribution_(rhs.joint_distribution_),
        observed_log_probability_table_(rhs.observed_log_probability_table_),
        workspace_is_current_(false),
        wsp_(rhs.wsp_)
  {
    for (int i = 0; i < rhs.conditional_observed_given_true_.size(); ++i) {
      conditional_observed_given_true_.push_back(
          rhs.conditional_observed_given_true_[i]->clone());
    }
    set_observers();
  }

  ErrorCorrectionModel *ErrorCorrectionModel::clone() const {
    return new ErrorCorrectionModel(*this);
  }

  double ErrorCorrectionModel::logp(double y) const {
    ensure_workspace_current();
    double ans = observed_log_probability_table_[category_map(y)];
    if (std::isnan(ans)) {
      report_error("Found a NaN in logp.");
    }
    return ans;
  }

  // If there are 3 atoms, this function returns 0, 1, or 2 if y is one of the
  // atomic values.  Otherwise it returns 3 if y is not NaN, and 4 if it is NaN.
  int ErrorCorrectionModel::category_map(double y) const {
    if (std::isnan(y)) {
      return observed_log_probability_table_.size() - 1;
    } else {
      for (int i = 0; i < atoms_.size(); ++i) {
        if (fabs(y - atoms_[i]) < 1e-6) {
          return i;
        }
      }
    }
    return observed_log_probability_table_.size() - 2;
  }

  void ErrorCorrectionModel::ensure_workspace_current() const {
    if (workspace_is_current_) {
      return;
    }

    int true_dim = atoms_.size() + 1;
    int obs_dim = true_dim + 1;
    for (int truth = 0; truth < true_dim; ++truth) {
      for (int obs = 0; obs < obs_dim; ++obs) {
        joint_distribution_(truth, obs) =
            marginal_of_true_data_->pi(truth)
            * conditional_observed_given_true_[truth]->pi(obs);
      }
    }
    observed_log_probability_table_ = log(joint_distribution_.col_sums());
    check_for_nan(observed_log_probability_table_);
    workspace_is_current_ = true;
  }

  void ErrorCorrectionModel::set_conjugate_prior_for_true_categories(
      const Vector &prior_counts) {
    marginal_of_true_data_->clear_methods();
    NEW(ConstrainedMultinomialDirichletSampler, sampler)(
        marginal_of_true_data_.get(), prior_counts);
    marginal_of_true_data_->set_method(sampler);
  }

  void ErrorCorrectionModel::set_conjugate_prior_for_observation_categories(
      const Matrix &prior_counts) {
    for (int i = 0; i < conditional_observed_given_true_.size(); ++i) {
      conditional_observed_given_true_[i]->clear_methods();
      NEW(ConstrainedMultinomialDirichletSampler, sampler)(
          conditional_observed_given_true_[i].get(),
          prior_counts.row(i));
      conditional_observed_given_true_[i]->set_method(sampler);
    }
  }

  void ErrorCorrectionModel::sample_posterior() {
    marginal_of_true_data_->sample_posterior();
    for (size_t i = 0; i < conditional_observed_given_true_.size(); ++i) {
      conditional_observed_given_true_[i]->sample_posterior();
    }
  }

  double ErrorCorrectionModel::logpri() const {
    double ans = marginal_of_true_data_->logpri();
    for (const auto &el : conditional_observed_given_true_) {
      ans += el->logpri();
    }
    return ans;
  }

  void ErrorCorrectionModel::clear_data() {
    marginal_of_true_data_->clear_data();
    for (int i = 0; i < conditional_observed_given_true_.size(); ++i) {
      conditional_observed_given_true_[i]->clear_data();
    }
  }

  int ErrorCorrectionModel::impute_atom(double observed, RNG &rng,
                                        bool update) {
    ensure_workspace_current();
    int observed_atom = category_map(observed);
    wsp_ = joint_distribution_.col(observed_atom);
    wsp_.normalize_prob();
    int true_atom = rmulti_mt(rng, wsp_);
    if (update) {
      marginal_of_true_data_->suf()->update_raw(true_atom);
      conditional_observed_given_true_[true_atom]->suf()->update_raw(observed_atom);
    }
    return true_atom;
  }

  double ErrorCorrectionModel::true_value(
      int true_atom, double observed_value) const {
    if (true_atom < atoms_.size()) {
      return atoms_[true_atom];
    } else {
      if (category_map(observed_value) == atoms_.size()) {
        return observed_value;
      } else {
        return std::numeric_limits<double>::quiet_NaN();
      }
    }
  }

  // When the true value is atomic the numeric value is NaN.
  double ErrorCorrectionModel::numeric_value(int true_atom,
                                             double observed_value) const {
    if (true_atom == atoms_.size() && category_map(observed_value) == atoms_.size()) {
      return observed_value;
    } else {
      return std::numeric_limits<double>::quiet_NaN();
    }
  }

  Matrix ErrorCorrectionModel::atom_error_probs() const {
    Matrix ans(number_of_atoms() + 1, number_of_atoms() + 2);
    for (int i = 0; i < number_of_atoms(); ++i) {
      ans.row(i) = conditional_observed_given_true_[i]->pi();
    }
    ans.last_row() = conditional_observed_given_true_.back()->pi();
    return ans;
  }

  void ErrorCorrectionModel::set_atom_error_probs(const Matrix &probs) {
    for (int i = 0; i < conditional_observed_given_true_.size(); ++i) {
      conditional_observed_given_true_[i]->set_pi(probs.row(i));
    }
  }

  void ErrorCorrectionModel::combine_sufficient_statistics(
      const ErrorCorrectionModel &other) {
    marginal_of_true_data_->suf()->combine(
        other.marginal_of_true_data_->suf());
    for (size_t m = 0; m < conditional_observed_given_true_.size(); ++m) {
      conditional_observed_given_true_[m]->suf()->combine(
          other.conditional_observed_given_true_[m]->suf());
    }
  }

  void ErrorCorrectionModel::copy_parameters(
      const ErrorCorrectionModel &other) {
    set_atom_probs(other.atom_probs());
    set_atom_error_probs(other.atom_error_probs());

  }

  void ErrorCorrectionModel::set_observers() {
    marginal_of_true_data_->Pi_prm()->add_observer(
        [this]() {
          this->workspace_is_current_ = false;
        });

    for (int i = 0; i < conditional_observed_given_true_.size(); ++i) {
      conditional_observed_given_true_[i]->Pi_prm()->add_observer(
          [this]() {
            this->workspace_is_current_ = false;
          });
    }
  }

  //===========================================================================
  CICM::ConditionallyIndependentCategoryModel(const std::vector<Vector> &atoms)
  {
    for (int i = 0; i < atoms.size(); ++i) {
      NEW(ErrorCorrectionModel, model)(atoms[i]);
      observed_data_models_.push_back(model);
    }
  }

  CICM::ConditionallyIndependentCategoryModel(const CICM &rhs) {
    for (int i = 0; i < rhs.ydim(); ++i) {
      observed_data_models_.push_back(rhs.observed_data_models_[i]->clone());
    }
  }

  double CICM::logp(const Vector &observed) const {
    double ans = 0;
    for (int i = 0; i < observed.size(); ++i) {
      ans += observed_data_models_[i]->logp(observed[i]);
      if (std::isnan(ans)) {
        report_error("logp produced a NaN");
      }
    }
    return ans;
  }

  void CICM::clear_data() {
    DataPolicy::clear_data();
    for (size_t i = 0; i < observed_data_models_.size(); ++i) {
      observed_data_models_[i]->clear_data();
    }
  }

  void CICM::impute_atoms(Imputer::CompleteData &data, RNG &rng,
                          bool update_complete_data_suf) {
    const Vector &observed(data.y_observed());
    for (int i = 0; i < observed.size(); ++i) {
      // True atom is the atom responsible for the true value.
      int true_atom = observed_data_models_[i]->impute_atom(
          observed[i], rng, update_complete_data_suf);
      data.set_y_true(i, observed_data_models_[i]->true_value(
          true_atom, observed[i]));
      data.set_y_numeric(i, observed_data_models_[i]->numeric_value(
          true_atom, observed[i]));
    }
  }

  void CICM::combine_sufficient_statistics(const CICM &other) {
    for (int i = 0; i < ydim(); ++i) {
      observed_data_models_[i]->combine_sufficient_statistics(other.model(i));
    }
  }

  void CICM::copy_parameters(const CICM &other) {
    for (int i = 0; i < ydim(); ++i) {
      observed_data_models_[i]->copy_parameters(other.model(i));
    }
  }

  //===========================================================================
  MvRegCopulaDataImputer::MvRegCopulaDataImputer(
      int num_clusters, const std::vector<Vector> &atoms, int xdim, RNG &seeding_rng)
      : cluster_mixing_distribution_(new MultinomialModel(num_clusters)),
        complete_data_model_(new MultivariateRegressionModel(xdim, atoms.size())),
        rng_(seed_rng(seeding_rng)),
        swept_sigma_(SpdMatrix(0)),
        swept_sigma_current_(false),
        worker_id_(-1)
  {
    for (int s = 0; s < num_clusters; ++s) {
      NEW(ConditionallyIndependentCategoryModel, component)(atoms);
      cluster_mixture_components_.push_back(component);
    }
    set_observers();
  }

  MvRegCopulaDataImputer::MvRegCopulaDataImputer(const MvRegCopulaDataImputer &rhs)
      : cluster_mixing_distribution_(rhs.cluster_mixing_distribution_->clone()),
        complete_data_model_(rhs.complete_data_model_->clone()),
        empirical_distributions_(rhs.empirical_distributions_),
        swept_sigma_(SpdMatrix(1)),
        swept_sigma_current_(false)
  {
    rng_.seed(seed_rng(rhs.rng_));
    for (int i = 0; i < rhs.cluster_mixture_components_.size(); ++i) {
      cluster_mixture_components_.push_back(
          rhs.cluster_mixture_components_[i]->clone());
    }
    set_observers();
  }

  MvRegCopulaDataImputer::~MvRegCopulaDataImputer() {
    shut_down_worker_pool();
  }

  void MvRegCopulaDataImputer::initialize_empirical_distributions(int ydim) {
    Vector probs(99);
    for (int i = 0; i < probs.size(); ++i) {
      probs[i] = (i + 1.0) / 100.0;
    }
    for (int i = 0; i < ydim; ++i) {
      empirical_distributions_.push_back(IQagent(probs, uint(1e+6)));
    }
  }

  //---------------------------------------------------------------------------
  MvRegCopulaDataImputer *MvRegCopulaDataImputer::clone() const {
    return new MvRegCopulaDataImputer(*this);
  }
  //---------------------------------------------------------------------------

  std::vector<Vector> MvRegCopulaDataImputer::atoms() const {
    std::vector<Vector> ans;
    for (int i = 0; i < ydim(); ++i) {
      ans.push_back(cluster_mixture_components_[0]->model(i).atoms());
    }
    return ans;
  }
  //---------------------------------------------------------------------------
  void MvRegCopulaDataImputer::clear_data() {
    DataPolicy::clear_data();
    complete_data_.clear();
    clear_client_data();
  }
  //---------------------------------------------------------------------------
  void MvRegCopulaDataImputer::clear_client_data() {
    cluster_mixing_distribution_->clear_data();
    for (size_t s = 0; s < cluster_mixture_components_.size(); ++s) {
      cluster_mixture_components_[s]->clear_data();
    }
    complete_data_model_->clear_data();
  }
  //---------------------------------------------------------------------------
  void MvRegCopulaDataImputer::add_data(const Ptr<MvRegData> &data) {
    NEW(Imputer::CompleteData, complete)(data);
    DataPolicy::add_data(data);

    if (empirical_distributions_.empty()) {
      initialize_empirical_distributions(data->y().size());
    }

    const Vector &y(data->y());
    for (size_t i = 0; i < y.size(); ++i) {
      const auto &model = cluster_mixture_components_[0]->model(i);
      int atom = model.category_map(y[i]);
      if (atom == model.number_of_atoms()) {
        empirical_distributions_[i].add(y[i]);
      }
    }

    complete_data_.push_back(complete);
  }
  //---------------------------------------------------------------------------
  void MvRegCopulaDataImputer::remove_data(const Ptr<Data> &data) {
    DataPolicy::remove_data(data);
    for (auto it = complete_data_.begin(); it != complete_data_.end(); ++it) {
      const Ptr<Imputer::CompleteData> &data_point(*it);
      if (data_point->observed_data() == data.get()) {
        complete_data_.erase(it);
      }
    }
  }
  //---------------------------------------------------------------------------
  double MvRegCopulaDataImputer::logpri() const {
    return negative_infinity();
  }

  //---------------------------------------------------------------------------
  int MvRegCopulaDataImputer::impute_cluster(
      Ptr<Imputer::CompleteData> &data, RNG &rng) const {
    int S = cluster_mixture_components_.size();
    wsp_ = cluster_mixing_distribution_->logpi();
    for (uint s = 0; s < S; ++s) {
      wsp_[s] += cluster_mixture_components_[s]->logp(data->y_observed());
    }
    wsp_.normalize_logprob();
    int component = rmulti_mt(rng, wsp_);
    return component;
  }
  //---------------------------------------------------------------------------
  int MvRegCopulaDataImputer::impute_cluster(
      Ptr<Imputer::CompleteData> &data,
      RNG &rng,
      bool update_complete_data_suf) {

    int component = impute_cluster(data, rng);
    if (update_complete_data_suf) {
      cluster_mixing_distribution_->suf()->update_raw(component);
    }
    return component;
  }

  //---------------------------------------------------------------------------
  void MvRegCopulaDataImputer::impute_row(Ptr<Imputer::CompleteData> &data,
                                          RNG &rng,
                                          bool update_complete_data_suf) {
    ensure_swept_sigma_current();
    int component = impute_cluster(data, rng, update_complete_data_suf);

    // Fill y_true and y_numeric with values, which might include missing
    // values.  If y_true is an atomic value then y_numeric will be missing.  If
    // y_true is non-atomic and missing then both y_numeric and y_true will be
    // missing.
    cluster_mixture_components_[component]->impute_atoms(
        *data, rng, update_complete_data_suf);

    // The remainder of this function fills in the missing numeric values.

    // Determine which numeric values need to be imputed.  As of this point the
    // numeric values have not been transformed to normality.
    Vector imputed_numeric = data->y_numeric();
    Selector observed(imputed_numeric.size(), true);
    for (int i = 0; i < imputed_numeric.size(); ++i) {
      if (std::isnan(imputed_numeric[i])) {
        // Can't transform to normality.
        observed.drop(i);
      } else {
        // imputed_numeric[i] = log1p(imputed_numeric[i]);
        // Transform to normality.
         double uniform = empirical_distributions_[i].cdf(imputed_numeric[i]);
         double shrinkage = .999;
         uniform = shrinkage * uniform + (1 - shrinkage) / 2.0;
         if (uniform <= 0.0 || uniform >= 1.0) {
           report_error("Need to shrink the extremes.");
         }
         imputed_numeric[i] = qnorm(uniform);
      }
    }

    // Impute those numeric values that need imputing.
    if (observed.nvars() < observed.nvars_possible()) {
      Vector mean = complete_data_model_->predict(data->x());
      if (observed.nvars() == 0) {
        imputed_numeric = rmvn_mt(rng, mean, complete_data_model_->Sigma());
      } else {
        swept_sigma_.SWP(observed);
        Vector conditional_mean = swept_sigma_.conditional_mean(
            observed.select(imputed_numeric), mean);
        Vector imputed_values = rmvn_mt(rng, conditional_mean, swept_sigma_.residual_variance());
        observed.fill_missing_elements(imputed_numeric, imputed_values);
      }

      // Transform imputed data back to observed scale.
      Vector y_true = data->y_true();
      for (int i = 0; i < imputed_numeric.size(); ++i) {
        if (std::isnan(y_true[i])) {
          y_true[i] = empirical_distributions_[i].quantile(
              pnorm(imputed_numeric[i]));
          // y_true[i] = expm1(imputed_numeric[i]);
        }
      }
      data->set_y_numeric(imputed_numeric);
      data->set_y_true(y_true);
    } else {
      // Handle the fully observed case.
      data->set_y_numeric(imputed_numeric);
    }

    if (update_complete_data_suf) {
      check_for_nan(data->y_numeric());
      complete_data_model_->suf()->update_raw_data(
          data->y_numeric(), data->x(), 1.0);
    }

  }

  //---------------------------------------------------------------------------
  void MvRegCopulaDataImputer::set_default_priors() {
    set_default_regression_prior();
    set_default_prior_for_mixing_weights();
    for (size_t s = 0; s < cluster_mixture_components_.size(); ++s) {
      Ptr<ConditionallyIndependentCategoryModel> component =
          cluster_mixture_components_[s];
      for (int j = 0; j < component->ydim(); ++j) {
        Ptr<ErrorCorrectionModel> model = component->mutable_model(j);
        int num_atoms = model->number_of_atoms();
        Vector truth_prior_counts(num_atoms + 1, .1 / num_atoms);
        truth_prior_counts.back() = .9;
        model->set_conjugate_prior_for_true_categories(truth_prior_counts);

        Matrix error_prior_counts(num_atoms + 1, num_atoms + 2, -1.0);
        for (int i = 0; i < num_atoms; ++i) {
          error_prior_counts(i, i) = 1.0;
          error_prior_counts(i, num_atoms + 1) = 1.0;
        }
        error_prior_counts.last_row() = 1.0;
        model->set_conjugate_prior_for_observation_categories(
            error_prior_counts);
      }
    }
  }

  Matrix MvRegCopulaDataImputer::imputed_data() const {
    Matrix ans(complete_data_.size(), ydim());
    for (int i = 0; i < complete_data_.size(); ++i) {
      ans.row(i) = complete_data_[i]->y_true();
    }
    return ans;
  }

  Matrix MvRegCopulaDataImputer::impute_data_set(
      const std::vector<Ptr<MvRegData>> &data) {
    int sample_size = data.size();
    Matrix ans(sample_size, ydim());
    for (int i = 0; i < sample_size; ++i) {
      NEW(Imputer::CompleteData, complete_data)(data[i]);
      impute_row(complete_data, rng_, false);
      ans.row(i) = complete_data->y_true();
    }
    return ans;
  }


  std::vector<IqAgentState>
  MvRegCopulaDataImputer::empirical_distribution_state() const {
    std::vector<IqAgentState> ans;
    for (int i = 0; i < empirical_distributions_.size(); ++i) {
      ans.push_back(empirical_distributions_[i].save_state());
    }
    return ans;
  }

  void MvRegCopulaDataImputer::restore_empirical_distributions(
      const std::vector<IqAgentState> &state) {
    empirical_distributions_.clear();
    for (int i = 0; i < state.size(); ++i) {
      empirical_distributions_.push_back(IQagent(state[i]));
    }
  }

  void MvRegCopulaDataImputer::set_default_regression_prior() {
    int xdim = complete_data_model_->xdim();
    int ydim = complete_data_model_->ydim();
    NEW(MultivariateRegressionSampler, regression_sampler)(
        complete_data_model_.get(),
        Matrix(xdim, ydim, 0.0),
        1.0,
        ydim + 1,
        SpdMatrix(ydim, 1.0));
    complete_data_model_->set_method(regression_sampler);
  }

  //---------------------------------------------------------------------------
  void MvRegCopulaDataImputer::sample_posterior() {
    for (int i = 0; i < empirical_distributions_.size(); ++i) {
      empirical_distributions_[i].update_cdf();
    }
    if (!workers_.empty()) {
      impute_latent_data_multithreaded();
    } else {
      impute_all_rows();
      // clear_client_data();
      // for (int i = 0; i < complete_data_.size(); ++i) {
      //   impute_row(complete_data_[i], rng_, true);
      // }
    }
    cluster_mixing_distribution_->sample_posterior();
    for (int s = 0; s < cluster_mixture_components_.size(); ++s) {
      cluster_mixture_components_[s]->sample_posterior();
    }
    complete_data_model_->sample_posterior();
  }

  //---------------------------------------------------------------------------
  void MvRegCopulaDataImputer::impute_latent_data_multithreaded() {
    ensure_data_distribution();
    broadcast_parameters();
    std::vector<std::future<void>> futures;
    for (int i = 0; i < workers_.size(); ++i) {
      MvRegCopulaDataImputer *worker = workers_[i].get();
      futures.emplace_back(thread_pool_.submit(
          [worker]() {
            worker->impute_all_rows();
          }));
    }
    for (int i = 0; i < workers_.size(); ++i) {
      futures[i].get();
    }
    reduce_sufficient_statistics();
  }

  void MvRegCopulaDataImputer::ensure_data_distribution() {
    size_t nobs = 0;
    for (size_t i = 0; i < workers_.size(); ++i) {
      nobs += workers_[i]->complete_data_.size();
    }
    if (nobs != complete_data_.size()) {
      distribute_data_to_workers();
    }
  }

  void MvRegCopulaDataImputer::setup_worker_pool(int nworkers) {
    shut_down_worker_pool();
    if (nworkers <= 0) {
      return;
    } else {
      for (int i = 0; i < nworkers; ++i) {
        // Check the copy constructor.  Workers don't sample their own
        // parameters, so no need to set priors on workers.
        workers_.push_back(clone());
        workers_.back()->worker_id_ = i;
      }
      thread_pool_.set_number_of_threads(nworkers);
    }
  }

  void MvRegCopulaDataImputer::shut_down_worker_pool() {
    thread_pool_.set_number_of_threads(0);
    workers_.clear();
  }

  void MvRegCopulaDataImputer::distribute_data_to_workers() {
    size_t data_per_worker = complete_data_.size() / workers_.size();
    auto b = complete_data_.begin();
    auto e = complete_data_.end();
    for (size_t i = 0; i < workers_.size(); ++i) {
      workers_[i]->complete_data_.clear();
      if (i + 1 == workers_.size()) {
        std::copy(b, e, std::back_inserter(workers_[i]->complete_data_));
      } else {
        std::copy(b, b + data_per_worker,
                  std::back_inserter(workers_[i]->complete_data_));
        b += data_per_worker;
      }

      workers_[i]-> empirical_distributions_ = empirical_distributions_;
    }
  }

  void MvRegCopulaDataImputer::impute_all_rows() {
    clear_client_data();
    for (size_t i = 0; i < complete_data_.size(); ++i) {
      impute_row(complete_data_[i], rng_, true);
    }
  }

  void MvRegCopulaDataImputer::reduce_sufficient_statistics() {
    clear_client_data();
    for (size_t worker = 0; worker < workers_.size(); ++worker) {
      complete_data_model_->suf()->combine(
          workers_[worker]->complete_data_model_->suf());
      cluster_mixing_distribution_->suf()->combine(
          workers_[worker]->cluster_mixing_distribution_->suf());
      for (int cluster = 0; cluster < nclusters(); ++cluster) {
        cluster_mixture_components_[cluster]->combine_sufficient_statistics(
            *(workers_[worker]->cluster_mixture_component(cluster)));
      }
    }
  }

  void MvRegCopulaDataImputer::broadcast_parameters() {
    for (size_t i = 0; i < workers_.size(); ++i) {
      workers_[i]->complete_data_model_->set_Beta(
          complete_data_model_->Beta());
      workers_[i]->complete_data_model_->set_Sigma(
          complete_data_model_->Sigma());

      workers_[i]->cluster_mixing_distribution_->set_pi(
          cluster_mixing_distribution_->pi());

      for (int s = 0; s < nclusters(); ++s) {
        workers_[i]->cluster_mixture_components_[s]->copy_parameters(
            *cluster_mixture_components_[s]);
      }
    }
  }
  //---------------------------------------------------------------------------
  const Vector &MvRegCopulaDataImputer::atom_probs(
      int cluster, int variable_index) const {
    return cluster_mixture_components_[cluster]->model(
        variable_index).atom_probs();
  }

  void MvRegCopulaDataImputer::set_atom_probs(
      int cluster, int variable_index, const Vector &values) {
    cluster_mixture_components_[cluster]->mutable_model(
        variable_index)->set_atom_probs(values);
  }
  //---------------------------------------------------------------------------
  Matrix MvRegCopulaDataImputer::atom_error_probs(
      int cluster, int variable_index) const {
    return cluster_mixture_components_[cluster]->model(
        variable_index).atom_error_probs();
  }

  void MvRegCopulaDataImputer::set_atom_error_probs(
      int cluster, int variable_index, const Matrix &values) {
    cluster_mixture_components_[cluster]->mutable_model(
        variable_index)->set_atom_error_probs(values);
  }

  //---------------------------------------------------------------------------
  void MvRegCopulaDataImputer::set_atom_prior(const Vector &prior_counts,
                                              int variable_index) {
    for (int s = 0; s < cluster_mixture_components_.size(); ++s) {
      cluster_mixture_components_[s]->mutable_model(
          variable_index)->set_conjugate_prior_for_true_categories(prior_counts);
    }
  }

  //---------------------------------------------------------------------------
  void MvRegCopulaDataImputer::set_atom_error_prior(
      const Matrix &prior_counts, int variable_index) {
    for (int s = 0; s < cluster_mixture_components_.size(); ++s) {
      cluster_mixture_components_[s]->mutable_model(
          variable_index)->set_conjugate_prior_for_observation_categories(
              prior_counts);
    }
  }

  //---------------------------------------------------------------------------
  void MvRegCopulaDataImputer::set_default_prior_for_mixing_weights() {
    NEW(MultinomialDirichletSampler, sampler)(
        cluster_mixing_distribution_.get(),
        Vector(nclusters(), 1.0 / nclusters()));
    cluster_mixing_distribution_->set_method(sampler);
  }

  //---------------------------------------------------------------------------
  void MvRegCopulaDataImputer::ensure_swept_sigma_current() const {
    if (swept_sigma_current_) return;
    swept_sigma_ = SweptVarianceMatrix(complete_data_model_->Sigma());
    swept_sigma_current_ = true;
  }

  void MvRegCopulaDataImputer::set_observers() {
    complete_data_model_->Sigma_prm()->add_observer(
        [this]() {this->swept_sigma_current_ = false;} );
  }

}  // namespace BOOMx
