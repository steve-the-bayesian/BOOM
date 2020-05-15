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

#include "Models/Impute/MvRegCopulaDataImputer.hpp"
#include "Models/PosteriorSamplers/MultinomialDirichletSampler.hpp"
#include "distributions.hpp"

namespace BOOM {

  namespace {
    using CICM = ConditionallyIndependentCategoryModel;
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
    marginal_of_true_data_->Pi_prm()->add_observer(
        [this]() {
          this->workspace_is_current_ = false;
        });

    for (int i = 0; i <= atoms.size(); ++i) {
      NEW(MultinomialModel, conditional_model)(atoms.size() + 2);
      conditional_model->Pi_prm()->add_observer(
          [this]() {
            this->workspace_is_current_ = false;
          });
      conditional_observed_given_true_.push_back(conditional_model);
    }
  }

  ErrorCorrectionModel *ErrorCorrectionModel::clone() const {
    return new ErrorCorrectionModel(*this);
  }

  double ErrorCorrectionModel::logp(double y) const {
    ensure_workspace_current();
    return observed_log_probability_table_[category_map(y)];
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

  double ErrorCorrectionModel::true_value(int true_atom,
                                          double observed_value) const {
    if (true_atom < atoms_.size()) {
      return atoms_[true_atom];
    } else {
      return observed_value;
    }
  }

  double ErrorCorrectionModel::numeric_value(int true_atom,
                                             double observed_value) const {
    if (true_atom == atoms_.size()) {
      return observed_value;
    } else {
      return std::numeric_limits<double>::quiet_NaN();
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

  double CICM::logp(const Vector &observed) const {
    double ans = 0;
    for (int i = 0; i < observed.size(); ++i) {
      ans += observed_data_models_[i]->logp(observed[i]);
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

  //===========================================================================
  MvRegCopulaDataImputer::MvRegCopulaDataImputer(
      int num_clusters, const std::vector<Vector> &atoms, int xdim, RNG &seeding_rng)
      : cluster_mixing_distribution_(new MultinomialModel(num_clusters)),
        complete_data_model_(new MultivariateRegressionModel(xdim, atoms.size())),
        rng_(seed_rng(seeding_rng)),
        swept_sigma_(SpdMatrix(0))
  {
    for (int s = 0; s < num_clusters; ++s) {
      NEW(ConditionallyIndependentCategoryModel, component)(atoms);
      cluster_mixture_components_.push_back(component);
    }
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
    int component = impute_cluster(data, rng, update_complete_data_suf);
    cluster_mixture_components_[component]->impute_atoms(
        *data, rng, update_complete_data_suf);

    Vector numeric = data->y_numeric();
    Selector observed(numeric.size(), true);
    for (int i = 0; i < numeric.size(); ++i) {
      if (std::isnan(numeric[i])) {
        observed.drop(i);
      } else {
        numeric[i] = qnorm(empirical_distributions_[i].cdf(numeric[i]));
      }
    }

    if (observed.nvars() < observed.nvars_possible()) {
      Vector mean = complete_data_model_->predict(data->x());
      Vector imputed_numeric;
      if (observed.nvars() == 0) {
        imputed_numeric = rmvn_mt(rng, mean, complete_data_model_->Sigma());
      } else {
        swept_sigma_.SWP(observed);
        Vector conditional_mean = swept_sigma_.conditional_mean(
            observed.select(numeric), mean);
        Vector imputed_values = rmvn_mt(rng, conditional_mean, swept_sigma_.residual_variance());

        observed.fill_missing_elements(numeric, imputed_values);

        // TODO: update the empirical CDF.

        // transform back.
        for (int i = 0; i < numeric.size(); ++i) {
          if (observed[i]) {
            numeric[i] = data->y_observed()[i];
          } else {
            numeric[i] = empirical_distributions_[i].quantile(pnorm(numeric[i]));
          }
        }
        data->set_y_numeric(numeric);
      }
    }
  }

  //---------------------------------------------------------------------------
  void MvRegCopulaDataImputer::sample_posterior() {
    for (int i = 0; i < empirical_distributions_.size(); ++i) {
      empirical_distributions_[i].update_cdf();
    }
    clear_client_data();
    swept_sigma_ = SweptVarianceMatrix(complete_data_model_->Sigma());
    for (int i = 0; i < complete_data_.size(); ++i) {
      impute_row(complete_data_[i], rng_, true);
    }
    cluster_mixing_distribution_->sample_posterior();
    for (int s = 0; s < cluster_mixture_components_.size(); ++s) {
      cluster_mixture_components_[s]->sample_posterior();
    }
    complete_data_model_->sample_posterior();
  }

  //---------------------------------------------------------------------------
  const Vector &MvRegCopulaDataImputer::atom_probs(
      int cluster, int variable_index) const {
    return cluster_mixture_components_[cluster]->model(
        variable_index).atom_probs();
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

}  // namespace BOOMx
