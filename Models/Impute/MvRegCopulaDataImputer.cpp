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

namespace BOOM {

  namespace {
    using CICM = ConditionallyIndependentCategoryModel;
  }  // namespace
  //===========================================================================
  ErrorCorrectionModel::ErrorCorrectionModel(const Vector &atoms)
      : atoms_(atoms),
        marginal_of_true_data_(new MultinomialModel(atoms.size() + 1)),
        observed_log_probability_table_(atoms.size() + 2),
        observed_log_probability_table_current_(false)
  {
    marginal_of_true_data_->Pi_prm()->add_observer(
        [this]() {
          this->observed_log_probability_table_current_ = false;
        });
    for (int i = 0; i < atoms.size(); ++i) {
      NEW(MultinomialModel, conditional_model)(atoms.size() + 2);
      conditional_model->Pi_prm()->add_observer(
          [this]() {
            this->observed_log_probability_table_current_ = false;
          });
      conditional_observed_given_true_.push_back(conditional_model);
    }
  }

  ErrorCorrectionModel *ErrorCorrectionModel::clone() const {
    return new ErrorCorrectionModel(*this);
  }

  double ErrorCorrectionModel::logp(double y) const {
    ensure_observed_log_probability_table_current();
    return observed_log_probability_table_[category_map(y)];
  }

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

  void ErrorCorrectionModel::ensure_observed_log_probability_table_current() const {
    if (observed_log_probability_table_current_) {
      return;
    }

    int true_dim = atoms_.size() + 1;
    int obs_dim = true_dim + 1;
    Matrix probs(true_dim, obs_dim);
    for (int truth = 0; truth < true_dim; ++truth) {
      for (int obs = 0; obs < obs_dim; ++obs) {
        probs(truth, obs) =
            marginal_of_true_data_->pi(truth)
            * conditional_observed_given_true_[truth]->pi(obs);
      }
    }
    observed_log_probability_table_ = log(probs.col_sums());
    observed_log_probability_table_current_ = true;
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

  //===========================================================================
  CICM::ConditionallyIndependentCategoryModel(const std::vector<Vector> &atoms)
  {
    for (int i = 0; i < atoms.size(); ++i) {
      NEW(ErrorCorrectionModel, model)(atoms[i]);
      observed_data_models_.push_back(model);
    }
  }

  double CICM::pdf(const Data *dp, bool logscale) const {
    const PartiallyObservedVectorData *data = dynamic_cast<
      const PartiallyObservedVectorData*>(dp);
    double logp = 0;
    for (int i = 0; i < observed_data_models_.size(); ++i) {
      logp += observed_data_models_[i]->logp(data->value()[i]);
    }
    return logp;
  }

  //===========================================================================
  MvRegCopulaDataImputer::MvRegCopulaDataImputer(
      int num_clusters, const std::vector<Vector> &atoms, int xdim, RNG &seeding_rng)
      : complete_data_model_(new MultivariateRegressionModel(xdim, atoms.size())),
        rng_(seed_rng(seeding_rng))
  {
    NEW(MultinomialModel, mixing_weights)(num_clusters);
    for (int s = 0; s < num_clusters; ++s) {
      NEW(ConditionallyIndependentCategoryModel, component)(atoms);
      mixture_components_.push_back(component);
    }
    cluster_model_.reset(new FiniteMixtureModel(
        mixture_components_, mixing_weights));
  }

  //---------------------------------------------------------------------------
  double MvRegCopulaDataImputer::logpri() const {
    return negative_infinity();
  }

  //---------------------------------------------------------------------------
  Vector MvRegCopulaDataImputer::impute_row(
      const Vector &input,
      const ConstVectorView &predictors,
      RNG &rng) const {

    Vector ans(input.size());
    Vector mean = complete_data_model_->predict(predictors);
    return mean;
  }

  Vector MvRegCopulaDataImputer::impute_continuous_values(
      const Vector &y,
      const ConstVectorView &x,
      RNG &rng) const {
    return Vector(0);
  }

  //---------------------------------------------------------------------------
  void MvRegCopulaDataImputer::sample_posterior() {
    cluster_model_->sample_posterior();

    complete_data_model_->suf()->clear_y_keep_x();
    for (int i = 0; i < dat().size(); ++i) {
      auto data_point = dat()[i];
      Vector y = impute_continuous_values(
          data_point->y(), data_point->x(), rng_);
      complete_data_model_->suf()->update_y_not_x(y, data_point->x(), 1.0);
    }
    complete_data_model_->sample_posterior();
  }

}  // namespace BOOMx
