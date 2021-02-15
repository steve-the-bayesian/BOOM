/*
  Copyright (C) 2020 Steven L. Scott

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

#include "Models/Impute/MixedDataImputerWithErrorCorrection.hpp"
#include "distributions.hpp"
#include "cpputil/lse.hpp"
#include "Models/PosteriorSamplers/MultinomialDirichletSampler.hpp"

namespace BOOM {

  namespace {
    void check_for_nan(const Vector &v) {
      for (int i = 0; i < v.size(); ++i) {
        if (std::isnan(v[i])) {
          report_error("Found a NaN where it shouldn't exist.");
        }
      }
    }
  }

  //===========================================================================

  namespace MixedImputation {
    namespace {
      using NECM = NumericErrorCorrectionModel;
    }  // namespace

    NECM::NumericErrorCorrectionModel(int index, const Vector &atoms)
        : ScalarModelBase(index),
          impl_(new ErrorCorrectionModel(atoms))
    {}

    NECM::NumericErrorCorrectionModel(const NECM &rhs)
        : ScalarModelBase(rhs),
          impl_(rhs.impl_->clone())
    {}

    NECM & NECM::operator=(const NECM &rhs) {
      if (&rhs != this) {
        ScalarModelBase::operator=(rhs);
        impl_.reset(rhs.impl_->clone());
      }
      return *this;
    }

    NECM *NECM::clone() const {return new NECM(*this);}

    double NECM::logp(const MixedMultivariateData &data) const {
      const DoubleData &scalar(data.numeric(index()));
      double value = std::numeric_limits<double>::quiet_NaN();
      if (scalar.missing() == Data::missing_status::observed) {
        value = scalar.value();
      }
      return logp(value);
    }

    //==========================================================================

    namespace {
      using CECM = CategoricalErrorCorrectionModel;
    }  // namespace

    CECM::CategoricalErrorCorrectionModel(int index,
                                          const Ptr<CatKey> &levels)
        : ScalarModelBase(index),
          levels_(levels),
          truth_model_(new MultinomialModel(levels_->max_levels())),
          workspace_is_current_(false)
    {
      for (int i = 0; i < levels_->max_levels(); ++i) {
        obs_models_.push_back(
            new MultinomialModel(levels_->max_levels() + 1));
      }
      build_atom_index();
      set_observers();
    }

    CECM::CategoricalErrorCorrectionModel(const CECM &rhs)
        : ScalarModelBase(rhs),
          levels_(rhs.levels_),
          truth_model_(rhs.truth_model_->clone()),
          workspace_is_current_(false)
    {
      for (int i = 0; i < rhs.obs_models_.size(); ++i) {
        obs_models_.push_back(rhs.obs_models_[i]->clone());
      }
      build_atom_index();
      set_observers();
    }

    CECM & CECM::operator=(const CECM &rhs) {
      if (&rhs != this) {
        ScalarModelBase::operator=(rhs);
        levels_ = rhs.levels_;
        truth_model_.reset(rhs.truth_model_->clone());
        obs_models_.clear();
        for (int i = 0; i < rhs.obs_models_.size(); ++i) {
          obs_models_.push_back(rhs.obs_models_[i]->clone());
        }
        build_atom_index();
        set_observers();
      }
      return *this;
    }

    CECM *CECM::clone() const {return new CECM(*this);}

    double CECM::logpri() const {
      double ans = truth_model_->logpri();
      for (int i = 0; i < obs_models_.size(); ++i) {
        ans += obs_models_[i]->logpri();
      }
      return ans;
    }

    void CECM::clear_data() {
      truth_model_->clear_data();
      for (auto &el : obs_models_) {
        el->clear_data();
      }
    }

    void CECM::update_complete_data_suf(int true_level, int observed_level) {
      truth_model_->suf()->update_raw(true_level);
      obs_models_[true_level]->suf()->update_raw(observed_level);
    }

    void CECM::sample_posterior() {
      truth_model_->sample_posterior();
      for (int i = 0; i < obs_models_.size(); ++i) {
        obs_models_[i]->sample_posterior();
      }
    }

    double CECM::logp(const MixedMultivariateData &data) const {
      const LabeledCategoricalData &scalar(data.categorical(index()));
      ensure_workspace_is_current();
      return log_marginal_observed_[atom_index(scalar)];
    }

    double CECM::logp(const std::string &label) const {
      ensure_workspace_is_current();
      return log_marginal_observed_[atom_index(label)];
    }

    Vector CECM::true_level_log_probability(
        const LabeledCategoricalData &observed) {
      ensure_workspace_is_current();
      return log_joint_distribution_.col(atom_index(levels_->label(
          observed.value())));
    }

    int CECM::atom_index(const LabeledCategoricalData &data) const {
      if (data.missing() != Data::missing_status::observed) {
        return levels_->max_levels() + 1;
      } else {
        return atom_index(levels_->label(data.value()));
      }
    }

    int CECM::atom_index(const std::string &label) const {
      auto it = atom_index_.find(label);
      if (it == atom_index_.end()) {
        return levels_->max_levels();
      } else {
        return it->second;
      }
    }

    void CECM::set_conjugate_prior_for_levels(const Vector &counts) {
      truth_model_->clear_methods();
      NEW(ConstrainedMultinomialDirichletSampler, sampler)(
          truth_model_.get(), counts);
      truth_model_->set_method(sampler);
    }

    void CECM::set_conjugate_prior_for_observations(const Matrix &counts) {
      for (int i = 0; i < obs_models_.size(); ++i) {
        obs_models_[i]->clear_methods();
        NEW(ConstrainedMultinomialDirichletSampler, sampler)(
            obs_models_[i].get(),
            counts.row(i));
        obs_models_[i]->set_method(sampler);
      }
    }

    // Ensure that the log joint distribution is up to date.
    void CECM::ensure_workspace_is_current() const {
      if (workspace_is_current_) {
        return;
      } else {
        // Update the joint distribution of true values (rows) and observed
        // values (columns).
        int nlevels = levels_->max_levels();
        log_joint_distribution_.resize(nlevels, nlevels + 1);
        for (int i = 0; i < nlevels + 1; ++i) {
          log_joint_distribution_.col(i) = truth_model_->logpi();
        }
        for (int i = 0; i < nlevels; ++i) {
          log_joint_distribution_.row(i) += obs_models_[i]->logpi();
        }

        // Update the marginal distribution of observed values.
        log_marginal_observed_.resize(nlevels + 1);
        for (int i = 0; i < nlevels + 1; ++i) {
          log_marginal_observed_[i] = lse(log_joint_distribution_.col(i));
        }

        workspace_is_current_ = true;
      }
    }

    void CECM::set_observers() {
      auto observer = [this]() {this->workspace_is_current_ = false;};
      truth_model_->Pi_prm()->add_observer(observer);
      for (int i = 0; i < obs_models_.size(); ++i) {
        obs_models_[i]->Pi_prm()->add_observer(observer);
      }
    }

    void CECM::build_atom_index() {
      atom_index_.clear();
      for (int i = 0; i < levels_->max_levels(); ++i) {
        std::string label = levels_->label(i);
        atom_index_[label] = i;
      }
    }

    //==========================================================================

    RowModelWithErrorCorrection::RowModelWithErrorCorrection() {
    }

    RowModelWithErrorCorrection::RowModelWithErrorCorrection(
        const RowModelWithErrorCorrection &rhs)
        : RowModelBase(rhs)
    {
      populate_ec_scalar_models();
    }

    RowModelWithErrorCorrection & RowModelWithErrorCorrection::operator=(
        const RowModelWithErrorCorrection &rhs) {
      if (&rhs != this) {
        RowModelBase::operator=(rhs);
        populate_ec_scalar_models();
      }
      return *this;
    }

    void RowModelWithErrorCorrection::populate_ec_scalar_models() {
      const std::vector<Ptr<ScalarModelBase>> &models(scalar_models());
      categorical_ec_models_.clear();
      numeric_ec_models_.clear();
      for (int i = 0; i < models.size(); ++i) {
        if (models[i]->variable_type() == VariableType::numeric) {
          numeric_ec_models_.push_back(
              models[i].dcast<NumericErrorCorrectionModel>());
        } else if (models[i]->variable_type() == VariableType::categorical) {
          categorical_ec_models_.push_back(
              models[i].dcast<CategoricalErrorCorrectionModel>());
        } else {
          report_error("Unsupported model type.");
        }
      }
    }

    void RowModelWithErrorCorrection::add_numeric(
        const Ptr<NumericErrorCorrectionModel> &model) {
      RowModelBase::add_scalar_model(model);
      numeric_ec_models_.push_back(model);
    }

    void RowModelWithErrorCorrection::add_categorical(
        const Ptr<CategoricalErrorCorrectionModel> &model) {
      RowModelBase::add_scalar_model(model);
      categorical_ec_models_.push_back(model);
    }

    RowModelWithErrorCorrection *RowModelWithErrorCorrection::clone() const {
      return new RowModelWithErrorCorrection(*this);
    }


    // Impute the categorical data given the numeric data.  Later this can be
    // improved by marginalizing over the numeric data and imputing given y_obs.
    void RowModelWithErrorCorrection::impute_categorical(
        Ptr<MixedImputation::CompleteData> &row,
        RNG &rng,
        bool update_complete_data_suf,
        const Ptr<DatasetEncoder> &encoder,
        const std::vector<Ptr<EffectsEncoder>> &encoders,
        const Ptr<MultivariateRegressionModel> &numeric_model) {

      Vector &predictors(row->x());
      predictors.resize(encoder->dim());
      int start = 0;
      if (encoder->add_intercept()) {
        predictors[0] = 1;
        start = 1;
      }
      const Vector &y_numeric(row->y_numeric());
      std::vector<int> imputed_categorical_data = row->true_categories();
      const std::vector<Ptr<LabeledCategoricalData>> observed_categories(
          row->observed_categories());

      for (int i = 0; i < encoders.size(); ++i) {
        // 'i' indexes the set of categorical variables in the row.

        VectorView view(predictors, start, encoders[i]->dim());

        // truth_logp is the log probability that each level is the true value.
        Vector truth_logp =
            categorical_ec_models_[i]->true_level_log_probability(
                *observed_categories[i]);

        for (int level = 0; level < truth_logp.size(); ++level) {
          if (std::isfinite(truth_logp[level])) {
            encoders[i]->encode(level, view);
            Vector yhat = numeric_model->predict(predictors);
            truth_logp[level] -=
                0.5 * numeric_model->Siginv().Mdist(y_numeric - yhat);
          }
        }
        truth_logp.normalize_logprob();
        imputed_categorical_data[i] = rmulti_mt(rng, truth_logp);
        view = encoders[i]->encode(imputed_categorical_data[i]);
        if (update_complete_data_suf) {
          categorical_ec_models_[i]->update_complete_data_suf(
              imputed_categorical_data[i], observed_categories[i]->value());
        }
      }
      row->set_true_categories(imputed_categorical_data);
    }

    void RowModelWithErrorCorrection::impute_atoms(
        Ptr<MixedImputation::CompleteData> &row,
        RNG &rng,
        bool update_complete_data_suf) {
      const Vector &observed(row->y_observed());
      for (int i = 0; i < observed.size(); ++i) {
        // True atom is the atom responsible for the true value.
        int true_atom = numeric_ec_models_[i]->impute_atom(
            observed[i], rng, update_complete_data_suf);
        row->set_y_true(i, numeric_ec_models_[i]->true_value(
            true_atom, observed[i]));
        row->set_y_numeric(i, numeric_ec_models_[i]->numeric_value(
            true_atom, observed[i]));
      }
    }
  }  // namespace MixedImputation

  //===========================================================================
  MixedDataImputerWithErrorCorrection::MixedDataImputerWithErrorCorrection(
      int num_clusters,
      const DataTable &data,
      const std::vector<Vector> &atoms,
      RNG &seeding_rng)
      : MixedDataImputerBase(num_clusters, data, atoms, seeding_rng)
  {}

  MixedDataImputerWithErrorCorrection::MixedDataImputerWithErrorCorrection(
      const MixedDataImputerWithErrorCorrection &rhs)
      : MixedDataImputerBase(rhs)
  {
    for (int i = 0; i < rhs.mixture_components_.size(); ++i) {
      mixture_components_.push_back(rhs.mixture_components_[i]->clone());
    }
  }

  MixedDataImputerWithErrorCorrection &
  MixedDataImputerWithErrorCorrection::operator=(
      const MixedDataImputerWithErrorCorrection &rhs) {
    if (&rhs != this) {
      MixedDataImputerBase::operator=(rhs);
      mixture_components_.clear();
      for (int i = 0; i < rhs.mixture_components_.size(); ++i) {
        mixture_components_.push_back(rhs.mixture_components_[i]->clone());
      }
    }
    return *this;
  }

  MixedDataImputerWithErrorCorrection *
  MixedDataImputerWithErrorCorrection::clone() const {
    return new MixedDataImputerWithErrorCorrection(*this);
  }

  void MixedDataImputerWithErrorCorrection::impute_numerics_given_atoms(
      Ptr<MixedImputation::CompleteData> &data,
      RNG &rng,
      bool update_complete_data_suf) {
    ensure_swept_sigma_current();
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
        double uniform = empirical_distribution(i).cdf(imputed_numeric[i]);
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
      Vector mean = numeric_data_model()->predict(data->x());
      if (observed.nvars() == 0) {
        imputed_numeric = rmvn_mt(rng, mean, numeric_data_model()->Sigma());
      } else {
        swept_sigma().SWP(observed);
        Vector conditional_mean = swept_sigma().conditional_mean(
            observed.select(imputed_numeric), mean);
        Vector imputed_values = rmvn_mt(
            rng, conditional_mean, swept_sigma().residual_variance());
        observed.fill_missing_elements(imputed_numeric, imputed_values);
      }

      // Transform imputed data back to observed scale.
      Vector y_true = data->y_true();
      for (int i = 0; i < imputed_numeric.size(); ++i) {
        if (std::isnan(y_true[i])) {
          y_true[i] = empirical_distribution(i).quantile(
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
      numeric_data_model()->suf()->update_raw_data(
          data->y_numeric(), data->x(), 1.0);
    }
  }

  void MixedDataImputerWithErrorCorrection::initialize_mixture(
      int num_clusters,
      const std::vector<Vector> &atoms,
      const std::vector<Ptr<CatKey>> &levels,
      const std::vector<VariableType> &variable_type) {
    for (int c = 0; c < num_clusters; ++c) {
      auto num_it = atoms.begin();
      auto cat_it = levels.begin();
      Ptr<MixedImputation::RowModelWithErrorCorrection> row_model(
          new MixedImputation::RowModelWithErrorCorrection);
      for (int j = 0; j < variable_type.size(); ++j) {
        switch (variable_type[j]) {
          case VariableType::numeric:
            row_model->add_numeric(
                new MixedImputation::NumericErrorCorrectionModel(j, *num_it));
            ++num_it;
            break;

          case VariableType::categorical:
            row_model->add_categorical(
                new MixedImputation::CategoricalErrorCorrectionModel(
                    j, *cat_it));
            ++cat_it;
            break;

          default:
            report_error(
                "Only numeric or categorical varaibles are supported.");
        }
      }
      mixture_components_.push_back(row_model);
    }
  }
}  // namespace BOOM
