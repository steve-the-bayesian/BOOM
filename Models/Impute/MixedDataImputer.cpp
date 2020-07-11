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

#include "Models/Impute/MixedDataImputer.hpp"
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

    CompleteData::CompleteData(const Ptr<MixedMultivariateData> &observed)
        : observed_data_(observed),
          y_true_(observed->numeric_dim()),
          y_numeric_(observed->numeric_dim()),
          true_categories_(observed->categorical_dim(), 0),
          observed_categories_(observed_data_->categorical_data())
    {}

    CompleteData::CompleteData(const CompleteData &rhs)
        : Data(rhs),
          observed_data_(rhs.observed_data_->clone()),
          y_true_(rhs.y_true_),
          y_numeric_(rhs.y_numeric_),
          true_categories_(rhs.true_categories_),
          observed_categories_(observed_data_->categorical_data())
    {}

    CompleteData & CompleteData::operator=(const CompleteData &rhs) {
      if (&rhs != this) {
        observed_data_ = rhs.observed_data_->clone();
        y_true_ = rhs.y_true_;
        y_numeric_ = rhs.y_numeric_;
        true_categories_ = rhs.true_categories_;
        observed_categories_ = observed_data_->categorical_data();
      }
      return *this;
    }

    CompleteData *CompleteData::clone() const {
      return new CompleteData(*this);
    }

    std::ostream &CompleteData::display(std::ostream &out) const {
      out << *observed_data_ << "\n"
          << "y_true = " << y_true_ << "\n"
          << "y_numeric_ = " << y_numeric_ << "\n"
          << "true_categories_ = " ;
      for (int i = 0; i < true_categories_.size(); ++i) {
        out << true_categories_[i] << ' ';
      }
      out << std::endl;
      return out;
    }

    void CompleteData::fill_data_table_row(DataTable &table, int row) {
      int numeric_counter = 0;
      int categorical_counter = 0;
      for (int i = 0; i < table.nvars(); ++i) {
        VariableType type = table.variable_type(i);
        if (type == VariableType::numeric) {
          table.set_numeric_value(row, i, y_true_[numeric_counter++]);
        } else if (type == VariableType::categorical) {
          table.set_nominal_value(row, i, true_categories_[categorical_counter++]);
        } else {
          report_error("Only numeric and categorical data types are supported.");
        }
      }
    }

    //===========================================================================
    namespace {
      using NECM = NumericErrorCorrectionModel;
    }  // namespace

    NECM::NumericErrorCorrectionModel(int index, const Vector &atoms)
        : ErrorCorrectionModelBase(index),
          impl_(new ErrorCorrectionModel(atoms))
    {}

    NECM::NumericErrorCorrectionModel(const NECM &rhs)
        : ErrorCorrectionModelBase(rhs),
          impl_(rhs.impl_->clone())
    {}

    NECM & NECM::operator=(const NECM &rhs) {
      if (&rhs != this) {
        ErrorCorrectionModelBase::operator=(rhs);
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

    //===========================================================================
    namespace {
      using CECM = CategoricalErrorCorrectionModel;
    }  // namespace

    CECM::CategoricalErrorCorrectionModel(int index,
                                          const Ptr<CatKey> &levels)
        : ErrorCorrectionModelBase(index),
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
        : ErrorCorrectionModelBase(rhs),
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
        ErrorCorrectionModelBase::operator=(rhs);
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
      const CategoricalData &scalar(data.categorical(index()));
      ensure_workspace_is_current();
      return log_marginal_observed_[atom_index(scalar)];
    }

    double CECM::logp(const std::string &label) const {
      ensure_workspace_is_current();
      return log_marginal_observed_[atom_index(label)];
    }

    Vector CECM::true_level_log_probability(const CategoricalData &observed) {
      ensure_workspace_is_current();
      return log_joint_distribution_.col(atom_index(levels_->label(
          observed.value())));
    }

    int CECM::atom_index(const CategoricalData &data) const {
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

    //===========================================================================
    RowModel::RowModel() {
    }

    void RowModel::add_numeric(
        const Ptr<NumericErrorCorrectionModel> &model) {
      scalar_models_.push_back(model);
      numeric_models_.push_back(model);
    }

    void RowModel::add_categorical(
        const Ptr<CategoricalErrorCorrectionModel> &model) {
      scalar_models_.push_back(model);
      categorical_models_.push_back(model);
    }


    RowModel::RowModel(const RowModel &rhs)
        : Model(rhs),
          CompositeParamPolicy(rhs),
          NullDataPolicy(rhs),
          NullPriorPolicy(rhs) {
      for (int i = 0; i < rhs.scalar_models_.size(); ++i) {
        scalar_models_.push_back(rhs.scalar_models_[i]->clone());
      }
    }

    RowModel *RowModel::clone() const { return new RowModel(*this); }

    void RowModel::clear_data() {
      for (size_t i = 0; i < scalar_models_.size(); ++i){
        scalar_models_[i]->clear_data();
      }
    }

    double RowModel::logp(const MixedMultivariateData &data) const {
      double ans = 0;
      for (const auto &el : scalar_models_) {
        ans += el->logp(data);
      }
      return ans;
    }

    // Impute the categorical data given the numeric data.  Later this can be
    // improved by marginalizing over the numeric data and imputing given y_obs.
    void RowModel::impute_categorical(
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
      const std::vector<Ptr<CategoricalData>> observed_categories(
          row->observed_categories());

      for (int i = 0; i < encoders.size(); ++i) {
        // 'i' indexes the set of categorical variables in the row.

        VectorView view(predictors, start, encoders[i]->dim());

        // truth_logp is the log probability that each level is the true value.
        Vector truth_logp = categorical_models_[i]->true_level_log_probability(
            *observed_categories[i]);

        for (int level = 0; level < truth_logp.size(); ++level) {
          if (std::isfinite(truth_logp[level])) {
            encoders[i]->encode(level, view);
            Vector yhat = numeric_model->predict(predictors);
            truth_logp[level] -= 0.5 * numeric_model->Siginv().Mdist(y_numeric - yhat);
          }
        }
        truth_logp.normalize_logprob();
        imputed_categorical_data[i] = rmulti_mt(rng, truth_logp);
        view = encoders[i]->encode(imputed_categorical_data[i]);
        if (update_complete_data_suf) {
          categorical_models_[i]->update_complete_data_suf(
              imputed_categorical_data[i], observed_categories[i]->value());
        }
      }
      row->set_true_categories(imputed_categorical_data);
    }


    void RowModel::impute_atoms(
        Ptr<MixedImputation::CompleteData> &row,
        RNG &rng,
        bool update_complete_data_suf) {
      const Vector &observed(row->y_observed());
      for (int i = 0; i < observed.size(); ++i) {
        // True atom is the atom responsible for the true value.
        int true_atom = numeric_models_[i]->impute_atom(
            observed[i], rng, update_complete_data_suf);
        row->set_y_true(i, numeric_models_[i]->true_value(
            true_atom, observed[i]));
        row->set_y_numeric(i, numeric_models_[i]->numeric_value(
            true_atom, observed[i]));
      }
    }

    void RowModel::sample_posterior() {
      for (auto &el : scalar_models_) {
        el->sample_posterior();
      }
    }

  }  // namespace MixedImputation
  //===========================================================================

  MixedDataImputer::MixedDataImputer(
      int num_clusters,
      const DataTable &data,
      const std::vector<Vector> &atoms,
      RNG &seeding_rng)
      : rng_(seed_rng(seeding_rng)),
        swept_sigma_(SpdMatrix(1)),
        swept_sigma_current_(false)
  {
    for (size_t i = 0; i < data.nrow(); ++i) {
      add_data(data.row(i));
    }
    create_encoders(data);
    initialize_empirical_distributions(data, atoms);
    initialize_regression_component();

    std::vector<Ptr<CatKey>> levels;
    for (int j = 0; j < data.nvars(); ++j) {
      if (data.variable_type(j) == VariableType::categorical) {
        levels.push_back(data.get_nominal(j).key());
      }
    }
    initialize_mixture(num_clusters, atoms, levels, data.variable_types());
    set_observers();
  }

  MixedDataImputer::MixedDataImputer(const MixedDataImputer &rhs)
      : mixing_distribution_(rhs.mixing_distribution_->clone()),
        numeric_data_model_(rhs.numeric_data_model_->clone()),
        empirical_distributions_(rhs.empirical_distributions_),
        rng_(seed_rng(rhs.rng_)),
        swept_sigma_(rhs.swept_sigma_),
        swept_sigma_current_(false)
  {
    encoder_.reset(new DatasetEncoder(rhs.encoder_->add_intercept()));
    for (int i = 0; i < rhs.encoders_.size(); ++i) {
      encoders_.push_back(rhs.encoders_[i]->clone());
      encoder_->add_encoder(encoders_.back());
    }

    for (int i = 0; i < rhs.mixture_components_.size(); ++i) {
      mixture_components_.push_back(rhs.mixture_components_[i]->clone());
    }
    set_observers();
  }

  MixedDataImputer &MixedDataImputer::operator=(const MixedDataImputer &rhs) {
    if (&rhs != this) {
      mixing_distribution_.reset(rhs.mixing_distribution_->clone());
      numeric_data_model_.reset(rhs.numeric_data_model_->clone());
      empirical_distributions_ = rhs.empirical_distributions_;

      encoder_.reset(new DatasetEncoder(rhs.encoder_->add_intercept()));
      encoders_.clear();
      for (int i = 0; i < rhs.encoders_.size(); ++i) {
        encoders_.push_back(rhs.encoders_[i]->clone());
        encoder_->add_encoder(encoders_.back());
      }

      swept_sigma_ = rhs.swept_sigma_;
      swept_sigma_current_ = false;
      mixture_components_.clear();
      for (int i = 0; i < rhs.mixture_components_.size(); ++i) {
        mixture_components_.push_back(rhs.mixture_components_[i]->clone());
      }
      set_observers();
    }
    return *this;
  }

  MixedDataImputer * MixedDataImputer::clone() const {
    return new MixedDataImputer(*this);
  }

  void MixedDataImputer::add_data(const Ptr<MixedMultivariateData> &data) {
    complete_data_.push_back(new MixedImputation::CompleteData(data));
  }

  void MixedDataImputer::clear_data() {
    clear_client_data();
    complete_data_.clear();
  }

  void MixedDataImputer::clear_client_data() {
    numeric_data_model_->clear_data();
    mixing_distribution_->clear_data();
    for (int s = 0;  s < mixture_components_.size(); ++s) {
      mixture_components_[s]->clear_data();
    }
  }

  void MixedDataImputer::impute_data_set(
      std::vector<Ptr<MixedImputation::CompleteData>> &rows) {
    for (auto &el : rows) {
      impute_row(el, rng_);
    }
  }

  void MixedDataImputer::impute_row(Ptr<MixedImputation::CompleteData> &row,
                                    RNG &rng,
                                    bool update_complete_data_suf)  {
    ensure_swept_sigma_current();
    int component = impute_cluster(row, rng, update_complete_data_suf);

    // This step will fill in the "true_categories" data element in *row.
    mixture_components_[component]->impute_categorical(
        row,
        rng,
        update_complete_data_suf,
        encoder_,
        encoders_,
        numeric_data_model_);

    mixture_components_[component]->impute_atoms(
        row, rng, update_complete_data_suf);

    impute_numerics_given_atoms(row, rng, update_complete_data_suf);
  }

  void MixedDataImputer::impute_numerics_given_atoms(
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
      Vector mean = numeric_data_model_->predict(data->x());
      if (observed.nvars() == 0) {
        imputed_numeric = rmvn_mt(rng, mean, numeric_data_model_->Sigma());
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
      numeric_data_model_->suf()->update_raw_data(
          data->y_numeric(), data->x(), 1.0);
    }
  }

  void MixedDataImputer::impute_all_rows() {
    clear_client_data();
    for (size_t i = 0; i < complete_data_.size(); ++i) {
      impute_row(complete_data_[i], rng_, true);
    }
  }

  int MixedDataImputer::impute_cluster(
      Ptr<MixedImputation::CompleteData> &row, RNG &rng) const {
    int S = mixture_components_.size();
    wsp_ = mixing_distribution_->logpi();
    for (int s = 0; s < S; ++s) {
      wsp_[s] += mixture_components_[s]->logp(row->observed_data());
    }
    wsp_.normalize_logprob();
    int component = rmulti_mt(rng, wsp_);
    return component;
  }

  int MixedDataImputer::impute_cluster(
      Ptr<MixedImputation::CompleteData> &row,
      RNG &rng,
      bool update_complete_data_suf) {
    int component = impute_cluster(row, rng);
    if (update_complete_data_suf) {
      mixing_distribution_->suf()->update_raw(component);
    }
    return component;
  }

  void MixedDataImputer::sample_posterior() {
    for (int i = 0; i < empirical_distributions_.size(); ++i) {
      empirical_distributions_[i].update_cdf();
    }
    impute_all_rows();
    mixing_distribution_->sample_posterior();
    for (int s = 0; s < mixture_components_.size(); ++s) {
      mixture_components_[s]->sample_posterior();
    }
    numeric_data_model_->sample_posterior();
  }

  void MixedDataImputer::ensure_swept_sigma_current() const {
    if (swept_sigma_current_) return;
    swept_sigma_ = SweptVarianceMatrix(numeric_data_model_->Sigma());
    swept_sigma_current_ = true;
  }

  void MixedDataImputer::create_encoders(const DataTable &table) {
    encoder_.reset(new DatasetEncoder(true));
    if (!complete_data_.empty()) {
      for (int i = 0; i < table.nvars(); ++i) {
        if (table.variable_type(i) == VariableType::categorical) {
          NEW(EffectsEncoder, encoder)(i, table.get_nominal(i).key());
          encoders_.push_back(encoder);
          encoder_->add_encoder(encoder);
        }
      }
    }
  }

  void MixedDataImputer::initialize_regression_component() {
    if (!complete_data_.empty()) {
      int xdim = encoder_->dim();
      int ydim = complete_data_[0]->observed_data().numeric_dim();
      numeric_data_model_.reset(new MultivariateRegressionModel(xdim, ydim));
    }
  }

  void MixedDataImputer::initialize_empirical_distributions(
      const DataTable &data, const std::vector<Vector> &atoms) {
    Vector probs(99);
    for (int i = 0; i < probs.size(); ++i) {
      probs[i] = (i + 1.0) / 100.0;
    }

    for (int i = 0; i < data.nvars(); ++i) {
      if (data.variable_type(i) == VariableType::numeric) {
        empirical_distributions_.push_back(IQagent(probs, uint(1e+6)));
        empirical_distributions_.back().add(data.getvar(i));
      }
    }
  }


  void MixedDataImputer::set_observers() {
    numeric_data_model_->Sigma_prm()->add_observer(
        [this]() {this->swept_sigma_current_ = false;});
  }


  void MixedDataImputer::initialize_mixture(
      int num_clusters,
      const std::vector<Vector> &atoms,
      const std::vector<Ptr<CatKey>> &levels,
      const std::vector<VariableType> &variable_type) {
    mixing_distribution_.reset(new MultinomialModel(num_clusters));
    for (int c = 0; c < num_clusters; ++c) {
      auto num_it = atoms.begin();
      auto cat_it = levels.begin();
      Ptr<MixedImputation::RowModel> row_model(new MixedImputation::RowModel);
      for (int j = 0; j < variable_type.size(); ++j) {
        switch (variable_type[j]) {
          case VariableType::numeric:
            row_model->add_numeric(
                new MixedImputation::NumericErrorCorrectionModel(j, *num_it));
            ++num_it;
            break;

          case VariableType::categorical:
            row_model->add_categorical(
                new MixedImputation::CategoricalErrorCorrectionModel(j, *cat_it));
            ++cat_it;
            break;

          default:
            report_error("Only numeric or categorical varaibles are supported.");
        }
      }
      mixture_components_.push_back(row_model);
    }
  }

} // namespace BOOM
