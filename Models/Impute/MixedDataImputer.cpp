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
          table.set_nominal_value(
              row, i, true_categories_[categorical_counter++]);
        } else {
          report_error(
              "Only numeric and categorical data types are supported.");
        }
      }
    }

    MixedMultivariateData CompleteData::to_mixed_multivariate_data() const {
      std::vector<Ptr<DoubleData>> numerics;
      std::vector<Ptr<LabeledCategoricalData>> categoricals;
      int numeric_counter = 0;
      int categorical_counter = 0;
      for (int i = 0; i < observed_data_->dim(); ++i) {
        VariableType vtype = observed_data_->variable_type(i);
        switch (vtype) {
          case VariableType::numeric:
            {
              numerics.push_back(new DoubleData(y_true_[numeric_counter++]));
            }
            break;

          case VariableType::categorical:
            {
              categoricals.push_back(observed_data_->categorical(i).clone());
              categoricals.back()->set(true_categories_[categorical_counter++]);
            }
            break;

          default:
            report_error(
                "Only numeric and categorical data types are supported.");
        }
      }
      return MixedMultivariateData(observed_data_->type_index(), numerics,
                                   categoricals);
    }

    //==========================================================================
    NumericScalarModel::NumericScalarModel(int index, const Vector &atoms)
        : ScalarModelBase(index),
          atom_model_(new MultinomialModel(atoms.size() + 1)),
          atoms_(atoms)
    {
      ParamPolicy::add_model(atom_model_);
    }

    NumericScalarModel::NumericScalarModel(const NumericScalarModel &rhs)
        : ScalarModelBase(rhs),
          atom_model_(rhs.atom_model_->clone()),
          atoms_(rhs.atoms_)
    {
      ParamPolicy::add_model(atom_model_);
      atom_model_->suf()->clear();
      atom_model_->suf()->combine(rhs.atom_model_->suf());
    }

    NumericScalarModel & NumericScalarModel::operator=(
        const NumericScalarModel &rhs) {
      if (&rhs != this) {
        atoms_ = rhs.atoms_;
        atom_model_ = rhs.atom_model_->clone();
        ParamPolicy::clear();
        ParamPolicy::add_model(atom_model_);
      }
      return *this;
    }

    NumericScalarModel * NumericScalarModel::clone() const {
      return new NumericScalarModel(*this);
    }

    double NumericScalarModel::logp(
        const MixedMultivariateData &data) const {
      const DoubleData &scalar(data.numeric(index()));
      double value = std::numeric_limits<double>::quiet_NaN();
      if (scalar.missing() == Data::missing_status::observed) {
        value = scalar.value();
      } else {
        // return atom_model_->entropy();
        return 0;
      }
      return logp(value);
    }

    double NumericScalarModel::logp(double observed) const {
      if (std::isnan(observed)) {
        return 0;
        // report_error("Argument cannot be NaN.");
      }
      return atom_model_->logpi()[category_map(observed)];
    }

    void NumericScalarModel::set_conjugate_prior(const Vector &counts) {
      if (counts.size() != atoms_.size() + 1) {
        std::ostringstream err;
        err << "Counts vector is size " << counts.size()
            << " expected " << atoms_.size() + 1;
        report_error(err.str());
      }
      atom_model_->clear_methods();
      NEW(ConstrainedMultinomialDirichletSampler, sampler)(
          atom_model_.get(), counts);
      atom_model_->set_method(sampler);
    }

    int NumericScalarModel::impute_atom(
        double observed_value, RNG &rng, bool update) {
      int atom = category_map(observed_value);
      if (atom == -1) {
        atom = atom_model_->sim(rng);
      }
      if (update) {
        atom_model_->suf()->update_raw(atom);
      }
      return atom;
    }

    int NumericScalarModel::category_map(double value) const {
      const double tiny_number = 1e-6;
      if (std::isnan(value)) return -1;
      for (int i = 0; i < atoms_.size(); ++i) {
        if (fabs(atoms_[i] - value) < tiny_number) {
          return i;
        }
      }
      return atoms_.size();
    }

    double NumericScalarModel::true_value(
        int true_atom, double observed_value) const {
      if (atoms_.empty()) {
        return observed_value;
      } else if (true_atom >= 0 && true_atom < atoms_.size()) {
        return atoms_[true_atom];
      } else if (category_map(observed_value) == atoms_.size()) {
        return observed_value;
      } else {
        std::ostringstream msg;
        msg << "Illegal value: true_atom = " << true_atom
            << " observed_value = " << observed_value << ".";
        report_error(msg.str());
        return -1;
      }
    }

    // When the true value is atomic the numeric value is NaN.
    double NumericScalarModel::numeric_value(
        int true_atom, double observed_value) const {
      if (true_atom == atoms_.size()
          && category_map(observed_value) == atoms_.size()) {
        return observed_value;
      } else {
        return std::numeric_limits<double>::quiet_NaN();
      }
    }

    //==========================================================================
    namespace {
      using CSM = CategoricalScalarModel;
    }  // namespace

    CategoricalScalarModel::CategoricalScalarModel(
        int index, const Ptr<CatKey> &levels):
        ScalarModelBase(index),
        levels_(levels),
        model_(new MultinomialModel(levels_->max_levels()))
    {
      build_atom_index();
    }

    CategoricalScalarModel::CategoricalScalarModel(
        const CategoricalScalarModel &rhs)
        : ScalarModelBase(rhs),
          levels_(rhs.levels_),
          atom_index_(rhs.atom_index_),
          model_(rhs.model_->clone())
    {}

    CategoricalScalarModel & CategoricalScalarModel::operator=(
        const CategoricalScalarModel &rhs) {
      if (&rhs != this) {
        ScalarModelBase::operator=(rhs);
        levels_ = rhs.levels_;
        atom_index_ = rhs.atom_index_;
        model_ = rhs.model_->clone();
      }
      return *this;
    }

    CategoricalScalarModel * CategoricalScalarModel::clone() const {
      return new CategoricalScalarModel(*this);
    }

    double CategoricalScalarModel::logp(const std::string &label) const {
      int index = atom_index(label);
      if (index >= 0) {
        return model_->logpi()[index];
      } else {
        return 0;
        // std::ostringstream err;
        // err << "Illegal level value: " << label << ".";
        // report_error(err.str());
        // return negative_infinity();
      }
    }

    double CategoricalScalarModel::logp(
        const MixedMultivariateData &data) const {
      const LabeledCategoricalData &scalar(data.categorical(index()));
      if (scalar.missing() != Data::missing_status::observed) {
        // return model_->entropy();
        return 0.0;
      } else {
        return logp(scalar.label());
      }
    }

    void CategoricalScalarModel::update_complete_data_suf(int observed_level) {
      model_->suf()->update_raw(observed_level);
    }

    void CategoricalScalarModel::set_conjugate_prior(const Vector &counts) {
      if (counts.size() != levels_->max_levels()) {
        std::ostringstream err;
        err << "Counts vector is size " << counts.size()
            << " expected " << levels_->max_levels() << ".";
        report_error(err.str());
      }
      model_->clear_methods();
      NEW(ConstrainedMultinomialDirichletSampler, sampler)(
          model_.get(), counts);
      model_->set_method(sampler);
    }

    void CategoricalScalarModel::build_atom_index() {
      atom_index_.clear();
      for (int i = 0; i < levels_->max_levels(); ++i) {
        std::string label = levels_->label(i);
        atom_index_[label] = i;
      }
    }

    int CategoricalScalarModel::atom_index(const std::string &label) const {
      auto it = atom_index_.find(label);
      if (it == atom_index_.end()) {
        return levels_->max_levels();
      } else {
        return it->second;
      }
    }

    //==========================================================================
    RowModelBase::RowModelBase() {}

    RowModelBase::RowModelBase(const RowModelBase &rhs) {
      operator=(rhs);
    }

    RowModelBase & RowModelBase::operator=(const RowModelBase &rhs) {
      if (&rhs != this) {
        scalar_models_.clear();
        ParamPolicy::clear();
        for (int i = 0; i < rhs.scalar_models_.size(); ++i) {
          add_scalar_model(rhs.scalar_models_[i]->clone());
        }
      }
      return *this;
    }

    void RowModelBase::add_scalar_model(const Ptr<ScalarModelBase> &model) {
        scalar_models_.push_back(model);
        ParamPolicy::add_model(model);
      }

    double RowModelBase::logp(const MixedMultivariateData &data) const {
      double ans = 0;
      for (const auto &el : scalar_models()) {
        ans += el->logp(data);
      }
      return ans;
    }

    void RowModelBase::clear_data() {
      for (size_t i = 0; i < scalar_models_.size(); ++i){
        scalar_models_[i]->clear_data();
      }
    }

    void RowModelBase::sample_posterior() {
      for (auto &model : scalar_models_) {
        model->sample_posterior();
      }
    }
    //==========================================================================
    RowModel::RowModel() {}

    RowModel::RowModel(const RowModel &rhs)
        : Model(rhs),
          RowModelBase(rhs)
    {
      populate_numeric_and_categorical_models();
    }

    RowModel & RowModel::operator=(const RowModel &rhs) {
      if (&rhs != this) {
        RowModelBase::operator=(rhs);
        populate_numeric_and_categorical_models();
      }
      return *this;
    }

    RowModel *RowModel::clone() const {return new RowModel(*this);}

    void RowModel::add_numeric(const Ptr<NumericScalarModel> &model) {
      RowModelBase::add_scalar_model(model);
      numeric_models_.push_back(model);
    }

    void RowModel::add_categorical(const Ptr<CategoricalScalarModel> &model) {
      RowModelBase::add_scalar_model(model);
      categorical_models_.push_back(model);
    }

    void RowModel::impute_atoms(Ptr<CompleteData> &row,
                                RNG &rng,
                                bool update_complete_data_suf) {
      const Vector &observed(row->y_observed());
      for (int i = 0; i < observed.size(); ++i) {
        // Impute the atom responsible for the true value.  If the observed
        // value contains no missing data this is a quick lookup.
        int atom = numeric_models_[i]->impute_atom(
            observed[i], rng, update_complete_data_suf);
        row->set_y_true(i, numeric_models_[i]->true_value(
            atom, observed[i]));
        row->set_y_numeric(i, numeric_models_[i]->numeric_value(
            atom, observed[i]));
      }
    }

    //----------------------------------------------------------------------
    // Impute the categorical data given the numeric data.  Later this can be
    // improved by marginalizing over the numeric data and imputing given y_obs.
    void RowModel::impute_categorical(
        Ptr<MixedImputation::CompleteData> &row,
        RNG &rng,
        bool update_complete_data_suf,
        const Ptr<DatasetEncoder> &encoder,
        const std::vector<Ptr<EffectsEncoder>> &encoders,
        const Ptr<MultivariateRegressionModel> &numeric_model) {

      // Ensure that the predictor vector is the right size.  This vector will
      // be filled with
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

      auto observed = Data::missing_status::observed;
      for (int i = 0; i < encoders.size(); ++i) {
        // 'i' indexes the set of categorical variables in the row.

        VectorView view(predictors, start, encoders[i]->dim());

        if (observed_categories[i]->missing() == observed) {
          // If the category was observed there is not need to impute it.
          imputed_categorical_data[i] = observed_categories[i]->value();
        } else {
          // The imputation distribution for the missing value is ...

          // p(x) * p(y | x)
          Vector imputation_distribution = categorical_models_[i]->log_probs();

          for (int level = 0; level < imputation_distribution.size(); ++level) {
            encoders[i]->encode(level, view);
            Vector yhat = numeric_model->predict(predictors);
            imputation_distribution[level] -=
                0.5 * numeric_model->Siginv().Mdist(y_numeric - yhat);
          }
          imputation_distribution.normalize_logprob();
          imputed_categorical_data[i] = rmulti_mt(rng, imputation_distribution);
          encoders[i]->encode(imputed_categorical_data[i], view);

        }
        if (update_complete_data_suf) {
          categorical_models_[i]->update_complete_data_suf(
              imputed_categorical_data[i]);
        }

      }
      row->set_true_categories(imputed_categorical_data);
    }

    void RowModel::populate_numeric_and_categorical_models() {
      numeric_models_.clear();
      categorical_models_.clear();
      for (int i = 0; i < scalar_models().size(); ++i) {
        Ptr<NumericScalarModel> model =
            scalar_models()[i].dcast<NumericScalarModel>();
        if (model) {
          numeric_models_.push_back(model);
        } else {
          categorical_models_.push_back(
              scalar_models()[i].dcast<CategoricalScalarModel>());
        }
      }
    }

  }  // namespace MixedImputation
  //===========================================================================

  MixedDataImputerBase::MixedDataImputerBase(
      int num_clusters,
      const DataTable &data,
      const std::vector<Vector> &atoms,
      RNG &seeding_rng)
      : data_types_(data.type_index()),
        mixing_distribution_(new MultinomialModel(num_clusters)),
        rng_(seed_rng(seeding_rng)),
        swept_sigma_(SpdMatrix(1)),
        swept_sigma_current_(false)
  {
    for (size_t i = 0; i < data.nrow(); ++i) {
      add_data(data.row(i));
    }
    summarize_data(data);
    create_encoders(data);
    initialize_empirical_distributions(data, atoms);
    initialize_regression_component();
    set_numeric_data_model_observers();
  }

  void MixedDataImputerBase::initialize(const std::vector<Vector> &atoms) {
    // int num_clusters = mixing_distribution_->dim();
    auto data_point = dat()[0];
    std::vector<Ptr<CatKey>> levels;
    for (int j = 0; j < data_point->dim(); ++j) {
      if (data_point->variable_type(j) == VariableType::categorical) {
        levels.push_back(data_point->categorical(j).catkey());
      }
    }
    //    initialize_mixture(num_clusters, atoms, levels, variable_types);
  }

  MixedDataImputerBase::MixedDataImputerBase(const MixedDataImputerBase &rhs)
      : data_types_(rhs.data_types_),
        mixing_distribution_(rhs.mixing_distribution_->clone()),
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
    set_numeric_data_model_observers();
  }

  MixedDataImputerBase &MixedDataImputerBase::operator=(
      const MixedDataImputerBase &rhs) {
    if (&rhs != this) {
      data_types_ = rhs.data_types_;
      mixing_distribution_.reset(rhs.mixing_distribution_->clone());
      swept_sigma_ = rhs.swept_sigma_;
      set_numeric_data_model(rhs.numeric_data_model_->clone());
      empirical_distributions_ = rhs.empirical_distributions_;

      encoder_.reset(new DatasetEncoder(rhs.encoder_->add_intercept()));
      encoders_.clear();
      for (int i = 0; i < rhs.encoders_.size(); ++i) {
        encoders_.push_back(rhs.encoders_[i]->clone());
        encoder_->add_encoder(encoders_.back());
      }

    }
    return *this;
  }

  void MixedDataImputerBase::add_data(const Ptr<MixedMultivariateData> &data) {
    complete_data_.push_back(new MixedImputation::CompleteData(data));
  }

  void MixedDataImputerBase::clear_data() {
    clear_client_data();
    complete_data_.clear();
  }

  void MixedDataImputerBase::clear_client_data() {
    numeric_data_model_->clear_data();
    mixing_distribution_->clear_data();
    clear_mixture_component_data();
  }

  void MixedDataImputerBase::impute_data_set(
      std::vector<Ptr<MixedImputation::CompleteData>> &rows) {
    for (auto &el : rows) {
      impute_row(el, rng_, false);
    }
  }

  void MixedDataImputerBase::impute_all_rows() {
    clear_client_data();
    for (size_t i = 0; i < complete_data_.size(); ++i) {
      impute_row(complete_data_[i], rng_, true);
    }
  }

  void MixedDataImputerBase::sample_posterior() {
    // Ensure that the ECDF's describing the marginal distribution of Y are up
    // to date.  This only has an impact in the first MCMC draw.  After that it
    // is a no-op.
    for (int i = 0; i < empirical_distributions_.size(); ++i) {
      empirical_distribution(i).update_cdf();
    }
    impute_all_rows();
    mixing_distribution_->sample_posterior();
    for (int s = 0; s < number_of_mixture_components(); ++s) {
      row_model(s)->sample_posterior();
    }
    numeric_data_model()->sample_posterior();
  }

  Vector MixedDataImputerBase::ybar() const {
    Vector ans(data_types_.number_of_numeric_fields());
    int index = 0;
    for (int i = 0; i < data_types_.total_number_of_fields(); ++i) {
      if (data_types_.variable_type(i) == VariableType::numeric) {
        const std::string &vname = data_types_.variable_names()[i];
        const auto it = numeric_summaries_.find(vname);
        if (it == numeric_summaries_.end()) {
          report_error("Found an un-summarized numeric column.");
        } else {
          ans[index++] = it->second.mean();
        }
      }
    }
    return ans;
  }

  int MixedDataImputerBase::impute_cluster(
      const Ptr<MixedImputation::CompleteData> &row,
      RNG &rng,
      bool update_complete_data_suf) {
    int component = impute_cluster(row, rng);
    if (update_complete_data_suf) {
      mixing_distribution_->suf()->update_raw(component);
    }
    return component;
  }

  int MixedDataImputerBase::impute_cluster(
      const Ptr<MixedImputation::CompleteData> &row, RNG &rng) const {
    int S = number_of_mixture_components();
    wsp_ = mixing_distribution_->logpi();
    for (int s = 0; s < S; ++s) {
      wsp_[s] += row_model(s)->logp(row->observed_data());
    }
    wsp_.normalize_logprob();
    int component = rmulti_mt(rng, wsp_);
    return component;
  }

  void MixedDataImputerBase::summarize_data(const DataTable &data) {
    for (int i = 0; i < data.nvars(); ++i) {
      switch(data.variable_type(i)) {
        case VariableType::numeric :
          numeric_summaries_[data.vnames()[i]] = NumericSummary(data.getvar(i));
          break;

        case VariableType::categorical :
          categorical_summaries_[data.vnames()[i]] = CategoricalSummary(
              data.get_nominal(i));
          break;

        case VariableType::datetime :
        default:
          std::ostringstream msg;
          msg << "Unsupported variable type for variable "
              << data.vnames()[i] << ".";
          report_error(msg.str());
      }
    }
  }

  void MixedDataImputerBase::impute_row(
      Ptr<MixedImputation::CompleteData> &row,
      RNG &rng,
      bool update_complete_data_suf)  {
    ensure_swept_sigma_current();
    int cluster = impute_cluster(row, rng, update_complete_data_suf);

    // This step will fill in the "true_categories" data element in *row.
    row_model(cluster)->impute_categorical(
        row,
        rng,
        update_complete_data_suf,
        encoder_,
        encoders_,
        numeric_data_model_);
    row_model(cluster)->impute_atoms(row, rng, update_complete_data_suf);
    impute_numerics_given_atoms(row, rng, update_complete_data_suf);
  }

  void MixedDataImputerBase::ensure_swept_sigma_current() const {
    if (swept_sigma_current_) return;
    swept_sigma_ = SweptVarianceMatrix(numeric_data_model_->Sigma());
    swept_sigma_current_ = true;
  }

  void MixedDataImputerBase::set_numeric_data_model_observers() {
    this->swept_sigma_current_ = false;
    numeric_data_model_->Sigma_prm()->add_observer(
        this,
        [this]() {this->swept_sigma_current_ = false;});
  }

  void MixedDataImputerBase::initialize_empirical_distributions(
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

  void MixedDataImputerBase::initialize_regression_component() {
    if (!complete_data_.empty()) {
      int xdim = encoder_->dim();
      int ydim = complete_data_[0]->observed_data().numeric_dim();
      set_numeric_data_model(new MultivariateRegressionModel(xdim, ydim));
    }
  }

  void MixedDataImputerBase::clear_mixture_component_data() {
    for (int s = 0; s < number_of_mixture_components(); ++s) {
      row_model(s)->clear_data();
    }
  }

  void MixedDataImputerBase::create_encoders(const DataTable &table) {
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

  //===========================================================================

  MixedDataImputer::MixedDataImputer(int num_clusters,
                                     const DataTable &data,
                                     const std::vector<Vector> &atoms,
                                     RNG &seeding_rng)
      : MixedDataImputerBase(num_clusters, data, atoms, seeding_rng)
    {
      std::vector<Ptr<CatKey>> levels;
      std::vector<VariableType> variable_types;
      for (int i = 0; i < data.nvars(); ++i) {
        variable_types.push_back(data.variable_type(i));
        if (variable_types.back() == VariableType::categorical) {
          levels.push_back(data.get_nominal(i).key());
        }
      }
      initialize_mixture(num_clusters, atoms, levels, variable_types);
    }

  MixedDataImputer::MixedDataImputer(const MixedDataImputer &rhs)
      : MixedDataImputerBase(rhs)
  {
    for (int i = 0; i < rhs.mixture_components_.size(); ++i) {
      mixture_components_.push_back(rhs.mixture_components_[i]->clone());
    }
  }

  MixedDataImputer * MixedDataImputer::clone() const {
    return new MixedDataImputer(*this);
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


  void MixedDataImputer::initialize_mixture(
      int num_clusters,
      const std::vector<Vector> &atoms,
      const std::vector<Ptr<CatKey>> &levels,
      const std::vector<VariableType> &variable_type) {
    for (int cluster = 0; cluster < num_clusters; ++cluster) {
      auto num_it = atoms.begin();
      auto cat_it = levels.begin();
      Ptr<MixedImputation::RowModel> row_model(new MixedImputation::RowModel);
      for (int j = 0; j < variable_type.size(); ++j) {
        switch (variable_type[j]) {
          case VariableType::numeric:
            row_model->add_numeric(
                new MixedImputation::NumericScalarModel(j, *num_it));
            ++num_it;
            break;

          case VariableType::categorical:
            row_model->add_categorical(
                new MixedImputation::CategoricalScalarModel(j, *cat_it));
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

} // namespace BOOM
