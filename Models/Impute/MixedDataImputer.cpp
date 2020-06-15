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

namespace BOOM {

  //===========================================================================
  namespace MixedImputation {

    CompleteData::CompleteData(const Ptr<MixedMultivariateData> &data)
        : observed_data_(data),
          y_true_(data->numeric_dim()),
          y_numeric_(data->numeric_dim()),
          true_categories_(data->dim())
    {}

    CompleteData::CompleteData(const CompleteData &rhs)
        : Data(rhs),
          observed_data_(rhs.observed_data_->clone()),
          y_true_(rhs.y_true_),
          y_numeric_(rhs.y_numeric_),
          true_categories_(rhs.true_categories_)
    {}

    CompleteData & CompleteData::operator=(const CompleteData &rhs) {
      if (&rhs != this) {
        observed_data_ = rhs.observed_data_->clone();
        y_true_ = rhs.y_true_;
        y_numeric_ = rhs.y_numeric_;
        true_categories_ = rhs.true_categories_;
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
                                          const Ptr<CatKey> &levels,
                                          const Ptr<CatKey> &atoms)
        : ErrorCorrectionModelBase(index),
          levels_(levels),
          atoms_(atoms),
          truth_model_(new MultinomialModel(levels_->max_levels()))
    {
      for (int i = 0; i < atoms_->max_levels() + 2; ++i) {
        obs_models_.push_back(
            new MultinomialModel(levels_->max_levels()));
      }
    }

    CECM::CategoricalErrorCorrectionModel(const CECM &rhs)
        : ErrorCorrectionModelBase(rhs),
          levels_(rhs.levels_),
          atoms_(rhs.atoms_),
          truth_model_(rhs.truth_model_->clone())
    {
      for (int i = 0; i < rhs.obs_models_.size(); ++i) {
        obs_models_.push_back(rhs.obs_models_[i]->clone());
      }
    }

    CECM & CECM::operator=(const CECM &rhs) {
      if (&rhs != this) {
        ErrorCorrectionModelBase::operator=(rhs);
        levels_ = rhs.levels_;
        atoms_ = rhs.atoms_;
        truth_model_.reset(rhs.truth_model_->clone());
        obs_models_.clear();
        for (int i = 0; i < rhs.obs_models_.size(); ++i) {
          obs_models_.push_back(rhs.obs_models_[i]->clone());
        }
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

    void CECM::sample_posterior() {
      truth_model_->sample_posterior();
      for (int i = 0; i < obs_models_.size(); ++i) {
        obs_models_[i]->sample_posterior();
      }
    }

    double CECM::logp(const MixedMultivariateData &data) const {
      const CategoricalData &scalar(data.categorical(index()));

      bool missing = scalar.missing() != Data::missing_status::observed;
      int value;
      if (missing) {
        value = number_of_atoms() + 1;
      } else {
        value = scalar.value();
      }

      wsp_ = truth_model_->logpi();
      for (int i = 0; i <= number_of_atoms(); ++i) {
        wsp_[i] += obs_models_[i]->logpi()[value];
      }
      return lse(wsp_);
    }

    //===========================================================================
    RowModel::RowModel(const std::vector<InitInfo> &init) {
      for (int i = 0; i < init.size(); ++i) {
        if (!init[i].numeric_atoms.empty()) {
          scalar_models_.push_back(new NumericErrorCorrectionModel(
              i, init[i].numeric_atoms));
        } else {
          scalar_models_.push_back(new CategoricalErrorCorrectionModel(
              i, init[i].levels, init[i].categorical_atoms));
        }
      }
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

    // For imputation purposes
    void RowModel::impute_categorical(
        Ptr<MixedImputation::CompleteData> &row,
        RNG &rng,
        bool update_complete_data_suf) {


    }

  }  // namespace MixedImputation
  //===========================================================================

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

  void MixedDataImputer::impute_row(Ptr<MixedImputation::CompleteData> &row,
                                    RNG &rng,
                                    bool update_complete_data_suf)  {
    ensure_swept_sigma_current();
    int component = impute_cluster(row, rng, update_complete_data_suf);

    // This step will fill in the "true_categories" data element in *row.
    mixture_components_[component]->impute_categorical(
        row, rng, update_complete_data_suf);

    impute_numeric_given_categorical(row, rng, update_complete_data_suf);
  }

  void MixedDataImputer::impute_all_rows() {
    clear_client_data();
    for (size_t i = 0; i < complete_data_.size(); ++i) {
      impute_row(complete_data_[i], rng_, true);
    }
  }

  void MixedDataImputer::impute_numeric_given_categorical(
      Ptr<MixedImputation::CompleteData> &row,
      RNG &rng,
      bool update_complete_data_suf) {
    report_error("Not implemented.");
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


} // namespace BOOM
