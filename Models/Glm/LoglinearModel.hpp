#ifndef BOOM_GLM_LOGLINEAR_MODEL_HPP_
#define BOOM_GLM_LOGLINEAR_MODEL_HPP_
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

#include "Models/CategoricalData.hpp"
#include "Models/Policies/ParamPolicy_1.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/Glm/GlmCoefs.hpp"
#include "Models/Glm/Encoders.hpp"

#include "distributions/rng.hpp"
#include "stats/DataTable.hpp"

namespace BOOM {

  class MultivariateCategoricalData
      : public Data {
   public:
    MultivariateCategoricalData() {}

    MultivariateCategoricalData(const MultivariateCategoricalData &rhs);
    MultivariateCategoricalData(MultivariateCategoricalData &&rhs) = default;
    MultivariateCategoricalData & operator=(const MultivariateCategoricalData &rhs);
    MultivariateCategoricalData &operator=(MultivariateCategoricalData &&rhs) = default;

    MultivariateCategoricalData *clone() const override {
      return new MultivariateCategoricalData(*this);
    }

    std::ostream &display(std::ostream &out) const override;

    void push_back(const Ptr<CategoricalData> &scalar) {
      data_.push_back(scalar);
    }

    const CategoricalData & operator[](int i) const {
      return *data_[i];
    }

    int nvars() const { return data_.size(); }

   private:
    std::vector<Ptr<CategoricalData>> data_;
  };

  //===========================================================================
  // An encoder that only handles categorical data.

  class CategoricalDataEncoder {
   public:
    virtual Vector encode_categorical_data(
        const MultivariateCategoricalData &data) const = 0;
    virtual int dim() const = 0;
  };
  //---------------------------------------------------------------------------

  class CategoricalMainEffect :
      public EffectsEncoder,
      public CategoricalDataEncoder {
   public:
    CategoricalMainEffect(int which_variable, const Ptr<CatKey> &key);

    Vector encode_categorical_data(
        const MultivariateCategoricalData &data) const override;
    int dim() const override {return EffectsEncoder::dim();}
  };

  //---------------------------------------------------------------------------
  class CategoricalInteractionEncoder :
      public InteractionEncoder,
      public CategoricalDataEncoder {
   public:
    CategoricalInteractionEncoder(
        const Ptr<CategoricalMainEffect> &main1,
        const Ptr<CategoricalMainEffect> &main2);

    CategoricalInteractionEncoder(
        const Ptr<CategoricalMainEffect> &main,
        const Ptr<CategoricalInteractionEncoder> &interaction);

    CategoricalInteractionEncoder(
        const Ptr<CategoricalInteractionEncoder> &interaction1,
        const Ptr<CategoricalInteractionEncoder> &interaction2);

    Vector encode_categorical_data(
        const MultivariateCategoricalData &data) const;

    int dim() const override {return InteractionEncoder::dim();}

   private:
    CategoricalDataEncoder *enc1_;
    CategoricalDataEncoder *enc2_;
  };

  //---------------------------------------------------------------------------
  class CategoricalDatasetEncoder
      : public DatasetEncoder,
        public CategoricalDataEncoder {
   public:
    void add_main_effect(const Ptr<CategoricalMainEffect> &main);
    void add_interaction(const Ptr<CategoricalInteractionEncoder> &interaction);

    int dim() const override {return DatasetEncoder::dim();}

    Vector encode_categorical_data(
        const MultivariateCategoricalData &data) const override;

   private:
    // This set of encoders parallels the set maintined by DatasetEncoder.  The
    // raw pointers are okay because the base class maintains the objects in
    // Ptr's.
    std::vector<CategoricalDataEncoder *> categorical_encoders_;
  };

  //===========================================================================
  class LoglinearModel
      : public ParamPolicy_1<GlmCoefs>,
        public IID_DataPolicy<MultivariateCategoricalData>,
        public PriorPolicy {
   public:
    LoglinearModel();

    // Buid a LoglinearModel from the categorical variables in a DataTable.
    explicit LoglinearModel(const DataTable &table);

    LoglinearModel *clone() const override;

    // Perform one Gibbs sampling pass over the data point.
    void impute(MultivariateCategoricalData &data_point, RNG &rng);

    // The number of categorical variables being modeled.
    int nvars() const;

    void add_interaction(const std::vector<int> &variable_postiions);

    const GlmCoefs &coef() const {return prm_ref();}

    double logp(const MultivariateCategoricalData &data_point) const;

   private:
    CategoricalDatasetEncoder encoder_;
  };



}  // namespace BOOM

#endif  // BOOM_GLM_LOGLINEAR_MODEL_HPP_
