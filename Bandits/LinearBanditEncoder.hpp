#ifndef BOOM_BANDITS_LINEAR_BANDIT_ENCODER_HPP_
#define BOOM_BANDITS_LINEAR_BANDIT_ENCODER_HPP_
/*
  Copyright (C) 2005-2026 Steven L. Scott

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

#include "cpputil/RefCounted.hpp"
#include "stats/Design.hpp"
#include "stats/Encoders.hpp"
#include <ostream>

namespace BOOM {

  // Forward declaration.
  class LinearBanditEncoder;
  class ArmMap;
  
  //===========================================================================
  // An ArmMap is a bijective mapping between arm definitions and factor values.
  class ArmMap : private RefCounted {
   public:
    ArmMap(const ExperimentStructure &xp);

    int number_of_arms() const {
      return arm_values_.size();
    }

    int number_of_factors() const {
      return arm_values_[0].size();
    }

    const std::vector<std::string> &factor_names() const {
      return xp_.factor_names();
    }
    
    const std::vector<int> &integer_factor_levels(int arm) const {
      return arm_values_[arm];
    }

    std::vector<std::string> factor_level_names(int arm) const;

    const ExperimentStructure &structure() const {
      return xp_;
    }

    friend void intrusive_ptr_add_ref(ArmMap *am) {am->up_count();}
    friend void intrusive_ptr_release(ArmMap *am) {
      am->down_count();
      if (am->ref_count() == 0) {
        delete am;
      }
    }

    std::string to_string() const;
    std::ostream &print(std::ostream &out) const;
    
   private:
    void FillArmValues_(const ExperimentStructure &xp);
    ExperimentStructure xp_;

    // Each row of arm_values is the unique index of an arm.  Each column is a
    // factor.  Element (i, j) gives the (integer) index
    std::vector<std::vector<int>> arm_values_;
  };

  inline std::ostream &operator<<(std::ostream &out, const ArmMap &arm_map) {
    return arm_map.print(out);
  }
  
  //===========================================================================
  // An ExperimentArmEncoder is a wrapper around an EffectsEncoder.  It is
  // designed to owned by a LinearBanditEncoder and encode the value of an
  // experimental factor when the parent encoder is asked to encode an (arm,
  // context) pair.
  class ExperimentArmEncoder : public EffectsEncoderBase {
   public:
    ExperimentArmEncoder(const std::string &variable_name,
                             const Ptr<ArmMap> &arm_map,
                             const std::string &baseline_level = "");

    ExperimentArmEncoder(const ExperimentArmEncoder &rhs);
    ExperimentArmEncoder &operator=(const ExperimentArmEncoder &rhs);
    ExperimentArmEncoder(ExperimentArmEncoder &&rhs);
    ExperimentArmEncoder & operator=(ExperimentArmEncoder &&rhs);
    ExperimentArmEncoder * clone() const override;

    const CatKey &key() const override {return *key_;}

    int dim() const override;

    // The table argument is ignored here.
    Matrix encode_dataset(const DataTable &table) const override;
    Matrix encode(const CategoricalVariable &variable) const override;

    Vector encode_row(const MixedMultivariateData &data) const override;
    void encode_row(const MixedMultivariateData &data,
                    VectorView view) const override;

    std::vector<std::string> encoded_variable_names() const override;
    
    int baseline_level() const override;

    void set_current_experiment_level(int level) {
      current_level_ = level;
    }
    
   private:
    Ptr<ArmMap> arm_map_;
    Ptr<CatKey> key_;
    int current_level_;
    std::string baseline_level_;
    int baseline_level_index_;
  };

  //===========================================================================
  // Converts a combination of experimental factors and context data into a
  // design/predictor matrix suitable for feeding to a generalized linear model.
  //
  // The LinearBanditEncoder generalizes a 'DatasetEncoder'.  A DatasetEncoder
  // assumes that all the relevant variables are present in the data_table being
  // encoded.  A LinearBanditEncoder
  class LinearBanditEncoder : private RefCounted {
   public:

    // Args:
    //   arm_map:  Describes the arms of the experiment.
    //   dataset_encoder: The encoder responsible for encoding the data set.
    //     The dataset_encoder must include as main effects one experiment
    //     encoder for each experiment factor present in arm_map.
    LinearBanditEncoder(const Ptr<ArmMap> &arm_map,
                        const Ptr<DatasetEncoder> &dataset_encoder);

    // Args:
    //   arm: The arm describing the values to assume for the action variables
    //     in the experiment.
    //   context: The collection of context variables for this experimental
    //     unit.
    //
    // Returns:
    //   The vector of predictors for this experimental unit under the
    //   designated arm.
    Vector encode_row(int arm, const MixedMultivariateData &context);

    // Args:
    //   input_data: The data set to encode into a matrix of predictor
    //     variables.  It is assumed that this data set contains past data,
    //     including values for any action variables associated with each row.
    //
    // Returns:
    //   A Matrix of predictor values.
    Matrix encode_dataset(const DataTable &input_data) const;

    int number_of_arms() const {
      return arm_map_->number_of_arms();
    }
    
   private:
    // Loop over all the encoders in the data encoder.  Find all the
    // ExperimentArmEncoder objects.  Inspect each one and make sure it
    // corresponds to a variable name in the arm map, and that all arm_map
    // variables are covered.
    //
    // Effects:
    //   Populates experiment_encoders_.
    void ensure_arm_coverage();
    
    Ptr<ArmMap> arm_map_;
    Ptr<DatasetEncoder> dataset_encoder_;
    std::map<std::string, Ptr<ExperimentArmEncoder>> experiment_encoders_;

    friend void intrusive_ptr_add_ref(LinearBanditEncoder *d) { d->up_count(); }
    friend void intrusive_ptr_release(LinearBanditEncoder *d) {
      d->down_count();
      if (d->ref_count() == 0) delete d;
    }

  };

  
}  // namespace BOOM

#endif  // BOOM_BANDITS_LINEAR_BANDIT_ENCODER_HPP_
