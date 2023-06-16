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

#include <map>
#include <cstdint>

#include "Models/CategoricalData.hpp"
#include "Models/Sufstat.hpp"
#include "Models/Policies/ParamPolicy_1.hpp"
#include "Models/Policies/SufstatDataPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/Glm/GlmCoefs.hpp"

#include "LinAlg/Array.hpp"

#include "distributions/rng.hpp"
#include "stats/DataTable.hpp"
#include "stats/Encoders.hpp"

namespace BOOM {

  //===========================================================================
  // A data type representing a collection of categorical variables.
  class MultivariateCategoricalData
      : public Data {
   public:
    // Args:
    //   data: The vector of categorical data defining an observation.  If
    //     desired, this can be empty at construction time and built up using
    //     calls to push_back.
    //   frequency: The number of observations represented by this data point.
    //     For data representing individual observations in a data table,
    //     frequency will be 1.  For data representing cells in a contingency
    //     table frequency is typically an integer > 1.
    explicit MultivariateCategoricalData(
        const std::vector<Ptr<CategoricalData>> &data = {},
        double frequency = 1.0)
        : data_(data), frequency_(frequency) {}

    // Create a data point by reading the categorical variables from a row of
    // the data table.  If there are no categorical variables in the table the
    // resulting data point has nvars == 0.
    MultivariateCategoricalData(const DataTable &table, int row_number, double frequency = 1.0);

    MultivariateCategoricalData(const MultivariateCategoricalData &rhs);
    MultivariateCategoricalData(MultivariateCategoricalData &&rhs) = default;
    MultivariateCategoricalData & operator=(const MultivariateCategoricalData &rhs);
    MultivariateCategoricalData &operator=(
        MultivariateCategoricalData &&rhs) = default;

    MultivariateCategoricalData *clone() const override {
      return new MultivariateCategoricalData(*this);
    }

    std::ostream &display(std::ostream &out) const override;

    // Add a new categorical variable to the back of the collection.
    void push_back(const Ptr<CategoricalData> &scalar) {
      data_.push_back(scalar);
    }

    Ptr<CategoricalData> mutable_element(int i) {
      return data_[i];
    }

    // Recover the variable in position i.
    const CategoricalData & operator[](int i) const {
      return *data_[i];
    }

    // The number of variables in the collection.
    int nvars() const { return data_.size(); }

    double frequency() const {return frequency_;}

    std::vector<int> to_vector() const;

   private:
    std::vector<Ptr<CategoricalData>> data_;
    double frequency_;
  };

  //===========================================================================
  // Convert a categorical variable to a Vector suitable for analysis by a
  // LoglinearModel.
  class CategoricalDataEncoder : private RefCounted {
   public:

    // A vector containing a 1/0/-1 effects encoding of the input data.
    virtual Vector encode(const MultivariateCategoricalData &data) const = 0;

    // Args:
    //   data: The full data vector.  Each element is an entry in
    //     MultivariateCategoricalData.
    virtual Vector encode(const std::vector<int> &data) const = 0;

    // The number of columns in the Vector returned by 'encode'.
    virtual int dim() const = 0;

    // The indices of the variables driving this effect.
    virtual const std::vector<int> &which_variables() const = 0;

    // The number of levels in each variable.
    virtual const std::vector<int> &nlevels() const = 0;

   private:
    friend void intrusive_ptr_add_ref(CategoricalDataEncoder *d) {
      d->up_count();
    }
    friend void intrusive_ptr_release(CategoricalDataEncoder *d) {
      d->down_count();
      if (d->ref_count() == 0) {
        delete d;
      }
    }
  };

  //---------------------------------------------------------------------------
  // A CategoricalDataEncoder focusing on a single variable.
  class CategoricalMainEffect : public CategoricalDataEncoder {
   public:
    CategoricalMainEffect(int which_variable, const Ptr<CatKeyBase> &key);

    Vector encode(const MultivariateCategoricalData &data) const override;
    Vector encode(const std::vector<int> &data) const override;
    int dim() const override {return encoder_.dim();}
    const std::vector<int> &which_variables() const override {
      return which_variables_;
    }
    const std::vector<int> &nlevels() const override {return nlevels_;}

   private:
    EffectsEncoder encoder_;

    // Identifies the index of the relevant variable.
    std::vector<int> which_variables_;

    // The number of levels in the relevant variable.
    std::vector<int> nlevels_;
  };

  //---------------------------------------------------------------------------
  // A CategoricalDataEncoder representing the interaction between lower-order
  // effects.  Interactions are built from two effects at a time.  Higher order
  // interactions are built from the interaction of two lower order interactions
  // or main effects.
  class CategoricalInteraction :
      public CategoricalDataEncoder {
   public:
    CategoricalInteraction(const Ptr<CategoricalDataEncoder> &enc1,
                           const Ptr<CategoricalDataEncoder> &enc2);

    Vector encode(const MultivariateCategoricalData &data) const override;
    Vector encode(const std::vector<int> &data) const override;
    int dim() const override {return enc1_->dim() * enc2_->dim();}
    const std::vector<int> &which_variables() const override {
      return which_variables_;
    }
    const std::vector<int> &nlevels() const override {return nlevels_;}

   private:
    Ptr<CategoricalDataEncoder> enc1_;
    Ptr<CategoricalDataEncoder> enc2_;
    std::vector<int> which_variables_;
    std::vector<int> nlevels_;
  };

  //---------------------------------------------------------------------------
  // The "parent" encoder class containing main effects and interactions.
  class MultivariateCategoricalEncoder {
   public:
    explicit MultivariateCategoricalEncoder(bool add_intercept=true)
        : add_intercept_(add_intercept),
          dim_(add_intercept_)
    {}

    void add_effect(const Ptr<CategoricalDataEncoder> &effect);
    int number_of_effects() const {return encoders_.size();}
    int effect_position(const std::vector<int> &which_variables) const;

    int dim() const {return dim_;}

    Vector encode(const MultivariateCategoricalData &data) const;
    Vector encode(const std::vector<int> &data) const;

    const CategoricalDataEncoder &encoder(int i) const;
    const CategoricalDataEncoder &encoder(
        const std::vector<int> &which_variables) const;

   private:
    // Encoders are stored in a vector, to preserve order, and simultaneously in
    // a map, for easy lookup by index.
    std::vector<Ptr<CategoricalDataEncoder>> encoders_;
    std::map<std::vector<int>, Ptr<CategoricalDataEncoder>> encoders_by_index_;

    // effect_position_[eff] is the index in the parameter vector corresponding
    // to the first element of effect eff.  The entries in eff_ are unique and
    // sorted in ascending order.
    std::map<std::vector<int>, int> effect_position_;

    // If true then prepend a constant '1' corresponding to the intercept term
    // when calling 'encode'.
    bool add_intercept_;

    int dim_;
  };

  //===========================================================================
  // The sufficient statistics for a log linear model are the marginal cross
  // tabulations for each effect in the model.
  class LoglinearModelSuf : public SufstatDetails<MultivariateCategoricalData> {
   public:
    LoglinearModelSuf() : sample_size_(0), valid_(true) {}
    LoglinearModelSuf *clone() const override {
      return new LoglinearModelSuf(*this);
    }

    std::ostream &print(std::ostream &out) const override;

    // vectorize/unvectorize packs the data but not the sizes or model
    // structure.
    Vector vectorize(bool minimal = true) const override;
    Vector::const_iterator unvectorize(
        Vector::const_iterator &v, bool minimal=true) override;
    Vector::const_iterator unvectorize(
        const Vector &v, bool minimal=true) override;

    // Add a main effect or interaction to the model structure.
    //
    // If data has already been allocated to the object, adding an effect
    // invalidates the object.  To put it back in a valid state call "refresh"
    // and pass the original data.
    //
    // If all elements of model structure are added prior to calling
    // Update. Then no refreshing is needed.
    void add_effect(const Ptr<CategoricalDataEncoder> &effect);

    // Clear the data but keep the information about model structure.  Set the
    // valid_ flag to true.
    void clear() override;

    // Clear everything.
    void clear_data_and_structure();

    // Clear the data and recompute the sufficient statistics.
    void refresh(const std::vector<Ptr<MultivariateCategoricalData>> &data);

    // It is an error to update the sufficient statistics with new data when the
    // object is in an invalid state.  The easiest way to prevent this from
    // happening is to add all elements of model structure before calling
    // update.
    void Update(const MultivariateCategoricalData &data) override;

    // Note that 'combine' assumes that the two suf's being combined have the
    // same structure.
    void combine(const LoglinearModelSuf &suf);
    void combine(const Ptr<LoglinearModelSuf> &suf);
    LoglinearModelSuf *abstract_combine(Sufstat *s) override;

    // Args:
    //   index: The indices of the variables in the desired margin.  For main
    //     effects the index will just contain one number.  For 2-way
    //     interactions it will contain 2 numbers, and for k-way interactions it
    //     will contain k numbers.  The elements of 'index' should be in
    //     increasing order: [0, 3, 4] is okay.  [3, 0, 4] is not.
    //
    // Returns:
    //   An array with dimensions corresponding to the variables in the desired
    //   margin.  The index of each array dimension corresponds to the level of
    //   that variable.  The array entry at (for example) (i, j, k) is the
    //   number of times X0 == i, X1 == j, and X2 == k.
    const Array &margin(const std::vector<int> &index) const;

    std::int64_t sample_size() const {return sample_size_;}

   private:
    std::vector<Ptr<CategoricalDataEncoder>> effects_;

    // Cross tabulations are indexed by a vector containing the indices of the
    // tabulated variables.  For example, a 3-way interaction might include
    // variables 0, 2, and 5.  The indices must be in order.
    std::map<std::vector<int>, Array> cross_tabulations_;

    std::int64_t sample_size_;

    // The state of the object.  The state becomes invalid each time an effect
    // is added.  The state can be made valid by calling clear() or refresh().
    bool valid_;
  };

  //===========================================================================
  class LoglinearModel
      : public ParamPolicy_1<GlmCoefs>,
        public SufstatDataPolicy<MultivariateCategoricalData, LoglinearModelSuf>,
        public PriorPolicy {
   public:

    // An empty LoglinearModel.  The fisrt time this model calls add_data main
    // effects will be added for each variable in the added data point.
    // If interactions are later added, then
    LoglinearModel();

    // Build a LoglinearModel from the categorical variables in a DataTable.
    //
    // A model built with this constructor must call refresh_suf() after all
    // model structure is added.
    explicit LoglinearModel(const DataTable &table);

    LoglinearModel *clone() const override;

    void add_data(const Ptr<MultivariateCategoricalData> &data_point) override;
    void add_data(const Ptr<Data> &dp) override { add_data(DAT(dp)); }
    void add_data(MultivariateCategoricalData *dp) override {
      add_data(Ptr<MultivariateCategoricalData>(dp));
    }

    // Perform one Gibbs sampling pass over the data point.
    void impute(MultivariateCategoricalData &data_point, RNG &rng);

    // The number of categorical variables being modeled.
    int nvars() const;

    // The number of elements in the (dense) parameter vector.
    int dim() const { return coef().nvars_possible(); }

    void add_interaction(const std::vector<int> &variable_postiions);

    void refresh_suf();

    const GlmCoefs &coef() const {return prm_ref();}

    void set_effect_coefficients(const Vector &coefficients,
                                 int encoder_index);
    void set_effect_coefficients(const Vector &coefficients,
                                 const std::vector<int> &index);

    double logp(const MultivariateCategoricalData &data_point) const;
    double logp(const std::vector<int> &data_point) const;

    // Fill each missing value in the data set with draws from the discrete
    // uniform distribution over the range of that variable.
    void initialize_missing_data(RNG &rng);

    // Make one Gibbs sampling pass through the data, filling in each
    // observation conditional on observed and all other missing data.
    void impute_missing_data(RNG &rng);

    int number_of_effects() const {return encoder_.number_of_effects();}
    const CategoricalDataEncoder &encoder(int i) const {
      return encoder_.encoder(i);
    }

   private:
    // Implementation for impute_missing_data.
    void impute_single_variable(
        MultivariateCategoricalData &observation, int position, RNG &rng,
        std::vector<int> &workspace);

    // Add the effect to the encoder, to the sufficient statistics, and resize
    // the coefficient vector.
    void add_effect(const Ptr<CategoricalDataEncoder> &effect);

    // The main_effects are used to build interaction terms.
    std::vector<Ptr<CategoricalMainEffect>> main_effects_;

    MultivariateCategoricalEncoder encoder_;
  };

}  // namespace BOOM

#endif  // BOOM_GLM_LOGLINEAR_MODEL_HPP_
