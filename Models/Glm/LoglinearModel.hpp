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

#include "Models/CategoricalData.hpp"
#include "Models/Sufstat.hpp"
#include "Models/Policies/ParamPolicy_1.hpp"
#include "Models/Policies/SufstatDataPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/Glm/GlmCoefs.hpp"
#include "Models/Glm/Encoders.hpp"

#include "LinAlg/Array.hpp"

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
  // An encoder that only handles categorical data.  The behavior of a
  // CategoricalDataEncoder is similar to a DataEncoder from Encoders.hpp, but
  // the purpose is to encode data from a MultivariateCategoricalData, not from
  // a DataTable.
  //
  // The classes are similar, but different enough that concrete
  // CategoricalDataEncoder classes should be implemented through composition
  // rather than inheritance.
  class CategoricalDataEncoder : private RefCounted {
   public:

    // A vector containing a 1/0/-1 effects encoding of the input data.
    virtual Vector encode(const MultivariateCategoricalData &data) const = 0;

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
  class CategoricalMainEffect : public CategoricalDataEncoder {
   public:
    CategoricalMainEffect(int which_variable, const Ptr<CatKeyBase> &key);

    Vector encode(const MultivariateCategoricalData &data) const override;
    int dim() const override {return encoder_.dim();}
    const std::vector<int> &which_variables() const override {
      return which_variables_;
    }
    const std::vector<int> &nlevels() const override {return nlevels_;}

   private:
    EffectsEncoder encoder_;
    std::vector<int> which_variables_;
    std::vector<int> nlevels_;
  };

  //---------------------------------------------------------------------------
  class CategoricalInteractionEncoder :
      public CategoricalDataEncoder {
   public:
    CategoricalInteractionEncoder(const Ptr<CategoricalDataEncoder> &enc1,
                                  const Ptr<CategoricalDataEncoder> &enc2);

    Vector encode(const MultivariateCategoricalData &data) const;
    int dim() const override {return enc1_->dim() * enc2_->dim();}
    const std::vector<int> &which_variables() const override {
      return which_variables_;
    }
    const std::vector<int> &nlevels() const {return nlevels_;}

   private:
    Ptr<CategoricalDataEncoder> enc1_;
    Ptr<CategoricalDataEncoder> enc2_;
    std::vector<int> which_variables_;
    std::vector<int> nlevels_;
  };

  //---------------------------------------------------------------------------
  class CategoricalDatasetEncoder {
   public:
    CategoricalDatasetEncoder() : dim_(0) {}

    void add_effect(const Ptr<CategoricalDataEncoder> &effect);

    int dim() const {return dim_;}

    Vector encode(const MultivariateCategoricalData &data) const;

   private:
    // This set of encoders parallels the set maintined by DatasetEncoder.  The
    // raw pointers are okay because the base class maintains the objects in
    // Ptr's.
    std::vector<Ptr<CategoricalDataEncoder>> encoders_;
    int dim_;
  };

  //===========================================================================
  // The sufficient statistics for a log linear model are the marginal cross
  // tabulations for each effect in the model.
  class LoglinearModelSuf : public SufstatDetails<MultivariateCategoricalData> {
   public:
    LoglinearModelSuf() : valid_(false) {}
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

    void add_effect(const Ptr<CategoricalDataEncoder> &effect);

    // Clear the data but keep the information about model structure.
    void clear() override;

    // Clear everything.
    void clear_data_and_structure();

    void Update(const MultivariateCategoricalData &data) override;

    // Note that 'combine' assumes that the two suf's being combined have the
    // same structure.
    void combine(const LoglinearModelSuf &suf);
    void combine(const Ptr<LoglinearModelSuf> &suf);
    LoglinearModelSuf *abstract_combine(Sufstat *s);

   private:
    std::vector<Ptr<CategoricalDataEncoder>> effects_;

    // Cross tabulations are indexed by a vector containing the indices of the
    // tabulated variables.  For example, a 3-way interaction might include
    // variables 0, 2, and 5.  The indices must be in order.
    std::map<std::vector<int>, Array> cross_tabulations_;

    bool valid_;
  };

  //===========================================================================
  class LoglinearModel
      : public ParamPolicy_1<GlmCoefs>,
        public SufstatDataPolicy<MultivariateCategoricalData, LoglinearModelSuf>,
        public PriorPolicy {
   public:
    LoglinearModel();

    // Buid a LoglinearModel from the categorical variables in a DataTable.
    explicit LoglinearModel(const DataTable &table);

    LoglinearModel *clone() const override;

    void add_data(const Ptr<MultivariateCategoricalData> &data_point) override;
    void add_data(const Ptr<Data> &dp) { add_data(DAT(dp)); }
    void add_data(MultivariateCategoricalData *dp) {
      add_data(Ptr<MultivariateCategoricalData>(dp));
    }

    // Perform one Gibbs sampling pass over the data point.
    void impute(MultivariateCategoricalData &data_point, RNG &rng);

    // The number of categorical variables being modeled.
    int nvars() const;

    void add_interaction(const std::vector<int> &variable_postiions);

    const GlmCoefs &coef() const {return prm_ref();}

    double logp(const MultivariateCategoricalData &data_point) const;

   private:
    // Add the effect to the encoder, to the sufficient statistics, and resize
    // the coefficient vector.
    void add_effect(const Ptr<CategoricalDataEncoder> &effect);

    // The main_effects are used to build interaction terms.
    std::vector<Ptr<CategoricalMainEffect>> main_effects_;

    CategoricalDatasetEncoder encoder_;
  };

}  // namespace BOOM

#endif  // BOOM_GLM_LOGLINEAR_MODEL_HPP_
