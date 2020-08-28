#ifndef BOOM_GLM_ENCODERS_HPP_
#define BOOM_GLM_ENCODERS_HPP_

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

#include "cpputil/RefCounted.hpp"
#include "stats/DataTable.hpp"
#include "LinAlg/Vector.hpp"
#include "LinAlg/Matrix.hpp"
#include "Models/CategoricalData.hpp"

namespace BOOM {

  // A DataEncoder creates a design matrix from one ore more columns in a
  // DataTable.
  class DataEncoder : private RefCounted {
   public:
    virtual ~DataEncoder() {}
    virtual int dim() const = 0;
    virtual Matrix encode_dataset(const DataTable &data) const = 0;
    virtual Vector encode_row(const MixedMultivariateData &data) const = 0;
    virtual void encode_row(
        const MixedMultivariateData &data, VectorView v) const = 0;

   private:
    friend void intrusive_ptr_add_ref(DataEncoder *d) {d->up_count();}
    friend void intrusive_ptr_release(DataEncoder *d) {
      d->down_count();
      if (d->ref_count() == 0) {
        delete d;
      }
    }
  };

  //===========================================================================
  // An encoder that depends on at most 1 variable.
  class MainEffectsEncoder : public DataEncoder {
   public:
    explicit MainEffectsEncoder(int which_variable):
        which_variable_(which_variable)
    {}
    virtual MainEffectsEncoder * clone() const = 0;
    int which_variable() const {return which_variable_;}

   private:
    int which_variable_;
  };

  //===========================================================================
  class EffectsEncoder : public MainEffectsEncoder {
   public:
    // The reference level is the last listed level in key.
    explicit EffectsEncoder(int which_variable, const Ptr<CatKeyBase> &key);
    EffectsEncoder(const EffectsEncoder &rhs);
    EffectsEncoder &operator=(const EffectsEncoder &rhs);
    EffectsEncoder(EffectsEncoder &&rhs) = default;
    EffectsEncoder & operator=(EffectsEncoder &&rhs) = default;

    EffectsEncoder * clone() const override;

    int dim() const override;
    Vector encode(const CategoricalData &data) const;
    void encode(const CategoricalData &data, VectorView view) const;

    Vector encode(int level) const;
    void encode(int level, VectorView view) const;

    Matrix encode(const CategoricalVariable &variable) const;

    Matrix encode_dataset(const DataTable &data) const override;
    Vector encode_row(const MixedMultivariateData &row) const override;
    void encode_row(const MixedMultivariateData &row, VectorView view) const override;

   private:
    Ptr<CatKeyBase> key_;
  };

  //===========================================================================
  class InteractionEncoder : public DataEncoder {
   public:
    InteractionEncoder(const Ptr<DataEncoder> &encoder1,
                       const Ptr<DataEncoder> &encoder2);

    int dim() const override {
      return encoder1_->dim() * encoder2_->dim();
    }

    Matrix encode_dataset(const DataTable &table) const override {
      Matrix m1 = encoder1_->encode_dataset(table);
      Matrix m2 = encoder2_->encode_dataset(table);

      Matrix ans(table.nrow(), dim());
      int index = 0;
      for (int i = 0; i < m1.ncol(); ++i) {
        for (int j = 0; j < m2.ncol(); ++j) {
          ans.col(index++) = m1.col(i) * m2.col(j);
        }
      }
      return ans;
    }

    void encode_row(const MixedMultivariateData &data,
                    VectorView ans) const override {
      encoder1_->encode_row(data, VectorView(wsp1_));
      encoder2_->encode_row(data, VectorView(wsp2_));
      int index = 0;
      for (int i = 0; i < wsp1_.size(); ++i) {
        for (int j = 0; j < wsp2_.size(); ++j) {
          ans[index++] = wsp1_[i] * wsp2_[j];
        }
      }
    }

    Vector encode_row(const MixedMultivariateData &data) const override {
      Vector ans(dim());
      encode_row(data, VectorView(ans));
      return ans;
    }

   private:
    Ptr<DataEncoder> encoder1_;
    Ptr<DataEncoder> encoder2_;
    mutable Vector wsp1_, wsp2_;
  };

  //===========================================================================
  class DatasetEncoder : public DataEncoder {
   public:
    DatasetEncoder(bool add_intercept = true)
        : dim_(add_intercept),
          add_intercept_(add_intercept)
    {}

    void add_encoder(const Ptr<DataEncoder> &encoder) {
      encoders_.push_back(encoder);
      dim_ += encoder->dim();
    }

    int dim() const override {return dim_;}
    bool add_intercept() const {return add_intercept_;}

    Matrix encode_dataset(const DataTable &data) const override;
    Vector encode_row(const MixedMultivariateData &row) const override;
    void encode_row(
        const MixedMultivariateData &row, VectorView ans) const override;

    const std::vector<Ptr<DataEncoder>> &encoders() const {return encoders_;}

   private:
    int dim_;
    bool add_intercept_;
    std::vector<Ptr<DataEncoder>> encoders_;
  };

}  // namespace BOOM

#endif  // BOOM_GLM_ENCODERS_HPP_
