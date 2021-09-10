/*
  Copyright (C) 2005-2021 Steven L. Scott

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

#include "Models/Glm/RegressionSlabPrior.hpp"
#include "distributions.hpp"

namespace BOOM {

  RegressionSlabPrior::RegressionSlabPrior(const SpdMatrix &xtx,
                                           const Ptr<UnivParams> &sigsq,
                                           double sample_mean,
                                           double data_sample_size,
                                           double prior_sample_size,
                                           double diagonal_shrinkage)
      : ParamPolicy(new VectorParams(xtx.nrow(), 0.0),
                    new UnivParams(prior_sample_size)),
        modified_xtx_(scale_xtx(xtx, data_sample_size, diagonal_shrinkage)),
        data_sample_size_(data_sample_size),
        diagonal_shrinkage_(diagonal_shrinkage),
        sigsq_(sigsq),
        wsp_current_(false),
        var_wsp_(new SpdParams(xtx))
  {
    set_observers();
  }

  RegressionSlabPrior::~RegressionSlabPrior() {
    remove_observers();
  }

  RegressionSlabPrior::RegressionSlabPrior(const RegressionSlabPrior &rhs)
      : Model(rhs),
        MvnBase(rhs),
        ParamPolicy(rhs),
        NullDataPolicy(rhs),
        NullPriorPolicy(rhs),
        modified_xtx_(rhs.modified_xtx_),
        wsp_current_(false),
        var_wsp_(rhs.var_wsp_->clone())
  {
    set_observers();
  }

  RegressionSlabPrior * RegressionSlabPrior::clone() const {
    return new RegressionSlabPrior(*this);
  }

  void RegressionSlabPrior::set_observers() {
    sigsq_->add_observer(this, [this]() { this->wsp_current_ = false; });
    SampleSize_prm()->add_observer(this, [this]() { this->wsp_current_ = false; });
  }

  void RegressionSlabPrior::remove_observers() {
    sigsq_->remove_observer(this);
    SampleSize_prm()->remove_observer(this);
  }

  SpdMatrix RegressionSlabPrior::scale_xtx(
      const SpdMatrix &xtx,
      double data_sample_size,
      double diagonal_shrinkage) {
    SpdMatrix ans = xtx / data_sample_size;
    if (diagonal_shrinkage > 1.0 or diagonal_shrinkage < 0.0) {
      report_error("diagonal_shrinkage must be between 0 and 1");
    } else if (diagonal_shrinkage >= 1.0) {
      // This branch handles the case where diagonal_shrinkage == 1.0.
      Vector diagonal_elements = ans.diag();
      ans = 0.0;
      ans.diag() = diagonal_elements;
    } else if (diagonal_shrinkage > 0.0) {
      ans *= (1 - diagonal_shrinkage);
      ans.diag() /= (1 - diagonal_shrinkage);
    } else {
      // In this branch the diagonal_shrinkage is zero, so do nothing.
    }
    return ans;
  }




}  // namespace BOOM
