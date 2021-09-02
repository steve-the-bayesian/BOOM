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
                                           dobule diagonal_shrinkage)
      : ParamPolicy(new VectorParams(xtx.nrow(), 0.0),
                    new UnivParams(prior_sample_size)),
        sigsq_(sigsq),
        wsp_current_(false),
        var_wsp_(new SpdParams(xtx))
  {
    set_modified_xtx(xtx, data_sample_size, diagonal_shrinkage);
    set_observers();
  }

  void RegressionSlabPrior::set_observers() {
    sigsq_->add_observer([this]() { this->wsp_current_ = false; });
    SampleSize_prm()->add_observer([this]() { this->wsp_current_ = false; });
  }

  void RegressionSlabPrior::set_modified_xtx(
      const SpdMatrix &xtx,
      double data_sample_size,
      double diagonal_shrinkage) {
    modified_xtx_ = xtx / data_sample_size;
    if (diagonal_shrinkage > 1.0 or diagonal_shrinkage < 0.0) {
      report_error("diagonal_shrinkage must be between 0 and 1");
    } else if (diagonal_shrinkage >= 1.0) {
      // This branch handles the case where diagonal_shrinkage == 1.0.
      Vector diagonal_elements = modified_xtx_.diag();
      modified_xtx_ = 0.0;
      modified_xtx_.diag() = diagonal_elements;
    } else if (diagonal_shrinkage > 0.0) {
      modified_xtx_ *= (1 - diagonal_shrinkage);
      modified_xtx_.diag() /= (1 - diagonal_shrinkage);
    } else {
      // In this branch the diagonal_shrinkage is zero, so do nothing.
    }
  }




}  // namespace BOOM
