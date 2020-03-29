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

#include "Models/StateSpace/DynamicRegression.hpp"

namespace BOOM {
  namespace StateSpace {

    //=========================================================================
    namespace {
      using RDTP = RegressionDataTimePoint;
    }  // namespace

    RDTP::RegressionDataTimePoint(const RegressionDataTimePoint &rhs)
        : xdim_(rhs.xdim_),
          suf_(nullptr)
    {
      if (!!rhs.suf_) {
        suf_.reset(rhs.suf_->clone());
      } else {
        for (int i = 0; i < rhs.raw_data_.size(); ++i) {
          raw_data_.push_back(rhs.raw_data_[i]->clone());
        }
      }
    }

    std::ostream &RDTP::display(std::ostream &out) const {
      if (!!suf_) {
        out << "sufficient statistics for " << suf_->n() << " observations."
            << std::endl;
      } else {
        for (int i = 0; i < raw_data_.size(); ++i) {
          out << *raw_data_[i] << std::endl;
        }
      }
      return out;
    }

    void RDTP::add_data(const Ptr<RegressionData> &dp) {
      if (xdim_ == -1) {
        xdim_ = dp->xdim();
      } else {
        if (dp->xdim() != xdim_) {
          std::ostringstream err;
          err << "Attempt to add ata point of dimension " << dp->xdim()
              << " to RegressionDataTimePoint of dimension " << xdim_ << ".";
          report_error(err.str());
        }
      }
      if (suf_) {
        suf_->update(dp);
      } else {
        raw_data_.push_back(dp);
        yty_ += square(dp->y());
        if (raw_data_.size() >= dp->xdim()) {
          suf_.reset(new NeRegSuf(dp->xdim()));
          for (const auto &el : raw_data_) {
            suf_->update(el);
          }
          raw_data_.clear();
          yty_ = negative_infinity();
        }
      }
    }

    int RDTP::sample_size() const {
      if (!suf_) {
        return raw_data_.size();
      } else {
        return lround(suf_->n());
      }
    }

    std::pair<SpdMatrix, Vector> RDTP::xtx_xty(const Selector &inc) const {
      if (inc.nvars() == 0) {
        return std::make_pair(SpdMatrix(0), Vector(0));
      }

      if (!suf_) {
        SpdMatrix xtx(inc.nvars(), 0.0);
        Vector xty(inc.nvars(), 0.0);
        for (const auto &el : raw_data_) {
          Vector x = inc.select(el->x());
          xtx.add_outer(x, 1.0, false);
          xty.axpy(x, el->y());
        }
        return std::make_pair(xtx, xty);
      } else {
        return std::make_pair(suf_->xtx(inc), suf_->xty(inc));
      }
    }

    double RDTP::yty() const {
      if (!suf_) {
        return yty_;
      } else {
        return suf_->yty();
      }
    }

  }  // namespace StateSpace


  //===========================================================================
  namespace {
    using TSRDP = TimeSeriesRegressionDataPolicy;
  }  // namespace

  TSRDP::TimeSeriesRegressionDataPolicy(int xdim)
      : xdim_(xdim) {}

  void TSRDP::add_data(const Ptr<Data> &dp) {
    Ptr<RegressionData> reg_ptr = dp.dcast<RegressionData>();
    if (!!reg_ptr) {
      add_data(reg_ptr);
      return;
    }

    Ptr<StateSpace::RegressionDataTimePoint> time_point_ptr =
        dp.dcast<StateSpace::RegressionDataTimePoint>();
    if (!!time_point_ptr) {
      add_data(time_point_ptr);
      return;
    }
    std::ostringstream err;
    err << "Data point " << *dp << " could not be converted to either "
        << "RegressionData or RegressionDataTimePoint.";
    report_error(err.str());
  }

  void TSRDP::add_data(const Ptr<RegressionData> &dp) {
    if (data_.empty()) {
      data_.push_back(new StateSpace::RegressionDataTimePoint(xdim_));
    }
    data_.back()->add_data(dp);
  }

  void TSRDP::add_data(
      const Ptr<RegressionData> &dp,
      int time) {
    while (time >= data_.size()) {
      data_.push_back(new StateSpace::RegressionDataTimePoint(xdim_));
    }
    data_[time]->add_data(dp);
  }

  void TSRDP::add_data(
      const Ptr<StateSpace::RegressionDataTimePoint> &dp) {
    data_.push_back(dp);
  }

  void TSRDP::clear_data() {
    data_.clear();
  }

  void TSRDP::combine_data(
      const Model &other_model, bool just_suf) {
    report_error("Not implemented.");
  }

  //===========================================================================

  namespace {
    using DRM = DynamicRegressionModel;
  }  // namespace

  DRM::DynamicRegressionModel(int xdim) : TSRDP(xdim) {}


  DRM::DynamicRegressionModel(const DRM &rhs)
      : Model(rhs),
        ManyParamPolicy(rhs),
        TimeSeriesRegressionDataPolicy(rhs),
        PriorPolicy(rhs),
        residual_variance_(rhs.residual_variance_->clone())
  {
    ManyParamPolicy::clear();
    ManyParamPolicy::add_params(residual_variance_);
    for (int i = 0; i < rhs.coefficients_.size(); ++i) {
      coefficients_.push_back(rhs.coefficients_[i]->clone());
      ManyParamPolicy::add_params(coefficients_.back());
    }
  }

}  // namespace BOOM
