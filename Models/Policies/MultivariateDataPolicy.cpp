/*
  Copyright (C) 2005-2024 Steven L. Scott

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

#include "Models/Policies/MultivariateDataPolicy.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  MultivariateDataPolicy::MultivariateDataPolicy()
      : data_(new DataTable)
  {}

  void MultivariateDataPolicy::add_data(const Ptr<Data> &data_point) {
    Ptr<MixedMultivariateData> row = data_point.dcast<MixedMultivariateData>();
    if (!!row) {
      add_data(row);
      return;
    }

    Ptr<DataTable> table = data_point.dcast<DataTable>();
    if (!!table) {
      add_data(table);
      return;
    }
    report_error("data_point could not be cast to either "
                 "MixedMultivariateData or DataTable.");
  }

  void MultivariateDataPolicy::add_data(const Ptr<MixedMultivariateData> &data_point) {
    data_->append_row(*data_point);
  }

  void MultivariateDataPolicy::add_data(const Ptr<DataTable> &data_table) {
    data_->rbind(*data_table);
  }

  void MultivariateDataPolicy::clear_data() {
    data_.reset(new DataTable);
  }

  void MultivariateDataPolicy::combine_data(const Model &other_model, bool) {
    try {
      const MultivariateDataPolicy &other_policy(
          dynamic_cast<const MultivariateDataPolicy&>(other_model));
      data_->rbind(*other_policy.data_);
    } catch (const std::exception &e) {
      std::ostringstream err;
      err << "Could not convert other_model to MultivariateDataPolicy.\n"
          << e.what();
      report_error(err.str());
    }
  }

}  // namespace BOOM
