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

  void MultivariateDataPolicy::add_data(const Ptr<Data> &data_point) {
    if (Ptr<DataTable> dtable = data_point.dcast<DataTable>()) {
      add_data(dtable);
    } else if (Ptr<MixedMultivariateData> dpoint =
               data_point.dcast<MixedMultivariateData>()) {
      add_data(dpoint);
    } else {
      report_error("Data could not be cast to either DataTable "
                   "or MixedMultivariateData.");
    }
  }

  void MultivariateDataPolicy::add_data(const Ptr<DataTable> &data_table) {
    if (!!data_) {
      data_->rbind(*data_table);
    } else {
      data_ = data_table;
    }
  }

  void MultivariateDataPolicy::add_data(
      const Ptr<MixedMultivariateData> &data_point) {
    if (!!data_) {
      data_->append_row(*data_point);
    } else {
      data_.reset(new DataTable);
      data_->append_row(*data_point);
    }
  }

  void MultivariateDataPolicy::clear_data() {
    data_ = nullptr;
  }

  void MultivariateDataPolicy::combine_data(const Model &other_model,
                                            bool just_suf) {
    const MultivariateDataPolicy &other(
        dynamic_cast<const MultivariateDataPolicy &>(other_model));
    add_data(other.data_);
  }

}  // namespace BOOM
