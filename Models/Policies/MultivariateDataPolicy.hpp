#ifndef BOOM_MODELS_POLICIES_MULTIVARIATE_DATA_POLICY_HPP_
#define BOOM_MODELS_POLICIES_MULTIVARIATE_DATA_POLICY_HPP_

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

#include "Models/ModelTypes.hpp"
#include "stats/DataTable.hpp"

namespace BOOM {

  // Holds a Ptr<DataTable> as a data structure.  The Ptr is nullptr until the
  // first data is added.
  class MultivariateDataPolicy
      : virtual public Model {
   public:
    MultivariateDataPolicy();

    // An exception is thrown unless Data resolves to either a DataTable or a
    // MixedMultivariateData.
    void add_data(const Ptr<Data> &data_dp) override;
    void add_data(const Ptr<DataTable> &data_table);
    void add_data(const Ptr<MixedMultivariateData> &data_point);

    void clear_data() override;
    void combine_data(const Model &other_model, bool just_suf = true) override;

    const DataTable &data() const {return *data_;}

   private:
    // Upon construction data_ points to an empty DataTable.
    Ptr<DataTable> data_;
  };

}  // namespace BOOM

#endif  // BOOM_MODELS_POLICIES_MULTIVARIATE_DATA_POLICY_HPP_
