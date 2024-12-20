#ifndef BOOM_STATS_FAKE_DATA_TABLE_HPP_
#define BOOM_STATS_FAKE_DATA_TABLE_HPP_

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

#include "stats/DataTable.hpp"
#include <vector>

namespace BOOM {

  // Generate a data table containing a specified number of Numeric and
  // Categorical variables.  Intended for testing purposes.
  //
  // Args:
  //   sample_size: The number of rows in the simulated table.
  //   num_numeric: The number of numeric variables to generate.
  //   nlevels: The number of levels to generate for each categorical variable.
  //     The number of entries in this vector is the number of categorical
  //     variables to generate.
  //   mix_order: If true then the numeric and categorical variables are mixed
  //     together in random order.  If false then the numeric variables are
  //     first, followed by the categorical variables.
  //
  // Returns:
  //   The simulated table.
  DataTable fake_data_table(size_t sample_size,
                            size_t num_numeric,
                            const std::vector<int> &nlevels,
                            bool mix_order = false);
}

#endif  // BOOM_STATS_FAKE_DATA_TABLE_HPP_
