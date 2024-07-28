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

#include "stats/fake_data_table.hpp"
#include "distributions.hpp"
#include "math/Permutation.hpp"

namespace BOOM {

  namespace {

    // Generate a categorical variable with levels 'level_1', 'level_2', etc.
    CategoricalVariable generate_categorical_variable(size_t sample_size, int nlevels) {
      std::vector<std::string> labels;
      for (int i = 0; i < nlevels; ++i) {
        labels.push_back(std::string("Level_") + std::to_string(i));
      }
      std::vector<int> levels;
      for (size_t i = 0; i < sample_size; ++i) {
        double u;
        do {
          u = runif();
        } while (u >= 1.0);
        int value = std::floor(u * nlevels);
        levels.push_back(value);
      }

      NEW(CatKey, key)(labels);

      CategoricalVariable ans(levels, key);
      return ans;
    }
  }

  DataTable fake_data_table(size_t sample_size,
                            size_t num_numeric,
                            const std::vector<int> &nlevels,
                            bool mix_order) {

    Matrix numeric(sample_size, num_numeric);
    numeric.randomize();

    int num_cat = nlevels.size();
    std::vector<CategoricalVariable> categorical;
    for (int i = 0; i < num_cat; ++i) {
      categorical.push_back(generate_categorical_variable(
          sample_size, nlevels[i]));
    }

    DataTable ans;
    for (int i = 0; i < num_numeric + num_cat; ++i) {
      std::string vname = std::string("V") + std::to_string(i + 1);
      if (i < num_numeric) {
        ans.append_variable(numeric.col(i), vname);
      } else {
        ans.append_variable(categorical[i - num_numeric], vname);
      }
    }

    if (mix_order) {
      Permutation perm = random_permutation(num_numeric + num_cat);
      DataTable permuted;
      for (int i = 0; i < ans.ncol(); ++i) {
        std::string vname = std::string("V") + std::to_string(i + 1);
        int I = perm[i];
        if (ans.variable_type(I) == VariableType::numeric) {
          permuted.append_variable(ans.getvar(I), vname);
        } else {
          permuted.append_variable(ans.get_nominal(I), vname);
        }
      }
      ans = permuted;
    }
    return ans;
  }

}  // namespace BOOM
