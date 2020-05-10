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

#include "Models/Impute/MvRegCopulaDataImputer.hpp"

namespace BOOM {

  void MvRegCopulaDataImputer::impute(RNG &rng) {
    impute_atoms(rng);
    impute_missing_status(rng);
    impute_mixture_class(rng);

    impute_missing_numerics(rng);
    refresh_ecdf();

    sample_regression_parameters(rng);
  }

  Vector MvRegCopulaDataImputer::impute_row(
      const Vector &input,
      const Selector &missing,
      const Vector &predictors,
      RNG &rng) {

    Vector ans(input.size());
    Vector mean = complete_data_model_->predict(predictors);
  }

  void MvRegCopulaDataImputer::impute_atoms(RNG &rng) {
    for (int i = 0; i < dat().size(); ++i) {

    }
  }

  void MvRegCopulaDataImputer::impute_missing_numerics(RNG &rng) {

  }




}  // namespace BOOMx
