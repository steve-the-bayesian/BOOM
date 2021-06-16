// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2012 Steven L. Scott

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2.1 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA
*/

#include "Models/SpdModel.hpp"
#include "Models/SpdData.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  double SpdModel::pdf(const Data *dp, bool logscale) const {
    if (!dp) {
      report_error("NULL data pointer passed to SpdModel::pdf");
    }
    const SpdData *d = dynamic_cast<const SpdData *>(dp);
    if (!d) {
      std::ostringstream err;
      err << "Data could not be cast to SpdData in SpdModel::pdf." << endl
          << "Data value was: " << endl
          << *dp << endl;
      report_error(err.str());
      return negative_infinity();
    } else {
      double ans = logp(d->value());
      return logscale ? ans : exp(ans);
    }
  }

}  // namespace BOOM
