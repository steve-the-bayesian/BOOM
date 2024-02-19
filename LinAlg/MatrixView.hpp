#ifndef BOOM_LINALG_MATRIX_VIEW_HPP_
#define BOOM_LINALG_MATRIX_VIEW_HPP_

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


namespace BOOM {

  // A MatrixView is a view into a rectangle of an existing Matrix object.
  class MatrixView : public SubMatrix {
   public:
    MatrixView(const double *v = 0, int nrow = 0, int ncol = 0);

    // Args:
    //   target:  The matrix to be viewed.
    //   first_row:  The first row of the view.
    //   first_col: The first column of the view.
    //   num_rows: The number of rows in the view.  A negative number indicates
    //     all rows after (but including) the first.
    //   num_cols: The number of columns in the view.  A negative number
    //     indicates all columns after (but including) the first.
    MatrixView(Matrix &target,
               int first_row = 0,
               int first_col = 0,
               int num_rows = -1,
               int num_cols = -1)
  };

}

#endif  // BOOM_LINALG_MATRIX_VIEW_HPP_
