/* The MIT License (MIT)

Copyright (c) 2016 source{d}.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

// Appropriated from github repo:
// https://github.com/src-d/lapjv
//
// They appropriated it from http://www.magiclogic.com/assignment.html
//
// On Dec 10, 2021, Steven L. Scott modified this code under the MIT license by
// stripping out a bunch of low level machine code.  This will cost a bit of
// performance in large problems, but gain quite a bit of portability.

#include <limits>
#include <memory>
#include <vector>

inline std::tuple<double, double, long, long>
find_umins(long dim, long i, const double *assign_cost,
                   const double *v) {
  const double *local_cost = &assign_cost[i * dim];
  double umin = local_cost[0] - v[0];
  long j1 = 0;
  long j2 = -1;
  double usubmin = std::numeric_limits<double>::max();
  for (long j = 1; j < dim; j++) {
    double h = local_cost[j] - v[j];
    if (h < usubmin) {
      if (h >= umin) {
        usubmin = h;
        j2 = j;
      } else {
        usubmin = umin;
        umin = h;
        j2 = j1;
        j1 = j;
      }
    }
  }
  return std::make_tuple(umin, usubmin, j1, j2);
}

/// @brief Exact Jonker-Volgenant algorithm.
/// @param dim in problem size
/// @param assign_cost in cost matrix
/// @param verbose in indicates whether to report the progress to stdout
/// @param rowsol out column assigned to row in solution / size dim
/// @param colsol out row assigned to column in solution / size dim
/// @param u out dual variables, row reduction numbers / size dim
/// @param v out dual variables, column reduction numbers / size dim
/// @return achieved minimum assignment cost
double lap(int dim,
           const double *assign_cost,
           long *rowsol,
           long *colsol,
           double *u,
           double *v) {

  // SLS: Replaced unique_ptr arrays with std::vector.
  //      Renamed 'free' with free_rows to avoid masking C's 'free', which might
  //      be expressed as a macro.
  std::vector<long> free_rows(dim);    // list of unassigned rows.
  std::vector<long> collist(dim);      // list of columns to be scanned in various ways.
  std::vector<long> matches(dim);      // counts how many times a row could be assigned.
  std::vector<double> d(dim);          // 'cost-distance' in augmenting path calculation.
  std::vector<long> pred(dim);         // row-predecessor of column in augmenting/alternating path.

  // init how many times a row will be assigned in the column reduction.
  #if _OPENMP >= 201307
  #pragma omp simd
  #endif
  for (long i = 0; i < dim; i++) {
    matches[i] = 0;
  }

  // COLUMN REDUCTION
  for (long j = dim - 1; j >= 0; j--) {   // reverse order gives better results.
    // find minimum cost over rows.
    double min = assign_cost[j];
    long imin = 0;
    for (long i = 1; i < dim; i++) {
      const double *local_cost = &assign_cost[i * dim];
      if (local_cost[j] < min) {
        min = local_cost[j];
        imin = i;
      }
    }
    v[j] = min;

    if (++matches[imin] == 1) {
      // init assignment if minimum row assigned for first time.
      rowsol[imin] = j;
      colsol[j] = imin;
    } else {
      colsol[j] = -1;        // row already assigned, column not assigned.
    }
  }

  // REDUCTION TRANSFER
  long numfree = 0;
  for (long i = 0; i < dim; i++) {
    const double *local_cost = &assign_cost[i * dim];
    if (matches[i] == 0) {  // fill list of unassigned 'free' rows.
      free_rows[numfree++] = i;
    } else if (matches[i] == 1) {  // transfer reduction from rows that are assigned once.
      long j1 = rowsol[i];
      double min = std::numeric_limits<double>::max();
      for (long j = 0; j < dim; j++) {
        if (j != j1) {
          if (local_cost[j] - v[j] < min) {
            min = local_cost[j] - v[j];
          }
        }
      }
      v[j1] = v[j1] - min;
    }
  }

  // AUGMENTING ROW REDUCTION
  for (int loopcnt = 0; loopcnt < 2; loopcnt++) {  // loop to be done twice.
    // scan all free rows.
    // in some cases, a free row may be replaced with another one to be scanned next.
    long k = 0;
    long prevnumfree = numfree;
    numfree = 0;  // start list of rows still free after augmenting row reduction.
    while (k < prevnumfree) {
      long i = free_rows[k++];

      // find minimum and second minimum reduced cost over columns.
      double umin, usubmin;
      long j1, j2;
      std::tie(umin, usubmin, j1, j2) = find_umins(dim, i, assign_cost, v);

      long i0 = colsol[j1];
      double vj1_new = v[j1] - (usubmin - umin);
      bool vj1_lowers = vj1_new < v[j1];  // the trick to eliminate the epsilon bug
      if (vj1_lowers) {
        // change the reduction of the minimum column to increase the minimum
        // reduced cost in the row to the subminimum.
        v[j1] = vj1_new;
      } else if (i0 >= 0) {  // minimum and subminimum equal.
        // minimum column j1 is assigned.
        // swap columns j1 and j2, as j2 may be unassigned.
        j1 = j2;
        i0 = colsol[j2];
      }

      // (re-)assign i to j1, possibly de-assigning an i0.
      rowsol[i] = j1;
      colsol[j1] = i;

      if (i0 >= 0) {  // minimum column j1 assigned earlier.
        if (vj1_lowers) {
          // put in current k, and go back to that k.
          // continue augmenting path i - j1 with i0.
          free_rows[--k] = i0;
        } else {
          // no further augmenting reduction possible.
          // store i0 in list of free rows for next phase.
          free_rows[numfree++] = i0;
        }
      }
    }
  }  // for loopcnt

  // AUGMENT SOLUTION for each free row.
  for (long f = 0; f < numfree; f++) {
    long endofpath;
    long freerow = free_rows[f];       // start row of augmenting path.

    // Dijkstra shortest path algorithm.
    // runs until unassigned column added to shortest path tree.
    #if _OPENMP >= 201307
    #pragma omp simd
    #endif
    for (long j = 0; j < dim; j++) {
      d[j] = assign_cost[freerow * dim + j] - v[j];
      pred[j] = freerow;
      collist[j] = j;  // init column list.
    }

    long low = 0; // columns in 0..low-1 are ready, now none.
    long up = 0;  // columns in low..up-1 are to be scanned for current minimum, now none.
                 // columns in up..dim-1 are to be considered later to find new minimum,
                 // at this stage the list simply contains all columns
    bool unassigned_found = false;
    // initialized in the first iteration: low == up == 0
    long last = 0;
    double min = 0;
    do {
      if (up == low) {        // no more columns to be scanned for current minimum.
        last = low - 1;
        // scan columns for up..dim-1 to find all indices for which new minimum occurs.
        // store these indices between low..up-1 (increasing up).
        min = d[collist[up++]];
        for (long k = up; k < dim; k++) {
          long j = collist[k];
          double h = d[j];
          if (h <= min) {
            if (h < min) {   // new minimum.
              up = low;      // restart list at index low.
              min = h;
            }
            // new index with same minimum, put on undex up, and extend list.
            collist[k] = collist[up];
            collist[up++] = j;
          }
        }

        // check if any of the minimum columns happens to be unassigned.
        // if so, we have an augmenting path right away.
        for (long k = low; k < up; k++) {
          if (colsol[collist[k]] < 0) {
            endofpath = collist[k];
            unassigned_found = true;
            break;
          }
        }
      }

      if (!unassigned_found) {
        // update 'distances' between freerow and all unscanned columns, via next scanned column.
        long j1 = collist[low];
        low++;
        long i = colsol[j1];
        const double *local_cost = &assign_cost[i * dim];
        double h = local_cost[j1] - v[j1] - min;
        for (long k = up; k < dim; k++) {
          long j = collist[k];
          double v2 = local_cost[j] - v[j] - h;
          if (v2 < d[j]) {
            pred[j] = i;
            if (v2 == min) {  // new column found at same minimum value
              if (colsol[j] < 0) {
                // if unassigned, shortest augmenting path is complete.
                endofpath = j;
                unassigned_found = true;
                break;
              } else {  // else add to list to be scanned right away.
                collist[k] = collist[up];
                collist[up++] = j;
              }
            }
            d[j] = v2;
          }
        }
      }
    } while (!unassigned_found);

    // update column prices.
    #if _OPENMP >= 201307
    #pragma omp simd
    #endif
    for (long k = 0; k <= last; k++) {
      long j1 = collist[k];
      v[j1] = v[j1] + d[j1] - min;
    }

    // reset row and column assignments along the alternating path.
    {
      long i;
      do {
        i = pred[endofpath];
        colsol[endofpath] = i;
        long j1 = endofpath;
        endofpath = rowsol[i];
        rowsol[i] = j1;
      } while (i != freerow);
    }
  }

  // calculate optimal cost.
  double lapcost = 0;
  #if _OPENMP >= 201307
  #pragma omp simd reduction(+:lapcost)
  #endif
  for (long i = 0; i < dim; i++) {
    const double *local_cost = &assign_cost[i * dim];
    long j = rowsol[i];
    u[i] = local_cost[j] - v[j];
    lapcost += local_cost[j];
  }

  return lapcost;
}
