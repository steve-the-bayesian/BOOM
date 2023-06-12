// Make the minn38 data available as a data table.

#include <vector>
#include <string>
#include "stats/DataTable.hpp"

namespace BOOM {

  // Returns the minn38 data set from MASS, which is used to illustrate
  // loglinear models.  From the R docs:
  //
  // The Minnesota high school graduates of 1938 were classified according to
  // four factors, described below. The minn38 data frame has 168 rows and 5
  // columns.
  //
  // This data frame contains the following columns:
  //   hs: high school rank: "L", "M" and "U" for lower, middle and upper third.
  //   phs: post high school status: Enrolled in college, ("C"), enrolled in
  //     non-collegiate school, ("N"), employed full-time, ("E") and other,
  //     ("O").
  //   fol: father's occupational level, (seven levels, "F1", "F2", …, "F7").
  //   sex: factor with levels "F" or "M".
  //   f:  frequency.
  DataTable minn38_data() {
    std::vector<std::string> hs = {
      "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L",
      "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "M", "M", "M", "M",
      "M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "M",
      "M", "M", "M", "M", "M", "M", "M", "M", "U", "U", "U", "U", "U", "U", "U", "U",
      "U", "U", "U", "U", "U", "U", "U", "U", "U", "U", "U", "U", "U", "U", "U", "U",
      "U", "U", "U", "U", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L",
      "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L",
      "M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "M",
      "M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "U", "U", "U", "U",
      "U", "U", "U", "U", "U", "U", "U", "U", "U", "U", "U", "U", "U", "U", "U", "U",
      "U", "U", "U", "U", "U", "U", "U", "U"
    };

    std::vector<std::string> phs = {
      "C", "C", "C", "C", "C", "C", "C", "N", "N", "N", "N", "N", "N", "N", "E", "E",
      "E", "E", "E", "E", "E", "O", "O", "O", "O", "O", "O", "O", "C", "C", "C", "C",
      "C", "C", "C", "N", "N", "N", "N", "N", "N", "N", "E", "E", "E", "E", "E", "E",
      "E", "O", "O", "O", "O", "O", "O", "O", "C", "C", "C", "C", "C", "C", "C", "N",
      "N", "N", "N", "N", "N", "N", "E", "E", "E", "E", "E", "E", "E", "O", "O", "O",
      "O", "O", "O", "O", "C", "C", "C", "C", "C", "C", "C", "N", "N", "N", "N", "N",
      "N", "N", "E", "E", "E", "E", "E", "E", "E", "O", "O", "O", "O", "O", "O", "O",
      "C", "C", "C", "C", "C", "C", "C", "N", "N", "N", "N", "N", "N", "N", "E", "E",
      "E", "E", "E", "E", "E", "O", "O", "O", "O", "O", "O", "O", "C", "C", "C", "C",
      "C", "C", "C", "N", "N", "N", "N", "N", "N", "N", "E", "E", "E", "E", "E", "E",
      "E", "O", "O", "O", "O", "O", "O", "O"
    };

    std::vector<std::string> fol = {
      "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F1", "F2", "F3", "F4", "F5",
      "F6", "F7", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F1", "F2", "F3",
      "F4", "F5", "F6", "F7", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F1",
      "F2", "F3", "F4", "F5", "F6", "F7", "F1", "F2", "F3", "F4", "F5", "F6",
      "F7", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F1", "F2", "F3", "F4",
      "F5", "F6", "F7", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F1", "F2",
      "F3", "F4", "F5", "F6", "F7", "F1", "F2", "F3", "F4", "F5", "F6", "F7",
      "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F1", "F2", "F3", "F4", "F5",
      "F6", "F7", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F1", "F2", "F3",
      "F4", "F5", "F6", "F7", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F1",
      "F2", "F3", "F4", "F5", "F6", "F7", "F1", "F2", "F3", "F4", "F5", "F6",
      "F7", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F1", "F2", "F3", "F4",
      "F5", "F6", "F7", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F1", "F2",
      "F3", "F4", "F5", "F6", "F7", "F1", "F2", "F3", "F4", "F5", "F6", "F7"
    };

    std::vector<std::string> sex = {
      "M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "M",
      "M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "M",
      "M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "M",
      "M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "M",
      "M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "M",
      "M", "M", "M", "M", "M", "M", "M", "M", "M", "F", "F", "F", "F", "F", "F",
      "F", "F", "F", "F", "F", "F", "F", "F", "F", "F", "F", "F", "F", "F", "F",
      "F", "F", "F", "F", "F", "F", "F", "F", "F", "F", "F", "F", "F", "F", "F",
      "F", "F", "F", "F", "F", "F", "F", "F", "F", "F", "F", "F", "F", "F", "F",
      "F", "F", "F", "F", "F", "F", "F", "F", "F", "F", "F", "F", "F", "F", "F",
      "F", "F", "F", "F", "F", "F", "F", "F", "F", "F", "F", "F", "F", "F", "F",
      "F", "F", "F"
    };

    Vector f = {
      87, 72, 52, 88, 32, 14, 20, 3, 6, 17, 9, 1, 2, 3, 17, 18, 14, 14, 12, 5,
      4, 105, 209, 541, 328, 124, 148, 109, 216, 159, 119, 158, 43, 24, 41, 4,
      14, 13, 15, 5, 6, 5, 14, 28, 44, 36, 7, 15, 13, 118, 227, 578, 304, 119,
      131, 88, 256, 176, 119, 144, 42, 24, 32, 2, 8, 10, 12, 2, 2, 2, 10, 22,
      33, 20, 7, 4, 4, 53, 95, 257, 115, 56, 61, 41, 53, 36, 52, 48, 12, 9, 3,
      7, 16, 28, 18, 5, 1, 1, 13, 11, 49, 29, 10, 15, 6, 76, 111, 521, 191,
      101, 130, 88, 163, 116, 162, 130, 35, 19, 25, 30, 41, 64, 47, 11, 13, 9,
      28, 53, 129, 62, 37, 22, 15, 118, 214, 708, 305, 152, 174, 158, 309, 225,
      243, 237, 72, 42, 36, 17, 49, 79, 57, 20, 10, 14, 38, 68, 284, 63, 21,
      19, 19, 89, 210, 448, 219, 95, 105, 93
    };

    DataTable table;
    table.append_variable(CategoricalVariable(hs), "hs");
    table.append_variable(CategoricalVariable(phs), "phs");
    table.append_variable(CategoricalVariable(fol), "fol");
    table.append_variable(CategoricalVariable(sex), "sex");
    table.append_variable(f, "f");
    return table;
  }

}  // namespace BOOM
