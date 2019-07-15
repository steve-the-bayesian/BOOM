// Copyright 2019 Steven L. Scott. All Rights Reserved.
// Author: steve.the.bayesian@gmail.com (Steve Scott)

#include "stats/Bspline.hpp"
#include "stats/Mspline.hpp"

#include "r_interface/boom_r_tools.hpp"
#include "r_interface/handle_exception.hpp"
#include "r_interface/list_io.hpp"
#include "r_interface/print_R_timestamp.hpp"
#include "r_interface/prior_specification.hpp"
#include "r_interface/seed_rng_from_R.hpp"
#include "utils.h"

#include "cpputil/Ptr.hpp"

namespace {
  using namespace BOOM;
  using namespace BOOM::RInterface;

}  // namespace

extern "C" {
  using namespace BOOM;  // NOLINT

  // Args:
  //   r_data_vector:  A vector of values to be expanded by the spline basis.

  //   r_sorted_knots_vector: A vector of knots that defines the spline.  The

  SEXP boom_spike_slab_Bspline_basis(SEXP r_data_vector,
                                     SEXP r_sorted_knots_vector) {
    try {
      Vector x = ToBoomVector(r_data_vector);
      Vector knots = ToBoomVector(r_sorted_knots_vector);
      Bspline spline(knots);
      Matrix basis(x.size(), spline.basis_dimension());
      for (int i = 0; i < x.size(); ++i) {
        basis.row(i) = spline.basis(x[i]);
      }
      return ToRMatrix(basis);
    } catch(std::exception &e) {
      handle_exception(e);
    } catch (...) {
      handle_unknown_exception();
    }
    return R_NilValue;
  }


  SEXP boom_spike_slab_Mspline_basis(SEXP r_data_vector,
                                     SEXP r_sorted_knots_vector) {
    try {
      Vector x = ToBoomVector(r_data_vector);
      Vector knots = ToBoomVector(r_sorted_knots_vector);
      Mspline spline(knots);
      Matrix basis(x.size(), spline.basis_dimension());
      for (int i = 0; i < x.size(); ++i) {
        basis.row(i) = spline.basis(x[i]);
      }
      return ToRMatrix(basis);
    } catch(std::exception &e) {
      handle_exception(e);
    } catch (...) {
      handle_unknown_exception();
    }
    return R_NilValue;
  }

  SEXP boom_spike_slab_Ispline_basis(SEXP r_data_vector,
                                     SEXP r_sorted_knots_vector) {
    try {
      Vector x = ToBoomVector(r_data_vector);
      Vector knots = ToBoomVector(r_sorted_knots_vector);
      Ispline spline(knots);
      Matrix basis(x.size(), spline.basis_dimension());
      for (int i = 0; i < x.size(); ++i) {
        basis.row(i) = spline.basis(x[i]);
      }
      return ToRMatrix(basis);
    } catch(std::exception &e) {
      handle_exception(e);
    } catch (...) {
      handle_unknown_exception();
    }
    return R_NilValue;
  }

}  // extern "C"
