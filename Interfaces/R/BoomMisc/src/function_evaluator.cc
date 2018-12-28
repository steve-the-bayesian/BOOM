#include "r_interface/boom_r_tools.hpp"
#include "r_interface/handle_exception.hpp"

// Contains code for evaluating an R function in C++.  This is primarily for
// testing.

namespace {
  using namespace BOOM;
  using namespace BOOM::RInterface;
}

extern "C" {
  using namespace BOOM;
  SEXP boom_evaluate_r_function(SEXP r_function, SEXP r_argument_value) {
    RErrorReporter error_reporter;
    RMemoryProtector protector;
    try {
      Vector x = ToBoomVector(r_argument_value);
      BOOM::RVectorFunction fun(r_function);
      return Rf_ScalarReal(fun(x));
    } catch (std::exception &e) {
      handle_exception(e);
    } catch (...) {
      handle_unknown_exception();
    }
    return R_NilValue;
  }
  
}  // extern "C"
