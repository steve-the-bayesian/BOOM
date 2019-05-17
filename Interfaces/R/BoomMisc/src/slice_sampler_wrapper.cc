#include "Samplers/UnivariateSliceSampler.hpp"

#include "r_interface/boom_r_tools.hpp"
#include "r_interface/handle_exception.hpp"
#include "r_interface/list_io.hpp"
#include "r_interface/print_R_timestamp.hpp"
#include "r_interface/prior_specification.hpp"
#include "r_interface/seed_rng_from_R.hpp"

namespace {
  using namespace BOOM;
  using namespace BOOM::RInterface;
}  // namespace

extern "C" {
  using namespace BOOM;  // NOLINT
  SEXP boom_misc_slice_sampler_wrapper(
      SEXP r_vector_function,
      SEXP r_initial_value,
      SEXP r_niter,
      SEXP r_ping,
      SEXP r_seed){
    RErrorReporter error_reporter;
    RMemoryProtector protector;
    try {
      seed_rng_from_R(r_seed);
      int niter = Rf_asInteger(r_niter);
      int ping = Rf_asInteger(r_ping);
      Vector theta = ToBoomVector(r_initial_value);
      RVectorFunction logf(r_vector_function);
      UnivariateSliceSampler sampler(logf);
      Matrix draws(niter, theta.size());
      for (int i = 0; i < niter; ++i) {
        if (RCheckInterrupt()) {
          error_reporter.SetError("Canceled by user.");
          return ToRMatrix(draws);
        }
        print_R_timestamp(i, ping);
        theta = sampler.draw(theta);
        draws.row(i) = theta;
      }
      return ToRMatrix(draws);
    } catch(std::exception &e) {
      handle_exception(e);
    } catch (...) {
      handle_unknown_exception();
    }
    return R_NilValue;
  }
}  // extern "C"
