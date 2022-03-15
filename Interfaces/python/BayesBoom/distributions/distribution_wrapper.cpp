#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <memory>

#include "distributions.hpp"

namespace py = pybind11;

namespace BayesBoom {
  using namespace BOOM;

  void distribution_def(py::module &boom) {

    py::class_<RNG>(boom, "RNG")
        .def("seed", [] (RNG &rng, long seed) {
            rng.seed(seed);
          }, "Seed the random number generator")
        .def("__call__", [](RNG &rng) {
            return rng();
          }, "Simulate a U[0, 1) random deviate.")
        ;

    py::class_<GlobalRng>(boom, "GlobalRng")
        .def_property_readonly_static(
            "rng",
            [](py::object /* self */) {
              return BOOM::GlobalRng::rng;
            },
            "The BOOM global random number generator.")
        ;

    boom.def("seed_global_rng", [](int seed) {
      BOOM::GlobalRng::rng.seed(seed);
    });

  }  // ends the distribution_def function.

}  // namespace BayesBoom
