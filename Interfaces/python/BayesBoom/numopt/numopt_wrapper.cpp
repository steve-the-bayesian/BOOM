#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "LinAlg/Vector.hpp"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/SpdMatrix.hpp"

#include "numopt/LinearAssignment.hpp"

namespace py = pybind11;

namespace BayesBoom {
  using namespace BOOM;

  void numopt_def(py::module &boom) {

    py::class_<LinearAssignment> (boom, "LinearAssignment")
        .def(py::init(
            [](const Matrix &cost) {
              return new LinearAssignment(cost);
            }),
             py::arg("cost"),
             "Args:\n\n"
             "  cost: A boom.Matrix of assigning task (column) j "
             "to worker (row) i.")
        .def("solve",
             [](LinearAssignment &lap) { return lap.solve(); },
             "Find the optimal solution.  The cost at the minimum is returned.")
        .def_property_readonly(
            "row_solution",
            [](const LinearAssignment &lap) {return lap.row_solution();})
        .def_property_readonly(
            "col_solution",
            [](const LinearAssignment &lap) {return lap.col_solution();})
        ;

  }

}  // namespace BayesBoom
