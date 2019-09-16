#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "LinAlg/Vector.hpp"
#include "stats/Bspline.hpp"

namespace py = pybind11;

namespace BayesBoom {
  using namespace BOOM;

  void Spline_def(py::module &boom) {
    py::class_<Bspline>(boom, "Bspline")
        .def(py::init<const Vector &, int>(), py::arg("knots"), py::arg("degree") = 3,
             "Create a Bspline basis.\n\n")
        .def("basis", &Bspline::basis, py::arg("x"),
             "The basis function expansion at x.")
        .def_property_readonly("dim", &Bspline::basis_dimension,
                               "The dimension of the expanded basis.")
        .def_property_readonly("order", &Bspline::order,
                               "The order of the spline. (1 + degree).")
        .def_property_readonly("degree", &Bspline::degree, "The degree of the spline.")
        .def("__repr__",
             [](const Bspline &s) {
               std::ostringstream out;
               out << "A Bspline basis of degree " << s.degree() << " with knots at ["
                   << s.knots() << "].";
               return out.str();
             })
        ;

  }  // ends the Spline_def function.

}  // namespace BayesBoom


