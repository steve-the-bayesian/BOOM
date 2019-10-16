#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "LinAlg/Vector.hpp"
#include "LinAlg/Matrix.hpp"
#include "stats/Spline.hpp"
#include "stats/Bspline.hpp"
#include "cpputil/Ptr.hpp"

namespace py = pybind11;

namespace BayesBoom {
  using namespace BOOM;

  void Spline_def(py::module &boom) {

    py::class_<SplineBase> (boom, "SplineBase")
        .def("basis", &SplineBase::basis, py::arg("x: float"),
             py::return_value_policy::copy,
             "Spline basis expansion at x.")
        .def("basis_matrix", &SplineBase::basis_matrix, py::arg("x: Vector"),
             py::return_value_policy::copy,
             "Spline basis matrix expansion of the Vector x.")
        .def_property_readonly("dim", &SplineBase::basis_dimension,
                               "The dimension of the expanded basis.")
        .def("add_knot", &SplineBase::add_knot, py::arg("knot: float"),
             "Add a knot at the specified value.  The support of the spline will be "
             "expanded to include 'knot' if necessary.")
        .def("remove_knot", &SplineBase::remove_knot, py::arg("which_knot: int"),
             "Remove the specified knot.  If which_knot corresponds to the "
             "largest or smallest knots then the support of the spline will be "
             "reduced.")
        .def("knots", &SplineBase::knots)
        .def("number_of_knots", &SplineBase::number_of_knots)
        ;


    py::class_<Bspline, SplineBase>(boom, "Bspline")
        .def(py::init<const Vector &, int>(), py::arg("knots"), py::arg("degree") = 3,
             "Create a Bspline basis.\n\n")
        .def("basis", (Vector (Bspline::*)(double)) &Bspline::basis, py::arg("x"),
             py::return_value_policy::copy,
             "The basis function expansion at x.")
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
