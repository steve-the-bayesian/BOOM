#include <pybind11/pybind11.h>

#include "LinAlg/Vector.hpp"
#include "Models/ParamTypes.hpp"
#include "Models/SpdParams.hpp"
#include "cpputil/Ptr.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;

  // Define the
  void Parameter_def(py::module &boom) {

    py::class_<UnivParams, Ptr<UnivParams>>(boom, "UnivParams")
        .def(py::init<double>(),
             py::arg("x") = 0,
             "Create a UnivParams with value x")
        .def("set",
             [](UnivParams &prm, double x) {
               prm.set(x);
             },
             "Set the parameter value to the given argument.")
        .def_property_readonly("value",
                               &UnivParams::value,
                               "The value of the parameter (float).")

        ;

    py::class_<VectorParams, Ptr<VectorParams>>(boom, "VectorParams")
        .def(py::init<const Vector &>(),
             py::arg("x") = 0,
             "Create a VectorParams with value x")
        .def("set",
             [](VectorParams &prm, const Vector &x) {
               prm.set(x);
             },
             "Set the parameter value to x.")
        .def_property_readonly("value",
                               &VectorParams::value,
                               "The value of the parameter (Vector).")
        ;

    py::class_<SpdParams,
               Ptr<SpdParams>>(boom, "SpdParams")
        .def(py::init<const SpdMatrix &, bool>(),
             py::arg("V"),
             py::arg("ivar") = false,
             "Create an SpdParams from a variance or precision matrix.\n\n"
             "Args:\n"
             "  V: SpdMatrix.  The initial value of the parameter.\n"
             "  ivar:  If True then V is a precision matrix.  If False then V is a variance.")
        .def("set_var",
             [](SpdParams &prm, const SpdMatrix &value) {
               prm.set_var(value);
             })
        .def("set_ivar",
             [](SpdParams &prm, const SpdMatrix &value) {
               prm.set_ivar(value);
             })
        .def_property_readonly(
            "var",
            &SpdParams::var,
            "The parameter's value as a variance (non-inverted).")
        .def_property_readonly(
            "ivar",
            &SpdParams::var,
            "The parameter's value as a precision (inverted).")
        ;

  }  // module

}  // namespace BayesBoom
