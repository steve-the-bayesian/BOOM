#include <pybind11/pybind11.h>

#include "LinAlg/Vector.hpp"
#include "Models/ParamTypes.hpp"
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
        ;

    py::class_<VectorParams, Ptr<VectorParams>>(boom, "VectorParams")
        .def(py::init<const Vector &>(),
             py::arg("x") = 0,
             "Create a VectorParams with value x")
        ;

  }

}  // namespace BayesBoom
