#include <pybind11/pybind11.h>

#include "Models/UniformModel.hpp"
#include "cpputil/Ptr.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;

  void UniformModel_def(py::module &boom) {

    // Do we want to inherit from DoubleModel in the class template
    py::class_<UniformModel,
               DoubleModel,
               BOOM::Ptr<UniformModel>>(boom, "UniformModel", py::multiple_inheritance())
        .def(py::init<double, double>(),
             py::arg("lo"),
             py::arg("hi"),
             "Args:\n\n"
             "lo, hi: the lower and upper limits of the model support.\n")
        ;

  }  // Module

}  // namespace BOOM
