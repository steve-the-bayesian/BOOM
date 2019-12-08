#include <pybind11/pybind11.h>

#include "Models/Glm/Glm.hpp"
#include "Models/Glm/RegressionModel.hpp"

#include "cpputil/Ptr.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;

  void GlmModel_def(py::module &boom) {

    py::class_<GlmModel,
               NumOptModel,
               Ptr<RegressionModel>>(boom, "GlmModel")
        .def_property_readonly("coef", &GlmModel::coef)   // TODO: handle const's.
        .def_property_readonly("xdim", &GlmModel::xdim)
        .def("add_all", &GlmModel::add_all)
        .def("drop_all", &GlmModel::drop_all)
        .def("drop_all_but_intercept", &GlmModel::drop_all_but_intercept)
        .def("add", &GlmModel::add, "Add the variable in the specified position.")
        .def("drop", &GlmModel::add, "Drop the variable in the specified position.")
        .def("flip", &GlmModel::add, "Flip the variable in the specified position.")

        ;


    py::class_<RegressionModel,
               GlmModel,
               NumOptModel,
               Ptr<RegressionModel>>(boom, "RegressionModel")
        ;


  }

}  // namespace BayesBoom
