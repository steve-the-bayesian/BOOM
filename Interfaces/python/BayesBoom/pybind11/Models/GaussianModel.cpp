#include <pybind11/pybind11.h>

#include "Models/GaussianModel.hpp"

namespace py = pybind11;

namespace BOOM {

  PYBIND11_MODULE(BayesBoom, boom) {

    boom.doc() = "A library for Bayesian modeling, and assorted "
        "other useful bits.";

    //    py::class_<GaussianModel, Model>(boom, "GaussianModel")
    py::class_<GaussianModel>(boom, "GaussianModel")    
        .def(py::init<double, double>(),
             py::arg("mean") = 0.0, py::arg("sd") = 1.0)
        .def("set_mean_sd", &GaussianModel::set_params,
             py::arg("mean"),
             py::arg("sd"))
        ;
    
  
  }  // Module

}  // namespace BOOM
