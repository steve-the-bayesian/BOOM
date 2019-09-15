#include <pybind11/pybind11.h>

#include "Models/DataTypes.hpp"
#include "Models/ParamTypes.hpp"
#include "Models/ModelTypes.hpp"
#include "Models/GaussianModel.hpp"


// Trampoline classes for pure virtual classes.  See
// https://pybind11.readthedocs.io/en/stable/advanced/classes.html#combining-virtual-functions-and-inheritance


PYBIND11_MODULE(Boom, boom) {
  boom.doc() = "A library for Bayesian modeling, and assorted "
      "other useful bits.";

    
  // Define Data abstract class
  py::class_<Data> data;
  
  // Define Model abstract class.

  py::class_<GaussianModelBase>(boom, "GaussianModelBase")

  py::class_<GaussianModel, GaussianModelBase>(boom, "GaussianModel")
      .def(py::init<double mean = 0.0, double sd = 1.0>())
      .def("set_params", &GaussianModel::set_params,
           "Set the mean and variance parameters.", 
           py::arg("mean"), py::arg("var"))
      ;
    
}
  
