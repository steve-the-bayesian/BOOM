#include <pybind11/pybind11.h>

#include "Models/GaussianModel.hpp"

namespace boompy {

  PYBIND11_MODULE(Boom, boom) {

    boom.doc() = "A library for Bayesian modeling, and assorted "
        "other useful bits.";

    py::class<GaussianModel, Model>(boom, "GaussianModel")
        .def(py::init<double mean = 0.0, double sd = 1.0>())
        .def("set_mean_sd", &GaussianModel::set_params)
        ;
    
  }
  
}  // namespace boompy
