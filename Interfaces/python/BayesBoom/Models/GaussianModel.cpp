#include <pybind11/pybind11.h>

#include "Models/GaussianModel.hpp"
#include "LinAlg/Vector.hpp"

namespace py = pybind11;

namespace BayesBoom {
  using namespace BOOM;
  
  template <class MODEL>
  void set_double_data(Ptr<MODEL> &model, const Vector &data) {
    for (int i = 0; i < data.size(); ++i) {
      NEW(DoubleData, data_point)(data[i]);
      model->add_data(data_point);
    }
  }

  
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
        .def("mean", &GaussianModel::mu)
        .def("sd", &GaussianModel::sigma)
        .def("variance", &GaussianModel::sigsq)
        //        .def("set_data", &set_double_data, this)
        ;
    
  
  }  // Module

}  // namespace BOOM
