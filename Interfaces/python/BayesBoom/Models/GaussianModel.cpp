#include <pybind11/pybind11.h>

#include "Models/GaussianModel.hpp"
#include "cpputil/Ptr.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;

  void GaussianModel_def(py::module &boom) {
    
    py::class_<GaussianModel, BOOM::Ptr<GaussianModel> >(boom, "GaussianModel")
        .def(py::init<double, double>(),
             py::arg("mean") = 0.0, py::arg("sd") = 1.0)
        .def("set_mean_sd", &GaussianModel::set_params,
             py::arg("mean"),
             py::arg("sd"))
        .def("mean", &GaussianModel::mu)
        .def("sd", &GaussianModel::sigma)
        .def("variance", &GaussianModel::sigsq)
        .def("mle", &GaussianModel::mle)
        .def("log_likelihood", &GaussianModel::Loglike)
        .def("set_data", [](GaussianModel &model, const Vector &data) {
            for (const auto &el: data) {
              NEW(DoubleData, data_point)(el);
              model.add_data(data_point);
            }
          }, "Assign the data in the supplied vector to the model.")
        .def("__repr__", [](const Ptr<GaussianModel> &model) {
            std::ostringstream out;
            out << "A BOOM Gaussian model with mean " << model->mu()
                << " and standard deviation " << model->sigma()
                << ".";
            return out.str();
          })
        ;
  
  }  // Module

}  // namespace BOOM
