#include <pybind11/pybind11.h>

#include "Models/GammaModel.hpp"
#include "Models/ChisqModel.hpp"
#include "cpputil/Ptr.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;

  void GammaModel_def(py::module &boom) {

    py::class_<GammaModelBase,
               DoubleModel,
               BOOM::Ptr<GammaModelBase>>(boom, "GammaModelBase")
        .def("alpha", &GammaModelBase::alpha)
        .def("beta", &GammaModelBase::beta)
        .def("mean", &GammaModelBase::mean)
        .def("variance", &GammaModelBase::variance)
        ;

    py::class_<GammaModel,
               GammaModelBase,
               BOOM::Ptr<GammaModel>>(boom, "GammaModel")
        .def(py::init<double, double>(),
             py::arg("a") = 1.0, py::arg("b") = 1.0,
             "Args:\n"
             "  a: The shape parameter.\n"
             "  b: The scale parameter, defined so that the mean of this "
             "distribution is a/b.")
        .def("mle", &GammaModel::mle)
        .def("set_data", [](GammaModel &model, const Vector &data) {
            for (const auto &el: data) {
              NEW(DoubleData, data_point)(el);
              model.add_data(data_point);
            }
          }, "Assign the data in the supplied vector to the model.")
        .def("__repr__", [](const Ptr<GammaModel> &model) {
            std::ostringstream out;
            out << "A BOOM Gamma model with alpha = " << model->alpha()
                << " and beta = " << model->beta()
                << ".";
            return out.str();
          })
        ;

    py::class_<ChisqModel,
               GammaModelBase,
               Ptr<ChisqModel>>(boom, "ChisqModel")
        .def(py::init<double, double>(),
             py::arg("df") = 1.0,
             py::arg("sigma_estimate") = 1.0,
             "Args:\n"
             "  df: A prior sample size.\n"
             "  sigma_estimate:  A prior guess at the standard deviation "
             "being modeled."
             )
        .def_property_readonly("sum_of_squares", &ChisqModel::sum_of_squares)
        ;

  }  // Module

}  // namespace BOOM
