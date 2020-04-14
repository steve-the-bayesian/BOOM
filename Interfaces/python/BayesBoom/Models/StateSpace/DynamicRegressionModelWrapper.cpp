#include <pybind11/pybind11.h>

#include "Models/StateSpace/DynamicRegression.hpp"

#include "cpputil/Ptr.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;
  using namespace BOOM::StateSpace;

  void DynamicRegressionModel_def(py::module &boom) {

    py::class_<RegressionDataTimePoint,
               Data,
               Ptr<RegressionDataTimePoint>>(boom, "RegressionDataTimePoint")
        ;


    py::class_<DynamicRegressionModel,
               Model,
               PriorPolicy,
               BOOM::Ptr<DynamicRegressionModel>>(boom, "DynamicRegressionModel")
        .def(py::init<int>(), py::arg("xdim"),
             "Args:\n\n"
             "  xdim: Number of potential predictor variables.")
        .def_property_readonly(
            "all_coefficients",
            [](const DynamicRegressionModel &model) {
              Matrix ans(model.xdim(), model.time_dimension());
              for (int t = 0; t < model.time_dimension(); ++t) {
                ans.col(t) = model.coef(t).Beta();
              }
              return ans;},
            "Matrix of dimension (xdim, time_dimension) containing the "
            "dynamic regression coefficients.")
        .def("coef", &DynamicRegressionModel::coef, py::arg("t"),
             "Args:\n\n"
             "  t: Time index.  A positive number less than time_dimension.")
        .def_property_readonly(
            "residual_sd",
            &DynamicRegressionModel::residual_sd,
             "Residual standard deviation")
        .def_property_readonly(
            "unscaled_innovation_sds",
            [](DynamicRegressionModel &model) {
              return sqrt(model.unscaled_innovation_variances());},
            "Vector of unscaled innovation standard deviations.  Multiply by "
            "residual SD to get the actual innovation standard deviations.")
        ;

  }  // StateSpaceModel_def

}  // namespace BayesBoom
