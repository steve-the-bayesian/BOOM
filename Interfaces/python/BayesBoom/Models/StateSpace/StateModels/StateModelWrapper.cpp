#include <pybind11/pybind11.h>

#include "Models/StateSpace/StateModels/LocalLevelStateModel.hpp"
#include "Models/StateSpace/StateModels/LocalLinearTrend.hpp"

#include "cpputil/Ptr.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;

  void StateModel_def(py::module &boom) {

    // Base class
    py::class_<LocalLevelStateModel,
               StateModel,
               ZeroMeanGaussianModel,
               BOOM::Ptr<LocalLevelStateModel>>(boom, "LocalLevelStateModel")
        .def(py::init<double>(),
             py::arg("sigma") = 1.0,
             "Args:\n"
             "  sigma: Standard deviation of the innovation errors.")
        .def_property_readonly(
            "state_dimension", &LocalLevelStateModel::state_dimension,
            "Dimension of the state vector.")
        .def_property_readonly(
            "state_error_dimension",
            &LocalLevelStateModel::state_error_dimension,
            "Dimension of the innovation term.")
        .def("set_initial_state_mean",
             &LocalLevelStateModel::set_initial_state_mean,
             py::arg("mean"),
             "Set the mean of the initial state distribution to the "
             "specified value.")
        .def("set_initial_state_variance",
             &LocalLevelStateModel::set_initial_state_variance,
             py::arg("variance"),
             "Set the variance of the initial state distribution to the "
             "specified value.")
        ;


  }  // StateSpaceModel_def

}  // namespace BayesBoom
