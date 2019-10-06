#include <pybind11/pybind11.h>

#include "Models/StateSpace/StateSpaceModelBase.hpp"
#include "Models/StateSpace/StateSpaceModel.hpp"

#include "cpputil/Ptr.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;

  void StateSpaceModel_def(py::module &boom) {

    // Base class
    py::class_<StateSpaceModelBase,
               Model,
               BOOM::Ptr<StateSpaceModelBase>>(boom, "StateSpaceModelBase")
        .def_property_readonly(
            "time_dimension", &StateSpaceModelBase::time_dimension,
            "The number of time points in the training data.")
        .def_property_readonly(
            "state_dimension", &StateSpaceModelBase::state_dimension,
            "The number of elements in the state vector.")
        .def_property_readonly(
            "number_of_state_models", &StateSpaceModelBase::number_of_state_models,
            "The number of logical components in the state for this model.")
        .def("add_state", &StateSpaceModelBase::add_state,
             "Expand the state definition by adding a state model to "
             "describe trend, seasonal, etc.\n\n"
             "Args:\n"
             "  state: state model to be added.   Posterior samplers and initial "
             "state priors should be set before adding.")
        .def("log_likelihood", [](StateSpaceModelBase &model) {
            return model.log_likelihood();
          },
             "The log likelihood associated with the current model parameters.  "
             "If the Kalman filter is current this is already computed.  If not, "
             "then computing log likelihood requires a Kalman filter pass through "
             "the data.")
        ;

    py::class_<ScalarStateSpaceModelBase,
               StateSpaceModelBase,
               BOOM::Ptr<ScalarStateSpaceModelBase>>(boom, "ScalarStateSpaceModelBase")
        .def("observation_variance",
             &ScalarStateSpaceModelBase::observation_variance,
             py::arg("t"),
             "Args:\n"
             "  t: time index where variance is desired.\n"
             "Returns:\n"
             "  The (residual) variance of the observed data y[t] given the state "
             "alpha[t] and all model parameters.")
        .def("one_step_prediction_errors",
             &ScalarStateSpaceModelBase::one_step_prediction_errors,
             py::arg("standardize") = false,
             "Args:\n"
             "  standardize: Should the prediction error at each time point be "
             "divided by its standard deviation from the Kalman filter? \n\n"
             "Returns:\n"
             "  A vector containing the one step prediction errors from the "
             "Kalman filter.")
        ;



  }  // StateSpaceModel_def

}  // namespace BayesBoom
