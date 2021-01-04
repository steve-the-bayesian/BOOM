#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Models/StateSpace/StateSpaceModelBase.hpp"
#include "Models/StateSpace/StateSpaceModel.hpp"
#include "Models/StateSpace/StateSpaceRegressionModel.hpp"
#include "Models/StateSpace/PosteriorSamplers/StateSpacePosteriorSampler.hpp"

#include "cpputil/Ptr.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;

  void StateSpaceModel_def(py::module &boom) {

    // Base class
    py::class_<StateSpaceModelBase,
               Model,
               BOOM::Ptr<StateSpaceModelBase>>(
                   boom,
                   "StateSpaceModelBase",
                   py::multiple_inheritance())
        .def_property_readonly(
            "time_dimension", &StateSpaceModelBase::time_dimension,
            "The number of time points in the training data.")
        .def_property_readonly(
            "state_dimension", &StateSpaceModelBase::state_dimension,
            "The number of elements in the state vector.")
        .def_property_readonly(
            "number_of_state_models", &StateSpaceModelBase::number_of_state_models,
            "The number of logical components in the state for this model.")
        .def_property_readonly(
            "state",
            [] (const StateSpaceModelBase &model) {return model.state();},
            "The state matrix. Rows are state variables, columns are time.")
        .def("add_state",
             [](StateSpaceModelBase &model, StateModel &state_model) {
               // TODO: This is a hack around pybind11 struggling to convert Ptr
               // from concrete to base classes.  It works only because BOOM's
               // pointers are intrusive.
               model.add_state(Ptr<StateModel>(&state_model));
             },
             "Expand the state definition by adding a state model to "
             "describe trend, seasonal, etc.\n\n"
             "Args:\n"
             "  state: state model to be added.   Posterior samplers and initial "
             "state priors should be set before adding.")
        .def_property_readonly(
            "log_likelihood",
            [](StateSpaceModelBase &model) { return model.log_likelihood(); },
             "The log likelihood associated with the current model parameters.  "
             "If the Kalman filter is current this is already computed.  If not, "
             "then computing log likelihood requires a Kalman filter pass through "
             "the data.")
        ;

    py::class_<ScalarStateSpaceModelBase,
               StateSpaceModelBase,
               BOOM::Ptr<ScalarStateSpaceModelBase>>(
                   boom,
                   "ScalarStateSpaceModelBase",
                   py::multiple_inheritance())
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

    py::class_<StateSpaceModel,
               ScalarStateSpaceModelBase,
               PriorPolicy,
               Ptr<StateSpaceModel>>(
                   boom,
                   "StateSpaceModel",
                   py::multiple_inheritance())
        .def(py::init<>(),
             "Create a state space model.")
        .def(py::init(
            [](const Vector &y,
               const std::vector<bool> &is_observed) {
              return Ptr<StateSpaceModel>(
                  new StateSpaceModel(y, is_observed));
            }),
            py::arg("y"),
            py::arg("is_observed"),
            "Args:\n"
            "  y: The time series of observations.\n"
            "  is_observed:  A vector of the same length as y, "
            "indicating which elements of y are observed.")
        .def_property_readonly(
            "observation_model",
            [](StateSpaceModel &model) {
              return model.observation_model();
            },
            "A ZeroMeanGaussianModel describing the observation errors.")
        .def_property_readonly(
            "residual_sd",
            [](const StateSpaceModel &model) {
              return model.observation_model()->sigma();
            },
            "The residual standard deviation parameter.")
        ;

    py::class_<StateSpaceRegressionModel,
               ScalarStateSpaceModelBase,
               PriorPolicy,
               Ptr<StateSpaceRegressionModel>>(
                   boom,
                   "StateSpaceRegressionModel",
                   py::multiple_inheritance())
        .def(py::init<int>(),
             py::arg("xdim"),
             "Args:\n\n"
             "  xdim:  The dimension of the predictor variables.")
        .def(py::init(
            [](const Vector &response,
               const Matrix &predictors,
               const std::vector<bool> &is_observed) {
              return new StateSpaceRegressionModel(
                  response, predictors, is_observed);
            }),
             py::arg("response"),
             py::arg("predictors"),
             py::arg("is_observed"),
             "Args:\n"
             "  response:  The time series to be modeled.\n"
             "  predictors:  The matrix of predictors.  The number of rows \n"
             "    must match the length of 'response'.\n"
             "  is_observed:  A boolean vector indicating which elements \n"
             "    of 'response' are observed.   All elements of 'predictors' \n"
             "    must be fully observed.\n")
        .def_property_readonly(
            "observation_model",
            [](StateSpaceRegressionModel &model) {
              return model.observation_model();
            },
            "A RegresionModel describing the observation errors.")
        .def_property_readonly(
            "residual_sd",
            [](const StateSpaceRegressionModel &model) {
              return model.observation_model()->sigma();
            },
            "The residual standard deviation parameter.")
        .def_property_readonly(
            "coef",
            [](const StateSpaceRegressionModel &model) {
              return model.observation_model()->coef();
            },
            "The GlmCoefs object describing the regression coefficients.")
        ;

    py::class_<StateSpacePosteriorSampler,
               PosteriorSampler,
               Ptr<StateSpacePosteriorSampler>>(
                   boom,
                   "StateSpacePosteriorSampler")
        .def(py::init(
            [] (StateSpaceModelBase &model, RNG &seeding_rng) {
              return new StateSpacePosteriorSampler(
                  &model, seeding_rng); }),
            py::arg("model"),
            py::arg("seeding_rng") = BOOM::GlobalRng::rng,
            "Args:\n\n"
            "  model: The model to be sampled.\n"
            "  seeding_rng: The random number generator used to seed the"
            "RNG in this sampler.")
          ;


  }  // StateSpaceModel_def

}  // namespace BayesBoom
