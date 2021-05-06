#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Models/StateSpace/StateSpaceModelBase.hpp"
#include "Models/StateSpace/StateSpaceModel.hpp"
#include "Models/StateSpace/StateSpaceRegressionModel.hpp"
#include "Models/StateSpace/StateSpaceStudentRegressionModel.hpp"
#include "Models/StateSpace/StateSpaceLogitModel.hpp"
#include "Models/StateSpace/StateSpacePoissonModel.hpp"
#include "Models/StateSpace/PosteriorSamplers/StateSpacePosteriorSampler.hpp"
#include "Models/StateSpace/PosteriorSamplers/StateSpaceStudentPosteriorSampler.hpp"
#include "Models/StateSpace/PosteriorSamplers/StateSpaceLogitPosteriorSampler.hpp"
#include "Models/StateSpace/PosteriorSamplers/StateSpacePoissonPosteriorSampler.hpp"

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
        .def_property_readonly(
            "regression_contribution",
            [](const ScalarStateSpaceModelBase &model) {
              return model.regression_contribution();
            },
            "The contribution of the regression component to the mean of y.   "
            "If no regression component is present, an empty Vector is "
            "returned.")
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
        .def("simulate_holdout_prediction_errors",
             [](ScalarStateSpaceModelBase &model,
                int niter,
                int cutpoint,
                bool standardize) {
               return model.simulate_holdout_prediction_errors(
                   niter, cutpoint, standardize);
             },
             py::arg("niter"),
             py::arg("cutpoint"),
             py::arg("standardize"),
             "Args:\n"
             "  niter:  The number of MCMC iterations for the simulation.  "
             "Controlling for burn-in is the responsibillity of the caller.\n"
             "  cutpoint:  An integer between 0 and the number "
             "of time points, giving the number of data in the initial "
             "training set of a train/test split.\n"
             "  standardize:  If True then each one step prediction errors "
             "will be divided by its forecast standard deviation from the "
             "Kalman filter.\n\n"
             "Returns:\n"
             "  A Matrix representing the distribution "
             "of one-step prediction errors.  The columns before the cutpoint "
             "are 'in-sample' while those after the cutpoint are "
             "'true-out-of-sample'.\n")
        .def("compute_prediction_errors",
             [] (ScalarStateSpaceModelBase &model,
                 int niter,
                 const std::vector<int> &cutpoints,
                 bool standardize) {
               return BOOM::StateSpaceUtils::compute_prediction_errors(
                       model, niter, cutpoints, standardize);
             },
             py::arg("niter"),
             py::arg("cutpoints"),
             py::arg("standardize"),
             "Compute the distribution of one step prediction errors for one "
             "or more holdout samples.\n\n"
             "Args:\n"
             "  niter:  The number of MCMC iterations for each simulation.  "
             "Controlling for burn-in is the responsibillity of the caller.\n"
             "  cutpoints:  A vector of integers between 0 and the number "
             "of time points.  Each number is the number of data points in "
             "the initial training set of a train/test split.\n"
             "  standardize:  If True then each one step prediction errors "
             "will be divided by its forecast standard deviation from the "
             "Kalman filter.\n\n"
             "Returns:\n"
             "  One Matrix for each cutpoint, representing the distribution "
             "of one-step prediction errors.  The columns before the cutpoint "
             "are 'in-sample' while those after the cutpoint are "
             "'true-out-of-sample'.\n")
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
        .def("simulate_forecast",
             [](StateSpaceModel &model, RNG &rng, int horizon, const Vector &final_state) {
               return model.simulate_forecast(rng, horizon, final_state);
             })
        .def("simulate_forecast_components",
             [](StateSpaceModel &model, RNG &rng, int horizon, const Vector &final_state) {
               return model.simulate_forecast_components(rng, horizon, final_state);
             })
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
        .def_property_readonly("xdim", [](StateSpaceRegressionModel &model) {
          return model.observation_model()->xdim();
        })
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
        .def("simulate_forecast",
             [](StateSpaceRegressionModel &model, RNG &rng, const Matrix &predictors,
                const Vector &final_state) {
               return model.simulate_forecast(rng, predictors, final_state);
             })
        .def("simulate_forecast_components",
             [](StateSpaceRegressionModel &model, RNG &rng, const Matrix &predictors,
                const Vector &final_state) {
               return model.simulate_forecast_components(rng, predictors, final_state);
             })
        ;


    py::class_<StateSpaceStudentRegressionModel,
               ScalarStateSpaceModelBase,
               PriorPolicy,
               Ptr<StateSpaceStudentRegressionModel>>(
                   boom,
                   "StateSpaceStudentRegressionModel",
                   py::multiple_inheritance())
        .def(py::init<int>(),
             py::arg("xdim"),
             "Args:\n\n"
             "  xdim:  The dimension of the predictor variables.  ")
        .def(py::init(
            [] (const Vector &response,
                const Matrix &predictors,
                const std::vector<bool> &response_is_observed) {
              return new StateSpaceStudentRegressionModel(
                  response, predictors, response_is_observed);
            }),
             py::arg("response"),
             py::arg("predictors"),
             py::arg("response_is_observed"),
             "Args:\n\n"
             "  response:  A BayesBoom.Vector containing the time series to be "
             "modeled.\n"
             "  predictors:  A BayesBoom.Matrix containing the predictor "
             "variables.\n"
             "  response_is_observed:  A numpy array of logical values "
             "indicating which elements of 'response' are observed.  "
             "False elements are considered missing values.\n")
        .def_property_readonly(
            "observation_model",
            [](StateSpaceStudentRegressionModel &model) {
              return model.observation_model();
            },
            "A BayesBoom.TRegression model object describing observation errors.")
        .def_property_readonly(
            "residual_sd",
            [] (const StateSpaceStudentRegressionModel &model) {
              return model.observation_model()->sigma();
            },
            "The scale parameter of the observation model describing residual "
            "errors.  Despite the name, this quantity is not in general the "
            "standard deviation of the errors, though it does approach the "
            "standard deviation when the tail parameter ('degrees of freedom')"
            "approaches infinity.\n")
        .def("set_residual_sd",
             [] (StateSpaceStudentRegressionModel &model, double residual_sd) {
               model.observation_model()->set_sigsq(residual_sd * residual_sd);
             },
             py::arg("residual_sd"),
             "Set the residual_sd parameter.\n"
             "Args:\n\n"
             "  residual_sd:  The new parameter value.\n")
        .def_property_readonly(
            "residual_df",
            [] (const StateSpaceStudentRegressionModel &model) {
              return model.observation_model()->nu();
            },
            "The tail thickness parameter of the observation model describing "
            "residual errors.  This is sometimes called the 'degrees of freedom' "
            "parameter\n")
        .def("set_residual_sd",
             [] (StateSpaceStudentRegressionModel &model, double residual_df) {
               model.observation_model()->set_nu(residual_df * residual_df);
             },
             py::arg("residual_df"),
             "Set the tail thickness parameter.\n"
             "Args:\n\n"
             "  residual_df:  The new parameter value.\n")
        .def_property_readonly(
            "coef",
            [](const StateSpaceStudentRegressionModel &model) {
              return model.observation_model()->coef();
            },
            "The GlmCoefs object describing the regression coefficients.")
        .def("simulate_forecast",
             [](StateSpaceStudentRegressionModel &model, RNG &rng, const Matrix &predictors,
                const Vector &final_state) {
               return model.simulate_forecast(rng, predictors, final_state);
             })
        .def("simulate_forecast_components",
             [](StateSpaceStudentRegressionModel &model, RNG &rng, const Matrix &predictors,
                const Vector &final_state) {
               return model.simulate_forecast_components(rng, predictors, final_state);
             })
        ;

    py::class_<StateSpaceLogitModel,
               ScalarStateSpaceModelBase,
               Ptr<StateSpaceLogitModel>>(
                   boom, "StateSpaceLogitModel", py::multiple_inheritance())
        .def(py::init(
            [](int xdim) {
              return new StateSpaceLogitModel(xdim);
            }),
             py::arg("xdim"),
             "Args\n\n"
             "  xdim: Dimension of the predictor vector.\n")
        .def(py::init(
            [](const Vector &successes,
               const Vector &trials,
               const Matrix &predictors,
               const std::vector<bool> &observed) {
              return new StateSpaceLogitModel(successes, trials, predictors, observed);
            }),
            py::arg("successes"),
            py::arg("trials"),
            py::arg("predictors"),
            py::arg("observed"),
            "Args:\n\n"
            "  successes:  The response variable.  The number of successes "
            "observed.  0 <= successes <= trials.\n"
            "  trials: The number of trials.  \n"
            "  predictors:  The matrix of predictors.\n"
            "  observed:  Is the response observed?\n")
        .def_property_readonly(
            "observation_model",
            [] (StateSpaceLogitModel *model) {
              return model->observation_model();
            })
        .def_property_readonly(
            "coef", [](const StateSpaceLogitModel *model) {
              return model->observation_model()->coef();
            })
        .def("simulate_forecast",
             [](StateSpaceLogitModel *model,
                RNG &rng,
                const Matrix &forecast_predictors,
                const Vector &trials,
                const Vector &final_state) {
               return model->simulate_forecast(
                   rng, forecast_predictors, trials, final_state);
             })
        .def("simulate_forecast_components",
             [](StateSpaceLogitModel *model,
                RNG &rng,
                const Matrix &forecast_predictors,
                const Vector &trials,
                const Vector &final_state) {
               return model->simulate_forecast_components(
                   rng, forecast_predictors, trials, final_state);
             })
        ;

    py::class_<StateSpacePoissonModel,
               ScalarStateSpaceModelBase,
               Ptr<StateSpacePoissonModel>>(
                   boom, "StateSpacePoissonModel", py::multiple_inheritance())
        .def(py::init(
            [](int xdim) {
              return new StateSpacePoissonModel(xdim);
            }),
             py::arg("xdim"),
             "Args\n\n"
             "  xdim: Dimension of the predictor vector.\n")
        .def(py::init(
            [](const Vector &counts,
               const Vector &exposure,
               const Matrix &predictors,
               const std::vector<bool> &observed) {
              return new StateSpacePoissonModel(
                  counts, exposure, predictors, observed);
            }),
            py::arg("counts"),
            py::arg("exposure"),
            py::arg("predictors"),
            py::arg("observed"),
            "Args:\n\n"
            "  counts:  The response variable. The number of events observed.\n"
            "  trials: The number of trials.  \n"
            "  predictors:  The matrix of predictors.\n"
            "  observed:  Is the response observed?\n")
        .def_property_readonly(
            "observation_model",
            [] (StateSpacePoissonModel *model) {
              return model->observation_model();
            })
        .def_property_readonly(
            "coef", [](const StateSpacePoissonModel *model) {
              return model->observation_model()->coef();
            })
        .def("simulate_forecast_components",
             [](StateSpacePoissonModel *model,
                RNG &rng,
                const Matrix &forecast_predictors,
                const Vector &exposure,
                const Vector &final_state) {
               return model->simulate_forecast_components(
                   rng, forecast_predictors, exposure, final_state);
             })
        ;


    //==========================================================================
    // Posterior Samplers
    //==========================================================================
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

    py::class_<StateSpaceStudentPosteriorSampler,
               PosteriorSampler,
               Ptr<StateSpaceStudentPosteriorSampler>>(
                   boom,
                   "StateSpaceStudentPosteriorSampler")
        .def(py::init(
            [] (StateSpaceStudentRegressionModel *model,
                TRegressionSpikeSlabSampler *observation_model_sampler,
                RNG &seeding_rng) {
              return new StateSpaceStudentPosteriorSampler(
                  model, observation_model_sampler, seeding_rng);
            }),
            py::arg("model"),
            py::arg("observation_model_sampler"),
            py::arg("seeding_rng") = BOOM::GlobalRng::rng,
             "Args:\n\n"
             "  model: The TRegressionModel to be sampled.\n"
             "  observation_model_sampler:  TRegressionSpikeSlabSampler for "
             "sampling the model.\n"
             "  seeding_rng: The random number generator used to seed the"
             "RNG in this sampler.\n")
          ;

    py::class_<StateSpaceLogitPosteriorSampler,
               PosteriorSampler,
               Ptr<StateSpaceLogitPosteriorSampler>>(
                   boom,
                   "StateSpaceLogitPosteriorSampler")
        .def(py::init(
            [] (StateSpaceLogitModel *model,
                BinomialLogitSpikeSlabSampler *observation_model_sampler,
                RNG &seeding_rng) {
              return new StateSpaceLogitPosteriorSampler(
                  model, observation_model_sampler, seeding_rng);
            }),
            py::arg("model"),
            py::arg("observation_model_sampler"),
            py::arg("seeding_rng") = BOOM::GlobalRng::rng,
             "Args:\n\n"
             "  model: The BinomialLogitModel to be sampled.\n"
             "  observation_model_sampler:  BinomialLogitSpikeSlabSampler for "
             "sampling the model.\n"
             "  seeding_rng: The random number generator used to seed the"
             "RNG in this sampler.\n")
          ;

    py::class_<StateSpacePoissonPosteriorSampler,
               PosteriorSampler,
               Ptr<StateSpacePoissonPosteriorSampler>>(
                   boom, "StateSpacePoissonPosteriorSampler")
        .def(py::init(
            [] (StateSpacePoissonModel *model,
                PoissonRegressionSpikeSlabSampler *observation_model_sampler,
                RNG &seeding_rng) {
              return new StateSpacePoissonPosteriorSampler(
                  model, observation_model_sampler, seeding_rng);
            }),
             py::arg("model"),
             py::arg("observation_model_sampler"),
             py::arg("seeding_rng") = GlobalRng::rng,
             "Args:\n\n"
             "  model: The boom.StateSpacePoissonModel to be sampled.\n"
             "  observation_model_sampler: A PoissonRegressionSpikeSlabSampler"
             "    responsible for sampling the observation model.\n"
             "  seeding_rng:  The random number generator used to seed "
             "this sampler's RNG.\n")
        ;

  }  // StateSpaceModel_def

}  // namespace BayesBoom
