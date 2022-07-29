#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Models/StateSpace/Multivariate/MultivariateStateSpaceRegressionModel.hpp"
#include "Models/StateSpace/Multivariate/PosteriorSamplers/MultivariateStateSpaceModelSampler.hpp"

#include "cpputil/math_utils.hpp"

#include "cpputil/Ptr.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;
  void MultivariateStateSpaceModel_def(py::module &boom) {

    py::class_<MultivariateStateSpaceModelBase,
               Model,
               BOOM::Ptr<MultivariateStateSpaceModelBase>>(
                   boom,
                   "MultivariateStateSpaceModelBase",
                   py::multiple_inheritance())
        .def_property_readonly(
            "nseries",
            [](const MultivariateStateSpaceModelBase &model) {
              return model.nseries();
            },
            "The number of time series described by the model.")
        .def_property_readonly(
            "time_dimension",
            [](const MultivariateStateSpaceModelBase &model) {
              return model.time_dimension();
            },
            "The number of time points in the model training data.")
        .def_property_readonly(
            "state_dimension",
            [](const MultivariateStateSpaceModelBase &model) {
              return model.state_dimension();
            },
            "The dimension of the state vector shared across all time series.")
        .def_property_readonly(
            "number_of_state_models",
            [](const MultivariateStateSpaceModelBase &model) {
              return model.number_of_state_models();
            },
            "The number state models defining the shared state vector.")
        .def_property_readonly(
            "log_likelihood",
            [](MultivariateStateSpaceModelBase &model) {
              return model.log_likelihood();
            },
            "The log likelihood under the current set of model parameters.")
        .def("state_contributions",
             [](const MultivariateStateSpaceModelBase &model, int which_state_model) {
               return model.state_contributions(which_state_model);
             },
             py::arg("which_state_model"),
             "Args:\n"
             "  which_state_model: The state model whose contribution is desired.\n"
             "\n"
             "Returns:\n"
             "  A Matrix.  Element (t, d) is the contrubtion of the specified "
             "state model to series d at time t.")
        .def_property_readonly(
            "shared_state",
            [](const MultivariateStateSpaceModelBase &model) {return model.shared_state();},
            "The full state matrix for the model")
        ;

    py::class_<ConditionallyIndependentMultivariateStateSpaceModelBase,
               MultivariateStateSpaceModelBase,
               BOOM::Ptr<ConditionallyIndependentMultivariateStateSpaceModelBase>>(
                   boom,
                   "ConditionallyIndependentMultivariateStateSpaceModelBase",
                   py::multiple_inheritance())
        ;

    py::class_<MultivariateTimeSeriesRegressionData,
               RegressionData,
               BOOM::Ptr<MultivariateTimeSeriesRegressionData>>(
                   boom,
                   "MultivariateTimeSeriesRegressionData")
        .def(py::init(
            [](double y, const Vector &x, int series, int timestamp) {
              return new MultivariateTimeSeriesRegressionData(
                  y, x, series, timestamp);
            }),
             py::arg("y"),
             py::arg("x"),
             py::arg("series"),
             py::arg("timestamp"),
             "Args:\n"
             "  y: The response variable.\n"
             "  x: A vector of predictors.\n"
             "  series: The identifier of the time series (0.. number of series - 1) to\n"
             "    which this observation belongs.\n"
             "  timestamp: The time-index of the time series (0.. sample_size - 1)\n"
             "    containing this observation.\n")
        ;


    py::class_<MultivariateStateSpaceRegressionModel,
               ConditionallyIndependentMultivariateStateSpaceModelBase,
               PriorPolicy,
               BOOM::Ptr<MultivariateStateSpaceRegressionModel>>(
                   boom,
                   "MultivariateStateSpaceRegressionModel",
                   py::multiple_inheritance())
        .def(py::init(
            [](int xdim, int nseries) {
              return new BOOM::MultivariateStateSpaceRegressionModel(
                  xdim, nseries);
            }),
             py::arg("xdim"),
             py::arg("nseries"),
             "Args:\n"
             "  xdim:  The dimension of the predictor variables.\n"
             "  nseries: The number of time series being modeled.\n")
        .def_property_readonly(
            "xdim",
            [](const MultivariateStateSpaceRegressionModel &model) {
              return model.xdim();
            },
            "Dimension of the vector of predictor variables.")
        .def("add_data",
             [](MultivariateStateSpaceRegressionModel &model,
                const Ptr<MultivariateTimeSeriesRegressionData> &data_point) {
               model.add_data(data_point);
             },
             py::arg("data_point"),
             "Add a single data point to the model.\n\n"
             "Args:\n"
             "  data_point: A MultivariateTimeSeriesRegressionData object\n"
             "    containing information for a single data point.\n")
        .def("add_data",
             [](MultivariateStateSpaceRegressionModel &model,
                const std::vector<int> &time_index,
                const std::vector<int> &series_index,
                const Vector &response,
                const Matrix &predictors) {
               size_t nobs = time_index.size();
               if (series_index.size() != nobs) {
                 report_error("The series_index and time_index must have "
                              "the same number of elements.");
               }
               if (response.size() != nobs) {
                 report_error("The response must have the same number of "
                              "elements as the time_index.");
               }
               if (predictors.nrow() != nobs) {
                 report_error("The matrix of predictors must have the same "
                              "number of rows as the time_index.");
               }
               for (size_t i = 0; i < nobs; ++i) {
                 NEW(MultivariateTimeSeriesRegressionData, data_point)(
                     response[i],
                     predictors.row(i),
                     series_index[i],
                     time_index[i]);
                 model.add_data(data_point);
               }
             },
             py::arg("time_index"),
             py::arg("series_index"),
             py::arg("response"),
             py::arg("predictors"),
             "Add a full data set to the model.\n\n"
             "Args:\n"
             "  time_index:  A list of integers indicating the time stamp "
             "(0, 1, 2...) associated with the observation.\n"
             "  series_index:  A list of integers indicating which series the "
             "observation describes.\n"
             "  response:  A boom.Vector giving the values of each series at "
             "the specified times.\n"
             "  predictors:  A boom.Matrix giving the row of predictor "
             "variables to use for each observation.\n\n"
             "Effect:\n"
             "  The model object is populated with the supplied data.\n")
        .def("add_state",
             [](MultivariateStateSpaceRegressionModel &model,
                SharedStateModel &state_model) {
               model.add_state(Ptr<SharedStateModel>(&state_model));
             },
             "Args:\n"
             "  state_model:  A SharedStateModel object defining an element of"
             " state.\n")
        .def("set_method",
             [](MultivariateStateSpaceRegressionModel &model,
                PosteriorSampler *sampler) {
               model.set_method(Ptr<PosteriorSampler>(sampler));
             })
        .def("set_regression_coefficients",
             [](MultivariateStateSpaceRegressionModel &model,
                const Matrix &coefficients) {
               if (coefficients.nrow() != model.nseries()) {
                 std::ostringstream err;
                 err << "The model describes " << model.nseries()
                     << " series but the input matrix has "
                     << coefficients.nrow() << " rows.";
                 report_error(err.str());
               }
               if (coefficients.ncol() != model.xdim()) {
                 std::ostringstream err;
                 err << "The model has predictor dimension "
                     << model.xdim() << " but the input matrix has "
                     << coefficients.ncol() << "columns.";
                 report_error(err.str());
               }
               for (int i = 0; i < coefficients.nrow(); ++i) {
                 model.observation_model()->model(i)->set_Beta(
                     coefficients.row(i));
               }
             },
             "Args:\n\n"
             "  coefficients:  A boom.Matrix with model.nseries rows and "
             "model.xdim columns.  Each row contains the regression "
             "coefficients for a specific series.\n")
        .def("set_regression_coefficients",
             [](MultivariateStateSpaceRegressionModel &model,
                const Vector &coefficients,
                int which_model) {
               model.observation_model()->model(which_model)->set_Beta(coefficients);
             },
             "Args:\n\n"
             "  coefficients:  The boom.Vector of regression coefficients for "
             "the regression model describing a single series.\n"
             "  which_model: The (integer) index of the model to update.\n")
        .def("set_residual_sd",
             [](MultivariateStateSpaceRegressionModel &model,
                const Vector &residual_sd) {
               if (residual_sd.size() != model.nseries()) {
                 std::ostringstream err;
                 err << "The model describes " << model.nseries()
                     << " series but the input vector has "
                     << residual_sd.size() << " entries.";
                 report_error(err.str());
               }
               for (int i = 0; i < model.nseries(); ++i) {
                 model.observation_model()->model(i)->set_sigsq(
                     square(residual_sd[i]));
               }
             },
             "Args:\n\n"
             "  residual_sd: A boom.Vector containing the residual standard "
             "deviation for each series.\n")
        .def("set_residual_sd",
             [](MultivariateStateSpaceRegressionModel &model,
                double residual_sd,
                int which_model) {
               model.observation_model()->model(which_model)->set_sigsq(
                   square(residual_sd));
             },
             "Args:\n\n"
             "  residual_sd:  The scalar valued residual standard deviation "
             "for a single model.\n"
             "  which_model: The (integer) index of the model to update.\n")
        .def("mle",
             [](MultivariateStateSpaceRegressionModel &model,
                double epsilon,
                int max_tries) {
               return model.mle(epsilon, max_tries);
             },
             py::arg("epsilon"),
             py::arg("max_tries") = 500,
             "Set model parameters to their maximum likelihood estimates.\n"
             "\n"
             "Args:\n"
             "  epsilon: A small positive number.  Absolute changes to log likelihood\n"
             "    less than this value indicate that the algorithm has converged.\n"
             "  max_tries:  Stop trying to optimzize after this many iterations.\n"
             "\n"
             "Returns:\n"
             "  The log likelihood value at the maximum.\n"
             "\n"
             "Effects:\n"
             "  Model parameters are set to the maximum likelihood estimates.\n")
        ;

    py::class_<MultivariateStateSpaceModelSampler,
               PosteriorSampler,
               Ptr<MultivariateStateSpaceModelSampler>>(
                   boom,
                   "MultivariateStateSpaceModelSampler")
        .def(py::init(
            [](MultivariateStateSpaceModelBase *model,
               RNG &seeding_rng = BOOM::GlobalRng::rng) {
              return new MultivariateStateSpaceModelSampler(model, seeding_rng);
            }))
        ;
  }
}  // namespace BayesBoom
