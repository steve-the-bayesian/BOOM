#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Models/StateSpace/Multivariate/MultivariateStateSpaceRegressionModel.hpp"
#include "Models/StateSpace/Multivariate/StudentMvssRegressionModel.hpp"
#include "Models/StateSpace/Multivariate/PosteriorSamplers/MultivariateStateSpaceModelSampler.hpp"

#include "cpputil/math_utils.hpp"

#include "cpputil/Ptr.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;

  namespace {
    using CIMSSMB = ConditionallyIndependentMultivariateStateSpaceModelBase;
  }

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
        .def("show_warnings",
             [](MultivariateStateSpaceModelBase &model, bool show) {
               model.show_warnings(show);
             },
             py::arg("show"),
             "Set whether warnings about non-positive definite matrices "
             "encountered during Kalman filtering should be shown to the "
             "user.\n\n"
             "Args:  \n"
             "  show:  True if warnings should be shown to the user. False "
             "otherwise.")
        .def("state_contributions",
             [](const MultivariateStateSpaceModelBase &model,
                int which_state_model) {
               return model.state_contributions(which_state_model);
             },
             py::arg("which_state_model"),
             "Args:\n"
             "  which_state_model: The state model whose contribution is "
             "desired.\n\n"
             "Returns:\n"
             "  A Matrix.  Element (t, d) is the contrubtion of the specified "
             "state model to series d at time t.")
        .def_property_readonly(
            "shared_state",
            [](const MultivariateStateSpaceModelBase &model) {
              return model.shared_state();
            },
            "The full state matrix for the model, as drawn by the most recent "
            "MCMC iteration.")
        .def("set_shared_state",
             [](MultivariateStateSpaceModelBase &model,
                const Matrix &shared_state) {
               model.set_shared_state(shared_state);
             },
             py::arg("shared_state"),
             "Args:\n\n"
             "  shared_state: A boom.Matrix containing the state values.  Rows "
             "are different components of the state vector.  Columns are "
             "different time points.")
        .def_property_readonly(
            "smoothed_state_mean",
            [](MultivariateStateSpaceModelBase &model) {
              model.kalman_filter();
              model.kalman_smoother();
              return model.state_mean();
            },
            "The smoothed state mean as computed using the non-stochastic "
            "Kalman filter and smoother.")
        .def_property_readonly(
            "final_state_mean",
            [](const MultivariateStateSpaceModelBase &model) {
              int t = model.time_dimension() - 1;
              if (t < 0) {
                report_error("Time dimension was zero.");
              }
              return model.get_filter()[t].state_mean();
            },
            "The mean of the state vector for the final node in the Kalman "
            "filter.")
        .def_property_readonly(
            "final_state_variance",
            [](const MultivariateStateSpaceModelBase &model) {
              int t = model.time_dimension() - 1;
              if (t < 0) {
                report_error("Time dimension was zero.");
              }
              return model.get_filter()[t].state_variance();
            },
            "The variance of the state vector for the final node in the Kalman "
            "filter.")
        ;

    py::class_<ConditionallyIndependentMultivariateStateSpaceModelBase,
               MultivariateStateSpaceModelBase,
               BOOM::Ptr<CIMSSMB>>(
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
             "  series: The identifier of the time series (0.. number of "
             "series - 1) to\n"
             "    which this observation belongs.\n"
             "  timestamp: The time-index of the time series "
             "(0.. sample_size - 1)\n"
             "    containing this observation.\n")
        ;

    //==========================================================================
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
        .def("series",
             [](const MultivariateStateSpaceRegressionModel &model, int series) {
               int max_time = model.time_dimension();
               Vector values;
               Selector inclusion(max_time, false);
               for (int t = 0; t < max_time; ++t) {
                 if (model.observed_status(t)[series]) {
                   values.push_back(model.observed_data(series, t));
                   inclusion.add(t);
                 }
               }
               return std::make_pair(values, inclusion);
             },
             py::arg("series"),
             "Return one of the the observed time series in the training data."
             "\n"
             "Args:\n\n"
             "  series:  The integer index of the time series to be returned."
             "\n\n"
             "Returns:\n"
             "  A pair.  The first element is the Vector of observed values "
             "from the requested series.  The second is a Selector indicating "
             "which time periods (from 0.. time_dimension - 1) are held in "
             "the Vector.\n")
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
        .def_property_readonly(
            "regression_coefficients",
            [](const MultivariateStateSpaceRegressionModel &model) {
              const IndependentRegressionModels *reg(model.observation_model());
              Matrix ans(reg->ydim(), reg->xdim());
              for (int i = 0; i < reg->ydim(); ++i) {
                ans.row(i) = reg->model(i)->Beta();
              }
              return ans;
            },
            "The matrix of regression coefficients.  Each row corresponds "
            "to a different time series.  The columns are the regression "
            "coefficients for that time series.")
        .def_property_readonly(
            "residual_sd",
            [](const MultivariateStateSpaceRegressionModel &model) {
              const IndependentRegressionModels *reg(model.observation_model());
              Vector ans(reg->ydim());
              for (int i = 0; i < reg->ydim(); ++i) {
                ans[i] = reg->model(i)->sigma();
              }
              return ans;
            },
            "The Vector of reisidual standard deviation parameters.")
        .def_property_readonly(
            "observation_model",
            [](const MultivariateStateSpaceRegressionModel &model) {
              return model.observation_model();
            },
            "Returns a boom.IndependentRegressionModels object.")
        .def("observation_coefficients",
             [](const MultivariateStateSpaceRegressionModel &model, int t) {
               Selector all_series(model.nseries(), true);
               return model.observation_coefficients(t, all_series)->dense();
             },
             "The matrix of coefficients linking the observed time series to "
             "the shared state.  Each row corresponds to a time series.  Each "
             "column to an element in the state vector.")
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
               model.observation_model()->model(
                   which_model)->set_Beta(coefficients);
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
             "  epsilon: A small positive number.  Absolute changes to log "
             "likelihood\n"
             "    less than this value indicate that the algorithm has "
             "converged.\n"
             "  max_tries:  Stop trying to optimzize after this many "
             "iterations.\n"
             "\n"
             "Returns:\n"
             "  The log likelihood value at the maximum.\n"
             "\n"
             "Effects:\n"
             "  Model parameters are set to the maximum likelihood estimates."
             "\n")
        .def("update_state_distribution",
             [](MultivariateStateSpaceRegressionModel &model,
                int time,
                const Vector &response,
                const Matrix &predictors,
                const Selector &which_series,
                Vector &state_mean,
                SpdMatrix &state_variance) {

               if (response.size() != which_series.nvars()) {
                 report_error("The size of the response Vector does not match "
                              "the inclusion number of the 'which_series' "
                              "Selector.");
               }
               if (response.size() != predictors.nrow()) {
                 report_error("The number of rows in 'predictors' does not "
                              "match the inclusion number of the "
                              "'which_series' Selector.");

               }
               if (predictors.ncol() != model.xdim()) {
                 std::ostringstream err;
                 err << "The number of columns in 'predictors' ("
                     << predictors.ncol()
                     << ") does not match the predictor dimension of the "
                     << "model ("
                     << model.xdim() << ").";
                 report_error(err.str());
               }
               if (model.has_series_specific_state()) {
                 report_error("Updates are not implmented for models with "
                              "series specific state.");
               }

               // 1) subtract off the regression and any series-specific effects
               //    from 'data'
               Vector adjusted_observation(response.size());
               for (int i = 0; i < which_series.nvars(); ++i) {
                 int I = which_series.expanded_index(i);
                 const RegressionModel *reg(
                     model.observation_model()->model(I));
                 adjusted_observation[i] =
                     response[i] - reg->predict(predictors.row(i));

               }

               // 2) Convert state_mean and state_variance from contemporaneous
               //    moments to forward moments.
               Ptr<SparseKalmanMatrix> transition =
                   model.state_transition_matrix(time - 1);
               Vector forward_state_mean = *transition * state_mean;
               SpdMatrix forward_state_variance =
                   transition->sandwich(state_variance);
               forward_state_variance +=
                   model.state_variance_matrix(time - 1)->dense();

               // 3) Update
               using Filter = ConditionallyIndependentKalmanFilter;
               Filter &filter(model.get_filter());
               filter.ensure_size(time + 1);
               filter[time - 1].set_state_mean(forward_state_mean);
               filter[time].set_state_mean(forward_state_mean);
               filter[time - 1].set_state_variance(forward_state_variance);
               filter[time].set_state_variance(forward_state_variance);

               filter[time].update(adjusted_observation, which_series);

               // 4) Copy contemporaneous moments back into state_mean and
               //    state_variance.
               state_mean = filter[time].contemporaneous_state_mean();
               Ptr<SparseKalmanMatrix> forecast_precision = filter[
                   time].sparse_forecast_precision();
               state_variance = filter[time].contemporaneous_state_variance(
                   forecast_precision);


             },
             py::arg("time"),
             py::arg("response"),
             py::arg("predictors"),
             py::arg("which_series"),
             py::arg("state_mean"),
             py::arg("state_variance"),
             "Perform one Kalman filtering step to compute the contemporaneous"
             " mean and variance of the model state given new data.\n\n"
             "Args:\n"
             "  time:  The integer-valued timestamp of the new data point.\n"
             "  response:  A Vector of observed data values at the new time "
             "point.\n"
             "  predictors:  A boom.Matrix containing the predictor values "
             "for the new data.  The number of rows must match the size of "
             "'response'.\n"
             "  which_series:  A Selector indicating which of the multivariate "
             "time series are observed in 'data'.\n"
             "  state_mean:  On input this is the contemporaneous state mean "
             "of the Kalman filter node at time 'time'-1 (i.e. the time point "
             "before the new data was observed).  On output it his the "
             "contemporaneous state mean of the time point corresponding to "
             "the new data.\n"
             "  state_variance:  On input this is the contemporaneous state "
             "variance of the Kalman filter node at time 'time'-1 (i.e. the "
             "time point before the new data was observed).  On output it "
             "is the contemporaneous state mean of the time point "
             "corresponding to the new data.\n")
        .def("__repr__",
            [](const MultivariateStateSpaceRegressionModel &model) {
              std::ostringstream out;
              out << "A boom.MultivariateStateSpaceRegressionModel with "
                  << model.number_of_state_models()
                  << " state component models, response dimension "
                  << model.nseries() << ", input dimension "
                  << model.xdim() << ", state dimension "
                  << model.state_dimension() << ", with "
                  << model.time_dimension() << " time points.";
              return out.str();
            })
        ;

    py::class_<StudentMvssRegressionModel,
               ConditionallyIndependentMultivariateStateSpaceModelBase,
               PriorPolicy,
               BOOM::Ptr<StudentMvssRegressionModel>>(
                   boom,
                   "StudentMvssRegressionModel",
                   py::multiple_inheritance())
        .def(py::init(
            [](int xdim, int nseries) {
              return new StudentMvssRegressionModel(
                  xdim, nseries);
            }),
             py::arg("xdim"),
             py::arg("nseries"),
             "Args:\n"
             "  xdim:  The dimension of the predictor variables.\n"
             "  nseries: The number of time series being modeled.\n")
        .def_property_readonly(
            "xdim",
            [](const StudentMvssRegressionModel &model) {
              return model.xdim();
            },
            "Dimension of the vector of predictor variables.")
        .def("add_data",
             [](StudentMvssRegressionModel &model,
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
                 NEW(StudentMultivariateTimeSeriesRegressionData, data_point)(
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
        .def("observed_data",
             [](const StudentMvssRegressionModel &model,
                int series, int time) {
               return model.observed_data(series, time);
             },
             py::arg("series"),
             py::arg("time"),
             "Args:\n\n"
             "  series:  The model series for which a data point is desired.\n"
             "  time:  The time point at which the data point is desired.\n"
             "\n"
             "Returns:\n"
             "  The time series value for the requested series at the requested "
             "time.\n")
        .def("series",
             [](const StudentMvssRegressionModel &model, int series) {
               int max_time = model.time_dimension();
               Vector values;
               Selector inclusion(max_time, false);
               for (int t = 0; t < max_time; ++t) {
                 if (model.observed_status(t)[series]) {
                   values.push_back(model.observed_data(series, t));
                   inclusion.add(t);
                 }
               }
               return std::make_pair(values, inclusion);
             },
             py::arg("series"),
             "Return one of the the observed time series in the training data."
             "\n"
             "Args:\n\n"
             "  series:  The integer index of the time series to be returned."
             "\n\n"
             "Returns:\n"
             "  A pair.  The first element is the Vector of observed values "
             "from the requested series.  The second is a Selector indicating "
             "which time periods (from 0.. time_dimension - 1) are held in "
             "the Vector.\n")
        .def("add_state",
             [](StudentMvssRegressionModel &model,
                SharedStateModel &state_model) {
               model.add_state(Ptr<SharedStateModel>(&state_model));
             },
             "Args:\n"
             "  state_model:  A SharedStateModel object defining an element of"
             " state.\n")
        // .def("set_method",
        //      [](StudentMvssRegressionModel &model,
        //         PosteriorSampler *sampler) {
        //        model.set_method(Ptr<PosteriorSampler>(sampler));
        //      })
        .def_property_readonly(
            "regression_coefficients",
            [](const StudentMvssRegressionModel &model) {
              const StudentMvssRegressionModel::ObservationModel
                  *reg(model.observation_model());
              Matrix ans(reg->ydim(), reg->xdim());
              for (int i = 0; i < reg->ydim(); ++i) {
                ans.row(i) = reg->model(i)->Beta();
              }
              return ans;
            },
            "The matrix of regression coefficients.  Each row corresponds "
            "to a different time series.  The columns are the regression "
            "coefficients for that time series.")
        .def_property_readonly(
            "residual_sd",
            [](const StudentMvssRegressionModel &model) {
              const StudentMvssRegressionModel::ObservationModel
                  *reg(model.observation_model());
              Vector ans(reg->ydim());
              for (int i = 0; i < reg->ydim(); ++i) {
                ans[i] = reg->model(i)->sigma();
              }
              return ans;
            },
            "The Vector of reisidual standard deviation parameters.")
        .def_property_readonly(
            "tail_thickness",
            [](const StudentMvssRegressionModel &model) {
              Vector ans(model.nseries());
              for (int i = 0; i < ans.size(); ++i) {
                ans[i] = model.observation_model()->model(i)->nu();
              }
              return ans;
            },
            "The Vector of tail thickness parameter values (nu).")
        .def_property_readonly(
            "observation_model",
            [](const StudentMvssRegressionModel &model) {
              return model.observation_model();
            },
            "Returns a ******* object.")
        .def("observation_coefficients",
             [](const StudentMvssRegressionModel &model, int t) {
               Selector all_series(model.nseries(), true);
               return model.observation_coefficients(t, all_series)->dense();
             },
             "The matrix of coefficients linking the observed time series to "
             "the shared state.  Each row corresponds to a time series.  Each "
             "column to an element in the state vector.")
        .def("set_regression_coefficients",
             [](StudentMvssRegressionModel &model,
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
             [](StudentMvssRegressionModel &model,
                const Vector &coefficients,
                int which_model) {
               model.observation_model()->model(
                   which_model)->set_Beta(coefficients);
             },
             "Args:\n\n"
             "  coefficients:  The boom.Vector of regression coefficients for "
             "the regression model describing a single series.\n"
             "  which_model: The (integer) index of the model to update.\n")
        .def("set_residual_sd",
             [](StudentMvssRegressionModel &model,
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
             [](StudentMvssRegressionModel &model,
                double residual_sd,
                int which_model) {
               model.observation_model()->model(which_model)->set_sigsq(
                   square(residual_sd));
             },
             "Args:\n\n"
             "  residual_sd:  The scalar valued residual standard deviation "
             "for a single model.\n"
             "  which_model: The (integer) index of the model to update.\n")
        .def("set_tail_thickness",
             [](StudentMvssRegressionModel &model,
                const Vector &tail_thickness_parameters) {
               if (tail_thickness_parameters.size() != model.nseries()) {
                 std::ostringstream err;
                 err << "The model describes " << model.nseries()
                     << " series but the input vector has "
                     << tail_thickness_parameters.size() << " entries.";
                 report_error(err.str());
               }
               for (int i = 0; i < tail_thickness_parameters.size(); ++i) {
                 model.observation_model()->model(i)->set_nu(
                     tail_thickness_parameters[i]);
               }
             },
             "Args:\n\n"
             "  tail_thickness_parameters:  A boom.Vector containing the tail "
             "thickness parameters (nu).  There is one element for each series"
             " in the model."
             )
        .def("update_state_distribution",
             [](StudentMvssRegressionModel &model,
                int time,
                const Vector &response,
                const Matrix &predictors,
                const Selector &which_series,
                Vector &state_mean,
                SpdMatrix &state_variance) {

               if (response.size() != which_series.nvars()) {
                 report_error("The size of the response Vector does not match "
                              "the inclusion number of the 'which_series' "
                              "Selector.");
               }
               if (response.size() != predictors.nrow()) {
                 report_error("The number of rows in 'predictors' does not "
                              "match the inclusion number of the "
                              "'which_series' Selector.");

               }
               if (predictors.ncol() != model.xdim()) {
                 std::ostringstream err;
                 err << "The number of columns in 'predictors' ("
                     << predictors.ncol()
                     << ") does not match the predictor dimension of the "
                     << "model ("
                     << model.xdim() << ").";
                 report_error(err.str());
               }
               if (model.has_series_specific_state()) {
                 report_error("Updates are not implmented for models with "
                              "series specific state.");
               }

               // 1) subtract off the regression and any series-specific effects
               //    from 'data'
               Vector adjusted_observation(response.size());
               for (int i = 0; i < which_series.nvars(); ++i) {
                 int I = which_series.expanded_index(i);
                 const CompleteDataStudentRegressionModel *reg(
                     model.observation_model()->model(I));
                 adjusted_observation[i] =
                     response[i] - reg->predict(predictors.row(i));

               }

               // 2) Convert state_mean and state_variance from contemporaneous
               //    moments to forward moments.
               Ptr<SparseKalmanMatrix> transition =
                   model.state_transition_matrix(time - 1);
               Vector forward_state_mean = *transition * state_mean;
               SpdMatrix forward_state_variance =
                   transition->sandwich(state_variance);
               forward_state_variance +=
                   model.state_variance_matrix(time - 1)->dense();

               // 3) Update
               using Filter = ConditionallyIndependentKalmanFilter;
               Filter &filter(model.get_filter());
               filter.ensure_size(time + 1);
               filter[time - 1].set_state_mean(forward_state_mean);
               filter[time].set_state_mean(forward_state_mean);
               filter[time - 1].set_state_variance(forward_state_variance);
               filter[time].set_state_variance(forward_state_variance);

               filter[time].update(adjusted_observation, which_series);

               // 4) Copy contemporaneous moments back into state_mean and
               //    state_variance.
               state_mean = filter[time].contemporaneous_state_mean();
               Ptr<SparseKalmanMatrix> forecast_precision = filter[
                   time].sparse_forecast_precision();
               state_variance = filter[time].contemporaneous_state_variance(
                   forecast_precision);


             },
             py::arg("time"),
             py::arg("response"),
             py::arg("predictors"),
             py::arg("which_series"),
             py::arg("state_mean"),
             py::arg("state_variance"),
             "Perform one Kalman filtering step to compute the contemporaneous"
             " mean and variance of the model state given new data.\n\n"
             "Args:\n"
             "  time:  The integer-valued timestamp of the new data point.\n"
             "  response:  A Vector of observed data values at the new time "
             "point.\n"
             "  predictors:  A boom.Matrix containing the predictor values "
             "for the new data.  The number of rows must match the size of "
             "'response'.\n"
             "  which_series:  A Selector indicating which of the multivariate "
             "time series are observed in 'data'.\n"
             "  state_mean:  On input this is the contemporaneous state mean "
             "of the Kalman filter node at time 'time'-1 (i.e. the time point "
             "before the new data was observed).  On output it his the "
             "contemporaneous state mean of the time point corresponding to "
             "the new data.\n"
             "  state_variance:  On input this is the contemporaneous state "
             "variance of the Kalman filter node at time 'time'-1 (i.e. the "
             "time point before the new data was observed).  On output it "
             "is the contemporaneous state mean of the time point "
             "corresponding to the new data.\n")
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
