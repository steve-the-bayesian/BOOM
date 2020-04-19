#include <pybind11/pybind11.h>

#include "Models/StateSpace/DynamicRegression.hpp"
#include "Models/StateSpace/PosteriorSamplers/DynamicRegressionDirectGibbs.hpp"

#include "cpputil/Ptr.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;
  using namespace BOOM::StateSpace;

  namespace {
    using DRDGS = DynamicRegressionDirectGibbsSampler;
  }

  void DynamicRegressionModel_def(py::module &boom) {

    py::class_<RegressionDataTimePoint,
               Ptr<RegressionDataTimePoint>>(boom, "RegressionDataTimePoint")
        .def(py::init<int>(), py::arg("xdim") = -1,
             "Args\n"
             "  xdim: The dimension of the predictor variable.  The default "
             "value of -1 is a signal that the dimension is unknown.  "
             "It will be set on the first call to add_data().")
        .def(py::init(
            [](const Matrix &X, const Vector&y) {
              return new RegressionDataTimePoint(X, y);
            }),
             py::arg("X"),
             py::arg("y"),
             "Args:\n"
             "  X:  Predictor matrix.\n"
             "  y:  Response vector.\n")
        .def("add_data",
             [](RegressionDataTimePoint &point,
                const Ptr<RegressionData> &data_point) {
               point.add_data(data_point);
             },
             "  Add an observation to the time point.")
        .def_property_readonly(
            "xdim",
            &RegressionDataTimePoint::xdim,
            "The dimension of the predictor variable.")
        .def_property_readonly(
            "sample_size",
            &RegressionDataTimePoint::sample_size,
            "The number of regression observations at this time point")
        ;


    py::class_<DynamicRegressionModel,
               PriorPolicy,
               BOOM::Ptr<DynamicRegressionModel>>(boom, "DynamicRegressionModel")
        .def(py::init<int>(),
             py::arg("xdim"),
             "Args:\n"
             "  xdim: Number of potential predictor variables.")
        .def_property_readonly(
            "all_coefficients",
            [](const DynamicRegressionModel &model) {
              Matrix ans(model.xdim(), model.time_dimension());
              for (int t = 0; t < model.time_dimension(); ++t) {
                ans.col(t) = model.coef(t).Beta();
              }
              return ans;},
            "Matrix of dimension (xdim, time_dimension) containing the \n"
            "dynamic regression coefficients.")
        .def_property_readonly(
            "inclusion_indicators",
            [](const DynamicRegressionModel &model) {
              Matrix ans(model.xdim(), model.time_dimension());
              for (int t = 0; t < model.time_dimension(); ++t) {
                const Selector &inc = model.inclusion_indicators(t);
                for (int j = 0; j < inc.nvars(); ++j) {
                  ans(inc.included_positions()[j], t) = 1.0;
                }
              }
              return ans;
            },
            "The matrix of inclusion indicators.  Rows correspond to "
            "predictor\nvariables, columns to time.")
        .def("set_inclusion_indicators",
             [](DynamicRegressionModel &model,
                const Matrix &inc_matrix) {
               for (int t = 0; t < inc_matrix.ncol(); ++t) {
                 Selector inc(inc_matrix.nrow());
                 for (int j = 0; j < inc_matrix.nrow(); ++j) {
                   if (inc_matrix(j, t) > .5) {
                     inc.add(j);
                   } else {
                     inc.drop(j);
                   }
                 }
                 model.set_inclusion_indicators(t, inc);
               }
             },
             py::arg("inc"),
             "Set the coefficient inclusion indicators to the given values.\n\n"
             "Args:\n"
             "  inc: A boom.Matrix with 'xdim' rows and 'time_dimension' \n"
             "    columns.  The matrix contains 1's and 0's, with 1's \n"
             "    indicating that a coefficient should have a nonzero value.\n")
        .def("set_coefficients",
             [](DynamicRegressionModel &model, const Matrix &coefficients) {
               if (nrow(coefficients) != model.xdim() ||
                   ncol(coefficients) != model.time_dimension()) {
                 std::ostringstream msg;
                 msg << "Matrix of coefficients should have " << model.xdim()
                     << " rows and " << model.time_dimension() << " columns.\n";
                 report_error(msg.str());
               }
               for (int t = 0; t < model.time_dimension(); ++t) {
                 model.coef(t).set_Beta(coefficients.col(t));
               }
             },
             py::arg("coefficients"),
             "Set the time series of regression coefficients to the \n"
             "specified values.  If an excluded coefficient is passed\n"
             "a nonzero value, it will be set to zero.\n\n"
             "Args:\n"
             "  coefficients: A matrix with 'xdim' rows and 'time_dimension'\n"
             "    columns.")
        .def("draw_coefficients_given_inclusion",
             [](DynamicRegressionModel &model,
                RNG &rng) {
               model.draw_coefficients_given_inclusion(rng);
             },
             "Simulate the coefficients conditional inclusion indicators.")
        .def_property_readonly("xdim", &DynamicRegressionModel::xdim,
                               "Number of potential predictor variables.")
        .def_property_readonly(
            "time_dimension",
            &DynamicRegressionModel::time_dimension,
            "Number of observed time points in the training data.")
        .def("add_data",
             [](DynamicRegressionModel &model,
                const Ptr<RegressionDataTimePoint> &time_point) {
               model.add_data(time_point); },
             py::arg("time_point"),
             "Add the time point as the most recent time point informing the "
             "model. \n")
        // TODO(steve): Look here in the event of a crash.  coef is being
        // returned by reference in C++.
        .def("coef",
             [](DynamicRegressionModel &model, int t) {
               return model.coef(t);
             },
             py::arg("t"),
             "Args:\n"
             "  t: Time index.  A positive number less than time_dimension."
             "Returns:\n"
             "  The  regression coefficients at time t.")
        .def_property_readonly(
            "residual_sd",
            [](const DynamicRegressionModel &model) {
              return model.residual_sd();
            },
             "Residual standard deviation")
        .def("set_residual_sd",
             [](DynamicRegressionModel &model, double residual_sd) {
               model.set_residual_variance(residual_sd * residual_sd);
             },
             "Set the residual standard deviation to the specified value.")
        .def_property_readonly(
            "unscaled_innovation_sds",
            [](DynamicRegressionModel &model) {
              return sqrt(model.unscaled_innovation_variances());
            },
            "Vector of unscaled innovation standard deviations.  Multiply by \n"
            "residual SD to get the actual innovation standard deviations.")
        .def("set_unscaled_innovation_sds",
             [](DynamicRegressionModel &model,
                const Vector unscaled_innovation_sds) {
               if (unscaled_innovation_sds.size() != model.xdim()) {
                 report_error("Vector of unscaled innovation sd's must have "
                              "length xdim.");
               }
               for (int i = 0; i < model.xdim(); ++i) {
                 model.innovation_error_model(i)->set_sigsq(
                     square(unscaled_innovation_sds[i]));
               }
             },
             "Set the unscaled innovation standard deviations to the specified "
             "values.\n"
             "Args:\n"
             "  unscaled_innovation_sds:  Vector of length self.xdim \n"
             "    containing the unscaled innovation standard deviations.  \n"
             "    Multipying by the residual standard deviation gives the \n"
             "    actual innovation SD."
             )
        .def("transition_probabilities",
             [](DynamicRegressionModel &model, int pred) {
               return model.transition_model(pred)->Q();
             },
             "Transition probability matrix for the requested coefficient.\n\n"
             "Args:\n"
             "  pred:  The index of a predictor.  An integer from 0 to "
             "xdim - 1.\n\n"
             "Returns:\n"
             "  A Matrix containing the transition probabilities of the \n"
             "  Markov chain defining the inclusion indicators for the \n"
             "  requested predictor.\n")
        .def("set_transition_probabilities",
             [](DynamicRegressionModel &model,
                const Vector &p00,
                const Vector &p11) {
               if (p00.size() != model.xdim() || p11.size() != model.xdim()) {
                 report_error("Both vectors must be of size 'xdim'.");
               }
               for (int i = 0; i < model.xdim(); ++i) {
                 if (p00[i] < 0 || p00[i] > 1) {
                   report_error("All elements of p00 must be probabilities.");
                 }
                 if (p11[i] < 0 || p11[i] > 1) {
                   report_error("All elements of p11 must be probabilities.");
                 }
                 Matrix transition_probabilities(2, 2);
                 transition_probabilities(0, 0) = p00[i];
                 transition_probabilities(0, 1) = 1 - p00[i];
                 transition_probabilities(1, 0) = 1 - p11[i];
                 transition_probabilities(1, 1) = p11[i];
                 model.transition_model(i)->set_Q(transition_probabilities);
               }
             },
             "Set the transition probability matrices for the Markov chain \n"
             "models defininng the prior inclusion probability at each time \n"
             "point.\n\n"
             "Args\n"
             "  p00:  Vector of dimension xdim giving the self-transition \n"
             "    probability for state 0, for each coefficient.  This is the\n"
             "    probability of remaining excluded if currently excluded.\n"
             "  p11:  Vector of dimension xdim giving the self-transition \n"
             "    probability for state 1, for each coefficient.  This is the\n"
             "    probability of remaining included if currently included.\n")
        ;

    py::class_<DynamicRegressionDirectGibbsSampler,
               PosteriorSampler,
               Ptr<DynamicRegressionDirectGibbsSampler>>(
                   boom, "DynamicRegressionDirectGibbsSampler")
        .def(py::init(
            [] (Ptr<DynamicRegressionModel> model,
                double residual_sd_prior_guess,
                double residual_sd_prior_sample_size,
                const Vector &innovation_sd_prior_guess,
                const Vector &innovation_sd_prior_sample_size,
                const Vector &prior_inclusion_probabilities,
                const Vector &expected_inclusion_duration,
                const Vector &transition_probability_prior_sample_size,
                RNG &seeding_rng) {
              return new DRDGS(
                  model.get(),
                  residual_sd_prior_guess,
                  residual_sd_prior_sample_size,
                  innovation_sd_prior_guess,
                  innovation_sd_prior_sample_size,
                  prior_inclusion_probabilities,
                  expected_inclusion_duration,
                  transition_probability_prior_sample_size,
                  seeding_rng);
            }),
             py::arg("model"),
             py::arg("residual_sd_prior_sample_size"),
             py::arg("residual_sd_prior_guess"),
             py::arg("innovation_sd_prior_sample_size"),
             py::arg("innovation_sd_prior_guess"),
             py::arg("prior_inclusion_probabilities"),
             py::arg("expected_inclusion_duratione"),
             py::arg("transition_probability_prior_sample_size"),
             py::arg("seeding_rng") = BOOM::GlobalRng::rng,
             "Args:\n"
             "  model: The model to be sampled.\n"
             "  residual_sd_prior_guess: An a priori estimate of the residual "
             "standard deviation parameter.\n"
             "  residual_sd_prior_sample_size: Number of observations worth of "
             "weight to put on residual_sd_prior_guess.\n"
             "  innovation_sd_prior_guess:  A vector containing a priori "
             "estimates of the 'unscaled' standard deviations describing the "
             "period-to-period changes in the regression coefficients.  \n"
             "  innovation_sd_prior_sample_size: The number of observations "
             "worth of weight to place on residual_sd_prior_guess.\n"
             "  prior_inclusion_probabilities:  A Vector containing a priori "
             "estimates of the steady state inclusion probability for each "
             "coefficient.\n"

             "  expected_inclusion_duration: A vector of positive numbers "
             "giving the expected duration of an inclusion event.  If the "
             "prior inclusion probability for a coefficient is less than 0.5 "
             "then this can be any number larger than 1.  Otherwise, the "
             "expected duration is constrained.  TODO: describe the constraint."
             "\n"
             "  transition_probability_prior_sample_size:  A vector of positive"
             " numbers giving the number of observations worth of weight to "
             "place on the prior inclusion probabilities and expected "
             "durations.\n"
             "  seeding_rng:  The random number generator used to seed the "
             "RNG in this sampler.\n")
        .def("draw",
             &DRDGS::draw,
             "Perform one posterior sampling iteration on the managed model.\n")
        .def("draw_residual_variance",
             &DRDGS::draw_residual_variance,
             "Draw the residual variance parameter.")
        .def("draw_inclusion_indicators",
             &DRDGS::draw_inclusion_indicators,
             "Simulate the vector of inclusion indicators at each time point.")
        .def("draw_unscaled_state_innovation_variance",
             &DRDGS::draw_unscaled_state_innovation_variance,
             "Simulate the state innovation error variances, up to the residual "
             "variance proportionality constant")
        .def("draw_transition_probabilities",
             &DRDGS::draw_transition_probabilities,
             "Simulate the Markov chain transition probabilities.")
        ;
  }  // StateSpaceModel_def

}  // namespace BayesBoom
