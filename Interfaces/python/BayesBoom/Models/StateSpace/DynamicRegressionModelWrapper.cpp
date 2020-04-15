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
             "Args\n\n"
             "  xdim: The dimension of the predictor variable.  The default "
             "value of -1 is a signal that the dimension is unknown.  "
             "It will be set on the first call to add_data().")
        .def("add_data", [](RegressionDataTimePoint &point,
                          const Ptr<RegressionData> &data_point) {
               point.add_data(data_point); },
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
        .def("add_data",
             [](DynamicRegressionModel &model,
                const Ptr<RegressionDataTimePoint> &time_point) {
               model.add_data(time_point); },
             py::arg("time_point"),
             "Add the time point as the most recent time point informing the "
             "model. \n")
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
             "Args:\n\n"
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
        .def("draw", &DRDGS::draw,
             "Perform one posterior sampling iteration on the managed model.\n")
        .def("draw_residual_variance", &DRDGS::draw_residual_variance)
        .def("draw_inclusion_indicators", &DRDGS::draw_inclusion_indicators)
        .def("draw_transition_probabilities",
             &DRDGS::draw_transition_probabilities)
        .def("draw_residual_variance", &DRDGS::draw_residual_variance)
        ;
  }  // StateSpaceModel_def

}  // namespace BayesBoom
