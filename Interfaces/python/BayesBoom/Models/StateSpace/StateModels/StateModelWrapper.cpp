#include <pybind11/pybind11.h>

#include "Models/StateSpace/StateModels/LocalLevelStateModel.hpp"
#include "Models/StateSpace/StateModels/LocalLinearTrend.hpp"
#include "Models/StateSpace/StateModels/SemilocalLinearTrend.hpp"
#include "Models/StateSpace/StateModels/SeasonalStateModel.hpp"

#include "Models/PosteriorSamplers/ZeroMeanGaussianConjSampler.hpp"
#include "Models/PosteriorSamplers/ZeroMeanMvnIndependenceSampler.hpp"

#include "cpputil/Ptr.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;

  void StateModel_def(py::module &boom) {

    py::class_<StateModel,
               BOOM::Ptr<StateModel>>(
                   boom, "StateModel", py::multiple_inheritance())
        ;

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
             [] (Ptr<LocalLevelStateModel> model, double mean) {
               model->set_initial_state_mean(mean);
             },
             py::arg("mean"),
             "Set the mean of the initial state distribution to the "
             "specified value.")
        .def("set_initial_state_variance",
             [] (Ptr<LocalLevelStateModel> model, double variance) {
               model->set_initial_state_variance(variance);
             },
             py::arg("variance"),
             "Set the variance of the initial state distribution to the "
             "specified value.")
        .def("set_posterior_sampler",
             [] (LocalLevelStateModel &model,
                 const Ptr<GammaModelBase> &prior,
                 RNG &seeding_rng) {
               NEW(ZeroMeanGaussianConjSampler, sampler)(
                   &model, prior, seeding_rng);
               model.set_method(sampler);
               return sampler; },
             py::arg("prior"),
             py::arg("rng") = BOOM::GlobalRng::rng,
             "Args:\n"
             "  prior:  Prior distribution on the innovation precision.\n\n"
             "Returns:\n"
             "  The posterior sampler, which has already been assigned to \n"
             "  the model.  Assigning it again will cause duplicate MCMC moves."
             )
        ;

    py::class_<LocalLinearTrendStateModel,
               StateModel,
               BOOM::Ptr<LocalLinearTrendStateModel>>(
                   boom,
                   "LocalLinearTrendStateModel")
        .def(py::init<>())
        .def_property_readonly(
            "state_dimension",
            &LocalLinearTrendStateModel::state_dimension)
        .def_property_readonly(
            "state_error_dimension",
            &LocalLinearTrendStateModel::state_error_dimension)
        .def_property_readonly(
            "sigma_level",
            [] (const LocalLinearTrendStateModel &model) {
              return sqrt(model.Sigma()(0, 0));
            },
            "Innovation standard deviation for the level component.")
        .def_property_readonly(
            "sigma_slope",
            [] (const LocalLinearTrendStateModel &model) {
              return sqrt(model.Sigma()(1, 1));
            },
            "Innovation standard deviation for the slope component.")
        .def("set_initial_state_mean",
             &LocalLinearTrendStateModel::set_initial_state_mean,
             py::arg("mean"),
             "Args:\n"
             "  mean:  A Vector of length 2 giving the prior mean of the \n"
             "    state at time 0.")
        .def("set_initial_state_variance",
             &LocalLinearTrendStateModel::set_initial_state_variance,
             py::arg("variance"),
             "Args:\n"
             "  variance:  SpdMatrix of dimension 2 giving the prior \n"
             "    variance of the state at time 0.")
        .def("set_posterior_sampler",
             [](LocalLinearTrendStateModel &state_model,
                GammaModelBase &level_sigma_prior,
                double level_sigma_upper_limit,
                GammaModelBase &slope_sigma_prior,
                double slope_sigma_upper_limit,
                BOOM::RNG &seeding_rng) {

               NEW(ZeroMeanMvnIndependenceSampler, sigma_level_sampler)(
                   &state_model,
                   Ptr<GammaModelBase>(&level_sigma_prior),
                   0,
                   seeding_rng);
               sigma_level_sampler->set_sigma_upper_limit(
                   level_sigma_upper_limit);
               state_model.set_method(sigma_level_sampler);

               NEW(ZeroMeanMvnIndependenceSampler, sigma_slope_sampler)(
                   &state_model,
                   Ptr<GammaModelBase>(&slope_sigma_prior),
                   1,
                   seeding_rng);
               sigma_slope_sampler->set_sigma_upper_limit(
                   slope_sigma_upper_limit);
               state_model.set_method(sigma_slope_sampler);
             })
        ;

    py::class_<SeasonalStateModel,
               StateModel,
               ZeroMeanGaussianModel,
               BOOM::Ptr<SeasonalStateModel>>(boom, "SeasonalStateModel")
        .def(py::init<int, int>(),
             py::arg("nseasons"),
             py::arg("season_duration") = 1,
             "Args:\n"
             "  nseasons: Number of seasons in the the model.\n"
             "  season_duration: Number of time periods each season lasts.\n")
        .def_property_readonly("nseasons", &SeasonalStateModel::nseasons)
        .def_property_readonly("season_duration",
                               &SeasonalStateModel::season_duration)
        .def_property_readonly(
            "state_dimension",
            &SeasonalStateModel::state_dimension,
            "Dimension of the state vector.")
        .def_property_readonly(
            "state_error_dimension",
            &SeasonalStateModel::state_error_dimension,
            "Dimension of the error term for this state component.")
        .def("set_initial_state_mean",
             &SeasonalStateModel::set_initial_state_mean,
             py::arg("mu"),
             "Args: \n"
             "  mu: Vector of size 'nseasons' - 1 giving the mean of the state "
             "at time 0.\n")
        .def("set_initial_state_variance",
             [] (SeasonalStateModel &seasonal,
                 const BOOM::SpdMatrix &variance) {
               seasonal.set_initial_state_variance(variance);
             },
             py::arg("variance"),
             "Args: \n"
             "  variance: SpdMatrix of size 'nseasons' - 1 giving the variance"
             " of the state at time 0.\n")
        .def("set_initial_state_variance",
             [] (SeasonalStateModel &seasonal, double variance) {
               seasonal.set_initial_state_variance(variance);
             },
             py::arg("variance"),
             "Args: \n"
             "  variance: The variance matrix is this constant times the "
             "identity.\n")
             ;

    py::class_<SemilocalLinearTrendStateModel,
               StateModel,
               PriorPolicy,
               BOOM::Ptr<SemilocalLinearTrendStateModel>>(
                   boom, "SemilocalLinearTrendStateModel")
        .def(py::init(
            [](const Ptr<ZeroMeanGaussianModel> &level,
               const Ptr<NonzeroMeanAr1Model> &slope) {
              return new SemilocalLinearTrendStateModel(
                  level, slope);
            }))
        .def_property_readonly(
            "state_dimension",
            &SemilocalLinearTrendStateModel::state_dimension,
            "Dimension of the state vector.")
        .def_property_readonly(
            "state_error_dimension",
            &SemilocalLinearTrendStateModel::state_error_dimension,
            "Dimension of the error term for this state component.")
        .def("set_initial_level_mean",
             &SemilocalLinearTrendStateModel::set_initial_level_mean,
             "Set the prior mean of the level component at time 0.")
        .def("set_initial_level_sd",
             &SemilocalLinearTrendStateModel::set_initial_level_sd,
             "Set the prior standard deviation of the level component "
             "at time 0.")
        .def("set_initial_slope_mean",
             &SemilocalLinearTrendStateModel::set_initial_slope_mean,
             "Set the prior mean of the slope component at time 0.")
        .def("set_initial_slope_sd",
             &SemilocalLinearTrendStateModel::set_initial_slope_sd,
             "Set the prior standard deviation of the slope component at "
             "time 0.")
        .def_property_readonly(
            "level_sd", &SemilocalLinearTrendStateModel::level_sd,
            "Innovation standard deviation for the level component.")
        .def_property_readonly(
            "slope_sd", &SemilocalLinearTrendStateModel::slope_sd,
            "Innovation standard deviation for the slope component.")
        .def_property_readonly(
            "slope_ar_coefficient",
            &SemilocalLinearTrendStateModel::slope_ar_coefficient,
            "AR1 coefficient for the slope component.")
        .def_property_readonly(
            "slope_mean",
            &SemilocalLinearTrendStateModel::slope_mean,
            "Long term mean for the slope component.")
        ;


  }  // StateSpaceModel_def

}  // namespace BayesBoom
