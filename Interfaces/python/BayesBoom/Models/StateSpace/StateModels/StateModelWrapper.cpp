#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Models/StateSpace/StateModels/ArStateModel.hpp"
#include "Models/StateSpace/StateModels/DynamicRegressionStateModel.hpp"
#include "Models/StateSpace/StateModels/LocalLevelStateModel.hpp"
#include "Models/StateSpace/StateModels/LocalLinearTrend.hpp"
#include "Models/StateSpace/StateModels/RegressionHolidayStateModel.hpp"
#include "Models/StateSpace/StateModels/SemilocalLinearTrend.hpp"
#include "Models/StateSpace/StateModels/SeasonalStateModel.hpp"
#include "Models/StateSpace/StateModels/GeneralSeasonalStateModel.hpp"
#include "Models/StateSpace/StateModels/StudentLocalLinearTrend.hpp"
#include "Models/StateSpace/StateModels/TrigStateModel.hpp"

#include "Models/PosteriorSamplers/ZeroMeanGaussianConjSampler.hpp"
#include "Models/PosteriorSamplers/ZeroMeanMvnIndependenceSampler.hpp"
#include "Models/StateSpace/PosteriorSamplers/DynamicRegressionPosteriorSampler.hpp"
#include "Models/StateSpace/PosteriorSamplers/StudentLocalLinearTrendPosteriorSampler.hpp"
#include "Models/StateSpace/StateModels/PosteriorSamplers/GeneralSeasonalLLTPosteriorSampler.hpp"

#include "Models/TimeSeries/ArModel.hpp"

#include "Models/StateSpace/StateModels/Holiday.hpp"

#include "cpputil/Date.hpp"
#include "cpputil/Ptr.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;

  void StateModel_def(py::module &boom) {

    py::class_<StateModelBase,
               PosteriorModeModel,
               BOOM::Ptr<StateModelBase>>(
                   boom, "StateModelBase", py::multiple_inheritance())
        .def_property_readonly(
            "state_dimension",
            [](const StateModelBase &state_model) {
              return state_model.state_dimension();
            },
            "The number of dimensions that this state model adds to "
            "the state vector.")
        .def_property_readonly(
            "state_error_dimension",
            [](const StateModelBase &state_model) {
              return state_model.state_error_dimension();
            },
            "The dimension of the error component of this state model.  "
            "This may be smaller than 'state_dimension'.")
        ;

    py::class_<StateModel,
               StateModelBase,
               BOOM::Ptr<StateModel>>(
                   boom, "StateModel", py::multiple_inheritance())
        .def("observe_time_dimension", [](StateModel &model, int t) {
          model.observe_time_dimension(t);
        },
          py::arg("time_dimension"),
          "Args:\n\n"
          "  time_dimension: Make state models that manage their own memory "
          "aware that there are 'time_dimension' time periods.\n")
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
        .def("set_sigma",
             [](LocalLevelStateModel &model, double sigma) {
               model.set_sigsq(sigma * sigma);
             })
        .def("set_sigsq",
             [](LocalLevelStateModel &model, double sigsq) {
               model.set_sigsq(sigsq);
             })
        .def_property_readonly(
            "state_error_dimension",
            &LocalLevelStateModel::state_error_dimension,
            "Dimension of the innovation term.")
        .def_property_readonly(
            "initial_state_mean",
            [](const LocalLevelStateModel &model) {
              return model.initial_state_mean()[0];
            })
        .def_property_readonly(
            "initial_state_variance",
            [](const LocalLevelStateModel &model) {
              return model.initial_state_variance()(0, 0);
            })
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
                 GammaModelBase *prior,
                 RNG &seeding_rng) {
               NEW(ZeroMeanGaussianConjSampler, sampler)(
                   &model, Ptr<GammaModelBase>(prior), seeding_rng);
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
        .def("initial_state_mean",
             [](const LocalLinearTrendStateModel &model) {
               return model.initial_state_mean();
             })
        .def("initial_state_variance",
             [](const LocalLinearTrendStateModel &model) {
               return model.initial_state_variance();
             })
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
        .def("set_sigma_level",
             [](LocalLinearTrendStateModel &state_model, double sigma) {
               SpdMatrix variance(state_model.Sigma());
               variance(0, 0) = sigma * sigma;
               state_model.set_Sigma(variance);
             })
        .def("set_sigma_slope",
             [](LocalLinearTrendStateModel &state_model, double sigma) {
               SpdMatrix variance(state_model.Sigma());
               variance(1, 1) = sigma * sigma;
               state_model.set_Sigma(variance);
             })
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
        .def_property_readonly(
            "initial_state_mean", [](SeasonalStateModel &model) {
              return model.initial_state_mean();
            })
        .def_property_readonly(
            "initial_state_variance", [](SeasonalStateModel &model) {
              return model.initial_state_variance();
            })
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
        .def_property_readonly(
            "initial_state_mean",
            &SemilocalLinearTrendStateModel::initial_state_mean)
        .def_property_readonly(
            "initial_state_variance",
            &SemilocalLinearTrendStateModel::initial_state_variance)
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
        .def("set_level_sd",
             [](SemilocalLinearTrendStateModel &model, double level_sd) {
               model.set_level_sd(level_sd);
             })
        .def("set_level_sd",
             [](SemilocalLinearTrendStateModel &model, double level_sd) {
               model.set_level_sd(level_sd);})
        .def("set_slope_sd",
             [](SemilocalLinearTrendStateModel &model, double sd) {
               model.set_slope_sd(sd);})
        .def("set_slope_mean",
             [](SemilocalLinearTrendStateModel &model, double slope) {
               model.set_slope_mean(slope);})
        .def("set_slope_ar_coefficient",
             [](SemilocalLinearTrendStateModel &model, double ar) {
               model.set_slope_ar_coefficient(ar);})
        ;

    py::class_<DynamicRegressionStateModel,
               StateModel,
               PriorPolicy,
               BOOM::Ptr<DynamicRegressionStateModel>>(
                   boom, "DynamicRegressionStateModel")
        .def(py::init(
            [](Matrix &predictors) {
              return new DynamicRegressionStateModel(predictors);
            }))
        .def_property_readonly(
            "sigma",
            [](DynamicRegressionStateModel *model) {
              Vector ans(model->xdim());
              for (int i = 0; i < model->xdim(); ++i) {
                ans[i] = sqrt(model->sigsq(i));
              }
              return ans;
            },
            "Standard deviation of the innovation term for each coefficient.")
        .def("set_sigma",
             [](DynamicRegressionStateModel *model,
                const Vector &sigma) {
               for (int i = 0; i < sigma.size(); ++i) {
                 model->set_sigsq(sigma[i] * sigma[i], i);
               }
             },
             "Set the standard deviations of the innovation terms for each "
             "coefficient.")
        .def("set_initial_state_mean",
             [] (DynamicRegressionStateModel *model,
                 const Vector &mean) {
               model->set_initial_state_mean(mean);
             })
        .def("set_initial_state_variance",
             [] (DynamicRegressionStateModel *model,
                 const SpdMatrix &variance) {
               model->set_initial_state_variance(variance);
             })
        ;

    py::class_<StudentLocalLinearTrendStateModel,
               StateModel,
               PriorPolicy,
               Ptr<StudentLocalLinearTrendStateModel>>(
                   boom, "StudentLocalLinearTrendStateModel")
        .def(py::init(
            [] () {
              return new StudentLocalLinearTrendStateModel;
            }))
        .def_property_readonly(
            "sigma_level",
            [] (StudentLocalLinearTrendStateModel *model) {
              return model->sigma_level();
            })
        .def("set_sigma_level",
             [] (StudentLocalLinearTrendStateModel *model, double sigma_level) {
               model->set_sigma_level(sigma_level);
             })
        .def_property_readonly(
            "nu_level",
            [] (StudentLocalLinearTrendStateModel *model) {
              return model->nu_level();
            })
        .def(
            "set_nu_level",
            [] (StudentLocalLinearTrendStateModel *model, double nu) {
              return model->set_nu_level(nu);
            })
        .def_property_readonly(
            "sigma_slope",
            [] (StudentLocalLinearTrendStateModel *model) {
              return model->sigma_slope();
            })
        .def("set_sigma_slope",
             [] (StudentLocalLinearTrendStateModel *model, double sigma_slope) {
               model->set_sigma_slope(sigma_slope);
             })
        .def_property_readonly(
            "nu_slope",
            [] (StudentLocalLinearTrendStateModel *model) {
              return model->nu_slope();
            })
        .def(
            "set_nu_slope",
            [] (StudentLocalLinearTrendStateModel *model, double nu) {
              return model->set_nu_slope(nu);
            })
        .def("set_initial_state_mean",
             [] (StudentLocalLinearTrendStateModel *model,
                 const Vector &initial_state_mean) {
               model->set_initial_state_mean(initial_state_mean);
             })
        .def("set_initial_state_variance",
             [] (StudentLocalLinearTrendStateModel *model,
                 const SpdMatrix &initial_state_variance) {
               model->set_initial_state_variance(initial_state_variance);
             })
        ;

    py::class_<TrigStateModel,
               StateModel,
               PriorPolicy,
               Ptr<TrigStateModel>>(boom, "TrigStateModel")
        .def(py::init(
            [] (double period, const Vector &frequencies) {
              return new TrigStateModel(period, frequencies);
            }),
             py::arg("period"),
             py::arg("frequencies"),
             "Args:\n\n"
             "  period: The (float) number of time steps in a full cycle.\n"
             "  frequencies: A vector of positive real numbers, giving the "
             "number of times a cycle repeats in a period.  One sine and "
             "one cosine term will be added to the state for each frequency.\n")
        .def_property_readonly(
            "error_distribution",
            [] (TrigStateModel *model) {
              return model->error_distribution();
            })
        .def("compute_state_contribution",
             [] (TrigStateModel *model, const Matrix &state) {
               int time_dimension = state.ncol();
               Vector ans(time_dimension);
               for (int t = 0; t < time_dimension; ++t) {
                 ans[t] = model->observation_matrix(t).dot(state.col(t));
               }
               return ans;
             })
        .def("set_initial_state_mean",
             [] (TrigStateModel *model, const Vector &initial_state_mean) {
               model->set_initial_state_mean(initial_state_mean);
             })
        .def("set_initial_state_variance",
             [] (TrigStateModel *model,
                 const SpdMatrix &initial_state_variance) {
               model->set_initial_state_variance(initial_state_variance);
             })
        ;

    py::class_<ArStateModel,
               StateModel,
               ArModel,
               PriorPolicy,
               Ptr<ArStateModel>>(boom, "ArStateModel")
        .def(py::init(
            [] (int lags) {
              return new ArStateModel(lags);
            }),
             py::arg("lags") = 1,
             "Args:\n\n"
             "  lags:  The number of lags in the AR process.\n")
        .def_property_readonly("sigma", [] (ArStateModel *model) {
          return model->sigma();
        })
        .def("set_sigma",
             [] (ArStateModel *model, double sigma) {
               model->set_sigma(sigma);
             })
        .def_property_readonly(
            "phi",
            [](ArStateModel *model) {
              return model->phi();
            },
            "The AR coefficients.")
        .def_property_readonly(
            "ar_coefficients",
            [](ArStateModel *model) {
              return model->phi();
            },
            "The AR coefficients.")
        .def("set_phi",
             [](ArStateModel *model, const Vector &phi) {
               model->set_phi(phi);
             })
        .def("set_initial_state_mean",
             [] (ArStateModel *model, const Vector &mean) {
               model->set_initial_state_mean(mean);
             })
        .def("set_initial_state_variance",
             [] (ArStateModel *model, const SpdMatrix &variance) {
               model->set_initial_state_variance(variance);
             })
        ;

    py::class_<RegressionHolidayStateModel,
               StateModel,
               Ptr<RegressionHolidayStateModel>>(
                   boom, "RegressionHolidayStateModel")
        .def(py::init(
            [] (Date &time0,
                ScalarStateSpaceModelBase *model,
                GaussianModel *prior,
                RNG &seeding_rng) {
              return new ScalarRegressionHolidayStateModel(
                  time0, model, prior, seeding_rng);
            }),
             py::arg("time0"),
             py::arg("parent_model"),
             py::arg("prior"),
             py::arg("seeding_rng") = GlobalRng::rng,
             "")
        .def("add_holiday",
             [] (RegressionHolidayStateModel *model,
                 Holiday *holiday) {
               model->add_holiday(holiday);
             })
        .def_property_readonly(
            "holiday_pattern",
            [] (const RegressionHolidayStateModel *model,
                int holiday_index) {
              return model->holiday_pattern(holiday_index);
            })
        .def("set_holiday_pattern",
             [] (RegressionHolidayStateModel *model,
                 int holiday_index,
                 const Vector &pattern) {
               model->set_holiday_pattern(holiday_index, pattern);
             })
        ;

    py::class_<GeneralSeasonalLLT,
               StateModel,
               PriorPolicy,
               BOOM::Ptr<GeneralSeasonalLLT>>(boom, "GeneralSeasonalLLT")
        .def(py::init<int>(),
             py::arg("nseasons"),
             "Args:\n"
             "  nseasons:  The number of seasons in a full cycle.")
        .def_property_readonly("nseasons", &GeneralSeasonalLLT::nseasons)
        .def_property_readonly(
            "state_dimension",
            &GeneralSeasonalLLT::state_dimension,
            "Dimension of the state vector.")
        .def_property_readonly(
            "state_error_dimension",
            &GeneralSeasonalLLT::state_error_dimension,
            "Dimension of the error_term for this state component.")
        .def_property_readonly(
            "sigma_level",
            [] (GeneralSeasonalLLT &model) {
              Vector ans(model.nseasons());
              for (int i = 0; i < model.nseasons(); ++i) {
                ans[i] = sqrt(model.subordinate_model(i)->Sigma()(0, 0));
              }
              return ans;
            })
        .def("set_sigma_level",
             [] (GeneralSeasonalLLT &model, const Vector &sigma_values) {
               for (int i = 0; i < model.nseasons(); ++i) {
                 SpdMatrix Sigma = model.subordinate_model(i)->Sigma();
                 Sigma(0, 0) = square(sigma_values[i]);
                 model.subordinate_model(i)->set_Sigma(Sigma);
               }
             },
             py::arg("sigma_values"),
             "Args:\n\n"
             "  sigma_values:  A Vector containing the innovation standard "
             "deviations for the level portion of the model.  There is one "
             "entry for each season.\n")
        .def_property_readonly(
            "sigma_slope",
            [] (GeneralSeasonalLLT &model) {
              Vector ans(model.nseasons());
              for (int i = 0; i < model.nseasons(); ++i) {
                ans[i] = sqrt(model.subordinate_model(i)->Sigma()(1, 1));
              }
              return ans;
            })
        .def("set_sigma_slope",
             [] (GeneralSeasonalLLT &model, const Vector &sigma_values) {
               for (int i = 0; i < model.nseasons(); ++i) {
                 SpdMatrix Sigma = model.subordinate_model(i)->Sigma();
                 Sigma(1, 1) = square(sigma_values[i]);
                 model.subordinate_model(i)->set_Sigma(Sigma);
               }
             },
             py::arg("sigma_values"),
             "Args:\n\n"
             "  sigma_values:  A Vector containing the innovation standard "
             "deviations for the slope portion of the model.  There is one "
             "entry for each season.\n")
        .def_property_readonly(
            "initial_state_mean",
            [](GeneralSeasonalLLT &model) {return model.initial_state_mean();},
            "Mean of the state at time 0.")
        .def_property_readonly(
            "initial_state_variance",
            [](GeneralSeasonalLLT &model) {
              return model.initial_state_variance();},
            "Variance of the state at time 0.")
        .def("set_initial_state_mean",
             [](GeneralSeasonalLLT &model, const Vector &mean) {
               model.set_initial_state_mean(mean);
             },
             "Set the mean of the initial state distribution to the "
             "specified Vector.")
        .def("set_initial_state_variance",
             [](GeneralSeasonalLLT &model, const SpdMatrix &variance) {
               model.set_initial_state_variance(variance);
             },
             "Set the variance of the initial state distribution to the "
             "specified value.")
        ;


    //==========================================================================
    // Posterior samplers
    //==========================================================================
    py::class_<DynamicRegressionIndependentPosteriorSampler,
               PosteriorSampler,
               Ptr<DynamicRegressionIndependentPosteriorSampler>>(
                   boom, "DynamicRegressionIndependentPosteriorSampler")
        .def(py::init(
            [] (DynamicRegressionStateModel *model,
                const std::vector<GammaModelBase *> &priors,
                RNG &seeding_rng) {
              return new DynamicRegressionIndependentPosteriorSampler(
                  model,
                  std::vector<Ptr<GammaModelBase>>(priors.begin(),
                                                   priors.end()),
                  seeding_rng);
            }),
             py::arg("model"),
             py::arg("priors"),
             py::arg("seeding_rng") = GlobalRng::rng,
            "Args:\n\n"
             "  model:  The boom.DynamicRegressionStateModel to be sampled.\n"
             "  priors:  A list of R.SdPrior objects giving the prior "
             "distributions for each coefficient's innovation errors.")
        ;

    py::class_<StudentLocalLinearTrendPosteriorSampler,
               PosteriorSampler,
               Ptr<StudentLocalLinearTrendPosteriorSampler>>(
                   boom, "StudentLocalLinearTrendPosteriorSampler")
        .def(py::init(
            [] (StudentLocalLinearTrendStateModel *model,
                GammaModelBase *sigsq_level_prior,
                DoubleModel *nu_level_prior,
                GammaModelBase *sigsq_slope_prior,
                DoubleModel *nu_slope_prior,
                RNG &seeding_rng) {
              return new StudentLocalLinearTrendPosteriorSampler(
                  model, sigsq_level_prior, nu_level_prior,
                  sigsq_slope_prior, nu_slope_prior, seeding_rng);
            }),
             py::arg("model"),
             py::arg("sigsq_level_prior"),
             py::arg("nu_level_prior"),
             py::arg("sigsq_slope_prior"),
             py::arg("nu_slope_prior"),
             py::arg("seeding_rng") = GlobalRng::rng,
             "Args: \n\n"
             "  sigsq_level_prior: an R.SdPrior on the variance of the "
             "level component.\n"
             "  nu_level_prior: an R.DoubleModel on the tail thickness"
             "(degrees of freedom) paramater for the level component.\n"
             "  sigsq_slope_prior: an R.SdPrior on the variance of the "
             "slope component.\n"
             "  nu_slope_prior: an R.DoubleModel on the tail thickness"
             "(degrees of freedom) paramater for the slope component.\n"
             "  seeding_rng:  The random number generator used to seed "
             "the RNG in this sampler.\n")
        .def("set_sigma_level_upper_limit",
             [] (StudentLocalLinearTrendPosteriorSampler *sampler,
                 double upper_limit) {
               sampler->set_sigma_level_upper_limit(upper_limit);
             })
        .def("set_sigma_slope_upper_limit",
             [] (StudentLocalLinearTrendPosteriorSampler *sampler,
                 double upper_limit) {
               sampler->set_sigma_slope_upper_limit(upper_limit);
             })
        ;

    py::class_<GeneralSeasonalLLTPosteriorSampler,
               PosteriorSampler,
               Ptr<GeneralSeasonalLLTPosteriorSampler>>(
                   boom, "GeneralSeasonalLLTPosteriorSampler")
        .def(py::init(
            [] (GeneralSeasonalLLT *model,
                const std::vector<Ptr<WishartModel>> &priors,
                RNG &seeding_rng) {
              return new GeneralSeasonalLLTPosteriorSampler(
                  model, priors, seeding_rng);
            }),
             py::arg("model"),
             py::arg("priors"),
             py::arg("seeding_rng") = GlobalRng::rng,
             "Args:\n\n"
             "  model: The boom.SeasonalLLT model to be sampled.\n"
             "  priors:  The priors on the precision matrices of the "
             "individual local linear trend models for each season in "
             "the cycle. \n.")
        ;

    py::class_<GeneralSeasonalLLTIndependenceSampler,
               PosteriorSampler,
               Ptr<GeneralSeasonalLLTIndependenceSampler>>(
                   boom, "GeneralSeasonalLLTIndependenceSampler")
        .def(py::init(
            [] (GeneralSeasonalLLT *model,
                const std::vector<Ptr<GammaModelBase>> &level_precision_priors,
                const std::vector<Ptr<GammaModelBase>> &slope_precision_priors,
                RNG &seeding_rng) {
              return new GeneralSeasonalLLTIndependenceSampler(
                  model,
                  level_precision_priors,
                  slope_precision_priors,
                  seeding_rng);
            }),
             py::arg("model"),
             py::arg("level_precision_priors"),
             py::arg("slope_precision_priors"),
             py::arg("seeding_rng") = GlobalRng::rng,
             "Args:\n\n"
             "  model: The boom.SeasonalLLT model to be sampled.\n"
             "  level_precision_priors:  Independent priors on the precision "
             "parameters (reciprocal variances) for the level components of the"
             " individual local linear trend models.\n"
             "  slope_precision_priors:  Independent priors on the precision "
             "parameters (reciprocal variances) for the slope components of the"
             " individual local linear trend models.\n")
        ;


    //======================================================================
    // Holiday tools
    //======================================================================

    py::class_<Holiday,
               Ptr<Holiday>>(boom, "Holiday")
        .def_property_readonly(
            "maximum_window_width",
            [] (Holiday *holiday) {
              return holiday->maximum_window_width();
            })
        ;

    py::class_<OrdinaryAnnualHoliday,
               Holiday,
               Ptr<OrdinaryAnnualHoliday>>(boom, "OrdinaryAnnualHoliday")
        .def("date", [] (OrdinaryAnnualHoliday *holiday, int year) {
          return holiday->date(year);
        },
          py::arg("year"),
          "The date of the holiday in a give year.\n")
        ;

    py::class_<FixedDateHoliday,
               OrdinaryAnnualHoliday,
               Ptr<FixedDateHoliday>>(boom, "FixedDateHoliday")
        .def(py::init(
            [] (int month, int day, int days_before, int days_after) {
              return new FixedDateHoliday(
                  MonthNames(month),
                  day,
                  days_before,
                  days_after);
            }),
             py::arg("month"),
             py::arg("day"),
             py::arg("days_before") = 1,
             py::arg("days_after") = 1,
             "Args:\n\n"
             "  month: Integer month identifier.  January is 1.\n"
             "  day: Day of the month 1--31.\n"
             "  days_before: The number of days (>= 0) before the holiday to "
             "look for influence.\n"
             "  days_after: The number of days (>= 0) after the holiday to "
             "look for influence.\n")
        ;


    py::class_<NthWeekdayInMonthHoliday,
               OrdinaryAnnualHoliday,
               Ptr<NthWeekdayInMonthHoliday>>(boom, "NthWeekdayInMonthHoliday")
        .def(py::init(
            [] (int week, int day, int month, int days_before, int days_after) {
              return new NthWeekdayInMonthHoliday(
                  week,
                  DayNames(day),
                  MonthNames(month),
                  days_before,
                  days_after);
            }),
             py::arg("week"),
             py::arg("day"),
             py::arg("month"),
             py::arg("days_before") = 1,
             py::arg("days_after") = 1,
             "For example: 2nd Tuesday in November.\n"
             "Args:\n\n"
             "  week: Which week (1 <= week <= 4) of the month.\n"
             "  day: Day of the week: 0--6.  Sunday is 0.\n"
             "  month: Integer month identifier.  January is 1.\n"
             "  days_before: The number of days (>= 0) before the holiday to "
             "look for influence.\n"
             "  days_after: The number of days (>= 0) after the holiday to "
             "look for influence.\n")
        ;


    py::class_<LastWeekdayInMonthHoliday,
               OrdinaryAnnualHoliday,
               Ptr<LastWeekdayInMonthHoliday>>(
                   boom, "LastWeekdayInMonthHoliday")
        .def(py::init(
            [] (int day, int month, int days_before, int days_after) {
              return new LastWeekdayInMonthHoliday(
                  DayNames(day),
                  MonthNames(month),
                  days_before,
                  days_after);
            }),
             py::arg("day"),
             py::arg("month"),
             py::arg("days_before") = 1,
             py::arg("days_after") = 1,
             "For example: 2nd Tuesday in November.\n"
             "Args:\n\n"
             "  week: Which week (1 <= week <= 4) of the month.\n"
             "  day: Day of the week: 0--6.  Sunday is 0.\n"
             "  month: Integer month identifier.  January is 1.\n"
             "  days_before: The number of days (>= 0) before the holiday to "
             "look for influence.\n"
             "  days_after: The number of days (>= 0) after the holiday to "
             "look for influence.\n")
        ;

    py::class_<DateRangeHoliday,
               Holiday,
               Ptr<DateRangeHoliday>>(boom, "DateRangeHoliday")
        .def(py::init(
            [] (const std::vector<Date> &start, const std::vector<Date> &end) {
              return new DateRangeHoliday(start, end);
            }))
        ;

    py::class_<EasterSunday,
               OrdinaryAnnualHoliday,
               Ptr<EasterSunday>>(boom, "EasterSunday")
        .def(py::init(
            [] (int days_before, int days_after) {
              return new EasterSunday(days_before, days_after);
            }))
        ;

    py::class_<USDaylightSavingsTimeBegins,
               OrdinaryAnnualHoliday,
               Ptr<USDaylightSavingsTimeBegins>>(
                   boom, "USDaylightSavingsTimeBegins")
        .def(py::init(
            [] (int days_before, int days_after) {
              return new USDaylightSavingsTimeBegins(days_before, days_after);
            }))
        ;

    py::class_<USDaylightSavingsTimeEnds,
               OrdinaryAnnualHoliday,
               Ptr<USDaylightSavingsTimeEnds>>(
                   boom, "USDaylightSavingsTimeEnds")
        .def(py::init(
            [] (int days_before, int days_after) {
              return new USDaylightSavingsTimeEnds(days_before, days_after);
            }))
        ;

    boom.def("create_named_holiday",
             [] (const std::string &holiday_name,
                 int days_before,
                 int days_after) {
               Ptr<Holiday> ans = CreateNamedHoliday(
                   holiday_name, days_before, days_after);
               return ans;
             },
             py::arg("holiday_name"),
             py::arg("days_before") = 1,
             py::arg("days_after") = 1,
             "Args:\n\n"
             "  holiday_name:  The name of the holiday.  See the C++ code.\n"
             "  days_before: The number of days (>= 0) before the holiday to "
             "look for influence.\n"
             "  days_after: The number of days (>= 0) after the holiday to "
             "look for influence.\n")
        ;

  }  // StateSpaceModel_def

}  // namespace BayesBoom
