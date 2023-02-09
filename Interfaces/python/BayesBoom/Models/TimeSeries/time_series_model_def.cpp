#include <pybind11/pybind11.h>

#include "Models/TimeSeries/NonzeroMeanAr1Model.hpp"
#include "Models/TimeSeries/PosteriorSamplers/NonzeroMeanAr1Sampler.hpp"

#include "Models/TimeSeries/ArModel.hpp"
#include "Models/TimeSeries/PosteriorSamplers/ArPosteriorSampler.hpp"
#include "Models/TimeSeries/PosteriorSamplers/ArSpikeSlabSampler.hpp"

#include "cpputil/Ptr.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace py = pybind11;

namespace BayesBoom {

  using namespace BOOM;

  void TimeSeries_def(py::module &boom) {
    py::class_<NonzeroMeanAr1Model,
               PriorPolicy,
               Ptr<NonzeroMeanAr1Model>>(boom, "NonzeroMeanAr1Model")
        .def(py::init<double, double, double>(),
             py::arg("mu") = 0.0,
             py::arg("ar1") = 0.0,
             py::arg("sigma") = 1.0,
             "Args:\n"
             "  mu:  The long run mean of the process.\n"
             "  ar1:  The AR1 coefficient.  Must be between -1 and 1 if a \n"
             "    stationary process is desired.\n"
             "  sigma:  The residual standard deviation.")
        .def_property_readonly(
            "mean", &NonzeroMeanAr1Model::mu,
            "The long run mean of the process.")
        .def_property_readonly(
            "ar1", &NonzeroMeanAr1Model::phi, "The Ar1 coefficient.")
        .def_property_readonly(
            "residual_sd", &NonzeroMeanAr1Model::sigma,
            "The residual standard deviation.")
        ;

    py::class_<ArModel,
               GlmModel,
               PriorPolicy,
               Ptr<ArModel>>(boom, "ArModel")
        .def(py::init([](int lags) {
          return new ArModel(lags);
        }),
          py::arg("lags") = 1,
          "Args:\n\n"
          "  lags:  The number of lags in the AR process.  This is the "
          "'p' in AR(p).")
        .def_property_readonly("sigma", [] (const ArModel *model) {
          return model->sigma();
        })
        .def("set_sigma", [](ArModel *model, double sigma) {
          model->set_sigma(sigma);
        })
        .def_property_readonly("phi", [] (const ArModel *model) {
          return model->phi();
        })
        .def("set_phi", [] (ArModel *model, const Vector &phi) {
          model->set_phi(phi);
        })
        .def("autocovariance", [] (const ArModel *model, int lags) {
          return model->autocovariance(lags);
        },
          py::arg("lags"),
          "Compute the first 'lag' elements of the autocovariance function "
          "implied by the model.\n")
        .def("simulate", [] (const ArModel *model, int sample_size, RNG &rng) {
          return model->simulate(sample_size, rng);
        },
          py::arg("sample_size"),
          py::arg("rng") = GlobalRng::rng,
          "Args:\n\n"
          "  sample_size: The number of time steps to simulate.\n"
          "  rng:  The random number generator to use for the simulation.\n")
        ;


    py::class_<NonzeroMeanAr1Sampler,
               PosteriorSampler,
               Ptr<NonzeroMeanAr1Sampler>>(boom, "NonzeroMeanAr1Sampler")
        .def(py::init([](
            NonzeroMeanAr1Model *model,
            GaussianModelBase *mean_prior,
            GaussianModelBase *ar1_coefficient_prior,
            GammaModelBase *residual_precision_prior,
            RNG &rng) {
               return new NonzeroMeanAr1Sampler(
                   model,
                   Ptr<GaussianModelBase>(mean_prior),
                   Ptr<GaussianModelBase>(ar1_coefficient_prior),
                   Ptr<GammaModelBase>(residual_precision_prior),
                   rng); }),
             py::arg("model"),
             py::arg("mean_prior"),
             py::arg("ar1_coefficient_prior"),
             py::arg("residual_precision_prior"),
             py::arg("rng") = GlobalRng::rng,
             "Args:\n"
             "  model: the model to be learned."
             "  mean_prior:  Prior distribution on the long run mean of the "
             "process.\n"
             "  ar1_coefficient_prior:  Prior distribution on the AR1 "
             "coefficient.\n"
             "  residual_precision_prior: Prior distribution on the reciprocal "
             "    of the error variance.\n"
             "  rng:  The random number generator used to seed the RNG in this "
             "sampler.\n")
        .def("set_sigma_upper_limit",
             &NonzeroMeanAr1Sampler::set_sigma_upper_limit,
             py::arg("limit"),
             "Limit the support of the standard deviation to (0, limit).\n"
             "Args:\n"
             "  limit:  The largest standard deviation in the support of the \n"
             "    prior.  This can be infinity.\n")
        .def("force_stationary",
             &NonzeroMeanAr1Sampler::force_stationary,
             "Limit the support of the AR1 coefficient to (-1, 1).")
        .def("force_positive",
             &NonzeroMeanAr1Sampler::force_ar1_positive,
             "Limit the support of the AR1 coefficient to (0, 1), if \n"
             "force_stationary is True, or (0, infinity) if force_stationary \n"
             "is False.")
        ;

    py::class_<ArPosteriorSampler,
               PosteriorSampler,
               Ptr<ArPosteriorSampler>>(boom, "ArPosteriorSampler")
        .def(py::init(
            [] (ArModel *model,
                GammaModelBase *siginv_prior,
                RNG &seeding_rng) {
              return new ArPosteriorSampler(model, siginv_prior, seeding_rng);
            }),
             py::arg("model"),
             py::arg("siginv_prior"),
             py::arg("seeding_rng") = GlobalRng::rng,
             "Args:\n\n"
             "  model:  The model to be sampled.\n"
             "  siginv_prior:  Prior distribution for the residual precision.\n"
             "  seeding_rng: RNG to seed the RNG in this sampler.")
        .def("set_sigma_upper_limit",
             [](ArPosteriorSampler *sampler, double limit) {
               sampler->set_sigma_upper_limit(limit);
             })
        ;

    py::class_<ArSpikeSlabSampler,
               PosteriorSampler,
               Ptr<ArSpikeSlabSampler>>(boom, "ArSpikeSlabSampler")
        .def(py::init(
            [] (ArModel *model,
                MvnBase *slab,
                VariableSelectionPrior *spike,
                GammaModelBase *residual_precision_prior,
                bool truncate_support_to_stationary_region,
                RNG &seeding_rng) {
              return new ArSpikeSlabSampler(
                  model, slab, spike, residual_precision_prior,
                  truncate_support_to_stationary_region, seeding_rng);
            }),
             py::arg("model"),
             py::arg("slab"),
             py::arg("spike"),
             py::arg("residual_precision_prior"),
             py::arg("enforce_stationarity"),
             py::arg("seeding_rng") = GlobalRng::rng,
             "Args:\n\n"
             "  model:  The model to be sampled.\n"
             "  slab:  The slab portion of the spike and slab sampler.\n"
             "  spike: The spike portion of the spike and slab sampler.\n"
             "  residual_precision_prior:  Prior on the residual precision "
             "parameter.\n"
             "  enforce_stationarity:  If True then the AR coefficients are "
             "constrained to the region that forces the model to remain "
             "stationary.  If False then the AR coefficients are "
             "unconstrained.\n"
             "  seeding_rng: The random number generator used to seed the "
             "RNG held by this sampler.\n")
        .def("limit_model_selection",
             [] (ArSpikeSlabSampler *sampler, int max_flips) {
               sampler->limit_model_selection(max_flips);
             },
             py::arg("max_flips"),
             "Args: \n\n"
             "  max_flips:  If positive and finite then limit the number"
             " of variable selection add/drop moves to 'max_flips'.\n")
        .def("set_sigma_upper_limit",
             [] (ArSpikeSlabSampler *sampler, double upper_limit) {
               sampler->set_sigma_upper_limit(upper_limit);
             },
             py::arg("upper_limit"),
             "Args:\n\n"
             "  upper_limit: If positive and finite then constrain the "
             "support of sigma to [0, upper_limit).\n")
        ;



  }


}  // namespace BayesBoom
