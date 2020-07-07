#include <pybind11/pybind11.h>

#include "Models/TimeSeries/NonzeroMeanAr1Model.hpp"
#include "Models/TimeSeries/PosteriorSamplers/NonzeroMeanAr1Sampler.hpp"

#include "Models/TimeSeries/ArModel.hpp"

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
            "mean", &NonzeroMeanAr1Model::mu, "The long run mean of the process.")
        .def_property_readonly(
            "ar1", &NonzeroMeanAr1Model::phi, "The Ar1 coefficient.")
        .def_property_readonly(
            "residual_sd", &NonzeroMeanAr1Model::sigma,
            "The residual standard deviation.")
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

  }


}  // namespace BayesBoom
