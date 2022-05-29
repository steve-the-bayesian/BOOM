#include <pybind11/pybind11.h>

#include "Models/ModelTypes.hpp"
#include "Models/DoubleModel.hpp"
#include "Models/SpdModel.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "cpputil/Ptr.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;

  void Model_def(py::module &boom) {

    py::class_<Model, Ptr<Model>>(boom, "Model")
        ;

    py::class_<DoubleModel, Model, Ptr<DoubleModel>>(boom, "DoubleModel", py::multiple_inheritance())
        .def("logp", [](const DoubleModel &m, double x) {
          return m.logp(x);
        },
          py::arg("x"),
          "The log density evaluated at 'x'.")
        ;

    py::class_<SpdModel, Model, Ptr<SpdModel>>(
        boom, "SpdModel", py::multiple_inheritance())
        .def("logp",
             [](const SpdModel &model, const SpdMatrix &x) {
               return model.logp(x);
             },
             py::arg("x"),
             "The log density evaluated at 'x'.")
        ;

    py::class_<PosteriorModeModel,
               Model,
               Ptr<PosteriorModeModel>>(
                   boom, "PosteriorModeModel", py::multiple_inheritance())
        .def("find_posterior_mode",
             [](PosteriorModeModel &model, double epsilon) {
               model.find_posterior_mode(epsilon);
             },
             py::arg("epsilon") = 1e-5,
             "Args:\n"
             "  epsilon:  If the mode finding algorithm is iterative, use "
             "epsilon as its convergence criterion.")
        .def_property_readonly(
            "can_find_posterior_mode",
            [](const PosteriorModeModel &model) {return model.can_find_posterior_mode();},
            "True iff the model has been assigned a PosteriorSampmler capable of "
            "finding its posterior mode.")
        ;

    py::class_<PosteriorSampler, Ptr<PosteriorSampler>>(
        boom, "PosteriorSampler")
        .def("draw", &PosteriorSampler::draw)
        ;


    py::class_<PriorPolicy, Model, Ptr<PriorPolicy>>(boom, "PriorPolicy")
        .def("set_method",
             [](PriorPolicy &model, PosteriorSampler *sampler) {
               model.set_method(sampler);
             },
             py::arg("sampler"),
             "Set 'sampler' as a posteriors sampling method.  More than one\n"
             "sampler can be set for the model (e.g. one for the mean and one \n"
             "for the variance).  If multiple samplers are present then each is \n"
             "called every time 'sample_posterior' is invoked.\n"
             )
        .def("sample_posterior", &PriorPolicy::sample_posterior,
             "Take one draw from the posterior distribution of model \n"
             "parameters given data.  The work for this draw is \n"
             "performed by any posterior samplers that have been assigned \n"
             "to this model by  'set_method'.\n")
        ;


  }  // Module

}  // namespace BOOM
