#include <pybind11/pybind11.h>

#include "Models/ModelTypes.hpp"
#include "Models/DoubleModel.hpp"
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

    py::class_<PosteriorSampler, Ptr<PosteriorSampler>>(
        boom, "PosteriorSampler")
        .def("draw", &PosteriorSampler::draw)
        ;


    py::class_<PriorPolicy, Model, Ptr<PriorPolicy>>(boom, "PriorPolicy")
        .def("set_method", &PriorPolicy::set_method,
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
