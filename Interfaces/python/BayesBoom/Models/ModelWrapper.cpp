#include <pybind11/pybind11.h>

#include "Models/ModelTypes.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "cpputil/Ptr.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;

  void Model_def(py::module &boom) {

    py::class_<Model, Ptr<Model>>(boom, "Model")
        .def("set_method", [] (
            const Ptr<Model> &model,
            const Ptr<PosteriorSampler> &sampler) {
               model->set_method(sampler);
             },
          py::arg("sampler"),
          "Set 'sampler' as a posteriors sampling method.  More than one "
          "sampler can be set for the model (e.g. one for the mean and one "
          "for the variance).  If multiple samplers are present then each is "
          "called every time 'sample_posterior' is invoked."
          )
        .def("sample_posterior", &Model::sample_posterior,
             "Take one draw from the posterior distribution of model "
             "parameters given data.  The work for this draw is "
             "performed by any posterior samplers that have been assigned "
             "to this model by  'set_method'.")
        ;

    py::class_<PriorPolicy, Model, Ptr<PriorPolicy>>(boom, "PriorPolicy")
        ;

    py::class_<PosteriorSampler, Ptr<PosteriorSampler>>(
        boom, "PosteriorSampler")
        .def("draw", &PosteriorSampler::draw)
        ;


  }  // Module

}  // namespace BOOM
