#include <pybind11/pybind11.h>

#include "Models/PoissonModel.hpp"
#include "Models/PosteriorSamplers/PoissonGammaSampler.hpp"
#include "cpputil/Ptr.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;

  void PoissonModel_def(py::module &boom) {

    py::class_<PoissonModel,
               PriorPolicy,
               BOOM::Ptr<PoissonModel>>(
                   boom, "PoissonModel", py::multiple_inheritance())
        .def(py::init(
            [](double lambda) {
              return new PoissonModel(lambda);
            }),
             py::arg("lambda"),
             "Args:\n\n"
             "  lambda:  The mean parameter for the Poisson distribution.\n")
        .def_property_readonly(
            "lambda",
            [](const PoissonModel &model) {
              return model.lambda();
            })
        .def_property_readonly(
            "mean",
            [](const PoissonModel &model) {
              return model.lambda();
            })
        .def_property_readonly(
            "variance",
            [](const PoissonModel &model) {
              return model.lambda();
            })
        ;

    
    py::class_<PoissonGammaSampler,
               PosteriorSampler,
               BOOM::Ptr<PoissonGammaSampler>>(
                   boom, "PoissonGammaSampler", py::multiple_inheritance())
        .def(py::init(
            [](PoissonModel *model,
               GammaModel *prior,
               RNG &seeding_rng) {
              return new PoissonGammaSampler(
                  model, prior, seeding_rng);
            }),
             py::arg("model"),
             py::arg("prior"),
             py::arg("seeding_rng"))
        ;


  }  // Module

}  // namespace BOOM
