#include <pybind11/pybind11.h>

#include "Models/BetaBinomialModel.hpp"
#include "Models/BetaModel.hpp"
#include "Models/DoubleModel.hpp"
#include "Models/PosteriorSamplers/BetaBinomialPosteriorSampler.hpp"
#include "cpputil/Ptr.hpp"
#include "uint.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;
  using BOOM::uint;

  void BetaBinomialModel_def(py::module &boom) {

    py::class_<BetaBinomialModel,
               PriorPolicy,
               BOOM::Ptr<BetaBinomialModel>>(boom,
                                             "BetaBinomialModel",
                                             py::multiple_inheritance())
        .def(py::init(
            [](double a, double b) {
              return new BetaBinomialModel(a, b);
            }),
             py::arg("a") = 1.0,
             py::arg("b") = 1.0,
             "Args:\n\n"
             "  a, b:  positive real numbers interpretable as success (a) "
             "and failure (b) counts.\n")
        .def_property_readonly("a", [](const BetaBinomialModel &model) {
                                      return model.a(); })
        .def_property_readonly("b", [](const BetaBinomialModel &model) {
                                      return model.b(); })
        .def("set_a",
             [](BetaBinomialModel &model, double a) {
               model.set_a(a);
             },
             "Set the 'a' parameter ('prior successes') to the given value.")
        .def("set_b",
             [](BetaBinomialModel &model, double b) {
               model.set_b(b );
             },
             "Set the 'b' parameter ('prior failures') to the given value.")
        .def("__repr__",
             [](const BetaBinomialModel &model) {
               std::ostringstream out;
               out << "A BOOM BetaBinomialModel with parameters ("
                   << model.a() << ", " << model.b() << ")\n";
               return out.str();
             })
        ;

    py::class_<BetaBinomialPosteriorSampler,
               PosteriorSampler,
               Ptr<BetaBinomialPosteriorSampler>>(
                   boom, "BetaBinomialPosteriorSampler")
        .def(py::init(
            [](BetaBinomialModel &model,
               Ptr<BetaModel> &mean_prior,
               DiffDoubleModel &sample_size_prior,
               RNG &seeding_rng) {
              return new BetaBinomialPosteriorSampler(
                  &model,
                  mean_prior,
                  Ptr<DiffDoubleModel>(&sample_size_prior),
                  seeding_rng);
            }),
             py::arg("model"),
             py::arg("mean_prior"),
             py::arg("sample_size_prior"),
             py::arg("seeding_rng") = BOOM::GlobalRng::rng,
             "Args:\n\n"
             "  model:  The model to be sampled.\n"
             "  mean_prior:  Prior distribution on the mean of the "
             "distribution, a/a+b\n."
             "  sample_size_prior:  Prior distribution on the sample "
             "size parameter, a+b.\n"
             "  seeding_rng:  The random number generator used to seed the "
             "sampler.\n")
        ;

  }  // Module

}  // namespace BayesBoom
