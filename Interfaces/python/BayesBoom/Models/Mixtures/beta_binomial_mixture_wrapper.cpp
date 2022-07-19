#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Models/BetaBinomialModel.hpp"
#include "Models/MultinomialModel.hpp"
#include "Models/Mixtures/BetaBinomialMixture.hpp"
#include "Models/Mixtures/PosteriorSamplers/BetaBinomialMixturePosteriorSampler.hpp"
#include "cpputil/Ptr.hpp"
#include "uint.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;
  using BOOM::uint;

  void BetaBinomialMixture_def(py::module &boom) {

    py::class_<BetaBinomialMixtureModel,
               PriorPolicy,
               BOOM::Ptr<BetaBinomialMixtureModel>>(
                   boom, "BetaBinomialMixtureModel", py::multiple_inheritance())
        .def(py::init(
            [](const std::vector<Ptr<BetaBinomialModel>> &components,
               const Ptr<MultinomialModel> &mixing_distribution) {
              return new BetaBinomialMixtureModel(components, mixing_distribution);
            }),
             py::arg("components"),
             py::arg("mixing_distribution"),
             "A finite mixture of beta binomial distributions.\n\n"
             "Args:\n\n"
             "  components:  The boom.BetaBinomial model objects representing "
             "the mixture components.\n"
             "  mixing_distribution:  The boom.MultinomialModel representing "
             "the mixing distribution (or the mixing weights).\n ")
        .def_property_readonly(
            "number_of_mixture_components",
            [](const BetaBinomialMixtureModel &model) {
              return model.number_of_mixture_components();
            })
        .def("add_data",
             [](BetaBinomialMixtureModel &model,
                const std::vector<int> &trials,
                const std::vector<int> &successes,
                const std::vector<int> &counts) {
               for (int i = 0; i < trials.size(); ++i) {
                 NEW(AggregatedBinomialData, data_point)(
                     trials[i], successes[i], counts[i]);
                 model.add_data(data_point);
               }
             },
             py::arg("trials"),
             py::arg("successes"),
             py::arg("count"),
             "Args:\n\n"
             "  trials: The number of binomial trials.\n"
             "  successes:  The number of successes.  0 <= successes "
             "<= trials.\n"
             "  counts:  The number of observations with the given number "
             "of trials and successes.\n"
             )
        .def_property_readonly(
            "mixing_distribution",
            [](BetaBinomialMixtureModel &model) {
              return model.mixing_distribution();
            })
        .def("mixture_component",
             [](BetaBinomialMixtureModel &model, int index) {
               return model.mixture_component(index);
             },
             py::arg("index"),
             "Args:\n\n"
             "  index:  The integer index of the desired component.\n")
        .def("__repr__",
             [](const BetaBinomialMixtureModel &model) {
               std::ostringstream out;
               out << "A BOOM BetaBinomialMixtureModel with "
                   << model.number_of_mixture_components()
                   << " components.\n";
               return out.str();
             })
        ;

    py::class_<BetaBinomialMixturePosteriorSampler,
               PosteriorSampler,
               BOOM::Ptr<BetaBinomialMixturePosteriorSampler>>(
                   boom, "BetaBinomialMixturePosteriorSampler",
                   py::multiple_inheritance())
        .def(py::init(
            [](BetaBinomialMixtureModel &model,
               RNG &seeding_rng) {
              return new BetaBinomialMixturePosteriorSampler(
                  &model, seeding_rng);
            }),
             py::arg("model"),
             py::arg("seeding_rng") = GlobalRng::rng,
             "Args:\n\n"
             "  model:  The model to be sampled. The model subcomponents must "
             "have their own posterior samplers set.  The component samplers "
             "are not managed by this object.\n"
             "  seeding_rng:  A random number generator used to seed the "
             "RNG of this sampler object.\n")
        ;

  }  // Module

}  // namespace BayesBoom
