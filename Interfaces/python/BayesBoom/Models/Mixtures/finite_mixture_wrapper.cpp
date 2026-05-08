#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Models/FiniteMixtureModel.hpp"
#include "Models/PosteriorSamplers/FiniteMixturePosteriorSampler.hpp"
#include "cpputil/Ptr.hpp"
#include "uint.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;
  using BOOM::uint;

  void FiniteMixtureModel_def(py::module &boom) {

    py::class_<FiniteMixtureModel,
               PriorPolicy,
               BOOM::Ptr<FiniteMixtureModel>>(
                   boom, "FiniteMixtureModel", py::multiple_inheritance())
        .def(py::init(
            [](std::vector<MixtureComponent *> &components,
               MultinomialModel &mixing_weights) {
              std::vector<Ptr<MixtureComponent>> component_ptrs;
              for (auto &component : components) {
                component_ptrs.push_back(Ptr<MixtureComponent>(component));
              }
              Ptr<MultinomialModel> mixing_weight_ptr(&mixing_weights);
              return new FiniteMixtureModel(component_ptrs, mixing_weight_ptr);
            }),
             py::arg("components"),
             py::arg("mixing_weights"),
             "Args:\n\n"
             "  components:  A list of models inheriting from "
             "'boom.MixtureComponent'.\n"
             "  mixing_weights:  A boom.MultinomialModel describing the mixing "
             "weights for the mixture.\n" )
        .def("add_data",
             [](FiniteMixtureModel *model, Data *data_point) {
               model->add_data(Ptr<Data>(data_point));
             })
        .def_property_readonly(
            "last_loglike",
            [](const FiniteMixtureModel &model) {
              return model.last_loglike();
            },
            "The log likelihood produced by the most recent call to "
            "sample_posterior.")
        .def_property_readonly(
            "class_membership_probability",
            [](const FiniteMixtureModel &model) {
              return model.class_membership_probability();
            },
            "A matrix with rows corresponding to observations, and columns to "
            "mixture components.\n"
            "Element (i, j) is the probability that unit i is a member of "
            "component j.\n")
        .def("__repr__",
             [](const FiniteMixtureModel &model) {
               std::ostringstream out;
               out << "A boom.FiniteMixtureModel with "
                   << model.number_of_mixture_components()
                   << " components.\n";
               return out.str();
             })
        ;

    py::class_<FiniteMixturePosteriorSampler,
               PosteriorSampler,
               BOOM::Ptr<FiniteMixturePosteriorSampler>>(
                   boom, "FiniteMixturePosteriorSampler",
                   py::multiple_inheritance())
        .def(py::init(
            [](FiniteMixtureModel &model, RNG &seeding_rng) {
              return new FiniteMixturePosteriorSampler(&model, seeding_rng);
            }),
             py::arg("model"),
             py::arg("seeding_rng") = GlobalRng::rng,
             "Args:\n\n"
             "  model:  The FiniteMixtureModel to be sampled.\n"
             "RNG of this sampler object.\n")
        ;

  }  // Module

}  // namespace BayesBoom
