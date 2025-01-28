#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Models/HMM/HMM2.hpp"
#include "Models/HMM/PosteriorSamplers/HmmPosteriorSampler.hpp"
#include "cpputil/Ptr.hpp"
#include "uint.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;
  using BOOM::uint;

  void HMM_def(py::module &boom) {

    py::class_<HiddenMarkovModel,
               PriorPolicy,
               BOOM::Ptr<HiddenMarkovModel>>(
                   boom, "HiddenMarkovModel", py::multiple_inheritance())
        .def(py::init(
            [](std::vector<MixtureComponent *> &components,
               MarkovModel &markov) {
              std::vector<Ptr<MixtureComponent>> component_ptrs;
              for (auto &component : components) {
                component_ptrs.push_back(Ptr<MixtureComponent>(component));
              }
              Ptr<MarkovModel> markov_ptr(&markov);
              return new HiddenMarkovModel(component_ptrs, markov_ptr);
            }),
             py::arg("components"),
             py::arg("markov"),
             "Args:\n\n"
             "  components:  A list of models inheriting from "
             "'boom.MixtureComponent'.\n"
             "  markov:  A boom.MarkovModel object for modeling the hidden "
             "Markov chain.\n" )
        .def("__repr__",
             [](const  HiddenMarkovModel &model) {
               std::ostringstream out;
               out << "A boom.HiddenMarkovModel with "
                   << model.state_space_size()
                   << " components.\n";
               return out.str();
             })
        ;

    py::class_<HmmPosteriorSampler,
               PosteriorSampler,
               BOOM::Ptr<HmmPosteriorSampler>>(
                   boom, "HmmPosteriorSampler",
                   py::multiple_inheritance())
        .def(py::init(
            [](HiddenMarkovModel &model,
               RNG &seeding_rng) {
              return new HmmPosteriorSampler(&model, seeding_rng);
            }),
             py::arg("model"),
             py::arg("seeding_rng") = GlobalRng::rng,
             "Args:\n\n"
             "  model: The boom.HiddenMarkovModel to be sampled.\n"
             "  seeding_rng:  The random number generator used to seed the "
             "RNG for the posterior sampler.\n"
             )
        ;
    

  }  // Module

}  // namespace BayesBoom
