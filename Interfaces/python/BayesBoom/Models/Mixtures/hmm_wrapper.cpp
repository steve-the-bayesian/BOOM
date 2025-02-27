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

  class MixtureComponentVector {
   public:
    void append(const Ptr<MixtureComponent> &component) {
      components_.push_back(component);
    }

    const std::vector<Ptr<MixtureComponent>> &get() const {
      return components_;
    }

   private:
    std::vector<Ptr<MixtureComponent>> components_;
  };

  void HMM_def(py::module &boom) {

    py::class_<MixtureComponentVector>(boom, "MixtureComponentVector")
        .def(py::init([](){return new MixtureComponentVector;}))
        .def("append",
             [](MixtureComponentVector &vec, MixtureComponent *component) {
               vec.append(component);
             })
        .def_property_readonly(
            "values",
            [](MixtureComponentVector &vec) {
              return vec.get();
            })
        ;

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
        .def_property_readonly(
            "loglike",
            [](const HiddenMarkovModel &hmm) {return hmm.loglike();},
            "Log likelihood of the data assigned to the model, assuming "
            "current model parameters.")
        .def_property_readonly(
            "markov_model",
            [](HiddenMarkovModel &hmm) {return hmm.mark();},
            "The boom.MarkovModel object containing parameters of the Markov "
            "process responsible for the hidden chain.")
        .def("add_data",
             [](HiddenMarkovModel &hmm, const std::vector<Data *> &data_series) {
               NEW(TimeSeries<Data>, data_series_ptrs)(
                   std::vector<Ptr<Data>>(
                       data_series.begin(),
                       data_series.end()));
               hmm.add_data_series(data_series_ptrs);
             },
             py::arg("data_series"),
             "Args:\n\n"
             "  data_series:  A time-ordered list of BOOM Data objects "
             "comprising the data values for a single subject.")
        .def("save_state_probs",
             [](HiddenMarkovModel &hmm) {
               hmm.save_state_probs();
             },
             "Save the marginal probabilities of the hidden states at each "
             "time point.")
        .def("imputed_states",
             [](const HiddenMarkovModel &hmm, int user_index) {
               // The HMM stores the imputed states in a map keyed by the user
               // data series.  That map is not available outside the C++
               // environment, so we have to do the mapping here between user 0,
               // user 1, etc and the respective data sets for those users.
               Ptr<Data> user_data_series();
               std::vector<int> state_draws = hmm.imputed_state(
                   hmm.dat(user_index));
               return Vector(state_draws.begin(), state_draws.end());
             },
             py::arg("user_index"),
             "Args:\n\n "
             "  user_index:  Users 0, 1, 2, etc are stored in the order "
             "their data was added to the HMM.\n\n"
             "Returns:\n"
             "  The state (hidden Markov chain) values imputed by the most "
             "recent draw from the forward-backward sampling algorithm.\n ")
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
