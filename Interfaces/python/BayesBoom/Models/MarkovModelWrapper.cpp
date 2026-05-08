#include <pybind11/pybind11.h>

#include "Models/MarkovModel.hpp"
#include "Models/PosteriorSamplers/MarkovConjSampler.hpp"
#include "cpputil/Ptr.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;

  void MarkovModel_def(py::module &boom) {

    py::class_<MarkovSuf,
               Data,
               BOOM::Ptr<MarkovSuf>>(boom, "MarkovSuf")
        .def(py::init(
            [](const Matrix &transition_counts,
               const Vector &initial_counts) {
              return new MarkovSuf(transition_counts, initial_counts);
            }),
             py::arg("transition_counts"),
             py::arg("initial_counts"),
             "Args:\n\n"
             "  transition_counts:  A square matrix with element r,s containing "
             "the number of transition from state r to state s.\n"
             "  initial_counts:  A vector of the same dimension as "

             "transition_counts containing the number of observations in each "
             "state at time 0.  If there is only one data sequence then this "
             "vector will be all 0's except for a single 1.  \n")
             ;
    
    py::class_<MarkovModel,
               PriorPolicy,
               MixtureComponent,
               BOOM::Ptr<MarkovModel>>(
                   boom, "MarkovModel", py::multiple_inheritance())
        .def(py::init(
            [](const Matrix &transition_probabilities,
               const Vector &initial_distribution) {
              return new MarkovModel(transition_probabilities, initial_distribution);
            }),
             py::arg("transition_probabilities"),
             py::arg("initial_distribution"),
             "Args:\n\n"
             "  transition_probabilities:  The transition probability matrix, "
             "with rows summing to 1.  Thus element (i, j) in the matrix is "
             "the conditional probability the next state is j, given that the "
             "current state is i.\n"
             "  initial_distribution:  A discrete probability distribution "
             "describing the state at the time of the first observation.\n")
        .def("fix_pi0",
             [](MarkovModel &model, const Vector &pi0) {
               model.fix_pi0(pi0);
             },
             py::arg("pi0"),
             "Args:\n\n"
             "  pi0:  A discrete probability distribution.\n"
             "\n"
             "Effects: \n"
             "  The initial distribution of the model is set to pi0. and "
             "cannot not be updated.\n")
        .def_property_readonly(
            "transition_probabilities",
            [](const MarkovModel &model) {
              return model.Q();
            })
        .def_property_readonly(
            "initial_distribution",
            [](const MarkovModel &model) {
              return model.pi0();
            })
        .def_property_readonly(
            "state_space_size",
            [](const MarkovModel &model) {
              return model.state_space_size();
            })
        .def_property_readonly(
            "number_of_observations",
            [](const MarkovModel &model) {
              return model.number_of_observations();
            })
        ;

    
    py::class_<MarkovConjSampler,
               PosteriorSampler,
               BOOM::Ptr<MarkovConjSampler>>(
                   boom, "MarkovConjugateSampler", py::multiple_inheritance())
        .def(py::init(
            [](MarkovModel *model,
               const Matrix &prior_transition_counts,
               RNG &seeding_rng) {
              return new MarkovConjSampler(
                  model,
                  prior_transition_counts,
                  seeding_rng);
            }),
             py::arg("model"),
             py::arg("prior_transition_counts"),
             py::arg("seeding_rng"))
        .def(py::init(
            [](MarkovModel *model,
               const Matrix &prior_transition_counts,
               const Vector &prior_initial_counts,
               RNG &seeding_rng) {
              return new MarkovConjSampler(
                  model,
                  prior_transition_counts,
                  prior_initial_counts,
                  seeding_rng);
            }),
             py::arg("model"),
             py::arg("prior_transition_counts"),
             py::arg("prior_initial_counts"),
             py::arg("seeding_rng"))
        ;


  }  // Module

}  // namespace BOOM
