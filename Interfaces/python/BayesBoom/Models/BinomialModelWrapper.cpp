#include <pybind11/pybind11.h>

#include "Models/BinomialModel.hpp"
#include "Models/BetaModel.hpp"
#include "Models/PosteriorSamplers/BetaBinomialSampler.hpp"

#include "cpputil/Ptr.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;
  using BOOM::uint;

  void BinomialModel_def(py::module &boom) {
    // =========================================================================
    // BinomialSuf: sufficient statistics for the binomial distribution.
    py::class_<BinomialSuf,
               Ptr<BinomialSuf>>(boom, "BinomialSuf")
        .def(py::init([]() { return new BinomialSuf(); }),
             "Create an empty set of sufficient statistics for the binomial "
             "distribution.\n")
        .def("update",
             [](BinomialSuf &suf, double y) { suf.update_raw(y); },
             py::arg("y"),
             "Add a single success/failure observation to the sufficient "
             "statistics.\n\n"
             "Args:\n"
             "  y: 1.0 for a success, 0.0 for a failure.\n")
        .def("batch_update",
             [](BinomialSuf &suf, double n, double y) {
               suf.batch_update(n, y);
             },
             py::arg("n"),
             py::arg("y"),
             "Add a batch of binomial observations.\n\n"
             "Args:\n"
             "  n: Number of trials.\n"
             "  y: Number of successes.\n")
        .def("clear",
             [](BinomialSuf &suf) { suf.clear(); },
             "Reset the sufficient statistics to zero.\n")
        .def_property_readonly(
            "sum",
            [](const BinomialSuf &suf) { return suf.sum(); },
            "The total number of observed successes.")
        .def_property_readonly(
            "nobs",
            [](const BinomialSuf &suf) { return suf.nobs(); },
            "The total number of trials.")
        .def("__repr__",
             [](const BinomialSuf &suf) {
               std::ostringstream out;
               out << "BinomialSuf(sum=" << suf.sum()
                   << ", nobs=" << suf.nobs() << ")";
               return out.str();
             })
        ;

    // =========================================================================
    // BinomialModel: models a binomial success probability.
    py::class_<BinomialModel,
               PriorPolicy,
               Ptr<BinomialModel>>(boom, "BinomialModel",
                                   py::multiple_inheritance())
        .def(py::init(
            [](double p) {
              return new BinomialModel(p);
            }),
             py::arg("p") = 0.5,
             "Args:\n\n"
             "  p: Initial value for the success probability.  "
             "Must satisfy 0 < p < 1.\n")
        .def_property(
            "prob",
            [](const BinomialModel &m) { return m.prob(); },
            [](BinomialModel &m, double p) { m.set_prob(p); },
            "The current success probability parameter.")
        .def_property_readonly(
            "suf",
            [](BinomialModel &m) { return m.suf();},
            "The sufficient statistics for the model.")
        .def("clear_data",
             [](BinomialModel &m) { m.clear_data(); },
             "Remove all observed data from the model.")
        .def("log_likelihood",
             [](const BinomialModel &m) { return m.log_likelihood(); },
             "Log likelihood of the observed data given the current parameters.")
        .def("__repr__",
             [](const BinomialModel &m) {
               std::ostringstream out;
               out << "BinomialModel(p=" << m.prob() << ")";
               return out.str();
             })
        ;

    // =========================================================================
    // BetaBinomialSampler: Beta-conjugate posterior sampler for BinomialModel.
    py::class_<BetaBinomialSampler,
               PosteriorSampler,
               Ptr<BetaBinomialSampler>>(boom, "BetaBinomialSampler")
        .def(py::init(
            [](BinomialModel *model,
               BetaModel *prior,
               RNG &seeding_rng) {
              return new BetaBinomialSampler(
                  model, Ptr<BetaModel>(prior), seeding_rng);
            }),
             py::arg("model"),
             py::arg("prior"),
             py::arg("seeding_rng") = BOOM::GlobalRng::rng,
             "Args:\n\n"
             "  model: The BinomialModel to be sampled.\n"
             "  prior: A BetaModel serving as the conjugate prior on the "
             "success probability.  The parameters a and b are interpretable "
             "as prior successes and failures respectively.\n"
             "  seeding_rng: Optional RNG used to seed this sampler's own "
             "RNG.\n")
        ;

  }  // Module

}  // namespace BOOM
