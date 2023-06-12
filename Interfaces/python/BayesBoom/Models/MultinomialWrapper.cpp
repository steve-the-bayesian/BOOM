#include <pybind11/pybind11.h>

#include "Models/MultinomialModel.hpp"
#include "Models/DirichletModel.hpp"
#include "Models/PosteriorSamplers/MultinomialDirichletSampler.hpp"
#include "cpputil/Ptr.hpp"
#include "uint.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;
  using BOOM::uint;

  void MultinomialModel_def(py::module &boom) {

    py::class_<MultinomialSuf,
               BOOM::Ptr<MultinomialSuf>>(boom, "MultinomialSuf")
        .def(py::init(
            [](const Vector &counts) {
              return new MultinomialSuf(counts);
            }),
             py::arg("counts"),
             "Args:\n"
             "  counts:  A vector of non-negative values interpretable as "
             "prior counts.\n")
        .def_property_readonly(
            "counts",
            [](const MultinomialSuf &suf) {return suf.n();})
        ;

    py::class_<MultinomialModel,
               PriorPolicy,
               BOOM::Ptr<MultinomialModel>>(boom,
                                            "MultinomialModel",
                                            py::multiple_inheritance())
        .def(py::init(
            [](int dim) {
              Vector probs(dim, 1.0 / dim);
              return new MultinomialModel(probs);
            }),
             py::arg("dim"),
             "Args:\n\n"
             "  dim: The dimension of the multinomial distribution.  I.e. The "
             "number of categories.\n")
        .def(py::init(
            [](const Vector &probs) {
              return new MultinomialModel(probs / probs.sum());
            }),
             py::arg("probs"),
             "Args:\n\n"
             "  probs:  A discrete probability distribution.\n")
        .def_property_readonly(
            "probs",
            [](const MultinomialModel &model) {
              return model.pi();
            })
        .def("set_probs",
             [](MultinomialModel &model, const Vector &probs) {
               model.set_pi(probs);
             })
        .def("__repr__",
             [](const MultinomialModel &model) {
               std::ostringstream out;
               out << "A BOOM MultinomialModel with parameter "
                   << model.pi() << "\n";
               return out.str();
             })
        ;

    py::class_<MultinomialDirichletSampler,
               PosteriorSampler,
               Ptr<MultinomialDirichletSampler>>(
                   boom, "MultinomialDirichletSampler")
        .def(py::init(
            [](MultinomialModel &model,
               Ptr<DirichletModel> &prior,
               RNG &seeding_rng) {
              return new MultinomialDirichletSampler(
                  &model, prior, seeding_rng);
            }),
             py::arg("model"),
             py::arg("prior"),
             py::arg("seeding_rng") = GlobalRng::rng,
             "Args:\n\n"
             "  model:  The model to be sampled.\n"
             "  prior:  Prior distribution on the multinomial probabilities.\n"
             )
        ;

  }  // Module

}  // namespace BayesBoom
