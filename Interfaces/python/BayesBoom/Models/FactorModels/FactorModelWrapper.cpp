#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Models/FactorModels/PoissonFactorModel.hpp"
#include "Models/FactorModels/PosteriorSamplers/PoissonFactorModelPosteriorSampler.hpp"
#include "cpputil/Ptr.hpp"
#include "cpputil/report_error.hpp"
#include "uint.hpp"

#include <sstream>

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;
  using BOOM::uint;

  void PoissonFactorModel_def(py::module &boom) {

    py::class_<PoissonFactorModel,
               PriorPolicy,
               BOOM::Ptr<PoissonFactorModel>>(
                   boom, "PoissonFactorModel", py::multiple_inheritance())
        .def(py::init(
            [](int num_classes) {
              return new PoissonFactorModel(num_classes);
                  }),
             py::arg("num_classes"),
             "Args:\n\n"
             "  num_classes:  The number of classes in the factor model.\n")
        .def("add_data",
             [](PoissonFactorModel &model,
                const std::vector<int> &visitor_id,
                const std::vector<int> &site_id,
                const std::vector<int> &num_visits) {
               if (visitor_id.size() != site_id.size()
                   || visitor_id.size() != num_visits.size()) {
                 report_error("visitor_id, site_id, and num_visits must "
                              "all have the same length.");
               }
               std::cout << "calling add_data in glue code for "
                         << visitor_id.size() << " data points.\n";
               for (size_t i = 0; i < visitor_id.size(); ++i) {
                 model.record_visit(visitor_id[i], site_id[i], num_visits[i]);
               }
             },
             py::arg("visitor_id"),
             py::arg("site_id"),
             py::arg("num_visits"),
             "Args:\n\n"
             "  visitor_id: A vector of integer ID's identifying the visitor.\n"
             "  site_id:  A vector of integer ID's identifying the site.\n"
             "  num_visits:  A vector of integers giving the number of times "
             "each visitor visited the corresponding site.\n")
        .def_property_readonly(
            "site_ids",
            [](PoissonFactorModel &model) {
              std::vector<int> site_ids;
              for (const auto &site : model.sites()) {
                site_ids.push_back(site->id());
              }
              return site_ids;
            },
            "The IDs of the sites, in the order they are stored by the model.\n")
        .def_property_readonly(
            "lambdas",
            [](PoissonFactorModel &model) {
              size_t num_sites = model.sites().size();
              Matrix lambdas(num_sites, model.number_of_classes());
              for (int i = 0; i < num_sites; ++i) {
                lambdas.row(i) = model.sites()[i]->lambda();
              }
              return lambdas;
            },
            "The visitation rate parameters for each of the sites in the model. "
            " Each row is a site (in the same order as the 'site_id' attribute) "
            "and each column is a conditional rate for one of the latent "
            "categories.\n")
        .def_property_readonly(
            "visitor_ids",
            [](PoissonFactorModel &model) {
              std::vector<int> visitor_ids;
              for (const auto &visitor : model.visitors()) {
                visitor_ids.push_back(visitor->id());
              }
              return visitor_ids;
            })
        .def("__repr__",
             [](PoissonFactorModel &model) {
               std::ostringstream out;
               out << "A boom.PoissonFactorModel with "
                   << model.number_of_classes()
                   << " latent classes.\n";
               return out.str();
             })
        ;

    py::class_<PoissonFactorModelPosteriorSampler,
               PosteriorSampler,
               BOOM::Ptr<PoissonFactorModelPosteriorSampler>>(
                   boom, "PoissonFactorModelPosteriorSampler",
                   py::multiple_inheritance())
        .def(py::init(
            [](PoissonFactorModel *model,
               const Vector &prior_class_membership_probabilities,
               RNG &seeding_rng) {
              return new PoissonFactorModelPosteriorSampler(
                  model, prior_class_membership_probabilities, seeding_rng);
            }),
             py::arg("model"),
             py::arg("prior_class_membership_probabilities"),
             py::arg("seeding_rng") = GlobalRng::rng)
        ;

  }  // Module

}  // namespace BayesBoom
