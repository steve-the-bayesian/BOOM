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

    py::class_<PoissonFactor::Site,
               BOOM::Ptr<::BOOM::PoissonFactor::Site>>(
                   boom, "PoissonFactorModelSite")
        .def_property_readonly(
            "id",
            [](PoissonFactor::Site &site) {
              return site.id();
            },
            "The ID of the site.")
        .def_property_readonly(
            "rate_params",
            [](PoissonFactor::Site &site) {
              return site.lambda();
            },
            "The poisson rate parameters for the Site.")
        .def_property_readonly(
            "num_visits",
            [](PoissonFactor::Site &site) {
              size_t num_visits = 0;
              for (const auto &it : site.observed_visitors()) {
                num_visits += it.second;
              }
              return num_visits;
            },
            "The number of times the site was visited.")
        .def_property_readonly(
            "num_visitors",
            [](PoissonFactor::Site &site) {
              return site.observed_visitors().size();
            },
            "The number of distinct visitors to the site.")
        .def_property_readonly(
            "visitor_counts",
            [](PoissonFactor::Site &site) {
              return site.visitor_counts();
            },
            "Returns a 2-column matrix containing the visit counts (column 0) "
            "and exposures (column 1) for each level of the latent category.\n")
        .def_property_readonly(
            "prior_a",
            [](const PoissonFactor::Site &site) {
              return site.prior_a();
            })
        .def_property_readonly(
            "prior_b",
            [](const PoissonFactor::Site &site) {
              return site.prior_b();
            })

        ;

    py::class_<PoissonFactor::Visitor, Ptr<PoissonFactor::Visitor>>(
        boom, "PoissonFactorModelVisitor")
        .def_property_readonly(
            "id",
            [](PoissonFactor::Visitor &visitor) {
              return visitor.id();
            },
            "The ID of the visitor.")
        .def_property_readonly(
            "imputed_class",
            [](PoissonFactor::Visitor &visitor) {
              return visitor.imputed_class_membership();
            },
            "The class membership indicator assigned to the Visitor "
            "in the most recent MCMC draw.")
        .def_property_readonly(
            "num_visits",
            [](PoissonFactor::Visitor &visitor) {
              size_t visits = 0;
              for (const auto &it : visitor.sites_visited()) {
                visits += it.second;
              }
              return visits;
            },
            "The number of visits (including repeat visits) the user "
            "made to any site.\n")
        .def_property_readonly(
            "number_of_distinct_sites_visited",
            [](PoissonFactor::Visitor &visitor) {
              return visitor.sites_visited().size();
            },
            "The number of distinct sites visited by the user.")
        ;


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
                const std::vector<std::string> &visitor_id,
                const std::vector<std::string> &site_id,
                const std::vector<int> &num_visits) {
               if (visitor_id.size() != site_id.size()
                   || visitor_id.size() != num_visits.size()) {
                 report_error("visitor_id, site_id, and num_visits must "
                              "all have the same length.");
               }
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
            "num_users",
            [](PoissonFactorModel &model) {
              return model.visitors().size();
            })
        .def_property_readonly(
            "num_sites",
            [](PoissonFactorModel &model) {
              return model.sites().size();
            })
        .def_property_readonly(
            "site_ids",
            [](PoissonFactorModel &model) {
              std::vector<std::string> site_ids;
              for (const auto &site_it : model.sites()) {
                site_ids.push_back(site_it.first);
              }
              return site_ids;
            },
            "The IDs of the sites, in the order they are stored by the model.\n")
        .def_property_readonly(
            "lambdas",
            [](PoissonFactorModel &model) {
              size_t num_sites = model.sites().size();
              Matrix lambdas(num_sites, model.number_of_classes());
              int i = 0;
              for (const auto &site_it : model.sites()) {
                lambdas.row(i++) = site_it.second->lambda();
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
              std::vector<std::string> visitor_ids;
              for (const auto &visitor_it : model.visitors()) {
                visitor_ids.push_back(visitor_it.first);
              }
              return visitor_ids;
            },
            "The visitor ID's, in the order stored by the model.\n")
        .def_property_readonly(
            "imputed_classes",
            [](PoissonFactorModel &model) {
              std::vector<int> classes;
              for (const auto &visitor_it : model.visitors()) {
                classes.push_back(
                    visitor_it.second->imputed_class_membership());
              }
              return classes;
            })
        .def_property_readonly(
            "site_params",
            [](PoissonFactorModel &model) {
              Matrix ans(model.number_of_sites(), model.number_of_classes());
              size_t i = 0;
              for (const auto &site_it : model.sites()) {
                ans.row(i++) = site_it.second->lambda();
              }
              return ans;
            })
        .def("site",
             [](PoissonFactorModel &model,
                const std::string &site_id) {
               return model.site(site_id);
             },
             py::arg("site_id"),
             "Args:\n\n"
             "  site_id:  string giving the ID of the site to be extracted.\n"
             "Returns:\n"
             "  The Site object managed by the model, corresponding to "
             "the requested ID.\n")
        .def("user",
             [](PoissonFactorModel &model,
                const std::string &user_id) {
               return model.visitor(user_id);
             },
             "Args:\n\n"
             "  user_id:  string giving the ID of the user to be extracted.\n"
             "Returns:\n"
             "  The Visitor object managed by the model, corresponding to "
             "the requested ID.\n")
        .def("set_prior_class_probabilities",
             [](PoissonFactorModel &model,
                const std::vector<std::string> &visitor_ids,
                const Matrix &prior_probs) {
               if (visitor_ids.size() != prior_probs.nrow()) {
                 report_error("The number of rows in 'prior_probs' must match "
                              "the length of 'visitor_ids'." );
               }
               if (prior_probs.ncol() != model.number_of_classes()) {
                 report_error("The number of columns in prior_probs must equal "
                              "the number of classes.");
               }
               for (size_t i = 0; i < visitor_ids.size(); ++i) {
                 model.visitor(
                     visitor_ids[i])->set_class_probabilities(
                         prior_probs.row(i));
               }
             },
             py::arg("visitor_ids"),
             py::arg("prior_probs"),
             "Args:\n\n"
             "  visitor_ids:  A list of strings giving the ID's of the "
             "visitors whose prior probabilities are to be set.\n"
             "  prior_probs: A Matrix, with one row per visitor giving the "
             "probability distribution of that visitor's latent class "
             "membership.\n"
             )
        .def("set_site_priors",
             [](PoissonFactorModel &model,
                const std::vector<std::string> &site_ids,
                const Matrix &prior_a,
                const Matrix &prior_b) {
               if (site_ids.size() != prior_a.nrow()) {
                 report_error("The number of rows in 'prior_a' must "
                              "match the length of 'site_ids'.");
               }
               if (site_ids.size() != prior_b.nrow()) {
                 report_error("The number of rows in 'prior_b' must "
                              "match the length of 'site_ids'.");
               }
               if (prior_a.ncol() != model.number_of_classes()) {
                 report_error("The number of columns in 'prior_a' must "
                              "match the number of latent classes.");
               }
               if (prior_b.ncol() != model.number_of_classes()) {
                 report_error("The number of columns in 'prior_b' must "
                              "match the number of latent classes.");
               }
               for (size_t i = 0; i < site_ids.size(); ++i) {
                 model.site(site_ids[i])->set_prior(
                     prior_a.row(i), prior_b.row(i));
               }
             },
             py::arg("site_ids"),
             py::arg("prior_a"),
             py::arg("prior_b"),
             "Args:\n\n"
             "  site_ids:  A list of strings giving the ID's of the sites "
             "whose prior is to be set.\n"
             "  prior_a:  A prior count representing visits to each site "
             "from users in each class.\n"
             "  prior_b:  A prior count representing the number of "
             "opportunites visitors in each class had to visit the site.  "
             "A 'prior exposure'.\n")
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
             py::arg("default_prior_class_membership_probabilities"),
             py::arg("seeding_rng") = GlobalRng::rng)
        .def("set_prior_class_probabilities",
             [](PoissonFactorModelPosteriorSampler &sampler,
                std::vector<std::string> &user_ids,
                const Matrix &probs) {
               if (user_ids.size() != probs.nrow()) {
                 std::ostringstream err;
                 err << "The vector of user ids's must have size ("
                     << user_ids.size()
                     << ") matching the number of rows in probs ("
                     << probs.nrow()
                     << ").";
                 report_error(err.str());
               }

               for (size_t i = 0; i < user_ids.size(); ++i) {
                 sampler.set_prior_class_probabilities(
                     user_ids[i], probs.row(i));
               }
             },
             py::arg("user_ids"),
             py::arg("probs"),
             "Args:\n\n"
             "  user_ids:  A list of user ID's to set.  Each ID must be "
             "present in the model's training data.  \n"
             "  probs:  A matrix giving the prior probability distribution "
             "of class membership for each user.  Each row of the matrix sums "
             "to 1, the number of rows must match the length of 'user_ids', "
             "and the number of columns must equal the number of latent "
             "classes.\n")
        ;

  }  // Module

}  // namespace BayesBoom
