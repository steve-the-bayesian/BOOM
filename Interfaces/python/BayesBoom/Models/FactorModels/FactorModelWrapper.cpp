#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Models/FactorModels/PoissonFactorModel.hpp"
#include "Models/FactorModels/MultinomialFactorModel.hpp"

#include "Models/FactorModels/PosteriorSamplers/PoissonFactorModelIndependentGammaPosteriorSampler.hpp"
#include "Models/FactorModels/PosteriorSamplers/PoissonFactorHierarchicalSampler.hpp"
#include "Models/FactorModels/PosteriorSamplers/MultinomialFactorModelPosteriorSampler.hpp"

#include "Models/MvnModel.hpp"

#include "cpputil/Ptr.hpp"
#include "cpputil/report_error.hpp"
#include "uint.hpp"

#include <sstream>

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;
  using BOOM::uint;

  void FactorModel_def(py::module &boom) {

    //===========================================================================
    py::class_<FactorModels::VisitorBase,
               BOOM::Ptr<FactorModels::VisitorBase>>(
                   boom, "FactorModelVisitorBase")
        .def_property_readonly(
            "id",
            [](FactorModels::VisitorBase &visitor) {
              return visitor.id();
            },
            "The ID of the visitor.")
        .def_property_readonly(
            "imputed_class",
            [](FactorModels::VisitorBase &visitor) {
              return visitor.imputed_class_membership();
            },
            "The class membership indicator assigned to the Visitor "
            "in the most recent MCMC draw.")
        .def_property_readonly(
            "num_visits",
            [](FactorModels::VisitorBase &visitor) {
              return visitor.number_of_visits();
            },
            "The number of visits (including repeat visits) the user "
            "made to any site.\n")
        .def_property_readonly(
            "num_sites_visited",
            [](FactorModels::VisitorBase &visitor) {
              return visitor.number_of_sites_visited();
            },
            "The number of distinct sites visited at least one time.\n")
        ;
    
    //===========================================================================
    py::class_<FactorModels::PoissonVisitor,
               FactorModels::VisitorBase,
               Ptr<FactorModels::PoissonVisitor>>(
                   boom, "PoissonFactorModelVisitor")
        .def_property_readonly(
            "number_of_distinct_sites_visited",
            [](FactorModels::PoissonVisitor &visitor) {
              return visitor.sites_visited().size();
            },
            "The number of distinct sites visited by the user.")
        .def_property_readonly(
            "visits",
            [](FactorModels::PoissonVisitor &visitor) {
              std::vector<std::string> site_names;
              std::vector<int> counts;
              py::dict result;
              for (const auto &it : visitor.sites_visited()) {
                result[it.first->id().c_str()] = it.second;
              }
              return result;
            },
            "Returns a dict, keyed by site id, containing the counts of "
            "visits to each site.\n")
        ;

    //===========================================================================
    py::class_<FactorModels::MultinomialVisitor,
               FactorModels::VisitorBase,
               Ptr<FactorModels::MultinomialVisitor>>(
                   boom,
                   "MultinomialFactorModelVisitor")
        .def_property_readonly(
            "number_of_distinct_sites_visited",
            [](FactorModels::MultinomialVisitor &visitor) {
              return visitor.sites_visited().size();
            },
            "The number of distinct sites visited by the user.")
        .def_property_readonly(
            "visits",
            [](FactorModels::MultinomialVisitor &visitor) {
              std::vector<std::string> site_names;
              std::vector<int> counts;
              py::dict result;
              for (const auto &it : visitor.sites_visited()) {
                result[it.first->id().c_str()] = it.second;
              }
              return result;
            },
            "Returns a dict, keyed by site id, containing the counts of "
            "visits to each site.\n")
        ;

    //===========================================================================
    py::class_<FactorModels::SiteBase,
               BOOM::Ptr<FactorModels::SiteBase>>(
                   boom,
                   "FactorModelSiteBase")
        .def_property_readonly(
            "id",
            [](FactorModels::SiteBase &site) {
              return site.id();
            },
            "The ID of the site.")
        .def_property_readonly(
            "num_visits",
            [](FactorModels::SiteBase &site) {
              return site.number_of_visits();
            },
            "The number of times visitors have visited the site.  Repeat "
            "visits by the same visitor add to this count.\n")
        .def_property_readonly(
            "num_visitors",
            [](FactorModels::SiteBase &site) {
              return site.number_of_visitors();
            },
            "The number of distinct visitors that have visited the site "
            "one or more times.")
        ;
    
    //===========================================================================
    py::class_<FactorModels::PoissonSite,
               FactorModels::SiteBase,
               BOOM::Ptr<FactorModels::PoissonSite>>(
                   boom, "PoissonFactorModelSite")
        .def_property_readonly(
            "rate_params",
            [](FactorModels::PoissonSite &site) {
              return site.lambda();
            },
            "The poisson rate parameters for the Site.")
        // .def_property_readonly(
        //     "num_visits",
        //     [](FactorModels::PoissonSite &site) {
        //       size_t num_visits = 0;
        //       for (const auto &it : site.observed_visitors()) {
        //         num_visits += it.second;
        //       }
        //       return num_visits;
        //     },
        //     "The number of times the site was visited.")
        // .def_property_readonly(
        //     "num_visitors",
        //     [](FactorModels::PoissonSite &site) {
        //       return site.observed_visitors().size();
        //     },
        //     "The number of distinct visitors to the site.")
        .def_property_readonly(
            "visitor_counts",
            [](FactorModels::PoissonSite &site) {
              return site.visitor_counts();
            },
            "Returns a 2-column matrix containing the visit counts (column 0) "
            "and exposures (column 1) for each level of the latent category.\n")
        ;

    //===========================================================================
    py::class_<FactorModels::MultinomialSite,
               FactorModels::SiteBase,
               BOOM::Ptr<FactorModels::MultinomialSite>>(
                   boom,
                   "MultinomialFactorModelSite")
        .def_property_readonly(
            "probs",
            [](const FactorModels::MultinomialSite &site) {
              return site.visit_probs();
            },
            "The visitation probabilities for the site.")
        // .def_property_readonly(
        //     "num_visits",
        //     [](FactorModels::MultinomialSite &site) {
        //       size_t num_visits = 0;
        //       for (const auto &it : site.observed_visitors()) {
        //         num_visits += it.second;
        //       }
        //       return num_visits;
        //     },
        //     "The number of times the site was visited.")
        // .def_property_readonly(
        //     "num_visitors",
        //     [](FactorModels::MultinomialSite &site) {
        //       return site.observed_visitors().size();
        //     },
        //     "The number of distinct visitors to the site.")
        ;

    //===========================================================================
    py::class_<MultinomialFactorModel,
               PriorPolicy,
               BOOM::Ptr<MultinomialFactorModel>>(
                   boom, "MultinomialFactorModel", py::multiple_inheritance())
        .def(py::init(
            [](int num_classes) {
              return new MultinomialFactorModel(num_classes);
            }),
             py::arg("num_classes"),
             "Args:\n\n"
             "  num_classes:  The number of classes in the factor model.\n")
        .def("add_data",
             [](MultinomialFactorModel &model,
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
            [](MultinomialFactorModel &model) {
              return model.visitors().size();
            })
        .def_property_readonly(
            "num_sites",
            [](MultinomialFactorModel &model) {
              return model.sites().size();
            })
        .def_property_readonly(
            "site_ids",
            [](MultinomialFactorModel &model) {
              std::vector<std::string> site_ids;
              for (const auto &site_it : model.sites()) {
                site_ids.push_back(site_it.first);
              }
              return site_ids;
            },
            "The IDs of the sites, in the order they are stored by the model.\n")


        .def_property_readonly(
            "probs",
            [](MultinomialFactorModel &model) {
              size_t num_sites = model.sites().size();
              Matrix probs(num_sites, model.number_of_classes());
              int i = 0;
              for (const auto &site_it : model.sites()) {
                probs.row(i++) = site_it.second->visit_probs();
              }
              return probs;
            },
            "The visitation probabilities for each of the sites in the model. "
            " Each row is a site (in the same order as the 'site_id' "
            "attribute) and each column is a conditional visitation "
            "probability for one of the latent categories.\n")
        .def_property_readonly(
            "visitor_ids",
            [](MultinomialFactorModel &model) {
              std::vector<std::string> visitor_ids;
              for (const auto &visitor_it : model.visitors()) {
                visitor_ids.push_back(visitor_it.first);
              }
              return visitor_ids;
            },
            "The visitor ID's, in the order stored by the model.\n")
        .def_property_readonly(
            "imputed_classes",
            [](MultinomialFactorModel &model) {
              std::vector<int> classes;
              for (const auto &visitor_it : model.visitors()) {
                classes.push_back(
                    visitor_it.second->imputed_class_membership());
              }
              return classes;
            })
        .def_property_readonly(
            "site_params",
            [](MultinomialFactorModel &model) {
              Matrix ans(model.number_of_sites(), model.number_of_classes());
              size_t i = 0;
              for (const auto &site_it : model.sites()) {
                ans.row(i++) = site_it.second->visit_probs();
              }
              return ans;
            })
        .def("set_site_parameters",
             [](MultinomialFactorModel &model,
                const std::vector<std::string> &site_ids,
                const Matrix &parameters) {
               if (site_ids.size() != parameters.nrow()) {
                 std::ostringstream err;
                 err << "Each row in 'parameters' must correspond to a site id, "
                     << "but 'site_ids' has " << site_ids.size()
                     << " elements and 'parameters' has "
                     << parameters.nrow() << " rows.";
                 report_error(err.str());
               }
               for (size_t i = 0; i < site_ids.size(); ++i) {
                 Ptr<FactorModels::MultinomialSite> site = model.site(site_ids[i]);
                 if (site) {
                   site->set_probs(parameters.row(i));
                 } else {
                   std::ostringstream err;
                   err << "Site " << site_ids[i] << " was not found.";
                   report_warning(err.str());
                 }
               }
             },
             py::arg("site_ids"),
             py::arg("parameters"),
             "Args:\n\n"
             "  site_ids:  The id's of the sites whose parameters are to be "
             "set.\n"
             "  parameters:  Row i contains the visitation probabilities for "
             "the site with site_ids[i].\n")
        .def("site",
             [](MultinomialFactorModel &model,
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
             [](MultinomialFactorModel &model,
                const std::string &user_id) {
               return model.visitor(user_id);
             },
             "Args:\n\n"
             "  user_id:  string giving the ID of the user to be extracted.\n"
             "Returns:\n"
             "  The Visitor object managed by the model, corresponding to "
             "the requested ID.\n")
        .def("set_prior_class_probabilities",
             [](MultinomialFactorModel &model,
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
        .def("__repr__",
             [](MultinomialFactorModel &model) {
               std::ostringstream out;
               out << "A boom.MultinomialFactorModel with "
                   << model.number_of_classes()
                   << " latent classes.\n";
               return out.str();
             })
        ;
    
    //===========================================================================
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
        .def("set_site_parameters",
             [](PoissonFactorModel &model,
                const std::vector<std::string> &site_ids,
                const Matrix &parameters) {
               if (site_ids.size() != parameters.nrow()) {
                 std::ostringstream err;
                 err << "Each row in 'parameters' must correspond to a site id, "
                     << "but 'site_ids' has " << site_ids.size()
                     << " elements and 'parameters' has "
                     << parameters.nrow() << " rows.";
                 report_error(err.str());
               }
               for (size_t i = 0; i < site_ids.size(); ++i) {
                 Ptr<FactorModels::PoissonSite> site = model.site(site_ids[i]);
                 if (site) {
                   site->set_lambda(parameters.row(i));
                 } else {
                   std::ostringstream err;
                   err << "Site " << site_ids[i] << " was not found.";
                   report_warning(err.str());
                 }
               }
             },
             py::arg("site_ids"),
             py::arg("parameters"),
             "Args:\n\n"
             "  site_ids:  The id's of the sites whose parameters are to be "
             "set.\n"
             "  parameters:  Row i contains the intensity parameters for site "
             "site_ids[i].\n")
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
        .def("__repr__",
             [](PoissonFactorModel &model) {
               std::ostringstream out;
               out << "A boom.PoissonFactorModel with "
                   << model.number_of_classes()
                   << " latent classes.\n";
               return out.str();
             })
        ;

    //===========================================================================
    py::class_<MultinomialFactorModelPosteriorSampler,
               PosteriorSampler,
               BOOM::Ptr<MultinomialFactorModelPosteriorSampler>>(
                   boom,
                   "MultinomialFactorModelPosteriorSampler")
        .def(py::init(
            [](MultinomialFactorModel *model,
               const Vector &default_prior_class_membership_probabilities,
               RNG &seeding_rng) {
              return new MultinomialFactorModelPosteriorSampler(
                  model,
                  default_prior_class_membership_probabilities,
                  seeding_rng);
            }),
             py::arg("model"),
             py::arg("default_prior_class_membership_probabilities"),
             py::arg("seeding_rng") = BOOM::GlobalRng::rng,
             "Args:\n\n"
             "  model:  The model to be sampled.\n"
             "  default_prior_class_membership_probabilities:  The prior "
             "distribution over a user's class membership to use when a "
             "user-specific prior has not been specified for that user.\n"
             "  seeding_rng:  The random number generator used to seed this "
             "sampler's RNG.\n")
        .def("set_prior_class_probabilities",
             [](MultinomialFactorModelPosteriorSampler &sampler,
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
    //===========================================================================
    py::class_<PoissonFactorPosteriorSamplerBase,
               PosteriorSampler,
               BOOM::Ptr<PoissonFactorPosteriorSamplerBase>>(
                   boom,
                   "PoissonFactorPosteriorSamplerBase",
                   py::multiple_inheritance())
        .def("prior_class_probabilities",
             [](PoissonFactorPosteriorSamplerBase &sampler,
                const std::string &user_id) {
               return sampler.prior_class_probabilities(user_id);
             },
             "Return the prior class probabilities for the requested user.  "
             "If the user is not found then the default prior is returned.\n")
        .def("set_prior_class_probabilities",
             [](PoissonFactorPosteriorSamplerBase &sampler,
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
        .def("impute_visitors",
             [](PoissonFactorPosteriorSamplerBase &sampler) {
               sampler.impute_visitors();
             })
        ;
        
    
    //===========================================================================
    py::class_<PoissonFactorHierarchicalSampler,
               PoissonFactorPosteriorSamplerBase,
               BOOM::Ptr<PoissonFactorHierarchicalSampler>>(
                   boom,
                   "PoissonFactorHierarchicalSampler",
                   py::multiple_inheritance())
        .def(py::init(
            [](PoissonFactorModel *model,
               const Vector &default_prior_class_membership_probabilities,
               const Vector &prior_mean,
               double kappa,
               const SpdMatrix &Sigma_guess,
               double prior_df,
               int MH_threshold,
               RNG &seeding_rng) {
              return new PoissonFactorHierarchicalSampler(
                  model, default_prior_class_membership_probabilities,
                  prior_mean, kappa, Sigma_guess, prior_df, MH_threshold,
                  seeding_rng);
            }),
             py::arg("model"),
             py::arg("default_prior_class_membership_probabilities"),
             py::arg("prior_mean"),
             py::arg("kappa"),
             py::arg("Sigma_guess"),
             py::arg("prior_df"),
             py::arg("MH_threshold") = 10,
             py::arg("seeding_rng") = GlobalRng::rng,
  " Posterior sampler for a PoissonFactorModel based on a hierarchical prior\n"
  " that borrows strength across sites for estimating a site's visitation\n"
  " profile.\n"
  "\n"
  " The PoissonFactorModel describes a collection of visitors as belonging to\n"
  " one of K latent categories.  Those visitors visit sites according to a\n"
  " Poisson process with category dependent rates lambda[j, 0], ...,\n"
  " lambda[j, K-1] (for site j).\n"
  "\n"
  " This sampler views a site's 'lambda' (category-specific intensity)\n"
  " parameters in terms of the total visitation rate alpha = sum(lambda) and\n"
  " a visitation profile pi = lambda / alpha.  In other words, pi is a\n"
  " discrete probability distribution.  This sampler assumes a flat prior on\n"
  " alpha, but assumes MultinomialLogit(pi[j]) ~ N(mu, Sigma).  The reference\n"
  " category for the multinomial logit transformation is pi[0].\n"
  "\n"
  " The prior parameters then are the parameters of a normal-invserse-Wishart\n"
  " prior on mu and Sigma.\n"
  "\n"
  " - Sigma_guess is a guess at the variance matrix describing variation in\n"
  "     the multinomial logits across sites.  The scale of the multinomial\n"
  "     logit tranformation is such that 1 unit is a fairly large shift, so a\n"
  "     prior guess at Sigma with unit diagonal emphasizes heterogeneity.\n"
  "\n"
  " - prior_df: A scalar 'prior sample size' indicating how much weight\n"
  "     should be assigned to Sigma_guess.  The relevant sample size here is\n"
  "     the number of sites in the PoissonFactorModel.  prior_df should be\n"
  "     larger than the dimension of Sigma.\n"
  "\n"
  " - prior_mean: The prior guess at the multinomial logit values.  Absent\n"
  "     strong prior knowledge about the rates at which various groups\n"
  "     generally visit sites, a vector of all 0's is an appropriate central\n"
  "     value.\n"
  "\n"
  " - kappa: A prior sample size indicating how many observations worth of\n"
  "     weight should be given to mu. The relevant sample size for comparison\n"
  "     is the number of sites in the PoissonFactorModel.\n")
        .def_property_readonly(
            "hyperprior",
            [](PoissonFactorHierarchicalSampler &sampler) {
              return sampler.hyperprior();
            },
            "The boom.MvnModel hyperprior distribution on the multinomial "
            "logit parameters.")
        .def(
            "set_MH_threshold",
            [](PoissonFactorHierarchicalSampler &sampler, int threshold) {
              sampler.set_MH_threshold(threshold);
            })
        .def_property_readonly(
            "sampling_report",
            [](PoissonFactorHierarchicalSampler &sampler) {
              return sampler.sampling_report();
            })
        ;

    
    //===========================================================================
    py::class_<PoissonFactorModelIndependentGammaPosteriorSampler,
               PoissonFactorPosteriorSamplerBase,
               BOOM::Ptr<PoissonFactorModelIndependentGammaPosteriorSampler>>(
                   boom, "PoissonFactorModelIndependentGammaPosteriorSampler",
                   py::multiple_inheritance())
        .def(py::init(
            [](PoissonFactorModel *model,
               const Vector &prior_class_membership_probabilities,
               const Vector &default_prior_a,
               const Vector &default_prior_b,
               RNG &seeding_rng) {
              if (default_prior_a.size() != default_prior_b.size()) {
                report_error("default_prior_a and default_prior_b must "
                             "be the same size.");
              }
              std::vector<Ptr<GammaModelBase>> default_site_priors;
              for (int i = 0; i < default_prior_a.size(); ++i) {
                default_site_priors.push_back(new GammaModel(
                    default_prior_a[i], default_prior_b[i]));
              }
              return new PoissonFactorModelIndependentGammaPosteriorSampler(
                  model,
                  prior_class_membership_probabilities,
                  default_site_priors,
                  seeding_rng);
            }),
             py::arg("model"),
             py::arg("default_prior_class_membership_probabilities"),
             py::arg("default_prior_a"),
             py::arg("default_prior_b"),
             py::arg("seeding_rng") = GlobalRng::rng)
        .def("set_site_priors",
             [](PoissonFactorModelIndependentGammaPosteriorSampler &sampler,
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
               if (prior_a.ncol() != sampler.number_of_classes()) {
                 report_error("The number of columns in 'prior_a' must "
                              "match the number of latent classes.");
               }
               if (prior_b.ncol() != sampler.number_of_classes()) {
                 report_error("The number of columns in 'prior_b' must "
                              "match the number of latent classes.");
               }
               for (size_t i = 0; i < site_ids.size(); ++i) {
                 std::vector<Ptr<GammaModelBase>> site_priors;
                 for (int j = 0; j < sampler.number_of_classes(); ++j) {
                   site_priors.push_back(new GammaModel(
                       prior_a(i, j), prior_b(i, j)));
                 }
                 sampler.set_intensity_prior(site_ids[i], site_priors);
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
        .def("draw_site_parameters",
             [](PoissonFactorModelIndependentGammaPosteriorSampler &sampler) {
               sampler.draw_site_parameters();
             })
        ;

  }  // Module

}  // namespace BayesBoom
