#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Models/CategoricalData.hpp"
#include "Models/MultinomialModel.hpp"
#include "Models/MultilevelMultinomialModel.hpp"
#include "Models/DirichletModel.hpp"
#include "Models/PosteriorSamplers/MultilevelMultinomialPosteriorSampler.hpp"

#include "cpputil/Ptr.hpp"
#include "uint.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;
  using BOOM::uint;

  void MultilevelMultinomialModel_def(py::module &boom) {

    py::class_<Taxonomy,
               BOOM::Ptr<Taxonomy>>(boom, "Taxonomy")
        .def(py::init(
            [](const std::vector<std::string> &levels,
               char sep) {
              return new Taxonomy(levels, sep);
            }),
             py::arg("levels"),
             py::arg("sep") = '/',
             "Args:\n\n"
             "  levels: element [i] is a single taxonomy element of the form "
             "shopping/clothes/shoes.\n\n"
             "  sep: The field delimiter used to separate values in 'levels'.")
        .def_property_readonly(
            "number_of_leaves",
            [](const Taxonomy &tax) {
              return tax.number_of_leaves();
            })
        .def_property_readonly(
            "interior_nodes",
            [](const Taxonomy &tax) {
              std::vector<std::string> ans;
              for (auto it = tax.begin(); it != tax.end(); ++it) {
                const Ptr<TaxonomyNode> &node(*it);
                if (!node->is_leaf()) {
                  ans.push_back(node->path_from_root());
                }
              }
              return ans;
            },
            "A list of all interior (non-leaf) nodes.  There is no 'top' node.")
        .def_property_readonly(
            "leaves",
            [](const Taxonomy &tax) {
              std::vector<std::string> ans;
              for (auto it = tax.begin(); it != tax.end(); ++it) {
                const Ptr<TaxonomyNode> &node(*it);
                if (node->is_leaf()) {
                  ans.push_back(node->path_from_root());
                }
              }
              return ans;
            },
            "A list of all leaf nodes in the taxonomy.\n")
        ;

    py::class_<MultilevelCategoricalData,
               Data,
               BOOM::Ptr<MultilevelCategoricalData>>(
                   boom,
                   "MultilevelCategoricalData")
        .def(py::init(
            [](const Ptr<Taxonomy> &taxonomy,
               const std::string &level,
               char sep) {
              return new MultilevelCategoricalData(taxonomy, level, sep);
            }),
             py::arg("taxonomy"),
             py::arg("level"),
             py::arg("sep") = '/',
             "Args:\n\n"
             "  taxonomy:  A Taxonomy object describing the allowed data "
             "values.\n"
             "  level:  A string of the form shopping/clothes/shoes giving "
             "the specific value of this observation. The level need not go "
             "to maximal depth.\n"
             "  sep:  The character used to delimit fields in 'level'.\n")
        ;
    
    py::class_<MultilevelMultinomialModel,
               PriorPolicy,
               MixtureComponent,
               BOOM::Ptr<MultilevelMultinomialModel>>(
                   boom,
                   "MultilevelMultinomialModel",
                   py::multiple_inheritance())
        .def(py::init(
            [](const Ptr<Taxonomy> &taxonomy) {
              return new MultilevelMultinomialModel(taxonomy);
            }),
             py::arg("taxonomy"),
             "Args:\n\n"
             "  taxonomy:  A Taxonomy object describing the allowed set of values.\n")
        .def("add_data",
             [](MultilevelMultinomialModel &model,
                const std::vector<std::string> &values,
                char sep) {
               Ptr<Taxonomy> taxonomy(model.taxonomy());
               for (const auto &el : values) {
                 NEW(MultilevelCategoricalData, data_point)(taxonomy, el, sep);
                 model.add_data(data_point);
               }
             },
             py::arg("values"),
             py::arg("sep") = '/',
             "Args:\n\n"
             "   values:  A sequence of strings indicating the taxonomy levels "
             "of each data point.\n"
             "   sep:  The field separator.\n")
        .def_property_readonly(
            "parameters",
            [](const MultilevelMultinomialModel &model) {
              std::vector<Vector> ans;
              ans.push_back(model.top_level_model()->pi());
              for (auto it = model.conditional_model_begin();
                   it != model.conditional_model_end(); ++it) {
                ans.push_back(it->second->pi());
              }
              return ans;
            },
            "A list of Vectors containing model parameters (conditional "
            "probabilities).  The list elements occur in the same order as "
            "'model_levels'.\n")
        .def_property_readonly(
            "model_levels",
            [](const MultilevelMultinomialModel &model) {
              std::vector<std::string> ans;
              ans.push_back("top");
              for (auto it = model.conditional_model_begin();
                   it != model.conditional_model_end(); ++it) {
                ans.push_back(it->first->path_from_root());
              }
              return ans;
            },
            "The list of taxonomy levels that have associated multinomial "
            "sub-models in this model.  This is all non-leaf taxonomy nodes, "
            "and a 'top' level that notionally roots the taxonomy. \n")
        .def_property_readonly(
            "top_level_model",
            [](MultilevelMultinomialModel &model) {
              return model.top_level_model();
            })
        .def("conditional_model",
             [](MultilevelMultinomialModel &model,
                std::string &level,
                char sep) {
               return model.conditional_model(level, sep);
             },
             py::arg("level"),
             py::arg("sep") = '/',
             "Args:\n\n"
             "   level:  The (parent) level of the taxonomy.  The returned "
             "model describes the child levels of the specfied taxonomy "
             "level.\n"
             "   sep:  The field separator.\n")
        .def("__repr__",
             [](const MultilevelMultinomialModel &model) {
               std::ostringstream out;
               out << "A BOOM MultilevelMultinomialModel. \n";
               return out.str();
             })
        ;

    py::class_<MultilevelMultinomialPosteriorSampler,
               PosteriorSampler,
               Ptr<MultilevelMultinomialPosteriorSampler>>(
                   boom, "MultilevelMultinomialPosteriorSampler")
        .def(py::init(
            [](MultilevelMultinomialModel *model,
               double default_prior_sum,
               RNG &seeding_rng) {
              return new MultilevelMultinomialPosteriorSampler(
                  model, default_prior_sum, seeding_rng);
            }),
             py::arg("model"),
             py::arg("default_prior_sum"),
             py::arg("seeding_rng") = GlobalRng::rng,
             "Args:\n\n"
             "  model:  The model to be sampled.\n"
             "  default_prior_sum:  See below.\n"
             "  seeding_rng:  The random number generator used to seed "
             "this sampler's RNG.\n" )
        ;

  }  // Module

}  // namespace BayesBoom
