#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Models/CategoricalData.hpp"
#include "Models/MultinomialModel.hpp"
#include "Models/MultilevelMultinomialModel.hpp"
#include "Models/DirichletModel.hpp"
#include "Models/PosteriorSamplers/MultilevelMultinomialPosteriorSampler.hpp"

#include "cpputil/Ptr.hpp"
#include "cpputil/StringSplitter.hpp"
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
        .def("child_levels",
             [](const Taxonomy &taxonomy, const std::string &parent_level, char sep) {
               return taxonomy.child_node_names(parent_level, sep);
             },
             py::arg("parent_level"),
             py::arg("sep") = '/',
             "Args:\n\n"
             "  parent_level:  The taxonomy value whose children are "
             "desired.  If this is the empty string then the values from the "
             "top level of the taxonomy are returned.\n"
             "  sep:  The character separating taxonomy levels.\n\n"
             "Returns:\n"
             "  A list of strings giving the values of the taxonomy levels directly "
             "underneath the parent node.  The parent node is not repeated.\n")
        .def("pop_level",
             [](const Taxonomy &tax, const std::string &level, char sep) {
               StringSplitter split(std::string(1, sep));
               return split.pop_back(level);
             },
             py::arg("level"),
             py::arg("sep") = '/',
             "Args:\n\n"
             "  level:  A string describing a taxonomy level.\n"
             "  sep:  The separator between levels.\n"
             "\n"
             "Returns:\n"
             "  A pair: the leading (parent) and trailing (child) levels "
             "in the argument.\n\n"
             "Examples:  foo/bar/baz -> 'foo/bar', 'baz'\n"
             "           foo         -> '', 'foo'\n"
             "           foo/        -> 'foo', '' \n"
             )
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
        .def("add_data",
             [](MultilevelMultinomialModel &model,
                MultilevelCategoricalData *data_point) {
               model.add_data(Ptr<MultilevelCategoricalData>(data_point));
             })
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
              ans.push_back("");
              for (auto it = model.conditional_model_begin();
                   it != model.conditional_model_end(); ++it) {
                ans.push_back(it->first->path_from_root());
              }
              return ans;
            },
            "The list of taxonomy levels that have associated multinomial "
            "sub-models in this model.  This is all non-leaf taxonomy nodes, "
            "and a top level (denoted with the empty string) that notionally "
            "roots the taxonomy. \n")
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
