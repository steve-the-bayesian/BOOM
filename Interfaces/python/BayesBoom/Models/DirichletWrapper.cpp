#include <pybind11/pybind11.h>

#include "Models/DirichletModel.hpp"
#include "cpputil/Ptr.hpp"
#include "uint.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;
  using BOOM::uint;

  void DirichletModel_def(py::module &boom) {

    py::class_<DirichletSuf,
               BOOM::Ptr<DirichletSuf>>(boom, "DirichletSuf")
        .def(py::init<uint>(),
             py::arg("dim"),
             "Create an empty set of sufficient statistics for the "
             "Dirichlet model of dimension 'dim'.\n\n"
             "Args:\n"
             "  dim:  The dimension of the distribution.\n")
        .def("update",
             [](DirichletSuf &suf, const Vector &probs) {
               suf.add_mixture_data(probs, 1.0);
             },
             py::arg("probs"),
             "Add data to the sufficient statistics.\n\n"
             "Args:\n"
             "  probs: A discrete probability distribution.\n")
        .def_property_readonly(
            "sumlog",
            [](const DirichletSuf &suf) {return suf.sumlog();},
            "The sums of the log probabilities in each dimension.")
        .def_property_readonly(
            "sample_size",
            [](const DirichletSuf &suf) {return suf.n();},
            "The number of observed distributions in the sample.")
        ;

    py::class_<DirichletModel,
               PriorPolicy,
               BOOM::Ptr<DirichletModel>>(boom, "DirichletModel",
                                          py::multiple_inheritance())
        .def(py::init(
            [](const Vector &prior_counts) {
              return new DirichletModel(prior_counts);
            }),
             py::arg("prior_counts"),
             "Args:\n\n"
             "  prior_counts:  A vector with all positive entries containing "
             "the prior counts in each dimension.\n")
        .def("__repr__",
             [](const DirichletModel &model) {
               std::ostringstream out;
               out << "A BOOM DirichletModel with prior counts "
                   << model.nu() << "\n";
               return out.str();
             })
        ;


  }  // Module

}  // namespace BayesBoom
