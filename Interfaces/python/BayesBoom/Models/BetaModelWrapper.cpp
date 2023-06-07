#include <pybind11/pybind11.h>

#include "Models/BetaModel.hpp"
#include "cpputil/Ptr.hpp"
#include "uint.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;
  using BOOM::uint;

  void BetaModel_def(py::module &boom) {

    py::class_<BetaSuf,
               BOOM::Ptr<BetaSuf>>(boom, "BetaSuf")
        .def(py::init([](){return new BetaSuf();}),
             "Create an empty set of sufficient statistics for the "
             "Beta distribution.\n")
        .def("update",
             [](BetaSuf &suf, double p) {
               suf.update_raw(p);
             },
             py::arg("p"),
             "Add data to the sufficient statistics.\n\n"
             "Args:\n"
             "  p: An observed data point between 0 and 1.\n")
        .def_property_readonly(
            "sumlog",
            [](const BetaSuf &suf) {return suf.sumlog();},
            "The sums of the log probabilities.")
        .def_property_readonly(
            "sumlogc",
            [](const BetaSuf &suf) {return suf.sumlogc();},
            "The sums of the log probability complements.")
        .def_property_readonly(
            "sample_size",
            [](const BetaSuf &suf) {return suf.n();},
            "The number of observed distributions in the sample.")
        ;

    py::class_<BetaModel,
               PriorPolicy,
               DiffDoubleModel,
               BOOM::Ptr<BetaModel>>(boom, "BetaModel",
                                     py::multiple_inheritance())
        .def(py::init(
            [](double a, double b) {
              return new BetaModel(a, b);
            }),
             py::arg("a") = 1.0,
             py::arg("b") = 1.0,
             "Args:\n\n"
             "  a:  A positive real number interpretable as a prior success "
             "count.\n"
             "  b:  A positive real number interpretable as a prior failure "
             "count.\n")
        .def("__repr__",
             [](const BetaModel &model) {
               std::ostringstream out;
               out << "A BOOM BetaModel with parameters ("
                   << model.a() << ", " << model.b() << ")\n";
               return out.str();
             })
        ;


  }  // Module

}  // namespace BayesBoom
