#include <pybind11/pybind11.h>

#include "Models/WishartModel.hpp"
#include "cpputil/Ptr.hpp"
#include "uint.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;
  using BOOM::uint;

  void WishartModel_def(py::module &boom) {

    py::class_<WishartModel,
               SpdModel,
               BOOM::Ptr<WishartModel>>(
                   boom, "WishartModel", py::multiple_inheritance())
        .def(py::init<double, SpdMatrix>(),
             py::arg("df"),
             py::arg("variance_estimate"),
             "Args:\n\n"
             "  df: The sample size parameter.  This must be larger than the "
             "dimension of 'variance_estimate'.\n"
             "  variance_estimate:  A symmetric positive definite matrix "
             "representing a variance estimate.\n")
        .def("__repr__",
             [](const Ptr<WishartModel> &model) {
               std::ostringstream out;
               out << "A BOOM WishartModel with sample size " << model->nu()
                   << std::endl
                   << "and sum of squares matrix ";
               if (model->dim() > 10) {
                 out << "too large to display." << std::endl;
               } else {
                 out << "\n" << model->sumsq();
               }
               return out.str();
             })
        ;

  }  // Module

}  // namespace BayesBoom
