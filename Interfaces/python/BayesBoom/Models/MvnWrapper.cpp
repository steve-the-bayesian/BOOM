#include <pybind11/pybind11.h>

#include "Models/MvnBase.hpp"
#include "Models/MvnModel.hpp"
#include "Models/MvnGivenScalarSigma.hpp"
#include "cpputil/Ptr.hpp"
#include "uint.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;
  using BOOM::uint;

  void MvnModel_def(py::module &boom) {

    py::class_<MvnSuf,
               BOOM::Ptr<MvnSuf>>(boom, "MvnSuf")
        .def(py::init<int>(),
             py::arg("dim") = 0,
             "Args:\n"
             "  dim: The dimension of the model to be fit.")
        ;

    py::class_<MvnBase,
               BOOM::Ptr<MvnBase>>(boom, "MvnBase", py::multiple_inheritance())
        .def_property_readonly(
            "dim",
            &MvnBase::dim,
            "The dimension of the random variable being modeled.")
        .def_property_readonly(
            "mu",
            &MvnBase::mu,
            "Mean of the distribution")
        .def_property_readonly(
            "Sigma",
            &MvnBase::Sigma,
            "Variance of the distribution")
        .def_property_readonly(
            "siginv",
            &MvnBase::siginv,
            "Precision (inverse variance) of the distribution")
        ;

    py::class_<MvnBaseWithParams,
               MvnBase,
               BOOM::Ptr<MvnBaseWithParams>>(boom, "MvnBaseWithParams", py::multiple_inheritance())
        .def_property_readonly(
            "mu",
            &MvnBaseWithParams::mu,
            "Mean of the distribution.")
        .def_property_readonly(
            "Sigma",
            &MvnBaseWithParams::Sigma,
            "Variance of the distribution.")
        .def_property_readonly(
            "siginv",
            &MvnBaseWithParams::siginv,
            "Precision (inverse variance) of the distribution.")
        .def("set_mu",
             [](MvnBaseWithParams &model, const Vector &mu) {
               model.set_mu(mu);
             },
             "Args:\n"
             "  mu: Mean of the distribution (boom Vector).")
        .def("set_Sigma",
             [](MvnBaseWithParams &model, const SpdMatrix &Sigma) {
               model.set_Sigma(Sigma);
             },
             "Args:\n"
             "  Sigma: Variance of the distribution (boom SpdMatrix).")
        .def("set_siginv",
             [](MvnBaseWithParams &model, const SpdMatrix &siginv) {
               model.set_siginv(siginv);
             },
             "Args:\n"
             "  siginv: Precision (inverse variance) of the distribution (boom SpdMatrix).")
        ;

    py::class_<MvnModel,
               MvnBaseWithParams,
               PriorPolicy,
               BOOM::Ptr<MvnModel>>(boom, "MvnModel", py::multiple_inheritance())
        .def(py::init<uint, double, double>(),
             py::arg("dim"),
             py::arg("mu") = 0.0,
             py::arg("sd") = 1.0,
             "Create a mulitivariate normal model with identical means and "
             "standard deviations for each element.\n\n"
             "Args:\n"
             "  dim:  The dimension of the distribution.\n"
             "  mu:  The mean of each element.\n"
             "  sd:  The standard deviation of each element.\n")
        .def(py::init<Vector, SpdMatrix, bool>(),
             py::arg("mu"),
             py::arg("SigmaOrSiginv"),
             py::arg("ivar") = false,
             "Create a MvnModel by specifying moments with a vector and a matrix.\n\n"
             "Args:\n"
             "  mu:  mean of the distribution.\n"
             "  Sigma:  Variance or precision of the distribution.\n"
             "  ivar:  If True then Sigma is a precision (inverse variance).  \n"
             "         If False then Sigma is a variance matrix."
             )
        .def(py::init( [] (const Ptr<VectorParams> &mean,
                           const Ptr<SpdParams> &var) {
                         return new MvnModel(mean, var);
                       }),
             py::arg("mu"),
             py::arg("Sigma"),
             "Create a MvnModel using parameter objects.\n\n"
             "Args:\n"
             "  mu:  mean of the distribution.\n"
             "  Sigma:  Variance or precision of the distribution.\n"
             )
        .def("set_data",
             [](MvnModel &model, const Matrix &data) {
               int n = data.nrow();
               for (int i = 0; i < n; ++i) {
                 NEW(VectorData, data_point)(data.row(i));
                 model.add_data(data_point);
               }
             },
             "Assign a data set to the model.\n\n"
             "Args:\n"
             "  data: boom.Matrix.  Rows are observations.  Columns are variables."
             )
        .def_property_readonly(
            "mean_parameter",
            [](const MvnModel &model) { return model.Mu_prm();},
            "The VectorParams object representing the mean of the distribution.")
        .def_property_readonly(
            "variance_parameter",
            [](const MvnModel &model) { return model.Sigma_prm();},
            "The SpdParams object representing the variance of the distribution.")
        .def("mle",
             &MvnModel::mle,
             "Set parameters to their maximum likelihood estimates.")
        .def("sim",
             &MvnModel::sim,
             "Simulate a draw from the distribution.")
        .def("__repr__",
             [](const Ptr<MvnModel> &model) {
               std::ostringstream out;
               out << "A BOOM MvnModel with mean " << model->mu()
                   << std::endl
                   << "and variance matrix ";
               if (model->dim() > 20) {
                 out << "too large to display." << std::endl;
               } else {
                 out << "\n" << model->Sigma();
               }
               return out.str();
             })
        ;

    //=========================================================================

    py::class_<MvnGivenScalarSigma,
               MvnBase,
               PriorPolicy,
               Ptr<MvnGivenScalarSigma>>(boom, "MvnGivenScalarSigma", py::multiple_inheritance())
        .def(py::init<const SpdMatrix&, const Ptr<UnivParams> &>(),
             py::arg("ominv"),
             py::arg("sigsq"),
             ""
             )
        .def(py::init<const Vector &, const SpdMatrix&, const Ptr<UnivParams> &>(),
             py::arg("mu"),
             py::arg("ominv"),
             py::arg("sigsq"),
             ""
             )
        ;

  }  // Module

}  // namespace BayesBoom
