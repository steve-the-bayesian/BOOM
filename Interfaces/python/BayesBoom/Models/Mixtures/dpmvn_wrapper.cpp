#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Models/Mixtures/DirichletProcessMvnModel.hpp"
#include "Models/Mixtures/PosteriorSamplers/DirichletProcessMvnCollapsedGibbsSampler.hpp"
#include "cpputil/Ptr.hpp"
#include "uint.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;
  using BOOM::uint;

  void DirichletProcessMvn_def(py::module &boom) {

    py::class_<DirichletProcessMvnModel,
               PriorPolicy,
               BOOM::Ptr<DirichletProcessMvnModel>>(
                   boom, "DirichletProcessMvnModel", py::multiple_inheritance())
        .def(py::init(
            [](int dim, double concentration_parameter) {
              return new DirichletProcessMvnModel(dim, concentration_parameter);
            }),
             py::arg("dim"),
             py::arg("concentration_parameter") = 1.0,
             "A Dirichlet process mixture of multivariate normal "
             "distributions.\n\n"
             "Args:\n\n"
             "  dim:  The dimension of the data to be modeled.\n"
             "  concentration_parameter:  The concentration parameter of "
             "the Dirichlet process.  Smaller values lead to fewer clusters.\n")
        .def_property_readonly(
            "number_of_clusters",
            [](const DirichletProcessMvnModel &model) {
              return model.number_of_clusters();
            })
        .def_property_readonly(
            "concentration_parameter",
            [](const DirichletProcessMvnModel &model) {
              return model.alpha();
            })
        .def("set_concentration_parameter",
             [](DirichletProcessMvnModel &model, double alpha) {
               model.set_alpha(alpha);
             })
        .def("add_data",
             [](DirichletProcessMvnModel &model,
                const Matrix &data) {
               for (int i = 0; i < data.nrow(); ++i) {
                 model.add_data(new VectorData(data.row(i)));
               }
             })
        .def("add_data",
             [](DirichletProcessMvnModel &model,
                const Vector &data) {
               model.add_data(new VectorData(data));
             })
        .def("cluster",
            [](const DirichletProcessMvnModel &model, int i) {
              return model.cluster(i);
            },
            py::arg("cluster"),
            "Args:\n\n"
            "  cluster:  The index of the cluster to return.\n\n"
            "Returns:\n"
            "  A boom.MvnModel describing the requested cluster.\n")
        .def_property_readonly("cluster_labels",
             [](const DirichletProcessMvnModel &model) {
               return model.cluster_indicators();
             })
        .def_property_readonly(
            "log_likelihood",
            [](const DirichletProcessMvnModel &model) {
              return model.log_likelihood();
            })
        .def("__repr__",
             [](const DirichletProcessMvnModel &model) {
               std::ostringstream out;
               out << "A BOOM DirichletProcessMvnModel of dimension "
                   << model.dim()
                   << " with concentration parameter " << model.alpha()
                   << ".\n";
               return out.str();
             })
        ;

    py::class_<DirichletProcessMvnCollapsedGibbsSampler,
               PosteriorSampler,
               BOOM::Ptr<DirichletProcessMvnCollapsedGibbsSampler>>(
                   boom, "DirichletProcessMvnCollapsedGibbsSampler",
                   py::multiple_inheritance())
        .def(py::init(
            [](DirichletProcessMvnModel &model,
               const Ptr<MvnGivenSigma> &mean_base_measure,
               const Ptr<WishartModel> &precision_base_measure,
               RNG &seeding_rng) {
              return new DirichletProcessMvnCollapsedGibbsSampler(
                  &model,
                  mean_base_measure,
                  precision_base_measure,
                  seeding_rng);
            }),
             py::arg("model"),
             py::arg("mean_base_measure"),
             py::arg("precision_base_measure"),
             py::arg("seeding_rng") = GlobalRng::rng,
             "Args:\n\n"
             "  model:  The DirichletProcessMvnModel to be sampled.\n"
             "  mean_base_measure:  A MvnGivenSigma model describing the base "
             "measure for the mean of each mixture component.\n"
             "  precision_base_measure:  A WishartModel describing the base "
             "measure for the precision of each mixture component.\n"
             "  seeding_rng:  A random number generator used to seed the "
             "RNG of this sampler object.\n")
        ;

  }  // Module

}  // namespace BayesBoom
