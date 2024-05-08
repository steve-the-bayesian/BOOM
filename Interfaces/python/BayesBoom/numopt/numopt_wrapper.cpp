#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "LinAlg/Vector.hpp"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/SpdMatrix.hpp"

#include "numopt/LinearAssignment.hpp"
#include "numopt/ClassAssigner.hpp"

#include "distributions/rng.hpp"

namespace py = pybind11;

namespace BayesBoom {
  using namespace BOOM;

  void numopt_def(py::module &boom) {

    py::class_<LinearAssignment> (boom, "LinearAssignment")
        .def(py::init(
            [](const Matrix &cost) {
              return new LinearAssignment(cost);
            }),
             py::arg("cost"),
             "Args:\n\n"
             "  cost: A boom.Matrix of assigning task (column) j "
             "to worker (row) i.")
        .def("solve",
             [](LinearAssignment &lap) { return lap.solve(); },
             "Find the optimal solution.  The cost at the minimum is returned.")
        .def_property_readonly(
            "row_solution",
            [](const LinearAssignment &lap) {return lap.row_solution();})
        .def_property_readonly(
            "col_solution",
            [](const LinearAssignment &lap) {return lap.col_solution();})
        ;

    py::class_<ClassAssigner> (boom, "ClassAssigner")
        .def(py::init())
        .def("set_initial_temperature",
             [](ClassAssigner &assigner, double temp) {
               assigner.set_initial_temperature(temp);
             },
             py::arg("temp"),
             "Set the temperature to use at the start of a simulated "
             "annealing run.\n")
        .def("set_max_kl",
             [](ClassAssigner &assigner, double kl) {
               assigner.set_max_kl(kl);
             },
             py::arg(""),
             "Set the largest tolerable Kullback-Liebler divergence between "
             "the global target distribution and the empirical distribution "
             "of assignments.  The scale of a KL divergence is comparable to "
             "the scale of a likelihood ratio test.  So a KL divergence on "
             "the scale of 1 to 4 is potentially tolerable, but much smaller "
             "numbers can be chosen if closer agreement is desired.\n")
        .def("set_max_iterations",
             [](ClassAssigner &assigner, int niter) {
               assigner.set_max_iterations(niter);
             },
             py::arg("niter"),
             "Set the maximum number of simulated annealing iterations "
             "(function evaluations) per SA run.\n")
        .def("assign",
             [](ClassAssigner &assigner,
                const Matrix &marginal_posteriors,
                const Vector &global_target,
                RNG &rng) {
               return assigner.assign(
                   marginal_posteriors, global_target, rng);
             },
             py::arg("marginal_posteriors"),
             py::arg("global_target"),
             py::arg("rng") = ::BOOM::GlobalRng::rng,
             "Assign class memberships to a collection of object, while "
             "maintaining an empirical distribution of assigned classes "
             "close to a global target distribution.\n\n"
             "Args:\n"
             "  marginal_posteriors:  A boom.Matrix, with each row "
             "corresponding to an object, and each column to a potential class "
             "value.  marginal_posteriors[i, j] is the posterior probability "
             "that object i belongs to class j.\n"
             "  global_target: A boom.Vector containing the discrete "
             "probability distribution of class membership for the "
             "population.\n"
             "  rng: The random number generator used to drive the "
             "simulated annealing algorithm producing the assignment.\n"
             "\n"
             "Returns:\n"
             "A list of integers indicating the class assignment for each "
             "object.\n")
        .def_property_readonly(
            "kl",
            [](const ClassAssigner &assigner) {
              return assigner.kl();
            },
            "The Kullback-Liebler divergence between the target and "
            "empirical assignment distributions.")
        ;

  }

}  // namespace BayesBoom
