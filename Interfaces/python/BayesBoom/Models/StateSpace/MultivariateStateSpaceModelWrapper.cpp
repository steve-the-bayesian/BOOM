#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Models/StateSpace/Multivariate/MultivariateStateSpaceRegressionModel.hpp"

#include "Models/StateSpace/PosteriorSamplers/StateSpacePosteriorSampler.hpp"
#include "Models/StateSpace/PosteriorSamplers/StateSpaceStudentPosteriorSampler.hpp"
#include "Models/StateSpace/PosteriorSamplers/StateSpaceLogitPosteriorSampler.hpp"
#include "Models/StateSpace/PosteriorSamplers/StateSpacePoissonPosteriorSampler.hpp"

#include "cpputil/Ptr.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;
  void MultivariateStateSpaceModel_def(py::module &boom) {
    py::class_<MultivariateStateSpaceModelBase,
               Model,
               BOOM::Ptr<MultivariateStateSpaceModelBase>>(
                   boom,
                   "MultivariateStateSpaceModelBase",
                   py::multiple_inheritance())
        .def_property_readonly(
            "time_dimension",
            [](const MultivariateStateSpaceModelBase &model) {
              return model.time_dimension();
            },
            "The number of time points in the model training data.")
        .def_property_readonly(
            "state_dimension",
            [](const MultivariateStateSpaceModelBase &model) {
              return model.state_dimension();
            },
            "The dimension of the state vector shared across all time series.")
        .def_property_readonly(
            "number_of_state_models",
            [](const MultivariateStateSpaceModelBase &model) {
              return model.number_of_state_models();
            },
            "The number state models defining the shared state vector.")
        .def_property_readonly(
            "log_likelihood",
            [](MultivariateStateSpaceModelBase &model) {
              return model.log_likelihood();
            },
            "The log likelihood under the current set of model parameters.")
        .def("state_contributions",
             [](const MultivariateStateSpaceModelBase &model, int which_state_model) {
               return model.state_contributions(which_state_model);
             },
             py::arg("which_state_model"),
             "Args:\n"
             "  which_state_model: The state model whose contribution is desired.\n"
             "\n"
             "Returns:\n"
             "  A Matrix.  Element (t, d) is the contrubtion of the specified "
             "state model to series d at time t.")
        ;

    py::class_<ConditionallyIndependentMultivariateStateSpaceModelBase,
               MultivariateStateSpaceModelBase,
               BOOM::Ptr<ConditionallyIndependentMultivariateStateSpaceModelBase>>(
                   boom,
                   "ConditionallyIndependentMultivariateStateSpaceModelBase",
                   py::multiple_inheritance())
        ;

    py::class_<MultivariateStateSpaceRegressionModel,
               ConditionallyIndependentMultivariateStateSpaceModelBase,
               BOOM::Ptr<MultivariateStateSpaceRegressionModel>>(
                   boom,
                   "MultivariateStateSpaceRegressionModel",
                   py::multiple_inheritance())
        .def(py::init(
            [](int xdim, int nseries) {
              return new BOOM::MultivariateStateSpaceRegressionModel(
                  xdim, nseries);
            }),
             py::arg("xdim"),
             py::arg("nseries"),
             "Args:\n"
             "  xdim:  The dimension of the predictor variables.\n"
             "  nseries: The number of time series being modeled.\n")
        .def("mle",
             [](MultivariateStateSpaceRegressionModel &model,
                double epsilon,
                int max_tries) {
               return model.mle(epsilon, max_tries);
             },
             py::arg("epsilon"),
             py::arg("max_tries") = 500,
             "Set model parameters to their maximum likelihood estimates.\n"
             "\n"
             "Args:\n"
             "  epsilon: A small positive number.  Absolute changes to log likelihood\n"
             "    less than this value indicate that the algorithm has converged.\n"
             "  max_tries:  Stop trying to optimzize after this many iterations.\n"
             "\n"
             "Returns:\n"
             "  The log likelihood value at the maximum.\n"
             "\n"
             "Effects:\n"
             "  Model parameters are set to the maximum likelihood estimates.\n")
        ;
  }
}  // namespace BayesBoom
