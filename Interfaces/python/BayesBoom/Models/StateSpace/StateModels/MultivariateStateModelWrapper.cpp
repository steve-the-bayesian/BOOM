#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Models/StateSpace/Multivariate/StateModels/SharedLocalLevel.hpp"
#include "Models/StateSpace/Multivariate/StateModels/ScalarStateModelAdapter.hpp"

#include "cpputil/Ptr.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;

  void MultivariateStateModel_def(py::module &boom) {

    py::class_<SharedStateModel,
               StateModelBase,
               BOOM::Ptr<SharedStateModel>>(
                   boom, "SharedStateModel", py::multiple_inheritance())
        ;

    py::class_<SharedLocalLevelStateModelBase,
               SharedStateModel,
               BOOM::Ptr<SharedLocalLevelStateModelBase>>(
                   boom, "SharedLocalLevelStateModelBase", py::multiple_inheritance())
        .def_property_readonly(
            "number_of_factors",
            [](const SharedLocalLevelStateModelBase &state_model) {
              return state_model.number_of_factors();
            },
            "The number of factors in the model.")
        // .def_property_readonly(
        //     "state_dimension",
        //     [](const SharedLocalLevelStateModelBase &state_model) {
        //       return state_model.state_dimension();
        //     },
        //     "The number of dimensions this component adds to the shared "
        //     "state vector.")
        ;


    py::class_<ScalarStateModelMultivariateAdapter,
               SharedStateModel,
               Ptr<ScalarStateModelMultivariateAdapter>>(
                   boom, "ScalarStateModelMultivariateAdapter")
        .def("add_state",
             [](ScalarStateModelMultivariateAdapter *base,
                StateModel &state_model) {
               base->add_state(Ptr<StateModel>(&state_model));
             },
             "Add 'state_model' to the state tracked by the adapter.")
        ;

    py::class_<ConditionallyIndependentScalarStateModelMultivariateAdapter,
               ScalarStateModelMultivariateAdapter,
               Ptr<ConditionallyIndependentScalarStateModelMultivariateAdapter>>(
                   boom,
                   "ConditionallyIndependentScalarStateModelMultivariateAdapter")
        .def(py::init(
            [](ConditionallyIndependentMultivariateStateSpaceModelBase *host,
               int nseries) {
              using Adapter =
                  ConditionallyIndependentScalarStateModelMultivariateAdapter;
              return new Adapter(host, nseries);
            }),
             py::arg("host"),
             py::arg("nseries"),
             "Args:\n\n"
             "  host:  The multivariate state space model in which this object "
             "is a component of state.\n"
             "  nseries:  The number of series being modeled.\n")
        ;


  }  // MultivariateStateModel_def

}  // namespace BayesBoom
