#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Models/StateSpace/Multivariate/StateModels/SharedLocalLevel.hpp"

#include "Models/PosteriorSamplers/ZeroMeanGaussianConjSampler.hpp"
#include "Models/PosteriorSamplers/ZeroMeanMvnIndependenceSampler.hpp"
#include "Models/StateSpace/PosteriorSamplers/DynamicRegressionPosteriorSampler.hpp"
#include "Models/StateSpace/PosteriorSamplers/StudentLocalLinearTrendPosteriorSampler.hpp"
#include "Models/StateSpace/StateModels/PosteriorSamplers/GeneralSeasonalLLTPosteriorSampler.hpp"

#include "Models/TimeSeries/ArModel.hpp"

#include "Models/StateSpace/StateModels/Holiday.hpp"

#include "cpputil/Date.hpp"
#include "cpputil/Ptr.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;

  void MultivariateStateModel_def(py::module &boom) {


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
        .def_property_readonly(
            "state_dimension",
            [](const SharedLocalLevelStateModelBase &state_model) {
              return state_model.state_dimension();
            },
            "The number of dimensions this component adds to the shared "
            "state vector.")
        ;

  }  // MultivariateStateModel_def

}  // namespace BayesBoom
