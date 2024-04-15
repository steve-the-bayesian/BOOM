#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <cpputil/Ptr.hpp>
namespace py = pybind11;
using namespace BOOM;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  // Forward definitions of all the class definitions to be added in other
  // files.  Each of these is defined in a local cpp file, but invoked here.
  // That way all the definitions occur within the same module.
  void distribution_def(py::module &);
  void cpputil_def(py::module &);
  void LinAlg_def(py::module &);
  void boom_math_def(py::module &);
  void stats_def(py::module &);
  void numopt_def(py::module &);
  void Model_def(py::module &);
  void Data_def(py::module &);
  void Parameter_def(py::module &);
  void DirichletModel_def(py::module &);
  void BetaModel_def(py::module &);

  //   void BinomialModel_def(py::module &);
  void BetaBinomialModel_def(py::module &);
  void MultinomialModel_def(py::module &);

  void GaussianModel_def(py::module &);
  void GammaModel_def(py::module &);
  void UniformModel_def(py::module &m);
  void MvnModel_def(py::module &);
  void WishartModel_def(py::module &);

  void GlmModel_def(py::module &);
  void GpModel_def(py::module &);
  void MultinomialLogitModel_def(py::module &);

  void Imputation_def(py::module &);
  void TimeSeries_def(py::module &);
  void StateSpaceModel_def(py::module &);
  void StateModel_def(py::module &);
  void DynamicRegressionModel_def(py::module &);

  void MultivariateStateSpaceModel_def(py::module &);
  void MultivariateStateModel_def(py::module &);
  void DirichletProcessMvn_def(py::module &);
  void BetaBinomialMixture_def(py::module &);

  void FactorModel_def(py::module &);

  void test_utils_def(py::module &);

  PYBIND11_MODULE(_boom, boom) {
    boom.doc() = "BOOM stands for 'Bayesian Object Oriented Models'.  "
        "It is also the sound your computer makes when it crashes.\n\n"
        "BOOM is a C++ library written by Steven L. Scott.  It is a standalone "
        "C++ library, but also the engine behind a couple of useful R packages "
        "and (now) some python.\n\n"
        "The BayesBoom.boom package is intended for library writers and should "
        "probably not be used directly."
        ;

    // The functions declared above need to be called here to add their contents
    // to the module.
    cpputil_def(boom);
    distribution_def(boom);
    LinAlg_def(boom);

    Data_def(boom);
    // stats includes DataTable, which inherits from Data.  Thus it must be
    // defined after Models, where the Data class is defined.

    boom_math_def(boom);
    stats_def(boom);
    numopt_def(boom);

    Model_def(boom);
    Parameter_def(boom);
    BetaModel_def(boom);
    DirichletModel_def(boom);

    //    BinomialModel_def(boom);
    BetaBinomialModel_def(boom);
    MultinomialModel_def(boom);

    GaussianModel_def(boom);
    GammaModel_def(boom);
    MvnModel_def(boom);
    UniformModel_def(boom);
    WishartModel_def(boom);

    GlmModel_def(boom);
    GpModel_def(boom);
    MultinomialLogitModel_def(boom);
    TimeSeries_def(boom);
    StateSpaceModel_def(boom);
    StateModel_def(boom);

    MultivariateStateSpaceModel_def(boom);
    MultivariateStateModel_def(boom);

    Imputation_def(boom);

    DynamicRegressionModel_def(boom);

    DirichletProcessMvn_def(boom);
    BetaBinomialMixture_def(boom);

    FactorModel_def(boom);

    test_utils_def(boom);
  }  // Module BOOM

}  // namespace BayesBoom
