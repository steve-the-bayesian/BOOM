#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <cpputil/Ptr.hpp>

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;
  
  // Forward definitions of all the class definitions to be added in other
  // files.
  void LinAlg_def(py::module &);
  void GaussianModel_def(py::module &);

  
  PYBIND11_MODULE(BayesBoom, boom) {
    boom.doc() = "A library for Bayesian modeling, and assorted "
        "other useful bits.";
    
    // Calling these functions here defines the classes in the module.
    LinAlg_def(boom);
    GaussianModel_def(boom);
    
  }  // Module BOOM

}  // namespace BayesBoom
