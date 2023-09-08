#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "LinAlg/Vector.hpp"

#include "math/fft.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;

  void boom_math_dev(py::module &boom) {

    py::class_<FastFourierTransform> (boom, "FastFourierTransform")
        .def(py::init(
            []() {return new FastFourierTransform;}))
        .def("transform",
             [](FastFourierTransform &transform,
                const Vector &time_domain) {
               return transform.transform(time_domain);
             },
             py::arg("time_domain"),
             "Args:\n\n"
             "  time_domain:  A vector of observations in the time domain.\n"
             "\n"
             "Returns:\n"
             "  A "


  }  // boom_math_def

}  // namespace BayesBoom
