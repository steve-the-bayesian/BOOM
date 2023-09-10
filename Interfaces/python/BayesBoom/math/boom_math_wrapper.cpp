#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "LinAlg/Vector.hpp"
#include "cpputil/Ptr.hpp"
#include "math/fft.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;

  void boom_math_def(py::module &boom) {

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
             "  A complex vector of the same length as x.  Because this "
             "vector contains twice as many numbers as the input, there is "
             "some duplication of information.  The second half of the real "
             "part of the sequence is a reflection of the first half.  The "
             "second half of the imaginary part of the sequence is the "
             " negative reflection of the first half. \n")
        .def("inverse_transform",
             [](FastFourierTransform &fft, const std::vector<std::complex<double>> &z) {
               return fft.inverse_transform(z);
             },
             py::arg("z"),
             "Args:\n\n"
             "  z: A sequence of complex numbers to be inverse "
             "transformed.  The second half of the sequence is not accessed, "
             "and is assumed to be a reflection of the first half, as noted "
             "in the documentation to 'transform'.\n\n"
             "Returns:\n"
             "  A real sequence whose transform (after scaling) is 'z'.  "
             " Note that if x is a sequence of length n then "
             "inverse_transform(transform(x)) returns x * n.  This is the "
             "convention adopted by many other fft programs, and notably the "
             "one used by R\n.")
        ;


  }  // boom_math_def

}  // namespace BayesBoom
