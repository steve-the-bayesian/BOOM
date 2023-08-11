#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include <sstream>

#include "Models/GP/kernels.hpp"
#include "Models/GP/GaussianProcessRegressionModel.hpp"
#include "Models/GP/PosteriorSamplers/GaussianProcessRegressionPosteriorSampler.hpp"

#include "cpputil/Ptr.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;
  using BOOM::uint;

  void GpModel_def(py::module &boom) {

    py::class_<FunctionParams,
               Params,
               Ptr<FunctionParams>>(boom, "FunctionParams")
        .def("__call__",
             [](const FunctionParams &fun,
                const Vector &x1) {
               return fun(x1);
             },
             py::is_operator(),
             py::arg("x"),
             "Evaluate the function at the given argument, "
             "returning a scalar.\n")
        .def("__call__",
             [](const FunctionParams &fun,
                const Matrix &X) {
               return fun(X);
             },
             py::is_operator(),
             py::arg("X"),
             "Evaluate the function at each row of the given argument, "
             "returning a boom.Vector.\n")
        ;

    py::class_<ZeroFunction,
               FunctionParams,
               Ptr<ZeroFunction>>(boom, "ZeroFunction")
        .def(py::init([](){ return new ZeroFunction; }))
        ;

    py::class_<KernelParams,
               Params,
               Ptr<KernelParams>>(boom, "KernelParams")
        .def("__call__",
             [](const KernelParams &k,
                const Vector &x1,
                const Vector &x2) {
               return k(x1, x2);
             },
             py::is_operator(),
             py::arg("x1"),
             py::arg("x2"),
             "Evaluate the kernel function at the given arguments, "
             "returning a scalar.\n")
        .def("__call__",
             [](const KernelParams &k,
                const Matrix &X) {
               return k(X);
             },
             py::is_operator(),
             py::arg("X"),
             "Evaluate the kernel at every pair of rows in X, "
             "returning a boom.SpdMatrix.\n")
        ;

    py::class_<RadialBasisFunction,
               KernelParams,
               Ptr<RadialBasisFunction>>(boom, "RadialBasisFunction")
        .def(py::init(
            [](double scale) {
              return new RadialBasisFunction(scale);
            }),
            py::arg("scale"),
            "Args:\n\n"
            "  scale: The size of a 'standard deviation' over which "
            "the kernel should reach.\n")
        ;

    py::class_<MahalanobisKernel,
               KernelParams,
               Ptr<MahalanobisKernel>>(boom, "MahalanobisKernel")
        .def(py::init(
            [](int dim, double scale) {
              return new MahalanobisKernel(dim, scale);
            }),
             py::arg("dim"),
             py::arg("scale"),
             "Args:\n\n"
             "  dim:  The dimension of the Vectors that the kernel accepts.\n"
             "  scale:  The scale factor.\n")
        .def(py::init(
            [](const Matrix &X, double scale, double diagonal_shrinkage) {
              return new MahalanobisKernel(X, scale, diagonal_shrinkage);
            }),
             py::arg("X"),
             py::arg("scale") = 1.0,
             py::arg("diagonal_shrinkage") = 0.05,
             "Args:\n\n"
             "  X:  TBD\n"
             "  scale:  TBD\n"
             "  diagonal_shrinkage:  TBD\n")
        ;

  }

}  // namespace BayesBoom
