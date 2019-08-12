#ifndef BOOM_LINALG_PYBIND11_WRAPPER_HPP
#define BOOM_LINALG_PYBIND11_WRAPPER_HPP

#include <pybind11/pybind11.h>

#include "LinAlg/Vector.hpp"
#include "LinAlg/VectorView.hpp"

PYBIND11_MODULE(Boom /* name in python */,
                boom /* local object name */) {

  using namespace BOOM;
  py::class_<Vector>(boom, "Vector")
      .def(py::init<int, double>())
      .def(py::init<>())  // python list
      .def("all_finite", &Vector::all_finite,
           "Returns true iff all elements are finite.")
      .def("randomize", &Vector::randomize)
      .def("stride", &Vector::stride)
      .def("length", &Vector::length)
      ;

  py::class <VectorView>(boom, "VectorView")
      .def(py::init<Vector &>())
      ;

  py::class <ConstVectorView>(boom, "ConstVectorView")
      .def(py::init<const Vector &>())
      ;

  py::class <Matrix>(boom, "Matrix")
      ;

  py::class <SpdMatrix, Matrix>(boom, "SpdMatrix")
      ;
  
}  // Module BOOM


#endif  //  BOOM_LINALG_PYBIND11_WRAPPER_HPP

