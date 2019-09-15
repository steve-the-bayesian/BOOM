#ifndef BOOM_LINALG_PYBIND11_WRAPPER_HPP
#define BOOM_LINALG_PYBIND11_WRAPPER_HPP

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include "LinAlg/Vector.hpp"
#include "LinAlg/VectorView.hpp"

namespace py = pybind11;

namespace BayesBoom {
  using namespace BOOM;
  
  PYBIND11_MODULE(BayesBoom,   // name in python 
                  boom) {      // local object name.

    py::class_<Vector>(boom, "Vector")
        .def(py::init<int, double>())
        .def(py::init<>())  // python list
        .def("all_finite", &Vector::all_finite,
             "Returns true iff all elements are finite.")
        .def("randomize", &Vector::randomize)
        .def("stride", &Vector::stride)
        .def("length", &Vector::length)
        ;

    // py::class_ <VectorView>(boom, "VectorView")
    //     .def(py::init<Vector &>())
    //     ;

    // py::class_ <ConstVectorView>(boom, "ConstVectorView")
    //     .def(py::init<const Vector &>())
    //     ;

    // py::class_ <Matrix>(boom, "Matrix")
    //     ;

    // py::class_ <SpdMatrix, Matrix>(boom, "SpdMatrix")
    //     ;
  
  }  // Module BOOM

}  // namespace BayesBoom

#endif  //  BOOM_LINALG_PYBIND11_WRAPPER_HPP
