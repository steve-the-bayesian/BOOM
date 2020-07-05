#include <pybind11/pybind11.h>

#include <sstream>

#include "Models/DataTypes.hpp"
#include "Models/SpdData.hpp"
#include "cpputil/Ptr.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;

  // Define the
  void Data_def(py::module &boom) {

    py::class_<Data, Ptr<Data>>(boom, "Data")
        .def("__repr__",
             [](const Data &dp) {
               std::ostringstream out;
               dp.display(out);
               return out.str();
             })
        ;

    py::class_<DoubleData, Data, Ptr<DoubleData>>(boom, "DoubleData")
        .def(py::init<double>(),
             py::arg("x"),
             "Args:\n"
             "  x:  The value of the data point.")
        .def_property_readonly("value", &DoubleData::value)
        .def("set",
             [](DoubleData &dp, double x) {dp.set(x);},
             "Set the value of the data point.")
        ;


  }  // module

}  // namespace BayesBoom
