#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>

#include "Models/DataTypes.hpp"
#include "Models/CategoricalData.hpp"
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


    py::class_<CatKeyBase, Ptr<CatKeyBase>>
        (boom, "CatKeyBase")
        ;

    py::class_<UnboundedIntCatKey,
               CatKeyBase,
               Ptr<UnboundedIntCatKey>>(
        boom, "UnboundedIntCatKey")
        .def(py::init([]() {return new UnboundedIntCatKey();}))
        ;

    py::class_<FixedSizeIntCatKey,
               CatKeyBase,
               Ptr<FixedSizeIntCatKey>>(
        boom, "FixedSizeIntCatKey")
        .def(py::init([](int max_levels) {
          return new FixedSizeIntCatKey(max_levels);}))
        ;

    py::class_<CatKey,
               CatKeyBase,
               Ptr<CatKey>>(boom, "CatKey")
        .def(py::init(
            [](const std::vector<std::string> &labels) {
              return new CatKey(labels);
            }),
             py::arg("labels"),
             "Args:\n\n"
             "  labels:  The levels of the categorical variable.\n "
             )
        ;

    py::class_<CategoricalData, Data, Ptr<CategoricalData>>(
        boom, "CategoricalData")
        .def(py::init([](int value, CatKeyBase *key) {
                        return new CategoricalData(value, Ptr<CatKeyBase>(key));
                      }),
          py::arg("value"),
          py::arg("key"),
          "Args:\n\n"
          "  value: The numeric index of the categorical data value.\n"
          "  key:  The CatKeyBase object managing the set of available "
          "levels.\n")
        .def(py::init(
            [](std::string &value, CatKey *key) {
              return new CategoricalData(value, Ptr<CatKey>(key));
            }),
          py::arg("value"),
          py::arg("key"),
          "Args:\n\n"
          "  value: The label describing the categorical data value.\n"
          "  key:  The CatKeyBase object managing the set of available "
          "levels.\n")
        .def_property_readonly(
            "value",
            [](const CategoricalData &data) {return data.value();})
        ;

  }  // module

}  // namespace BayesBoom
