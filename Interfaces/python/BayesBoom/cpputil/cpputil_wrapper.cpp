#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>
#include <memory>
#include <cpputil/report_error.hpp>
#include <cpputil/Date.hpp>

namespace py = pybind11;

namespace BayesBoom {
  using namespace BOOM;

  void cpputil_def(py::module &boom) {

    py::class_<Date>(boom, "Date")
        .def(py::init(
            [] (int month, int day, int year) {
              return new Date(MonthNames(month), day, year);
            }),
             py::arg("month"),
             py::arg("day"),
             py::arg("year"),
             "Args:\n\n"
             "  month: Numeric month, starting with January = 1.\n"
             "  day:  Day of the month, 1-31.\n"
             "  year:  Full four-digit year.\n")
        .def("__repr__",
             [] (const Date &date) {
               std::ostringstream out;
               out << date;
               return out.str();
             })
        .def_property_readonly(
            "year",
            [] (const Date &d) {
              return d.year();
            })
        .def_property_readonly(
            "month",
            [] (const Date &d) {
              return static_cast<int>(d.month());
            })
        .def_property_readonly(
            "day",
            [] (const Date &d) {
              return static_cast<int>(d.day());
            })
        ;

  }  // ends the cpputil_def function.

}  // namespace BayesBoom
