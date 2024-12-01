#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>
#include <memory>
#include <cpputil/report_error.hpp>
#include <cpputil/Date.hpp>
#include <cpputil/DateTime.hpp>
#include <cpputil/find.hpp>

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

    py::class_<DateTime>(boom, "DateTime")
        .def(py::init(
            [](int month, int day, int year, double day_fraction) {
              Date date(MonthNames(month), day, year);
              return new DateTime(date, day_fraction);
            }),
             py::arg("month"),
             py::arg("day"),
             py::arg("year"),
             py::arg("day_fraction"),
             "Args:\n\n"
             "  month: Month of year 1-12.\n"
             "  day:  Day of month 1-31.\n"
             "  year: Four digit year.\n"
             "  day_fraction: The time of day, expressed as a real number "
             "in [0, 1) giving the fraction of the 24 hour day.\n")
        .def_property_readonly(
            "year",
            [](const DateTime &dt) {
              return dt.date().year();
            })
        .def_property_readonly(
            "month",
            [](const DateTime &dt) {
              return static_cast<int>(dt.date().month());
            })
        .def_property_readonly(
            "day",
            [](const DateTime &dt) {
              return dt.date().day();
            })
        .def_property_readonly(
            "hour",
            [](const DateTime &dt) {
              return dt.hour();
            })
        .def_property_readonly(
            "minute",
            [](const DateTime &dt) {
              return dt.minute();
            })
        .def_property_readonly(
            "second",
            [](const DateTime &dt) {
              return dt.second();
            })
        .def_property_readonly(
            "nanosecond",
            [](const DateTime &dt) {
              return dt.nanoseconds_after_second();
            })
        .def_property_readonly(
            "nanoseconds_since_epoch",
            [](const DateTime &dt) {
              return dt.nanoseconds_since_epoch();
            })
        .def_property_readonly(
            "hours_left_in_day",
            [](const DateTime &dt) {
              return dt.hours_left_in_day();
            })
        .def_property_readonly(
            "minutes_left_in_hour",
            [](const DateTime &dt) {
              return dt.minutes_left_in_hour();
            })
        .def_property_readonly(
            "seconds_left_in_minute",
            [](const DateTime &dt) {
              return dt.seconds_left_in_minute();
            })
        .def("__repr__",
             [] (const DateTime &dt) {
               std::ostringstream out;
               out << dt;
               return out.str();
             })
        ;

    boom.def("fast_find",
             [](const std::vector<std::string> &input,
                const std::vector<std::string> &target) {
               return find<std::string>(input, target);
             },
             "Find all the objects in 'input' by looking in 'target'."
             "\n\n"
             "Args:\n"
             "  input:  The set of strings to search for.\n"
             "  target:  The set of strings in which to search.\n"
             "\n"
             "Returns:  A vector of indices ans of the same length as 'input'"
             "  where ans[i] is the position in 'target' where input[i] is "
             "found.\n");


  }  // ends the cpputil_def function.

}  // namespace BayesBoom
