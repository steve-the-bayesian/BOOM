#include "nmath.hpp"
#include <sstream>
#include "cpputil/report_error.hpp"

namespace Rmath{
  void mathlib_error(const std::string &s){
    report_error(s); }

  void mathlib_error(const std::string &s, int d){
    std::ostringstream err;
    err << s << " " << d << std::endl;
    report_error(err.str());
  }
  void mathlib_error(const std::string &s, double d){
    std::ostringstream err;
    err << s << " " << d << std::endl;
    report_error(err.str());
  }
}
