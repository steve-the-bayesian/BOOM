#include "nmath.hpp"
#include <stdexcept>
#include <sstream>
#include <cpputil/report_error.hpp>
using namespace std;

namespace Rmath{
  void mathlib_error(const string &s){
    report_error(s); }

  void mathlib_error(const string &s, int d){
    ostringstream err;
    err << s << " " << d << std::endl;
    report_error(err.str());
  }
  void mathlib_error(const string &s, double d){
    ostringstream err;
    err << s << " " << d << std::endl;
    report_error(err.str());
  }
}
