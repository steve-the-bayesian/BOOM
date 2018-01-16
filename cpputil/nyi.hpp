#ifndef BOOM_NYI_HPP
#define BOOM_NYI_HPP

#include <sstream>
#include <cpputil/report_error.hpp>

namespace BOOM{

  inline void nyi(const std::string & thing){
    std::ostringstream err;
    err << thing << " is not yet implemented.\n";
    report_error(err.str());
  }
}
#endif // BOOM_NYI_HPP
