/*
  Copyright (C) 2005-2011 Steven L. Scott

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2.1 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA
*/

#include <time.h>
#include <string>
#include <Rinternals.h>

namespace BOOM {
  void print_R_timestamp(int iteration_number, int ping){
    if (ping <= 0) return;
    if ( (iteration_number % ping) == 0) {
      time_t rawtime;
      time(&rawtime);
#ifdef _WIN32
      // mingw does not include the re-entrant versions localtime_r
      // and asctime_r.
      std::string time_str(asctime(localtime(&rawtime)));
#else
      struct tm timeinfo;
      localtime_r(&rawtime, &timeinfo);
      char buf[28];
      std::string time_str(asctime_r(&timeinfo, buf));
      time_str = time_str.substr(0, time_str.find("\n"));
#endif
      const char *sep="=-=-=-=-=";
      Rprintf("%s Iteration %d %s %s\n", sep, iteration_number,
              time_str.c_str(), sep);
    }
  }
}
