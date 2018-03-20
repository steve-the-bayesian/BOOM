// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2017 Steven L. Scott

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

#include "Models/PosteriorSamplers/Imputer.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  // This function must appear in a cpp file because the exception handling that
  // it does caused problems when it appeared in the header file.
  void ParallelLatentDataImputer::impute_latent_data() {
    if (pool_.no_threads()) {
      for (int i = 0; i < workers_.size(); ++i) {
        workers_[i]->impute_latent_data();
        workers_[i]->combine_complete_data();
      }
    } else {
      std::vector<std::future<void>> jobs;
      jobs.reserve(workers_.size());
      for (int i = 0; i < workers_.size(); ++i) {
        jobs.emplace_back(
            pool_.submit(workers_[i]->data_imputation_callback()));
      }
      std::vector<std::string> error_messages;
      for (int i = 0; i < jobs.size(); ++i) {
        try {
          jobs[i].get();
        } catch (std::exception &e) {
          std::string message = e.what();
          error_messages.push_back(message);
        } catch (...) {
          error_messages.push_back("Unknown exception.");
        }
      }
      if (!error_messages.empty()) {
        if (error_messages.size() == 1) {
          report_error(error_messages[0]);
        } else {
          std::ostringstream err;
          err << "There were " << error_messages.size() << " exceptions thrown."
              << std::endl;
          for (int i = 0; i < error_messages.size(); ++i) {
            err << "Error message from exception " << i + 1 << "." << std::endl
                << error_messages[i] << std::endl;
          }
          report_error(err.str());
        }
      }
    }
  }

}  // namespace BOOM
