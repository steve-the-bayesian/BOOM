// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2009 Steven L. Scott

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
#ifndef BOOM_PROGRAM_OPTIONS_HPP
#define BOOM_PROGRAM_OPTIONS_HPP
#include <boost/program_options.hpp>
#include <map>
#include <memory>
#include <vector>
#include "cpputil/string_utils.hpp"

namespace BOOM {

  namespace po = boost::program_options;
  typedef po::options_description OD;
  typedef po::variables_map VM;

  class ProgramOptions {
   public:
    ProgramOptions(const string &desc = "program options")
        : od_(desc.c_str()) {}

    void add_option(const string &option_name, const string &description);
    template <class T>
    void add_option(const string &option_name, const string &description) {
      od_.add_options()(option_name.c_str(), po::value<T>(),
                        description.c_str());
    }

    bool is_set(const string &option) const { return vm.count(option) > 0; }

    void add_option_family(const string &name, const string &description);

    void add_option_to_family(const string &family_name,
                              const string &option_name, const string &desc);

    template <class T>
    void add_option_type_to_family(const string &family_name,
                                   const string &option_name,
                                   const string &desc) {
      option_families_[family_name]->add_options()(option_name, po::value<T>(),
                                                   desc);
    }

    void process_command_line(int argc, char **argv);
    void process_command_line(const std::vector<std::string> &args);
    void process_cfg_file(const string &cfg_file_name);

    template <class T>
    T get_with_default(const string &name, const T &default_value) const {
      if (vm.count(name) == 0) return default_value;
      return vm[name].as<T>();
    }

    template <class T>
    T get_required_option(const string &name) const {
      if (vm.count(name) == 0) {
        ostringstream err;
        err << "Error!!  Required option " << name << " was not supplied."
            << endl;
        error(err.str());
      }
      return vm[name].as<T>();
    }

    ostream &print(ostream &) const;

   private:
    VM vm;
    OD od_;

    std::map<string, std::shared_ptr<OD> > option_families_;
    std::vector<string> option_family_names_;
    bool processed_;

    void error(const string &msg) const;
  };

  ostream &operator<<(ostream &out, const ProgramOptions &op);

}  // namespace BOOM
#endif  // BOOM_PROGRAM_OPTIONS_HPP
