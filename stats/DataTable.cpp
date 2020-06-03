// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005 Steven L. Scott

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

#include "stats/DataTable.hpp"
#include "stats/moments.hpp"

#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "Models/CategoricalData.hpp"
#include "cpputil/DefaultVnames.hpp"
#include "cpputil/Ptr.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"
#include "cpputil/string_utils.hpp"

namespace BOOM {
  using std::endl;


  //---------------------------------------------------------------------
  // Determine the type of variable stored in vs.

  MixedDataOrganizer::MixedDataOrganizer()
      : numeric_count_(0),
        categorical_count_(0),
        unknown_count_(0)
  {}

  void MixedDataOrganizer::add_variable(VariableType type) {
    int index = variable_types_.size();
    variable_types_.push_back(type);
    if (type == VariableType::numeric) {
      type_map_[index] = std::make_pair(type, numeric_count_++);
    } else if (type == VariableType::categorical) {
      type_map_[index] = std::make_pair(type, categorical_count_++);
    } else {
      ++unknown_count_;
      report_error("Numeric and categorical the the only currently supported"
                   " types.");
    }
  }

  std::pair<VariableType, int> MixedDataOrganizer::type_map(int i) const {
    auto it = type_map_.find(i);
    if (it == type_map_.end()) {
      return std::make_pair(VariableType::unknown, -1);
    } else {
      return it->second;
    }
  }

  void MixedDataOrganizer::diagnose_types(const std::vector<std::string> &vs) {
    uint nfields = vs.size();
    variable_types_ = std::vector<VariableType>(nfields, VariableType::unknown);
    for (uint i = 0; i < vs.size(); ++i) {
      variable_types_[i] =
          is_numeric(vs[i]) ? VariableType::numeric : VariableType::categorical;
    }
  }

  bool MixedDataOrganizer::check_type(int i, const std::string &s) const {
    VariableType type = variable_types_[i];
    if (is_numeric(s)) {
      if (type == VariableType::numeric) return true;
    } else {  // s is not numeric
      if (type == VariableType::categorical) return true;
    }
    return false;
  }

  bool MixedDataOrganizer::operator==(const MixedDataOrganizer &rhs) const {
    return numeric_count_ == rhs.numeric_count_
        && categorical_count_ == rhs.categorical_count_
        && unknown_count_ == rhs.unknown_count_
        && variable_types_ == rhs.variable_types_;
  }

  //===========================================================================
  CategoricalVariable::CategoricalVariable(
      const std::vector<std::string> &raw_data)
      : key_(make_catkey(raw_data)) {
    for (int i = 0; i < raw_data.size(); ++i) {
      Ptr<CategoricalData> dp = new CategoricalData(raw_data[i], key_);
      data_.push_back(dp);
    }
  }

  const Vector &get(const std::map<uint, Vector> &m, uint i) {
    return m.find(i)->second;
  }

  const CategoricalVariable &get(const std::map<uint, CategoricalVariable> &m,
                                 uint i) {
    return m.find(i)->second;
  }

  inline void field_length_error(const std::string &fname, uint line, uint nfields,
                                 uint prev_nfields) {
    std::ostringstream msg;
    msg << "file: " << fname << endl
        << " line number " << line << " has " << nfields
        << " fields.  Previous lines had " << prev_nfields << "fields." << endl;
    report_error(msg.str());
  }

  //-----------------------------------------------------------------
  inline void wrong_type_error(uint line_num, uint field_num) {
    std::ostringstream msg;
    msg << "line number " << line_num << " field number " << field_num << endl;
    report_error(msg.str());
  }

  inline void unknown_type() { report_error("unknown type"); }
  //-----------------------------------------------------------------

  DataTable::DataTable() {}

  DataTable::DataTable(const std::string &fname,
                       bool header,
                       const std::string &sep) {
    ifstream in(fname.c_str());
    if (!in) {
      std::string msg = "bad file name ";
      report_error(msg + fname);
    }

    StringSplitter split(sep);
    std::string line;
    uint nfields = 0;
    uint line_number = 0;

    std::vector<std::vector<std::string>> categorical_data;
    std::vector<Vector> numeric_data;

    if (header) {
      ++line_number;
      getline(in, line);
      vnames_ = split(line);
      nfields = vnames_.size();
    }

    while (in) {
      ++line_number;
      getline(in, line);
      if (is_all_white(line)) continue;
      std::vector<std::string> fields = split(line);

      if (nfields == 0) {
        // No data has yet been read.  Initialize the variable names and data
        // types based off the first row.
        nfields = fields.size();
        data_organizer_->diagnose_types(fields);
        numeric_data.resize(data_organizer_->number_of_numeric_fields());
        categorical_data.resize(data_organizer_->number_of_categorical_fields());
      }

      if (fields.size() != nfields) {  // check number of fields
        field_length_error(fname, line_number, nfields, fields.size());
      }

      for (uint i = 0; i < nfields; ++i) {
        if (!data_organizer_->check_type(i, fields[i])) {
          wrong_type_error(line_number, i + 1);
        }

        if (variable_type(i) == VariableType::numeric) {
          double tmp = std::stod(fields[i], nullptr);
          int index = data_organizer_->type_map(i).second;
          numeric_data[index].push_back(tmp);
        } else if (variable_type(i) == VariableType::categorical) {
          int index = data_organizer_->type_map(i).second;
          categorical_data[index].push_back(fields[i]);
        } else {
          unknown_type();
        }
      }
    }

    for (uint i = 0; i < nfields; ++i) {
      VariableType type;
      int index;
      std::tie(type, index) = data_organizer_->type_map(i);

      if (type == VariableType::numeric) {
        numeric_variables_.push_back(numeric_data[index]);
      } else if (type == VariableType::categorical) {
        categorical_variables_.emplace_back(categorical_data[i]);
      }
    }
    if (vnames_.empty()) vnames_ = default_vnames(nfields);
  }

  DataTable *DataTable::clone() const { return new DataTable(*this); }

  std::ostream &DataTable::display(std::ostream &out) const { return print(out); }

  //--- build a DataTable by appending variables ---
  void DataTable::append_variable(const Vector &v, const std::string &name) {
    // If there are no variables, ie the table is empty, append to the numeric
    // variables.  IMPORTANT: The first set of observations determines the size
    // of the data columns from then on! (since nobs() method refers to the
    // first appended vector of obsevations.)
    if (nvars() == 0) {
      numeric_variables_.push_back(v);
      data_organizer_->add_variable(VariableType::numeric);
      vnames_.push_back(name);
    } else {
      // If the table is NOT empty, check if the observations for the added
      // variable is same for the previous variables.
      if (nobs() > 0 && nobs() != v.size()) {
        report_error(
            "Wrong sized include vector in DataTable::append_variable");
      } else {
        numeric_variables_.push_back(v);
        vnames_.push_back(name);
        data_organizer_->add_variable(VariableType::numeric);
      }
    }
  }

  void DataTable::append_variable(const CategoricalVariable &cv,
                                  const std::string &name) {
    // If there are no variables, ie the table is empty, append to the numeric
    // variables.  IMPORTANT: The first set of observations determines the size
    // of the data columns from then on! (since nobs() method refers to the
    // first appended vector of obsevations.
    if (nvars() == 0) {
      categorical_variables_.push_back(cv);
      data_organizer_->add_variable(VariableType::categorical);
      vnames_.push_back(name);
    } else {
      // If the table is NOT empty, check if the number of observations
      // for the added variable is same for the previous variables.
      if (nobs() > 0 && nobs() != cv.size()) {
        report_error(
            "Wrong sized include vector in DataTable::append_variable");
      } else {
        categorical_variables_.push_back(cv);
        vnames_.push_back(name);
        data_organizer_->add_variable(VariableType::categorical);
      }
    }
  }

  std::vector<std::string> &DataTable::vnames() { return vnames_; }
  const std::vector<std::string> &DataTable::vnames() const { return vnames_; }

  //------------------------------------------------------------
  uint DataTable::nvars() const { return vnames_.size(); }

  LabeledMatrix DataTable::design(bool add_int) const {
    std::vector<bool> include(nvars(), true);
    return design(Selector(include), add_int);
  }

  //------------------------------------------------------------
  LabeledMatrix DataTable::design(const Selector &include, bool add_int) const {
    uint dimension = add_int ? 1 : 0;
    for (uint i = 0; i < include.nvars(); ++i) {
      uint J = include.indx(i);
      uint incremental_dimension = 1;
      if (variable_type(J) == VariableType::categorical) {
        incremental_dimension = nlevels(J) - 1;
      }
      dimension += incremental_dimension;
    }

    uint number_of_observations = nobs();
    Matrix X(number_of_observations, dimension);
    for (uint i = 0; i < number_of_observations; ++i) {
      if (add_int) X(i, 0) = 1.0;
      uint column = add_int ? 1 : 0;
      for (uint j = 0; j < include.nvars(); ++j) {
        uint J = include.indx(j);
        VariableType type;
        int index;
        std::tie(type, index) = data_organizer_->type_map(J);
        if (type == VariableType::numeric) {
          X(i, column++) = numeric_variables_[index][i];
        } else if (type == VariableType::categorical) {
          const Ptr<CategoricalData> x(categorical_variables_[index][i]);
          for (uint k = 1; k < x->nlevels(); ++k)
            X(i, column++) = (k == x->value() ? 1 : 0);
        } else {
          unknown_type();
        }
      }
    }

    std::vector<std::string> dimnames;
    if (add_int) {
      dimnames.push_back("Intercept");
    }
    for (uint j = 0; j < include.nvars(); ++j) {
      uint J = include.indx(j);
      int index;
      VariableType type;
      std::tie(type, index) = data_organizer_->type_map(J);
      if (type == VariableType::numeric) {
        dimnames.push_back(vnames_[J]);
      } else if (type == VariableType::categorical) {
        const Ptr<CategoricalData> x(categorical_variables_[index][0]);
        std::string stub = vnames_[J];
        std::vector<std::string> labs = categorical_variables_[index].labels();
        for (uint i = 1; i < labs.size(); ++i) {
          dimnames.push_back(stub + ":" + labs[i]);
        }
      }
    }
    return LabeledMatrix(X, std::vector<std::string>(), dimnames);
  }

  //----------------------------------------------------------------------
  DataTable &DataTable::rbind(const DataTable &rhs) {
    if (rhs.nobs() == 0) {
      return *this;
    }
    if (nobs() == 0) {
      *this = rhs;
      return *this;
    }
    if (*data_organizer_ != *rhs.data_organizer_) {
      report_error("Variable type mismatch in rbind(DataTable).");
    }
    for (int i = 0; i < numeric_variables_.size(); ++i) {
      numeric_variables_[i].concat(rhs.numeric_variables_[i]);
    }
    for (int i = 0; i < categorical_variables_.size(); ++i) {
      if (categorical_variables_[i].labels() !=
          rhs.categorical_variables_[i].labels()) {
        std::ostringstream err;
        err << "Labels for categorical variable " << i
            << " do not match in DataTable::rbind." << endl
            << "Labels from left hand side: " << endl
            << categorical_variables_[i].labels() << endl
            << "Labels from right hand side: " << endl
            << rhs.categorical_variables_[i].labels() << endl;
        report_error(err.str());
      }
      Ptr<CatKey> key = categorical_variables_[i].key();
      for (int j = 0; j < rhs.categorical_variables_[i].size(); ++j) {
        uint value = rhs.categorical_variables_[i][j]->value();
        categorical_variables_[i].push_back(new CategoricalData(value, key));
      }
    }
    return *this;
  }

  //======================================================================
  uint DataTable::nlevels(uint i) const {
    VariableType type;
    int index;
    std::tie(type, index) = data_organizer_->type_map(i);
    if (type == VariableType::numeric) return 1;
    return categorical_variables_[index][0]->nlevels();
  }

  int DataTable::nrow() const {
    if (numeric_variables_.empty() && categorical_variables_.empty()) {
      return 0;
    }
    if (numeric_variables_.empty()) {
      return categorical_variables_[0].size();
    } else {
      return numeric_variables_[0].size();
    }
    report_error("Can't determine size.");
    return -1;
  }

  Vector DataTable::getvar(uint n) const {
    VariableType type;
    int index;
    std::tie(type, index) = data_organizer_->type_map(n);
    if (type == VariableType::numeric) {
      return numeric_variables_[index];
    } else {
      Vector ans(nobs());
      for (uint i = 0; i < nobs(); ++i) {
        ans[i] = categorical_variables_[index][i]->value();
      }
      return ans;
    }
  }

  double DataTable::getvar(int row, int col) const {
    VariableType type;
    int index;
    std::tie(type, index) = data_organizer_->type_map(col);
    if (type == VariableType::numeric) {
      return numeric_variables_[index][row];
    } else {
      return negative_infinity();
    }
  }

  CategoricalVariable DataTable::get_nominal(uint n) const {
    VariableType type;
    int index;
    std::tie(type, index) = data_organizer_->type_map(n);
    if (type != VariableType::categorical) wrong_type_error(1, n);
    return categorical_variables_[index];
  }

  Ptr<CategoricalData> DataTable::get_nominal(int row, int col) const {
    VariableType type;
    int index;
    std::tie(type, index) = data_organizer_->type_map(col);
    if (type != VariableType::categorical) wrong_type_error(1, col);
    return categorical_variables_[index][row];
  }

  // DataTable::OrdinalVariable DataTable::get_ordinal(uint n)const{
  //   if (variable_types_[n]!=categorical) wrong_type_error(1, n);
  //   std::vector<Ptr<OrdinalData> > ans;
  //   const std::vector<Ptr<CategoricalData> > &v(categorical_variables_[n]);
  //   for (uint i=0; i<v.size(); ++i) {
  //     NEW(OrdinalData, dp)(v[i]->value(), v[0]->key());
  //     ans.push_back(dp);}
  //   return ans;
  // }

  // DataTable::OrdinalVariable DataTable::get_ordinal(
  //     uint n,
  //     const std::vector<std::string> &ord)const{
  //   std::vector<Ptr<OrdinalData> > ans(get_ordinal(n));
  //   set_order(ans, ord);
  //   return ans;
  // }

  //------------------------------------------------------------
  std::ostream &DataTable::print(std::ostream &out, uint from, uint to) const {
    if (to > nobs()) {
      to = nobs();
    }

    uint N = nvars();
    const std::vector<std::string> &vn(vnames());
    std::vector<uint> fw(nvars());
    uint padding = 2;
    for (uint i = 0; i < N; ++i) fw[i] = vn[i].size() + padding;

    using std::setw;
    std::vector<std::vector<std::string>> labmat(nvars());
    for (uint j = 0; j < nvars(); ++j) {
      std::vector<std::string> &v(labmat[j]);
      v.reserve(nobs());
      VariableType type;
      int index;
      std::tie(type, index) = data_organizer_->type_map(j);
      for (uint i = 0; i < nobs(); ++i) {
        std::ostringstream sout;
        if (type == VariableType::numeric) {
          sout << numeric_variables_[index][i];
        } else {
          sout << categorical_variables_[index].label(i);
        }
        std::string lab = sout.str();
        fw[j] = std::max<uint>(fw[j], lab.size() + padding);
        v.push_back(lab);
      }
    }

    for (uint j = 0; j < nvars(); ++j) out << setw(fw[j]) << vn[j];
    out << endl;

    for (uint i = from; i < to; ++i) {
      for (uint j = 0; j < nvars(); ++j) {
        out << setw(fw[j]) << labmat[j][i];
      }
      out << endl;
    }
    return out;
  }
  //------------------------------------------------------------
  std::ostream &operator<<(std::ostream &out, const DataTable &dt) {
    dt.print(out, 0, dt.nobs());
    return out;
  }

}  // namespace BOOM
