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
    std::vector<std::vector<double>> numeric_data;

    if (header) {
      ++line_number;
      getline(in, line);
      vnames_ = split(line);
    }

    while (in) {
      ++line_number;
      getline(in, line);
      if (is_all_white(line)) continue;
      std::vector<std::string> fields = split(line);

      if (nfields == 0) {  // getting started
        nfields = fields.size();
        diagnose_types(fields);
        numeric_data.resize(nfields);
        categorical_data.resize(nfields);
      }

      if (fields.size() != nfields) {  // check number of fields
        field_length_error(fname, line_number, nfields, fields.size());
      }

      for (uint i = 0; i < nfields; ++i) {
        if (!check_type(variable_types_[i], fields[i])) {
          wrong_type_error(line_number, i + 1);
        }

        if (variable_types_[i] == continuous) {
          double tmp = std::stod(fields[i], nullptr);
          numeric_data[i].push_back(tmp);
        } else if (variable_types_[i] == categorical) {
          categorical_data[i].push_back(fields[i]);
        } else {
          unknown_type();
        }
      }
    }

    for (uint i = 0; i < nfields; ++i) {
      if (variable_types_[i] == continuous) {
        continuous_variables_.push_back(
            Vector(numeric_data[i].begin(), numeric_data[i].end()));
      } else {
        continuous_variables_.push_back(Vector(0));
      }
    }

    for (uint i = 0; i < nfields; ++i) {
      if (variable_types_[i] == categorical) {
        categorical_variables_.emplace_back(categorical_data[i]);
      } else {
        CategoricalVariable empty;
        categorical_variables_.push_back(empty);
      }
    }

    if (vnames_.empty()) vnames_ = default_vnames(variable_types_.size());
  }

  DataTable *DataTable::clone() const { return new DataTable(*this); }

  std::ostream &DataTable::display(std::ostream &out) const { return print(out); }

  //--- build a DataTable by appending variables ---
  void DataTable::append_variable(const Vector &v, const std::string &name) {
    // If there are no variables, ie the table is empty, append to the
    // continuous variables.  IMPORTANT: The first set of observations
    // determines the size of the data columns from then on! (since nobs()
    // method refers to the first appended vector of obsevations.)
    if (nvars() == 0) {
      continuous_variables_.push_back(v);
      variable_types_.push_back(continuous);
      vnames_.push_back(name);

      // The empty categorical variable keeps indexing consistent with the
      // variable_types_ and vnames_ vectors.
      CategoricalVariable empty;
      categorical_variables_.push_back(empty);
    } else {
      // If the table is NOT empty, check if the observations for the added
      // variable is same for the previous variables.
      if (nobs() != v.size()) {
        report_error(
            "Wrong sized include vector in "
            "DataTable::append_variable");
      } else {
        continuous_variables_.push_back(v);
        vnames_.push_back(name);
        variable_types_.push_back(continuous);

        // The empty categorical data pointer vector to keep the indexing
        // consistent with the variable_types_ and vnames_ vectors.
        CategoricalVariable empty;
        categorical_variables_.push_back(empty);
      }
    }
  }

  void DataTable::append_variable(const CategoricalVariable &cv,
                                  const std::string &name) {
    // If there are no variables, ie the table is empty, append to the
    // continuous variables.  IMPORTANT: The first set of observations
    // determines the size of the data columns from then on! (since nobs()
    // method refers to the first appended vector of obsevations.
    if (nvars() == 0) {
      categorical_variables_.push_back(cv);
      variable_types_.push_back(categorical);
      vnames_.push_back(name);

      // The empty vector keeps the indexing consistent with the variable_types_
      // and vnames_ vectors.
      continuous_variables_.push_back(Vector(0));
    } else {
      // If the table is NOT empty, check if the number of observations
      // for the added variable is same for the previous variables.
      if (nobs() != cv.size()) {
        report_error(
            "Wrong sized include vector in "
            "DataTable::append_variable");
      } else {
        categorical_variables_.push_back(cv);
        vnames_.push_back(name);
        variable_types_.push_back(categorical);

        // The empty vector keeps the indexing consistent with the
        // variable_types_ and vnames_ vectors.
        continuous_variables_.push_back(Vector(0));
      }
    }
  }

  //---------------------------------------------------------------------
  // Determine the type of variable stored in vs.
  void DataTable::diagnose_types(const std::vector<std::string> &vs) {
    uint nfields = vs.size();
    variable_types_ = std::vector<VariableType>(nfields, unknown);
    for (uint i = 0; i < vs.size(); ++i) {
      variable_types_[i] = is_numeric(vs[i]) ? continuous : categorical;
    }
  }

  bool DataTable::check_type(VariableType type, const std::string &s) const {
    if (is_numeric(s)) {
      if (type == continuous) return true;
    } else {  // s is not numeric
      if (type == categorical) return true;
    }
    return false;
  }

  const std::vector<DataTable::VariableType>
      &DataTable::display_variable_types() const {
    return variable_types_;
  }

  std::vector<std::string> &DataTable::vnames() { return vnames_; }
  const std::vector<std::string> &DataTable::vnames() const { return vnames_; }

  //------------------------------------------------------------
  uint DataTable::nvars() const { return variable_types_.size(); }

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
      if (variable_types_[J] == categorical) {
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
        if (variable_types_[J] == continuous) {
          X(i, column++) = continuous_variables_[J][i];
        } else if (variable_types_[J] == categorical) {
          const Ptr<CategoricalData> x(categorical_variables_[J][i]);
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
      if (variable_types_[J] == continuous) {
        dimnames.push_back(vnames_[J]);
      } else if (variable_types_[J] == categorical) {
        const Ptr<CategoricalData> x(categorical_variables_[J][0]);
        std::string stub = vnames_[J];
        std::vector<std::string> labs = categorical_variables_[J].labels();
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
    if (variable_types_ != rhs.variable_types_) {
      report_error("Variable type mismatch in rbind(DataTable).");
    }
    for (int i = 0; i < continuous_variables_.size(); ++i) {
      if (!continuous_variables_[i].empty()) {
        continuous_variables_[i].concat(rhs.continuous_variables_[i]);
      }
    }
    for (int i = 0; i < categorical_variables_.size(); ++i) {
      if (!categorical_variables_[i].empty()) {
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
    }
    return *this;
  }

  //======================================================================
  template <class T>
  uint mapsize(const std::map<uint, T> &m) {
    if (m.empty()) return 0;
    const T &first_element(m.begin()->second);
    return first_element.size();
  }

  uint DataTable::nobs() const {
    if (variable_types_.empty()) {
      return 0;
    }
    if (variable_types_[0] == continuous)
      return continuous_variables_[0].size();
    return categorical_variables_[0].size();
  }

  uint DataTable::nlevels(uint i) const {
    if (variable_types_[i] == continuous) return 1;
    return categorical_variables_[i][0]->nlevels();
  }

  DataTable::VariableType DataTable::variable_type(uint which_column) const {
    return variable_types_[which_column];
  }

  Vector DataTable::getvar(uint n) const {
    if (variable_types_[n] == continuous) return continuous_variables_[n];
    Vector ans(nobs());
    for (uint i = 0; i < nobs(); ++i) {
      ans[i] = categorical_variables_[n][i]->value();
    }
    return ans;
  }

  CategoricalVariable DataTable::get_nominal(uint n) const {
    if (variable_types_[n] != categorical) wrong_type_error(1, n);
    return categorical_variables_[n];
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
      bool is_cont = variable_types_[j] == continuous;
      for (uint i = 0; i < nobs(); ++i) {
        std::ostringstream sout;
        if (is_cont) {
          sout << continuous_variables_[j][i];
        } else {
          sout << categorical_variables_[j].label(i);
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

  std::vector<VariableSummary> summarize(const DataTable &table) {
    std::vector<VariableSummary> ans;
    for (int i = 0; i < table.nvars(); ++i) {
      VariableSummary summary;
      summary.type = table.variable_type(i);
      if (summary.type == DataTable::VariableType::continuous) {
        Vector data = table.getvar(i);
        data.sort();
        summary.min = data[0];
        summary.max = data.back();
        summary.mean = mean(data);
        summary.standard_deviation = sd(data);
        summary.number_of_distinct_values = 1;
        for (int j = 1; j < data.size(); ++j) {
          if (data[j - 1] != data[j]) {
            ++summary.number_of_distinct_values;
          }
        }
      } else if (summary.type == DataTable::VariableType::categorical) {
        CategoricalVariable data = table.get_nominal(i);
        summary.mean = summary.standard_deviation = negative_infinity();
        summary.number_of_distinct_values = data[0]->nlevels();
        summary.min = 0;
        summary.max = summary.number_of_distinct_values - 1;
      }
      ans.push_back(summary);
    }
    return ans;
  }

}  // namespace BOOM
