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
// TODO: add this back when c++17 support is widely available.
// #include <filesystem>
#include <iomanip>
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

  DataTypeIndex::DataTypeIndex()
      : numeric_count_(0),
        categorical_count_(0),
        unknown_count_(0),
        vnames_(0)
  {}

  void DataTypeIndex::add_variable(VariableType type, const std::string &name) {
    vnames_.push_back(name);
    add_type(type);
  }

  void DataTypeIndex::add_type(VariableType type) {
    int index = type_map_.size();
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

  std::pair<VariableType, int> DataTypeIndex::type_map(int i) const {
    auto it = type_map_.find(i);
    if (it == type_map_.end()) {
      return std::make_pair(VariableType::unknown, -1);
    } else {
      return it->second;
    }
  }

  void DataTypeIndex::diagnose_types(const std::vector<std::string> &fields) {
    for (uint i = 0; i < fields.size(); ++i) {
      VariableType type =
          is_numeric(fields[i]) ? VariableType::numeric : VariableType::categorical;
      add_type(type);
    }
  }

  void DataTypeIndex::set_names(const std::vector<std::string> &variable_names) {
    if (!type_map_.empty()) {
      if (variable_names.size() != type_map_.size()) {
        std::ostringstream err;
        err << variable_names.size() << " names were given to a data set with "
            << type_map_.size() << " variables.";
        report_error(err.str());
      }
    }
    vnames_ = variable_names;
  }

  bool DataTypeIndex::check_type(
      int i,
      const std::string &variable_data_as_string) const {
    VariableType type = variable_type(i);
    if (is_numeric(variable_data_as_string)) {
      if (type == VariableType::numeric) return true;
    } else {  // data is not numeric
      if (type == VariableType::categorical) return true;
    }
    return false;
  }

  bool DataTypeIndex::operator==(const DataTypeIndex &rhs) const {
    return numeric_count_ == rhs.numeric_count_
        && categorical_count_ == rhs.categorical_count_
        && unknown_count_ == rhs.unknown_count_
        && type_map_ == rhs.type_map_;
  }

  int DataTypeIndex::position(const std::string &vname) const {
    auto ans = std::find(vnames_.begin(), vnames_.end(), vname);
    if (ans == vnames_.end()) {
      return -1;
    } else {
      return ans - vnames_.begin();
    }
  }

  //===========================================================================
  MixedMultivariateData::MixedMultivariateData()
      : type_index_(new DataTypeIndex)
  {}

  MixedMultivariateData::MixedMultivariateData(
      const Ptr<DataTypeIndex> &sorter,
      const std::vector<Ptr<DoubleData>> &numerics,
      const std::vector<Ptr<LabeledCategoricalData>> &categoricals)
      : type_index_(sorter),
        numeric_data_(numerics),
        categorical_data_(categoricals)
  {}

  MixedMultivariateData::MixedMultivariateData(const MixedMultivariateData &rhs)
      : type_index_(rhs.type_index_)
  {
    for (int i = 0; i < rhs.numeric_data_.size(); ++i) {
      numeric_data_.push_back(rhs.numeric_data_[i]->clone());
    }
    for (int i = 0; i < rhs.categorical_data_.size(); ++i) {
      categorical_data_.push_back(rhs.categorical_data_[i]->clone());
    }
  }

  MixedMultivariateData &MixedMultivariateData::operator=(
      const MixedMultivariateData &rhs) {
    if (&rhs != this) {
      type_index_ = rhs.type_index_;
      numeric_data_.clear();
      for (int i = 0; i < rhs.numeric_data_.size(); ++i) {
        numeric_data_.push_back(rhs.numeric_data_[i]->clone());
      }

      categorical_data_.clear();
      for (int i = 0; i < rhs.categorical_data_.size(); ++i) {
        categorical_data_.push_back(rhs.categorical_data_[i]->clone());
      }
    }
    return *this;
  }

  MixedMultivariateData *MixedMultivariateData::clone() const {
    return new MixedMultivariateData(*this);
  }

  std::ostream &MixedMultivariateData::display(std::ostream &out) const {
    // TODO: consider taking greater care with field widths.
    for (int i = 0; i < dim(); ++i) {
      out << variable(i) << " ";
    }
    out << std::endl;
    return out;
  }

  void MixedMultivariateData::add_numeric(const Ptr<DoubleData> &numeric,
                                          const std::string &name) {
    type_index_->add_variable(VariableType::numeric, name);
    numeric_data_.push_back(numeric);
  }

  void MixedMultivariateData::add_categorical(
      const Ptr<LabeledCategoricalData> &categorical,
      const std::string &name) {
    type_index_->add_variable(VariableType::categorical, name);
    categorical_data_.push_back(categorical);
  }

  const Data &MixedMultivariateData::variable(int i) const {
    VariableType type;
    int pos;
    std::tie(type, pos) = type_index_->type_map(i);
    if (type == VariableType::numeric) {
      return *numeric_data_[pos];
    } else if (type == VariableType::categorical) {
      return *categorical_data_[pos];
    } else {
      std::ostringstream err;
      err << "Variable in position " << i << " is neither categorical "
          << "nor numeric.";
      report_error(err.str());
    }
    return *numeric_data_[0];
  }

  const DoubleData &MixedMultivariateData::numeric(int i) const {
    VariableType type;
    int pos;
    std::tie(type, pos) = type_index_->type_map(i);
    if (type != VariableType::numeric) {
      std::ostringstream err;
      err << "Variable in position " << i << " is not numeric.";
      report_error(err.str());
    }
    return *numeric_data_[pos];
  }

  Ptr<DoubleData> MixedMultivariateData::mutable_numeric(int i) {
    VariableType type;
    int pos;
    std::tie(type, pos) = type_index_->type_map(i);
    if (type != VariableType::numeric) {
      std::ostringstream err;
      err << "Variable in position " << i << " is not numeric.";
      report_error(err.str());
    }
    return numeric_data_[pos];
  }

  const LabeledCategoricalData &MixedMultivariateData::categorical(int i) const {
    VariableType type;
    int pos;
    std::tie(type, pos) = type_index_->type_map(i);
    if (type != VariableType::categorical) {
      std::ostringstream err;
      err << "Variable in position " << i << " is not categorical.";
      report_error(err.str());
    }
    return *categorical_data_[pos];
  }

  Ptr<LabeledCategoricalData>
  MixedMultivariateData::mutable_categorical(int i) const {
    VariableType type;
    int pos;
    std::tie(type, pos) = type_index_->type_map(i);
    if (type != VariableType::categorical) {
      std::ostringstream err;
      err << "Variable in position " << i << " is not categorical.";
      report_error(err.str());
    }
    return categorical_data_[pos];
  }

  Vector MixedMultivariateData::numeric_data() const {
    Vector ans(numeric_data_.size());
    for (int i = 0; i < numeric_data_.size(); ++i) {
      ans[i] = numeric_data_[i]->value();
    }
    return ans;
  }

  //===========================================================================
  CategoricalVariable::CategoricalVariable(
      const std::vector<std::string> &raw_data)
      : key_(make_catkey(raw_data)) {
    for (int i = 0; i < raw_data.size(); ++i) {
      Ptr<LabeledCategoricalData> dp(new LabeledCategoricalData(raw_data[i], key_));
      data_.push_back(dp);
    }
  }

  CategoricalVariable::CategoricalVariable(
      const std::vector<int> &values,
      const Ptr<CatKey> &key)
      : key_(key) {
    for (const auto &el : values) {
      NEW(LabeledCategoricalData, dp)(el, key_);
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

  DataTable::DataTable()
      : type_index_(new DataTypeIndex)
  {}

  DataTable::DataTable(const std::string &fname,
                       bool header,
                       const std::string &sep)
      : type_index_(new DataTypeIndex)
  {
    read_file(fname, header, sep);
  }

  void DataTable::read_file(const std::string &fname, bool header, const std::string &sep) {
    ifstream in(fname.c_str());
    if (!in) {
      std::ostringstream err;
      err << "Could not open file: " << fname << "\n"
          // TODO add this line when C++17 support is widely available.
          // << "Program running from " << std::filesystem::current_path() << "\n"
      ;
      report_error(err.str());
    }

    StringSplitter split(sep);
    std::string line;
    uint nfields = 0;
    uint line_number = 0;

    std::vector<std::vector<std::string>> categorical_data;
    std::vector<Vector> numeric_data;
    std::vector<std::string> variable_names;

    if (header) {
      ++line_number;
      getline(in, line);
      variable_names = split(line);
      nfields = variable_names.size();
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
        variable_names = default_vnames(nfields);
      }

      if (type_index_->total_number_of_fields() == 0) {
        type_index_->diagnose_types(fields);
        type_index_->set_names(variable_names);
        numeric_data.resize(type_index_->number_of_numeric_fields());
        categorical_data.resize(type_index_->number_of_categorical_fields());
      }

      if (fields.size() != nfields) {  // check number of fields
        field_length_error(fname, line_number, nfields, fields.size());
      }

      for (uint i = 0; i < nfields; ++i) {
        if (variable_type(i) == VariableType::numeric) {
          if (!type_index_->check_type(i, fields[i])) {
            std::ostringstream err;
            err << "Expected a numeric value on line number " << line_number
                << " in field number " << i + 1
                << " (" << variable_names[i] << ").  Got "
                << fields[i] << ".";
            report_error(err.str());
          }
          double tmp = std::stod(fields[i], nullptr);
          int index = type_index_->type_map(i).second;
          numeric_data[index].push_back(tmp);
        } else if (variable_type(i) == VariableType::categorical) {
          int index = type_index_->type_map(i).second;
          categorical_data[index].push_back(fields[i]);
        } else {
          unknown_type();
        }
      }
    }

    for (uint i = 0; i < nfields; ++i) {
      VariableType type;
      int index;
      std::tie(type, index) = type_index_->type_map(i);
      if (type == VariableType::numeric) {
        numeric_variables_.push_back(numeric_data[index]);
      } else if (type == VariableType::categorical) {
        categorical_variables_.emplace_back(categorical_data[index]);
      }
    }
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
      type_index_->add_variable(VariableType::numeric, name);
    } else {
      // If the table is NOT empty, check if the observations for the added
      // variable is same for the previous variables.
      if (nobs() > 0 && nobs() != v.size()) {
        report_error(
            "Wrong sized include vector in DataTable::append_variable");
      } else {
        numeric_variables_.push_back(v);
        type_index_->add_variable(VariableType::numeric, name);
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
      type_index_->add_variable(VariableType::categorical, name);
    } else {
      // If the table is NOT empty, check if the number of observations
      // for the added variable is same for the previous variables.
      if (nobs() > 0 && nobs() != cv.size()) {
        report_error(
            "Wrong sized include vector in DataTable::append_variable");
      } else {
        categorical_variables_.push_back(cv);
        type_index_->add_variable(VariableType::categorical, name);
      }
    }
  }

  void DataTable::append_row(const MixedMultivariateData &row) {
    if (nobs() > 0) {
      if (row.dim() != nvars()) {
        report_error("The number of fields in the new row must match the "
                     "number of columns in the DataTable.");
      }

      int numeric_counter = 0;
      int categorical_counter = 0;
      for (int i = 0; i < nvars(); ++i) {
        VariableType vtype = variable_type(i);
        if (vtype != row.variable_type(i)) {
          std::ostringstream err;
          err << "variable type mismatch in field " << i << ".";
          report_error(err);
        }

        switch(vtype) {
          case VariableType::numeric:
            {
              numeric_variables_[numeric_counter++].push_back(
                  row.numeric(i).value());
            }
            break;

          case VariableType::categorical:
            {
              categorical_variables_[categorical_counter].push_back(
                  row.categorical_data()[categorical_counter]);
              ++categorical_counter;
            }
            break;

          default:
            report_error("Only numeric and categorical types are supported.");
        }
      }


    } else {
      type_index_ = row.type_index_;
      for (int i = 0; i < row.dim(); ++i) {
        VariableType vtype = row.variable_type(i);
        switch (vtype) {
          case VariableType::numeric :
            {
              double value = row.numeric(i).value();
              numeric_variables_.push_back(Vector(1, value));
            }
            break;

          case VariableType::categorical:
            {
              categorical_variables_.push_back(
                  CategoricalVariable(
                      std::vector<Ptr<LabeledCategoricalData>>(
                          1, row.mutable_categorical(i))));
            }
            break;

          default:
            report_error("Only numeric and categorical data types are supported.");

        }
      }
    }
  }

  const std::vector<std::string> &DataTable::vnames() const {
    return type_index_->variable_names();
  }

  //------------------------------------------------------------
  uint DataTable::nvars() const {
    return type_index_->total_number_of_fields();
  }

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
        std::tie(type, index) = type_index_->type_map(J);
        if (type == VariableType::numeric) {
          X(i, column++) = numeric_variables_[index][i];
        } else if (type == VariableType::categorical) {
          const Ptr<LabeledCategoricalData> x(categorical_variables_[index][i]);
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
      std::tie(type, index) = type_index_->type_map(J);
      if (type == VariableType::numeric) {
        dimnames.push_back(vnames()[J]);
      } else if (type == VariableType::categorical) {
        const Ptr<LabeledCategoricalData> x(categorical_variables_[index][0]);
        std::string stub = vnames()[J];
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
    if (*type_index_ != *rhs.type_index_) {
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
        categorical_variables_[i].push_back(new LabeledCategoricalData(value, key));
      }
    }
    return *this;
  }

  //======================================================================
  uint DataTable::nlevels(uint i) const {
    VariableType type;
    int index;
    std::tie(type, index) = type_index_->type_map(i);
    if (type == VariableType::numeric) return 1;
    return categorical_variables_[index][0]->nlevels();
  }

  int DataTable::numeric_dim() const {
    return type_index_->number_of_numeric_fields();
  }

  int DataTable::categorical_dim() const {
    return type_index_->number_of_categorical_fields();
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
  }

  Vector DataTable::getvar(uint n) const {
    VariableType type;
    int index;
    std::tie(type, index) = type_index_->type_map(n);
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

  Vector DataTable::get_numeric(const std::string &vname) const {
    int pos = type_index_->position(vname);
    if (pos < 0) {
      std::ostringstream err;
      err << "'" << vname << "' was not found among the column names.";
      report_error(err.str());
    }
    return getvar(pos);
  }

  double DataTable::getvar(int row, int col) const {
    VariableType type;
    int index;
    std::tie(type, index) = type_index_->type_map(col);
    if (type == VariableType::numeric) {
      return numeric_variables_[index][row];
    } else {
      return negative_infinity();
    }
  }

  CategoricalVariable DataTable::get_nominal(uint n) const {
    VariableType type;
    int index;
    std::tie(type, index) = type_index_->type_map(n);
    if (type != VariableType::categorical) {
      wrong_type_error(1, n);
    }
    return categorical_variables_[index];
  }

  CategoricalVariable DataTable::get_nominal(const std::string &vname) const {
    int position = type_index_->position(vname);
    if (position < 0) {
      std::ostringstream err;
      err << "'" << vname << "' was not found among the column names.";
      report_error(err.str());
    }
    return get_nominal(position);
  }

  Ptr<LabeledCategoricalData> DataTable::get_nominal(int row, int col) const {
    VariableType type;
    int index;
    std::tie(type, index) = type_index_->type_map(col);
    if (type != VariableType::categorical) wrong_type_error(1, col);
    return categorical_variables_[index][row];
  }

  void DataTable::set_numeric_value(int row, int column, double value) {
    VariableType type;
    int index;
    std::tie(type, index) = type_index_->type_map(column);
    if (type != VariableType::numeric) {
      report_error("Attempt to set numerical value to non-numeric variable.");
    }
    numeric_variables_[index][row] = value;
  }

  void DataTable::set_nominal_value(int row, int column, int value) {
    VariableType type;
    int index;
    std::tie(type, index) = type_index_->type_map(column);
    if (type != VariableType::categorical) {
      report_error(
          "Attempt to set categorical value to non-categorical variable.");
    }
    categorical_variables_[index][row]->set(value);
  }

  // DataTable::OrdinalVariable DataTable::get_ordinal(uint n)const{
  //   if (variable_types_[n]!=categorical) wrong_type_error(1, n);
  //   std::vector<Ptr<OrdinalData> > ans;
  //   const std::vector<Ptr<LabeledCategoricalData> > &v(categorical_variables_[n]);
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

  Ptr<MixedMultivariateData> DataTable::row(uint row_index) const {
    std::vector<Ptr<DoubleData>> numerics;
    for (int i = 0; i < numeric_variables_.size(); ++i) {
      numerics.push_back(new DoubleData(numeric_variables_[i][row_index]));
    }
    std::vector<Ptr<LabeledCategoricalData>> categoricals;
    for (int i = 0; i < categorical_variables_.size(); ++i) {
      categoricals.push_back(categorical_variables_[i][row_index]);
    }
    return new MixedMultivariateData(type_index_, numerics, categoricals);
  }

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
      std::tie(type, index) = type_index_->type_map(j);
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
