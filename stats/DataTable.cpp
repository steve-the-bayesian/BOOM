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
#include <cmath>
#include <fstream>
// TODO: add this back when c++17 support is widely available.
// #include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "Models/CategoricalData.hpp"
#include "cpputil/DefaultVnames.hpp"
#include "cpputil/Ptr.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"
#include "cpputil/string_utils.hpp"

#include "distributions.hpp"

namespace {

  // Returns true if s looks like an ISO 8601 date or datetime string.
  // Accepted formats: YYYY-MM-DD, YYYY-MM-DD HH:MM:SS, YYYY-MM-DDTHH:MM:SS
  bool is_datetime_string(const std::string &s) {
    if (s.size() < 10) return false;
    if (s[4] != '-' || s[7] != '-') return false;
    for (int i : {0, 1, 2, 3, 5, 6, 8, 9}) {
      if (!std::isdigit(static_cast<unsigned char>(s[i]))) return false;
    }
    int year  = std::stoi(s.substr(0, 4));
    int month = std::stoi(s.substr(5, 2));
    int day   = std::stoi(s.substr(8, 2));
    if (year < 1000 || year > 9999) return false;
    if (month < 1 || month > 12)    return false;
    if (day   < 1 || day   > 31)    return false;
    if (s.size() >= 19) {
      char sep = s[10];
      if (sep != ' ' && sep != 'T') return false;
      if (s[13] != ':' || s[16] != ':') return false;
      for (int i : {11, 12, 14, 15, 17, 18}) {
        if (!std::isdigit(static_cast<unsigned char>(s[i]))) return false;
      }
    }
    return true;
  }

  BOOM::DateTime parse_datetime_string(const std::string &s) {
    int year  = std::stoi(s.substr(0, 4));
    int month = std::stoi(s.substr(5, 2));
    int day   = std::stoi(s.substr(8, 2));
    double fraction = 0.0;
    if (s.size() >= 19 && (s[10] == ' ' || s[10] == 'T')) {
      int hour = std::stoi(s.substr(11, 2));
      int min  = std::stoi(s.substr(14, 2));
      int sec  = std::stoi(s.substr(17, 2));
      fraction = (hour * 3600.0 + min * 60.0 + sec) / 86400.0;
    }
    return BOOM::DateTime(BOOM::Date(month, day, year), fraction);
  }

}  // namespace

namespace BOOM {
  using std::endl;


  //---------------------------------------------------------------------
  // Determine the type of variable stored in vs.

  DataTypeIndex::DataTypeIndex()
      : numeric_count_(0),
        categorical_count_(0),
        datetime_count_(0),
        high_cardinality_count_(0),
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
    } else if (type == VariableType::datetime) {
      type_map_[index] = std::make_pair(type, datetime_count_++);
    } else if (type == VariableType::high_cardinality) {
      type_map_[index] = std::make_pair(type, high_cardinality_count_++);
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
      VariableType type;
      if (is_numeric(fields[i])) {
        type = VariableType::numeric;
      } else if (is_datetime_string(fields[i])) {
        type = VariableType::datetime;
      } else {
        type = VariableType::categorical;
      }
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
        && datetime_count_ == rhs.datetime_count_
        && high_cardinality_count_ == rhs.high_cardinality_count_
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
    for (int i = 0; i < rhs.datetime_data_.size(); ++i) {
      datetime_data_.push_back(rhs.datetime_data_[i]->clone());
    }
    for (int i = 0; i < rhs.high_cardinality_data_.size(); ++i) {
      high_cardinality_data_.push_back(rhs.high_cardinality_data_[i]->clone());
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

      datetime_data_.clear();
      for (int i = 0; i < rhs.datetime_data_.size(); ++i) {
        datetime_data_.push_back(rhs.datetime_data_[i]->clone());
      }

      high_cardinality_data_.clear();
      for (int i = 0; i < rhs.high_cardinality_data_.size(); ++i) {
        high_cardinality_data_.push_back(rhs.high_cardinality_data_[i]->clone());
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

  void MixedMultivariateData::add_datetime(
      const Ptr<DateTimeData> &dt,
      const std::string &name) {
    type_index_->add_variable(VariableType::datetime, name);
    datetime_data_.push_back(dt);
  }

  void MixedMultivariateData::add_high_cardinality(
      const Ptr<StringData> &value,
      const std::string &name) {
    type_index_->add_variable(VariableType::high_cardinality, name);
    high_cardinality_data_.push_back(value);
  }

  const Data &MixedMultivariateData::variable(int i) const {
    VariableType type;
    int pos;
    std::tie(type, pos) = type_index_->type_map(i);
    if (type == VariableType::numeric) {
      return *numeric_data_[pos];
    } else if (type == VariableType::categorical) {
      return *categorical_data_[pos];
    } else if (type == VariableType::datetime) {
      return *datetime_data_[pos];
    } else if (type == VariableType::high_cardinality) {
      return *high_cardinality_data_[pos];
    } else {
      std::ostringstream err;
      err << "Variable in position " << i << " has an unsupported type.";
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

  const DoubleData &MixedMultivariateData::numeric(
      const std::string &variable_name) const {
    return numeric(type_index_->position(variable_name));
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

  Ptr<DoubleData> MixedMultivariateData::mutable_numeric(
      const std::string &name) {
    return mutable_numeric(get_position(name));
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

  const LabeledCategoricalData &MixedMultivariateData::categorical(
      const std::string &variable_name) const {
    return categorical(type_index_->position(variable_name));
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

  Ptr<LabeledCategoricalData> MixedMultivariateData::mutable_categorical(
      const std::string &variable_name) {
    return mutable_categorical(type_index_->position(variable_name));
  }

  const DateTimeData &MixedMultivariateData::datetime(int i) const {
    VariableType type;
    int pos;
    std::tie(type, pos) = type_index_->type_map(i);
    if (type != VariableType::datetime) {
      std::ostringstream err;
      err << "Variable in position " << i << " is not datetime.";
      report_error(err.str());
    }
    return *datetime_data_[pos];
  }

  const DateTimeData &MixedMultivariateData::datetime(
      const std::string &variable_name) const {
    return datetime(type_index_->position(variable_name));
  }

  Ptr<DateTimeData> MixedMultivariateData::mutable_datetime(int i) {
    VariableType type;
    int pos;
    std::tie(type, pos) = type_index_->type_map(i);
    if (type != VariableType::datetime) {
      std::ostringstream err;
      err << "Variable in position " << i << " is not datetime.";
      report_error(err.str());
    }
    return datetime_data_[pos];
  }

  Ptr<DateTimeData> MixedMultivariateData::mutable_datetime(
      const std::string &variable_name) {
    return mutable_datetime(type_index_->position(variable_name));
  }

  const StringData &MixedMultivariateData::high_cardinality(int i) const {
    VariableType type;
    int pos;
    std::tie(type, pos) = type_index_->type_map(i);
    if (type != VariableType::high_cardinality) {
      std::ostringstream err;
      err << "Variable in position " << i << " is not high_cardinality.";
      report_error(err.str());
    }
    return *high_cardinality_data_[pos];
  }

  const StringData &MixedMultivariateData::high_cardinality(
      const std::string &variable_name) const {
    return high_cardinality(type_index_->position(variable_name));
  }

  Ptr<StringData> MixedMultivariateData::mutable_high_cardinality(int i) {
    VariableType type;
    int pos;
    std::tie(type, pos) = type_index_->type_map(i);
    if (type != VariableType::high_cardinality) {
      std::ostringstream err;
      err << "Variable in position " << i << " is not high_cardinality.";
      report_error(err.str());
    }
    return high_cardinality_data_[pos];
  }

  Ptr<StringData> MixedMultivariateData::mutable_high_cardinality(
      const std::string &variable_name) {
    return mutable_high_cardinality(get_position(variable_name));
  }

  Vector MixedMultivariateData::numeric_data() const {
    Vector ans(numeric_data_.size());
    for (int i = 0; i < numeric_data_.size(); ++i) {
      ans[i] = numeric_data_[i]->value();
    }
    return ans;
  }

  int MixedMultivariateData::get_position(const std::string &name) const {
    int position = type_index_->position(name);
    if (position < 0) {
      std::ostringstream err;
      err << "MixedMultivariateData contains no variable named " << name
          << ".";
      report_error(err.str());
    }
    return position;
  }

  //===========================================================================
  CategoricalVariable::CategoricalVariable(
      const std::vector<std::string> &raw_data)
      : key_(make_catkey(raw_data)) {
    for (int i = 0; i < raw_data.size(); ++i) {
      Ptr<LabeledCategoricalData> dp(
          new LabeledCategoricalData(raw_data[i], key_));
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

  void DataTable::read_file(const std::string &fname,
                            bool header,
                            const std::string &sep) {
    // Clear any existing data.
    numeric_variables_.clear();
    categorical_variables_.clear();
    datetime_variables_.clear();
    high_cardinality_variables_.clear();
    type_index_.reset(new DataTypeIndex);

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
    std::vector<std::string> variable_names;

    if (header) {
      ++line_number;
      getline(in, line);
      variable_names = split(line);
      nfields = variable_names.size();
    }

    // First pass: read all data as raw strings stored column-major.
    std::vector<std::vector<std::string>> raw_columns;
    if (nfields > 0) {
      raw_columns.resize(nfields);
    }
    while (in) {
      ++line_number;
      getline(in, line);
      if (is_all_white(line)) continue;
      std::vector<std::string> fields = split(line);

      if (nfields == 0) {
        nfields = fields.size();
        variable_names = default_vnames(nfields);
        raw_columns.resize(nfields);
      }

      if (fields.size() != nfields) {
        field_length_error(fname, line_number, nfields, fields.size());
      }

      for (uint i = 0; i < nfields; ++i) {
        raw_columns[i].push_back(fields[i]);
      }
    }

    if (nfields == 0 || raw_columns.empty() || raw_columns[0].empty()) {
      return;
    }

    // Second pass: classify each column and populate the DataTable.
    //
    // High cardinality threshold matches the Python definition in summary.py:
    //   cardinality_limit = max(5, int(cbrt(sample_size)))
    // A categorical column with >= cardinality_limit unique values is treated
    // as high cardinality.
    int sample_size = static_cast<int>(raw_columns[0].size());
    int high_cardinality_limit =
        std::max(5, static_cast<int>(std::cbrt(sample_size)));

    for (uint col = 0; col < nfields; ++col) {
      const std::vector<std::string> &column = raw_columns[col];
      const std::string &vname = variable_names[col];

      // Check numeric: all values parseable as a number.
      bool all_numeric = true;
      for (const auto &val : column) {
        if (!is_numeric(val)) { all_numeric = false; break; }
      }
      if (all_numeric) {
        Vector v(sample_size);
        for (int i = 0; i < sample_size; ++i) {
          v[i] = std::stod(column[i]);
        }
        append_variable(v, vname);
        continue;
      }

      // Check datetime: all values match ISO 8601 date/datetime format.
      bool all_datetime = true;
      for (const auto &val : column) {
        if (!is_datetime_string(val)) { all_datetime = false; break; }
      }
      if (all_datetime) {
        std::vector<DateTime> dt_values;
        dt_values.reserve(sample_size);
        for (const auto &val : column) {
          dt_values.push_back(parse_datetime_string(val));
        }
        append_variable(DateTimeVariable(dt_values), vname);
        continue;
      }

      // Categorical or high cardinality based on number of unique values.
      std::unordered_set<std::string> unique_values(column.begin(), column.end());
      if (static_cast<int>(unique_values.size()) >= high_cardinality_limit) {
        append_variable(HighCardinalityVariable(column), vname);
      } else {
        append_variable(CategoricalVariable(column), vname);
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

  void DataTable::append_variable(const DateTimeVariable &dt,
                                  const std::string &name) {
    if (nobs() > 0  && nobs() != dt.size()) {
        std::ostringstream err;
        err << "A DateTime variable with " << dt.size()
            << " observations cannot be added to a DataTable "
            << "with " << nobs() << " rows.";
        report_error(err.str());
    }
    datetime_variables_.push_back(dt);
    type_index_->add_variable(VariableType::datetime, name);
  }

  void DataTable::append_variable(const HighCardinalityVariable &id,
                                  const std::string &name) {
    if (nobs() > 0  && nobs() != id.size()) {
        std::ostringstream err;
        err << "A HighCardinalityVariable with " << id.size()
            << " observations cannot be added to a DataTable "
            << "with " << nobs() << " rows.";
        report_error(err.str());
    }
    high_cardinality_variables_.push_back(id);
    type_index_->add_variable(VariableType::high_cardinality, name);
  }

  void DataTable::append_row(const MixedMultivariateData &row) {
    if (nobs() > 0) {
      if (row.dim() != nvars()) {
        report_error("The number of fields in the new row must match the "
                     "number of columns in the DataTable.");
      }

      int numeric_counter = 0;
      int categorical_counter = 0;
      int datetime_counter = 0;
      int high_cardinality_counter = 0;
      for (int i = 0; i < nvars(); ++i) {
        VariableType vtype = variable_type(i);
        if (vtype != row.variable_type(i)) {
          std::ostringstream err;
          err << "variable type mismatch in field " << i << ".";
          report_error(err);
        }

        switch(vtype) {
          case VariableType::numeric:
            numeric_variables_[numeric_counter++].push_back(
                row.numeric(i).value());
            break;

          case VariableType::categorical:
            categorical_variables_[categorical_counter].push_back(
                row.categorical_data()[categorical_counter]);
            ++categorical_counter;
            break;

          case VariableType::datetime:
            datetime_variables_[datetime_counter++].push_back(
                row.datetime(i).value());
            break;

          case VariableType::high_cardinality:
            high_cardinality_variables_[high_cardinality_counter++].push_back(
                row.high_cardinality(i).value());
            break;

          default:
            report_error("Unsupported variable type in DataTable::append_row.");
        }
      }


    } else {
      type_index_ = row.type_index_;
      for (int i = 0; i < row.dim(); ++i) {
        VariableType vtype = row.variable_type(i);
        switch (vtype) {
          case VariableType::numeric:
            numeric_variables_.push_back(Vector(1, row.numeric(i).value()));
            break;

          case VariableType::categorical:
            categorical_variables_.push_back(
                CategoricalVariable(
                    std::vector<Ptr<LabeledCategoricalData>>(
                        1, row.mutable_categorical(i))));
            break;

          case VariableType::datetime:
            datetime_variables_.push_back(
                DateTimeVariable(
                    std::vector<DateTime>(1, row.datetime(i).value())));
            break;

          case VariableType::high_cardinality:
            high_cardinality_variables_.push_back(
                HighCardinalityVariable(
                    std::vector<std::string>(1, row.high_cardinality(i).value())));
            break;

          default:
            report_error("Unsupported variable type in DataTable::append_row.");
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

  LabeledMatrix DataTable::design(bool add_intercept) const {
    std::vector<bool> include(nvars(), true);
    return design(Selector(include), add_intercept);
  }

  //------------------------------------------------------------
  LabeledMatrix DataTable::design(const Selector &include,
                                  bool add_intercept) const {

    std::ostringstream err;
    err << "DataTable::design is deprecated.  Please migrate code to "
        "use the methods in 'stats/Encoders.hpp' instead.";
    report_warning(err.str());
    
    uint dimension = add_intercept ? 1 : 0;
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
      if (add_intercept) X(i, 0) = 1.0;
      uint column = add_intercept ? 1 : 0;
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
    if (add_intercept) {
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
    for (int i = 0; i < datetime_variables_.size(); ++i) {
      for (const DateTime &dt : rhs.datetime_variables_[i].data()) {
        datetime_variables_[i].push_back(dt);
      }
    }
    for (int i = 0; i < high_cardinality_variables_.size(); ++i) {
      for (const std::string &s : rhs.high_cardinality_variables_[i].data()) {
        high_cardinality_variables_[i].push_back(s);
      }
    }
    return *this;
  }

  DataTable &DataTable::cbind(const DataTable &rhs) {
    for (uint i = 0; i < rhs.ncol(); ++i) {
      const std::string &vname(rhs.vnames()[i]);

      switch(rhs.variable_type(i)) {
        case (VariableType::numeric): {
          append_variable(rhs.get_numeric(i), vname);
        }
          break;

        case (VariableType::categorical) : {
          append_variable(rhs.get_nominal(i), vname);
        }
          break;

        case (VariableType::datetime) : {
          append_variable(rhs.get_datetime(i), vname);
        }
          break;

        case (VariableType::high_cardinality) : {
          append_variable(rhs.get_high_cardinality(i), vname);
        }
          break;

        default:
          report_error("Unsupported variable type in DataTable::cbind.");
      }
    }
    return *this;
  }

  //======================================================================
  int DataTable::nlevels(uint i) const {
    VariableType type;
    int index;
    std::tie(type, index) = type_index_->type_map(i);
    if (type == VariableType::unknown) {
      return 0;
    } else if (type != VariableType::categorical) {
      return 1;
    } else {
      return categorical_variables_[index][0]->nlevels();
    }
  }

  int DataTable::numeric_dim() const {
    return type_index_->number_of_numeric_fields();
  }

  int DataTable::categorical_dim() const {
    return type_index_->number_of_categorical_fields();
  }

  int DataTable::datetime_dim() const {
    return type_index_->number_of_datetime_fields();
  }

  int DataTable::nrow() const {
    if (!numeric_variables_.empty()) return numeric_variables_[0].size();
    if (!categorical_variables_.empty()) return categorical_variables_[0].size();
    if (!datetime_variables_.empty()) return datetime_variables_[0].size();
    if (!high_cardinality_variables_.empty()) return high_cardinality_variables_[0].size();
    return 0;
  }

  Vector DataTable::getvar(uint which_column) const {
    VariableType type;
    int index;
    std::tie(type, index) = type_index_->type_map(which_column);
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

  DateTimeVariable DataTable::get_datetime(uint which_column) const {
    VariableType type;
    int index;
    std::tie(type, index) = type_index_->type_map(which_column);
    if (type != VariableType::datetime) {
      wrong_type_error(1, which_column);
    }
    return datetime_variables_[index];
  }

  DateTimeVariable DataTable::get_datetime(const std::string &vname) const {
    int position = type_index_->position(vname);
    if (position < 0) {
      std::ostringstream err;
      err << "'" << vname << "' was not found among the column names.";
      report_error(err.str());
    }
    return get_datetime(position);
  }

  HighCardinalityVariable DataTable::get_high_cardinality(uint which_column) const {
    VariableType type;
    int index;
    std::tie(type, index) = type_index_->type_map(which_column);
    if (type != VariableType::high_cardinality) {
      wrong_type_error(1, which_column);
    }
    return high_cardinality_variables_[index];
  }

  HighCardinalityVariable DataTable::get_high_cardinality(const std::string &vname) const {
    int position = type_index_->position(vname);
    if (position < 0) {
      std::ostringstream err;
      err << "'" << vname << "' was not found among the column names.";
      report_error(err.str());
    }
    return get_high_cardinality(position);
  }
  
  Ptr<MixedMultivariateData> DataTable::row(uint row_index) const {
    Ptr<MixedMultivariateData> result(new MixedMultivariateData);
    for (uint i = 0; i < nvars(); ++i) {
      VariableType type;
      int index;
      std::tie(type, index) = type_index_->type_map(i);
      const std::string &name = vnames()[i];
      switch (type) {
        case VariableType::numeric:
          result->add_numeric(
              new DoubleData(numeric_variables_[index][row_index]), name);
          break;
        case VariableType::categorical:
          result->add_categorical(
              categorical_variables_[index][row_index], name);
          break;
        case VariableType::datetime:
          result->add_datetime(
              new DateTimeData(datetime_variables_[index].data()[row_index]),
              name);
          break;
        case VariableType::high_cardinality:
          result->add_high_cardinality(
              new StringData(
                  high_cardinality_variables_[index].data()[row_index]),
              name);
          break;
        default:
          report_error("Unsupported variable type in DataTable::row().");
      }
    }
    return result;
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
        } else if (type == VariableType::categorical) {
          sout << categorical_variables_[index].label(i);
        } else if (type == VariableType::datetime) {
          sout << datetime_variables_[index].data()[i];
        } else if (type == VariableType::high_cardinality) {
          sout << high_cardinality_variables_[index].data()[i];
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

  DataTable repeat(const MixedMultivariateData &dp, int ntimes) {
    DataTable ans;
    if (ntimes <= 0) {
      return ans;
    }

    for (int i = 0; i < dp.nvars(); ++i) {
      const std::string &vname(dp.vnames()[i]);
      switch (dp.variable_type(i)) {
        case VariableType::numeric: {
          double value = dp.numeric(i).value();
          ans.append_variable(Vector(ntimes, value), vname);
        }
          break;

        case VariableType::categorical: {
          const LabeledCategoricalData &data_point(
              dp.categorical(i));
          ans.append_variable(
              CategoricalVariable(
                  std::vector<int>(ntimes, data_point.value()),
                  data_point.catkey()),
              vname);
        }
          break;

        case VariableType::datetime: {
          DateTime dt = dp.datetime(i).value();
          ans.append_variable(
              DateTimeVariable(std::vector<DateTime>(ntimes, dt)),
              vname);
        }
          break;

        case VariableType::high_cardinality: {
          const std::string &s = dp.high_cardinality(i).value();
          ans.append_variable(
              HighCardinalityVariable(std::vector<std::string>(ntimes, s)),
              vname);
        }
          break;

        default:
          report_error("Unknown variable type encountered in repeat().");
      }
    }
    return ans;
  }
  //======================================================================

  inline std::vector<std::string> random_strings(int sample_size, int string_length) {
    std::string char_set("", 26 + 26 + 10);
    for (int i = 0; i < 26; ++i) {
      char_set[i] = 'a' + i;
      char_set[26 + i] = 'A' + i;
    }
    for (int i = 0; i < 10; ++i) {
      char_set[26 + 26 + i] = '0' + i;
    }

    std::vector<std::string> ans;
    ans.reserve(sample_size);
    for (int i = 0; i < sample_size; ++i) {
      std::string my_string;
      my_string.reserve(string_length);
      for (int j = 0; j< string_length; ++j) {
        my_string.push_back(char_set[rmulti(0, string_length - 1)]);
      }
      ans.push_back(my_string);
    }
    return ans;
  }

  
  DataTable simulate_fake_data_table(
      int sample_size,
      const std::vector<std::string> &numeric_field_names,
      const std::map<std::string, std::vector<std::string>> &categorical_levels,
      const std::map<std::string, std::pair<DateTime, DateTime>> &datetime_fields,
      const std::map<std::string, int> high_cardinality_fields) {

    DataTable data;
    
    for (int i = 0; i < numeric_field_names.size(); ++i) {
      Vector variable(sample_size);
      variable.randomize();
      data.append_variable(variable, numeric_field_names[i]);
    }

    for (const auto &val : categorical_levels) {
      std::string variable_name = val.first;
      NEW(CatKey, key)(val.second);
      std::vector<int> indices(sample_size);
      if (indices.size() > 0) {
        for (int i = 0; i < sample_size; ++i) {
          indices[i] = rmulti(0, key->max_levels() - 1);
        }
      }
      data.append_variable(CategoricalVariable(indices, key),
                           variable_name);
    }

    for (const auto &val : datetime_fields) {
      std::string variable_name = val.first;
      DateTime start = val.second.first;
      DateTime stop = val.second.second;
      if (stop < start) {
        std::swap(stop, start);
      }
      double dt = stop - start;
      std::vector<DateTime> values(sample_size);
      DateTime::TimeScale days = DateTime::day_scale;
      for (int i = 0; i < sample_size; ++i) {
        double julian = runif(0, dt);
        values[i] = DateTime(julian, days);
      }
      data.append_variable(DateTimeVariable(values), variable_name);
    }

    for (const auto &val : high_cardinality_fields) {
      std::string variable_name = val.first;
      data.append_variable(
          HighCardinalityVariable(random_strings(sample_size, val.second)),
          variable_name);
    }
    
    return data;
  }


}  // namespace BOOM
