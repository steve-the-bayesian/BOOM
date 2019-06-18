// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2015 Steven L. Scott

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

#ifndef BOOM_DATA_TABLE_HPP
#define BOOM_DATA_TABLE_HPP

#include "uint.hpp"

#include "LinAlg/Matrix.hpp"
#include "LinAlg/Selector.hpp"
#include "LinAlg/Vector.hpp"

#include <limits>
#include "Models/CategoricalData.hpp"
#include "Models/DataTypes.hpp"

namespace BOOM {

  // A CategoricalVariable is a column of CategoricalData.  The data are
  // assumed to come in string format, so a CatKey is used to handle the
  // mapping between the string values and the values of the categorical data
  // elements.
  class CategoricalVariable {
   public:
    CategoricalVariable() = default;
    explicit CategoricalVariable(const std::vector<std::string> &raw_data);
    CategoricalVariable(const std::vector<Ptr<CategoricalData>> &data,
                        const Ptr<CatKey> &key)
        : key_(key), data_(data) {}

    Ptr<CategoricalData> operator[](uint i) { return data_[i]; }
    const Ptr<CategoricalData> operator[](uint i) const { return data_[i]; }
    const std::vector<std::string> &labels() const { return key_->labels(); }

    // Return the label of the ith data point.
    const std::string &label(int observation_number) const {
      return key_->label(data_[observation_number]->value());
    }

    int size() const { return data_.size(); }
    bool empty() const { return data_.empty(); }
    void push_back(const Ptr<CategoricalData> &element) {
      data_.push_back(element);
      key_->Register(element.get());
    }
    void set_order(const std::vector<std::string> &level_names) {
      key_->reorder(level_names);
    }

    Ptr<CatKey> key() { return key_; }
    const std::vector<Ptr<CategoricalData>> &data() const { return data_; }

   private:
    Ptr<CatKey> key_;
    std::vector<Ptr<CategoricalData>> data_;
  };

  // class OrdinalVariable {
  //   public:
  //   OrdinalVariable() = default;
  //   OrdinalVariable(const std::vector<std::string> &raw_data,
  //                   const std::vector<std::string> &order);
  //   private:
  //   Ptr<CatKey> key_;
  //   std::vector<Ptr<OrdinalData>> data_;
  // };

  // A DataTable is created by reading a plain text file and storing
  // "variables" in a table.  Variables can be extracted from the
  // DataTable either individually (e.g. to get the y variable for a
  // regression/classification problem) or as a design matrix with
  // column labels giving the variable names.  When building the
  // design matrix, special care is taken to properly name dummy
  // variables.
  class DataTable : public Data {
   public:
    typedef std::vector<double> dvector;
    enum VariableType { unknown = -1, continuous, categorical };
    typedef std::vector<std::string> StringVector;

    //--- constructors ---
    // Creates an empty data table.
    DataTable();

    // Creates a data table from a file
    // Args:
    //   fname:  The name of the file to read in.
    //   header: If 'true' then the first line of the file contains
    //     variable names.  If 'false' then the first lineof the file
    //     is the first observation, and variable names will be
    //     automatically generated.
    //   sep: The separator between fields in the data file.
    explicit DataTable(const std::string &fname, bool header = false,
                       const std::string &sep = "");

    DataTable *clone() const override;
    std::ostream &display(std::ostream &out) const override;

    //--- build a DataTable by appending variables ---
    void append_variable(const Vector &v, const std::string &name);
    void append_variable(const CategoricalVariable &cv, const std::string &name);

    //--- size  ---
    uint nvars() const;          // number of variables stored in the table
    uint nobs() const;           // number of observations
    uint nlevels(uint i) const;  // 1 for continuous, nlevels for categorical

    //--- look inside ---
    std::ostream &print(std::ostream &out, uint from = 0,
                        uint to = std::numeric_limits<uint>::max()) const;

    const std::vector<VariableType> &display_variable_types() const;

    StringVector &vnames();
    const StringVector &vnames() const;

    //--- extract variables ---
    // Get column 'which_column' from the table.
    VariableType variable_type(uint which_column) const;
    Vector getvar(uint which_column) const;
    CategoricalVariable get_nominal(uint which_column) const;
    //    OrdinalVariable get_ordinal(uint which_column) const;
    //    OrdinalVariable get_ordinal(uint which_column, const StringVector
    //    &ord) const;

    //--- Compute a design matrix ---
    LabeledMatrix design(bool add_icpt = false) const;
    LabeledMatrix design(const Selector &include, bool add_icpt = false) const;

    // Bind the rows of rhs below the rows of *this.  The variable
    // types of *this and rhs must match, and all categorical
    // variables must have the same levels.  A reference to *this is
    // returned.
    DataTable &rbind(const DataTable &rhs);

   private:
    std::vector<Vector> continuous_variables_;
    std::vector<CategoricalVariable> categorical_variables_;

    std::vector<VariableType> variable_types_;
    std::vector<std::string> vnames_;
    void diagnose_types(const std::vector<std::string> &);
    bool check_type(VariableType type, const std::string &s) const;
  };

  std::ostream &operator<<(std::ostream &out, const DataTable &dt);

  struct VariableSummary {
    DataTable::VariableType type;
    double min;
    double max;
    // mean and standard deviation are not used if type == 'categorical'
    double mean;
    double standard_deviation;
    int number_of_distinct_values;
  };

  std::vector<VariableSummary> summarize(const DataTable &table);

}  // namespace BOOM
#endif  // BOOM_DATA_TABLE_HPP
