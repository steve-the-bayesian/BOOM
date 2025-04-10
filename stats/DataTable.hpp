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

#include <limits>
#include "uint.hpp"

#include "LinAlg/Matrix.hpp"
#include "LinAlg/Selector.hpp"
#include "LinAlg/Vector.hpp"

#include "Models/CategoricalData.hpp"
#include "Models/DataTypes.hpp"

#include "cpputil/RefCounted.hpp"
#include "cpputil/DateTime.hpp"

namespace BOOM {

  // A DataTable is a rectangular array of mixed type data.  Each column in a
  // DataTable contains data of homogeneous type: numeric, categorical, ordinal,
  // datetime, etc.
  //
  // A single row of a DataTable is represented by the type
  // MixedMultivariateData.  The type information in a DataTable and a
  // MixedMultivariateData is kept in a DataTypeIndex, which can be shared by a
  // DataTable and potentially multiple MixedMultivariateData objects.
  //
  // Columns in a DataTable are represented by different data types.  Vector is
  // used for numeric data.  CategoricalVariable and DateTimeVariable are used
  // for categorical and DateTime objects, respectively.  Using these variable
  // types (instead of vectors of integers and DateTime objects) allows for
  // observation-by-observation data missingness.

  enum class VariableType {unknown = -1, numeric, categorical, datetime};

  using DateTimeData = UnivData<DateTime>;

  //===========================================================================
  // Tracks the data types in a mixed data settting.  A DataTable stores its
  // columns using vectors of homogeneous type.  If the first and third
  // variables in a three column table are numeric, and the second is
  // categorical, then the DataTable will have a vector of 2 numeric variables
  // and another vector containing one CategoricalVariable.
  //
  // This class helps keep these vectors of vectors lined up as the 'columns' of
  // the table.
  class DataTypeIndex : public RefCounted {
   public:

    DataTypeIndex();

    // Make the index aware that the DataTable it supports has added a variable
    // of the gien type with the given name.
    void add_variable(VariableType type, const std::string &name);

    // Return the type of variable in column i of the data table, and its
    // position in the data store for variables of that type.
    //
    // Example, in a 3-column data frame with column types numeric,
    // categorical, numeric, here are the return values:
    //
    // type_map(0): numeric, 0
    // type_map(1): categorical, 0
    // type_map(2): numeric, 1
    std::pair<VariableType, int> type_map(int i) const;

    // Fill the type_map_ mapping according to whether each element of 'fields'
    // can be converted to a numeric value.
    //
    // Args:
    //   fields: Each element is a string version of a field in a data table.
    //
    // Effects:
    //   The internal data structures of this object are inferred from the
    //   fields.  If a field is a numeric value, the corresponding column will
    //   be labelled as numeric.  Otherwise a field will be labelled as
    //   categorical. There is no (as of late 2024) check for whether a field is
    //   datetime.
    void diagnose_types(const std::vector<std::string> &fields);

    void set_names(const std::vector<std::string> &variable_names);

    VariableType variable_type(int col) const {
      return type_map_.find(col)->second.first;
    }

    // Return true iff the string 's' appears convertible to the type stored in
    // the data frame's column i.  The check only looks for numeric data.
    //
    // TODO: add a check for datetime.
    bool check_type(int i, const std::string &s) const;

    int number_of_numeric_fields() const {return numeric_count_;}
    int number_of_categorical_fields() const {return categorical_count_;}
    int number_of_datetime_fields() const {return datetime_count_;}
    int number_of_unknown_fields() const {return unknown_count_;}
    int total_number_of_fields() const {
      return numeric_count_
          + categorical_count_
          + datetime_count_
          + unknown_count_;
    }

    bool operator==(const DataTypeIndex &rhs) const;
    bool operator!=(const DataTypeIndex &rhs) const {
      return !(*this == rhs);
    }

    const std::vector<std::string> &variable_names() const {
      return vnames_;
    }

    const std::string &variable_name(int i) const {
      return vnames_[i];
    }

    int position(const std::string &vname) const;

   private:
    int numeric_count_;
    int categorical_count_;
    int datetime_count_;
    int unknown_count_;

    // To find the variable in slot i, get its type and its index in private
    // storage.  For example, if variable 7 is the 4th numeric variable (and
    // thus index 3) then type_map[7] = {numeric, 3}.
    std::map<int, std::pair<VariableType, int>> type_map_;

    // Names of the stored variables.
    std::vector<std::string> vnames_;

    bool check_type(VariableType type,
                    const std::string &variable_data_as_string) const;

    // Add a type to the type map.
    void add_type(VariableType type);

    friend void intrusive_ptr_add_ref(DataTypeIndex *d) {
      d->up_count();
    }

    friend void intrusive_ptr_release(DataTypeIndex *d) {
      d->down_count();
      if (d->ref_count() == 0) {
        delete d;
      }
    }
  };

  //===========================================================================
  // MixedMultivariateData is a "row" in a data table.  It can have categorical
  // or numeric data in each cell, with possible other types to be added later
  // (ordinal, datetime, ...).
  class MixedMultivariateData : public Data {
    // Friendship is merited here because a MixedMultivariateData is a row in a
    // data table.
    friend class DataTable;

   public:
    MixedMultivariateData();
    MixedMultivariateData(
        const Ptr<DataTypeIndex> &type_index,
        const std::vector<Ptr<DoubleData>> &numerics,
        const std::vector<Ptr<LabeledCategoricalData>> &categoricals);
    MixedMultivariateData(const MixedMultivariateData &rhs);
    MixedMultivariateData &operator=(const MixedMultivariateData &rhs);
    MixedMultivariateData(MixedMultivariateData &&rhs) = default;
    MixedMultivariateData &operator=(MixedMultivariateData &&rhs) = default;

    MixedMultivariateData *clone() const override;
    std::ostream &display(std::ostream &out) const override;

    void add_numeric(const Ptr<DoubleData> &numeric,
                     const std::string &name = "");
    void add_categorical(const Ptr<LabeledCategoricalData> &categorical,
                         const std::string &name = "");
    void add_datetime(const Ptr<DateTimeData> &dt,
                      const std::string &name = "");

    // Variable names.
    const std::vector<std::string> &vnames() const {
      return type_index_->variable_names();
    }

    // The number of numeric variables.
    int numeric_dim() const {
      return type_index_->number_of_numeric_fields();
    }

    // The number of categorical variables.  This is the number of categorical
    // columns in the "data frame", not its dummy variable expansion.
    int categorical_dim() const {
      return type_index_->number_of_categorical_fields();
    }

    // The total number of variables.
    int dim() const {return type_index_->total_number_of_fields();}

    // Similarly named size functions in DataTable.
    int nvars() const {return dim();}
    int nrow() const {return 1;}
    int nobs() const {return 1;}

    // The type of variable in cell i.
    VariableType variable_type(int i) const {
      return type_index_->variable_type(i);
    }

    const Data &variable(int i) const;

    // Return the entry in cell i if it is of the requested type.  If it is
    // not then raise an error.
    const DoubleData &numeric(int i) const;
    Ptr<DoubleData> mutable_numeric(int i);
    
    const DoubleData &numeric(const std::string &variable_name) const;
    Ptr<DoubleData> mutable_numeric(const std::string &variable_name);

    const LabeledCategoricalData &categorical(int i) const;
    Ptr<LabeledCategoricalData> mutable_categorical(int i) const;
    
    const LabeledCategoricalData &categorical(
        const std::string &variable_name) const;
    Ptr<LabeledCategoricalData> mutable_categorical(
        const std::string &variable_name);

    const DateTimeData &datetime(int i) const;
    Ptr<DateTimeData> mutable_datetime(int i);
    const DateTimeData &datetime(const std::string &variable_name) const;
    Ptr<DateTimeData> mutable_datetime(const std::string &variable_name);

    // Collapse all the numeric data into a vector.
    Vector numeric_data() const;

    const std::vector<Ptr<LabeledCategoricalData>> &categorical_data() const {
      return categorical_data_;
    }

    Ptr<DataTypeIndex> type_index() const {
      return type_index_;
    }

   private:
    // Returns the position number of the variable named 'name'.  If 'name' is
    // not the name of a variable then -1 gets returned.
    int get_position(const std::string &name) const;
    
    Ptr<DataTypeIndex> type_index_;
    std::vector<Ptr<DoubleData>> numeric_data_;
    std::vector<Ptr<LabeledCategoricalData>> categorical_data_;
    std::vector<Ptr<DateTimeData>> datetime_data_;
  };

  //===========================================================================
  // A CategoricalVariable is a column of LabeledCategoricalData.  The data are
  // assumed to come in string format, so a CatKey is used to handle the
  // mapping between the string values and the values of the categorical data
  // elements.
  class CategoricalVariable {
   public:
    CategoricalVariable() = default;
    explicit CategoricalVariable(const std::vector<std::string> &raw_data);
    CategoricalVariable(const std::vector<int> &values, const Ptr<CatKey> &key);
    explicit CategoricalVariable(
        const std::vector<Ptr<LabeledCategoricalData>> &data)
        : key_(data[0]->catkey()), data_(data) {}

    Ptr<LabeledCategoricalData> operator[](uint i) { return data_[i]; }
    const Ptr<LabeledCategoricalData> &operator[](uint i) const {
      return data_[i];
    }
    const std::vector<std::string> &labels() const { return key_->labels(); }

    // Return the label of the ith data point.
    const std::string &label(int observation_number) const {
      return key_->label(data_[observation_number]->value());
    }

    int size() const { return data_.size(); }
    bool empty() const { return data_.empty(); }
    void push_back(const Ptr<LabeledCategoricalData> &element) {
      data_.push_back(element);
      key_->Register(element.get());
    }
    void set_order(const std::vector<std::string> &level_names) {
      key_->reorder(level_names);
    }

    Ptr<CatKey> key() { return key_; }
    const std::vector<Ptr<LabeledCategoricalData>> &data() const {
      return data_;
    }

   private:
    Ptr<CatKey> key_;
    std::vector<Ptr<LabeledCategoricalData>> data_;
  };

  //===========================================================================
  // Timestamps are an important data type that are distinct from "numeric" or
  // "categorical" data.

  class DateTimeVariable {
   public:
    DateTimeVariable(const std::vector<DateTime> &dt)
        : data_(dt)
    {}

    int size() const {return data_.size();}

    const std::vector<DateTime> &data() const {
      return data_;
    }

   private:
    std::vector<DateTime> data_;
  };

  //===========================================================================
  // class OrdinalVariable {
  //   public:
  //   OrdinalVariable() = default;
  //   OrdinalVariable(const std::vector<std::string> &raw_data,
  //                   const std::vector<std::string> &order);
  //   private:
  //   Ptr<CatKey> key_;
  //   std::vector<Ptr<OrdinalData>> data_;
  // };

  //===========================================================================
  // A DataTable is created by reading a plain text file and storing "variables"
  // in a table.  Variables can be extracted from the DataTable either
  // individually (e.g. to get the y variable for a regression/classification
  // problem) or as a design matrix with column labels giving the variable
  // names.  When building the design matrix, special care is taken to properly
  // name dummy variables.
  class DataTable : public Data {
   public:
    // Empty data table that can be populated by calls to 'append_row',
    // 'append_variable', or 'read_file'.
    DataTable();

    // Create a data table from a file.
    // Args:
    //   fname:  The name of the file to read in.
    //   header: If 'true' then the first line of the file contains
    //     variable names.  If 'false' then the first lineof the file
    //     is the first observation, and variable names will be
    //     automatically generated.
    //   sep: The separator between fields in the data file.
    explicit DataTable(const std::string &fname,
                       bool header = false,
                       const std::string &sep = "");

    DataTable *clone() const override;

    // Print a data table to a stream.
    std::ostream &display(std::ostream &out) const override;

    // Clear the contenst of a data table, and repopulate all data members by
    // reading the given file.
    void read_file(const std::string &filename,
                   bool header = false,
                   const std::string &sep = "");

    //--- build a DataTable by appending variables/columns ---
    //
    // If the data table is empty, appending the first variable determines the
    // number of rows.
    void append_variable(const Vector &numeric,
                         const std::string &name);
    void append_variable(const CategoricalVariable &cv,
                         const std::string &name);
    void append_variable(const DateTimeVariable &dt,
                         const std::string &name);
    
    DataTable &cbind(const Vector &v, const std::string &name) {
      append_variable(v, name);
      return *this;
    }
    DataTable &cbind(const CategoricalVariable &cv, const std::string &name) {
      append_variable(cv, name);
      return *this;
    }
    DataTable &cbind(const DateTimeVariable &dt, const std::string &name) {
      append_variable(dt, name);
      return *this;
    }

    // If the data table is empty, appending the first row determines the number
    // and type of columns.
    void append_row(const MixedMultivariateData &row);
    DataTable &rbind(const MixedMultivariateData &row) {
      append_row(row);
      return *this;
    }

    //--- size  ---
    // Number of variables stored in the table
    uint nvars() const;
    uint ncol() const {return nvars();}
    int numeric_dim() const;      // number of numeric variables.
    int categorical_dim() const;  // number of categorical variables.
    int datetime_dim() const;     // number of datetime variables.

    // Number of rows.
    int nrow() const;            // number of rows
    int nobs() const {return nrow();}  // syntactic sugar.

    // The number of levels for variable i.  If the variable is numeric or
    // datetime then the answer is 1.  Otherwise the number of levels is
    // returned for categorical variables.
    int nlevels(uint i) const;


    //--- look inside ---
    std::ostream &print(std::ostream &out, uint from = 0,
                        uint to = std::numeric_limits<uint>::max()) const;

    // Set the variable names.
    void set_vnames(const std::vector<std::string> &names) {
      type_index_->set_names(names);
    }

    // The names of the variables stored in the table.  These are the "column
    // names."
    const std::vector<std::string> &vnames() const;

    //--- extract variables ---
    // Get column 'which_column' from the table.
    VariableType variable_type(uint which_column) const {
      return type_index_->variable_type(which_column);
    }

    // For numeric data
    Vector getvar(uint which_column) const;
    Vector get_numeric(const std::string &vname) const;
    Vector get_numeric(uint which_column) const {
      return getvar(which_column);
    }
    double getvar(int which_row, int which_column) const;
    void set_numeric_value(int row, int column, double value);

    // For categorical data
    CategoricalVariable get_nominal(uint which_column) const;
    CategoricalVariable get_nominal(const std::string &vname) const;

    Ptr<LabeledCategoricalData> get_nominal(
        int which_row, int which_column) const;
    //    OrdinalVariable get_ordinal(uint which_column) const;
    //    OrdinalVariable get_ordinal(uint which_column,
    //           const std::vector<std::string> &ord) const;
    void set_nominal_value(int row, int column, int value);

    // For datetiem data
    DateTimeVariable get_datetime(uint which_column) const;
    DateTimeVariable get_datetime(const std::string &vname) const;

    // Accessing a row of data involves memory allocations and copies.  If you
    // plan to repeatedly iterate through the rows consider saving a std::vector
    // of rows.
    Ptr<MixedMultivariateData> row(uint row_index) const;

    //--- Compute a design matrix ---
    //
    // ***DEPRECATED***
    //
    // To create a design matrix from a DataTable, use the methods in
    // .../stats/Encoders.hpp.
    LabeledMatrix design(bool add_icpt = false) const;
    LabeledMatrix design(const Selector &include, bool add_icpt = false) const;

    // Bind the rows of rhs below the rows of *this.  The variable
    // types of *this and rhs must match, and all categorical
    // variables must have the same levels.  A reference to *this is
    // returned.
    DataTable &rbind(const DataTable &rhs);

    // Add the columns of rhs to *this, and return *this.
    DataTable &cbind(const DataTable &rhs);

    // Return the dictionary
    const DataTypeIndex &type_index() const {
      return *type_index_;
    }

   private:
    // The data are organized as columns.  The type_index_ keeps track of
    // the variable type of column i, and which index in the relevant vector it
    // is stored.
    std::vector<Vector> numeric_variables_;
    std::vector<CategoricalVariable> categorical_variables_;
    std::vector<DateTimeVariable> datetime_variables_;
    Ptr<DataTypeIndex> type_index_;
  };

  std::ostream &operator<<(std::ostream &out, const DataTable &dt);

  // Create a DataTable by repeating a single row a fixed number of times.
  DataTable repeat(const MixedMultivariateData &row, int num_times);

  inline DataTable cbind(const DataTable &lhs, const DataTable &rhs) {
    DataTable ans(lhs);
    ans.cbind(rhs);
    return ans;
  }

}  // namespace BOOM
#endif  // BOOM_DATA_TABLE_HPP
