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

  enum class VariableType {unknown = -1, numeric, categorical, datetime};

  //===========================================================================
  class MixedDataOrganizer : public RefCounted {
   public:
    MixedDataOrganizer();
    void add_variable(VariableType type);
    std::pair<VariableType, int> type_map(int i) const;

    void diagnose_types(const std::vector<std::string> &);
    VariableType variable_type(int col) const {
      return variable_types_[col];
    }

    bool check_type(int i, const std::string &s) const;

    int number_of_numeric_fields() const {return numeric_count_;}
    int number_of_categorical_fields() const {return categorical_count_;}
    int number_of_unknown_fields() const {return unknown_count_;}
    int total_number_of_fields() const {
      return numeric_count_ + categorical_count_ + unknown_count_;
    }

    bool operator==(const MixedDataOrganizer &rhs) const;
    bool operator!=(const MixedDataOrganizer &rhs) const {
      return !(*this == rhs);
    }

   private:
    int numeric_count_;
    int categorical_count_;
    int unknown_count_;

    // The type of variable in each slot.
    std::vector<VariableType> variable_types_;

    // To find the variable in slot i, get its type and its index in private
    // storage.  For example, if variable 7 is the 4th numeric variable (and
    // thus index 3) then type_map[7] = {numeric, 3}.
    std::map<int, std::pair<VariableType, int>> type_map_;

    // The names
    bool check_type(VariableType type, const std::string &s) const;

    friend void intrusive_ptr_add_ref(MixedDataOrganizer *d) {
      d->up_count();
    }

    friend void intrusive_ptr_release(MixedDataOrganizer *d) {
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
   public:
    MixedMultivariateData();
    MixedMultivariateData(const Ptr<MixedDataOrganizer> &sorter);
    MixedMultivariateData(const MixedMultivariateData &rhs);
    MixedMultivariateData &operator=(const MixedMultivariateData &rhs);
    MixedMultivariateData(MixedMultivariateData &&rhs) = default;
    MixedMultivariateData &operator=(MixedMultivariateData &&rhs) = default;

    MixedMultivariateData *clone() const override;
    std::ostream &display(std::ostream &out) const override;

    void add_numeric(const Ptr<DoubleData> &numeric);
    void add_categorical(const Ptr<CategoricalData> &categorical);

    // The number of numeric variables.
    int numeric_dim() const {
      return data_sorter_->number_of_numeric_fields();
    }

    // The number of categorical variables.
    int categorical_dim() const {
      return data_sorter_->number_of_categorical_fields();
    }

    // The total number of variables.
    int dim() const {return data_sorter_->total_number_of_fields();}

    // The type of variable in cell i.
    VariableType vtype(int i) const { return data_sorter_->variable_type(i); }

    const Data &variable(int i) const;

    // Return the entry in cell i if it is of the requested type.  If it is
    // not then raise an error.
    const DoubleData &numeric(int i) const;
    Ptr<DoubleData> mutable_numeric(int i);
    const CategoricalData &categorical(int i) const;
    Ptr<CategoricalData> mutable_categorical(int i);

    // Collapse all the numeric data into a vector.
    Vector numeric_data() const;

    const std::vector<Ptr<CategoricalData>> &categorical_data() const {
      return categorical_data_;
    }

   private:
    Ptr<MixedDataOrganizer> data_sorter_;
    std::vector<Ptr<DoubleData>> numeric_data_;
    std::vector<Ptr<CategoricalData>> categorical_data_;
  };

  //===========================================================================
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

  //===========================================================================
  // Timestamps are an important data type that are distinct from "numeric" or
  // "categorical" data.
  class DateTimeVariable {
   public:
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
    typedef std::vector<double> dvector;
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
    //
    // If the data table is empty, appending the first variable determines the
    // number of rows.
    virtual void append_variable(const Vector &v, const std::string &name);
    virtual void append_variable(const CategoricalVariable &cv,
                                 const std::string &name);

    //--- size  ---
    uint nvars() const;          // number of variables stored in the table
    int nrow() const;            // number of rows
    int nobs() const {return nrow();}  // syntactic sugar.
    uint nlevels(uint i) const;  // 1 for numeric, nlevels for categorical

    //--- look inside ---
    std::ostream &print(std::ostream &out, uint from = 0,
                        uint to = std::numeric_limits<uint>::max()) const;

    // The names of the variables stored in the table.  These are the "column
    // names."
    StringVector &vnames();
    const StringVector &vnames() const;

    //--- extract variables ---
    // Get column 'which_column' from the table.
    VariableType variable_type(uint which_column) const {
      return data_organizer_->variable_type(which_column);
    }
    Vector getvar(uint which_column) const;
    double getvar(int which_row, int which_column) const;
    CategoricalVariable get_nominal(uint which_column) const;
    Ptr<CategoricalData> get_nominal(int which_row, int which_column) const;
    //    OrdinalVariable get_ordinal(uint which_column) const;
    //    OrdinalVariable get_ordinal(uint which_column, const StringVector
    //    &ord) const;

    // Accessing a row of data involves memory allocations and copies.  If you
    // plan to repeatedly iterate through the rows consider saving a std::vector
    // of rows.
    Ptr<MixedMultivariateData> row(uint row_index) const;

    //--- Compute a design matrix ---
    LabeledMatrix design(bool add_icpt = false) const;
    LabeledMatrix design(const Selector &include, bool add_icpt = false) const;

    // Bind the rows of rhs below the rows of *this.  The variable
    // types of *this and rhs must match, and all categorical
    // variables must have the same levels.  A reference to *this is
    // returned.
    DataTable &rbind(const DataTable &rhs);

   private:
    // The data are organized as columns.  The data_organizer_ keeps track of
    // the variable type of column i, and which index in the relevant vector it
    // is stored.
    std::vector<Vector> numeric_variables_;
    std::vector<CategoricalVariable> categorical_variables_;
    Ptr<MixedDataOrganizer> data_organizer_;
    std::vector<std::string> vnames_;
  };

  std::ostream &operator<<(std::ostream &out, const DataTable &dt);

  //===========================================================================
  // PartiallyMissingDataTable

  // class PartiallyMissingDataTable : public DataTable {
  //  public:
  //   PartiallyMissingDataTable();

  //   // The append_variable functions inherited from DataTable will be
  //   // implemented by append_potentially_missing_variable.
  //   void append_variable(const Vector &numeric, const std::string &vname) override;
  //   void append_variable(const CategoricalVariable &cv,
  //                        const std::string &name) override;

  //   // Each element of the CategoricalVariable indicates missingness by the
  //   // missing data flag it inherits from Data.
  //   //
  //   //
  //   void append_potentially_missing_variable(
  //       const Vector &numeric, const std::string &vname, double missing_value_key);

  //  private:
  // };

}  // namespace BOOM
#endif  // BOOM_DATA_TABLE_HPP
