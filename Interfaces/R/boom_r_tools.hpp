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

#ifndef BOOM_R_TOOLS_HPP_
#define BOOM_R_TOOLS_HPP_

#include <string>

#include "LinAlg/Vector.hpp"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/SpdMatrix.hpp"
#include "LinAlg/SubMatrix.hpp"
#include "LinAlg/Array.hpp"

#include "Models/CategoricalData.hpp"
#include "stats/DataTable.hpp"
#include "cpputil/Date.hpp"
//======================================================================
// Note that the functions listed here throw exceptions.  Code that
// uses them should be wrapped in a try-block where the catch
// statement catches the exception and calls Rf_error() with an
// appropriate error message.  The functions handle_exception(), and
// handle_unknown_exception (in handle_exception.hpp), are suitable
// defaults.  These try-blocks should be present in any code called
// directly from R by .Call.
//======================================================================

// If Rinternals.h has already been included and R_NO_REMAP has not yet been
// defined, then throw a compiler error to prevent madness caused by R's
// preprocessor renaming of things like length() and error().
#ifndef R_NO_REMAP
#define R_NO_REMAP
#ifdef R_INTERNALS_H_
#error Code that includes both boom_r_tools.hpp and Rinternals.h must either\
 (a) include them in that order or (b) define R_NO_REMAP.
#endif  // ifdef R_INTERNALS_H_
#endif  // ifndef R_NO_REMAP

#include <Rinternals.h>

namespace BOOM{
  // Returns list[[name]] if a list element with that name exists.
  // Returns R_NilValue otherwise.
  // Args:
  //   list: The list to search.
  //   name: The name of the list element to search for.
  //   expect_answer: If true, then print a warning message if the named element
  //     cannot be found.  This is useful in "printf debugging."
  //
  // Returns:
  //   If the requested element is found then it is returned.  If not then
  //   R_NilValue is returned.  If the first argument does not have a 'names'
  //   attribute then an error is reported.
  SEXP getListElement(SEXP list, const std::string &name,
                      bool expect_answer = false);

  // Extract the names from a list.  If the list has no names
  // attribute a vector of empty strings is returned.
  std::vector<std::string> getListNames(SEXP list);

  // Set the names attribute of 'list' to match 'list_names'.  BOOM's
  // error reporting mechanism is invoked if the length of 'list' does
  // not match the length of 'list_names'.
  // Returns 'list' with the new and improved set of 'list_names'.
  SEXP setListNames(SEXP list, const std::vector<std::string> &list_names);

  // Returns the levels attribute of the factor argument.  Throws an
  // exception if the argument is not a factor.
  std::vector<std::string> GetFactorLevels(SEXP factor);

  // Converts an R character vector into a c++ vector of strings.  If
  // the object is NULL an empty string vector is returned.  Otherwise
  // if the object is not a character vector an exception is thrown.
  std::vector<std::string> StringVector(SEXP r_character_vector);

  // Converts a C++ vector of strings into an R character vector.
  SEXP CharacterVector(const std::vector<std::string> &string_vector);

  // Creates a new list with the contents of the original 'list' with
  // new_element added.  The names of the original list are copied,
  // and 'name' is appended.  The original 'list' is not modified, so
  // it is possible to write:
  // my_list = appendListElement(my_list, new_thing, "blah");
  // Two things to note:
  // (1) The output is in new memory, so it is not PROTECTED by default
  // (2) Each time you call this function all the list entries are
  //     copied, and (more importantly) new space is allocated, so
  //     you're better off creating the list you want from the
  //     beginning if you can.
  SEXP appendListElement(SEXP list, SEXP new_element, const std::string &name);

  // Appends the collection of SEXP elements to the list.
  // Args:
  //   list:  The original list.
  //   new_elements:  The vector of new elements to add to the list.
  //   new_element_names: A vector of names for the new elements.  The
  //     length of this vector MUST match the length of the new
  //     elements.
  //
  // Returns:
  //   A new list containing the elements from the original list, with
  //   new_elements appended at the end.
  //   *** NOTE ***
  //   The return value is in new memory, so even if the name of the
  //   list is the same, it will have to be re-PROTECTed.
  SEXP appendListElements(
      SEXP list,
      const std::vector<SEXP> &new_elements,
      const std::vector<std::string> &new_element_names);

  // Creates a list from the C++ vector of SEXP elements.  The vector
  // of element_names can be empty, in which case the list will not
  // have names.  If non-empty, then the length of element_names must
  // match the length of elements.
  SEXP CreateList(
      const std::vector<SEXP> &elements,
      const std::vector<std::string> &element_names);

  // Returns the class attribute of the specified R object.  If no
  // class attribute exists an empty vector is returned.
  std::vector<std::string> GetS3Class(SEXP object);

  // Report and error, using BOOM's report_error mechanism, arising from an
  // object with the wrong class being passed to a function.
  //
  // Args:
  //   error_message: An error message to be printed giving the context in which
  //     the error occurred.  What sort object was expected, etc.
  //   r_object:  The passed object that led to the error.
  //
  // Effects:
  //   An error is reported.  The error message includes the first argument, a
  //   list of the classes associated with r_object, and an indicator of whether
  //   the object was NULL.
  void ReportBadClass(const std::string &error_message, SEXP r_object);
  
  // Returns a pair, with .first set to the number of rows, and
  // .second set to the number of columns.  If the argument is not a
  // matrix then an exception will be thrown.
  std::pair<int, int> GetMatrixDimensions(SEXP matrix);

  // Set the column names for a matrix.
  // Args:
  //   r_matrix:  The matrix whose column names are to be set.
  //   column_names: The vector of names to be assigned to the
  //     columns.
  // Returns:
  //   If column_names is empty then r_matrix is returned unchanged.
  //   If column_names is non-empty but contains a number of elements
  //   that differs from ncol(r_matrix) then an error is reported with
  //   report_error.  Otherwise the column names of r_matrix are set
  //   to the supplied values and the matrix with column names is
  //   returned.  Note that this is the same object as r_matrix, so
  //   it inherits the PROTECT status of the r_matrix argument.
  //
  //   Row names, if they previously existed, are removed.
  SEXP SetColnames(SEXP r_matrix, const std::vector<std::string> &column_names);

  // Set the dimnames attributes for an R multi-way array.
  // Args:
  //   r_array:  The array on which to set the dimnames attribute.
  //   dimnames: A collection of dimnames.
  // Returns:
  //  If dimnames is empty then r_array is returned unchanged.  If dimnames is
  //  non-empty then the returned object is r_array with dimnames assigned.
  //  Because it is the same object, it inherits the PROTECT status of the
  //  r_array argument.  It is the user's responsibility not to PROTECT twice.
  // Details:
  //   The dimnames argument can be empty, signifying that there are no dimnames
  //   for any dimension, or it can have length equal to dim(r_array).  In the
  //   latter case, any element can either be empty, signifying no dimnames for
  //   that dimension, or it can be a vector with length equal to the extent of
  //   that dimension.
  SEXP SetDimnames(SEXP r_array,
                   const std::vector<std::vector<std::string>> &dimnames);
  
  // Returns a vector of dimensions for an R multi-way array.  If the
  // argument is not an array, then an exception will be thrown.
  std::vector<int> GetArrayDimensions(SEXP array);

  // If 'my_list' contains a character vector named 'name' then the
  // first element of that character vector is returned.  If not then
  // an exception will be thrown.
  std::string GetStringFromList(SEXP my_list, const std::string &name);

  // If 'my_vector' is a numeric vector, it is converted to a BOOM::Vector.
  // Otherwise an exception will be thrown.  ToBoomVector makes a copy of the
  // underlying memory.  ToBoomVectorView accesses the memory in the R object,
  // without making a copy.
  Vector ToBoomVector(SEXP my_vector);
  ConstVectorView ToBoomVectorView(const SEXP my_vector);

  // Returns true iff x is set to R's NA value.
  bool isNA(double x);

  // Returns a Selector of the sizem size as v, with elements indicating which
  // elements of v are observed.  This is equivalent to the R expression
  // !is.na(v).
  Selector FindNonNA(const ConstVectorView &v);
  
  // If 'r_matrix' is an R matrix, it is converted to a BOOM::Matrix.
  // Otherwise an exception will be thrown.  ToBoomMatrix makes a copy
  // of the underlying memory.  ToBoomMatrixView accesses the memory
  // in the R object, without making a copy.
  Matrix ToBoomMatrix(SEXP r_matrix);
  ConstSubMatrix ToBoomMatrixView(SEXP r_matrix);
  SubMatrix ToBoomMutableMatrixView(SEXP r_matrix);

  // If 'r_array' is an R multi-way array then it is converted to an
  // equivalent BOOM::Array.  Otherwise an exception will be thrown.
  // A numeric vector is interpreted as a 1-d array, and a matrix as a
  // 2-d array.
  Array ToBoomArray(SEXP r_array);
  ConstArrayView ToBoomArrayView(SEXP r_array);

  // If 'r_data_frame' is an R data frame object, then it will be
  // converted into a BOOM::DataTable.  Otherwise an exception will be
  // thrown.  The return value is a copy, not a reference.
  DataTable ToBoomDataTable(SEXP r_data_frame);

  // If 'my_matrix' is an R matrix, it is converted to a BOOM::Spd.  If
  // the conversion fails then an exception will be thrown.
  SpdMatrix ToBoomSpdMatrix(SEXP my_matrix);

  // If 'my_vector' is an R logical vector, then it is converted to a
  // std::vector<bool>.  Otherwise an exception will be thrown.
  std::vector<bool> ToVectorBool(SEXP my_vector);

  // If r_int_vector is an R vector of integers then it is converted
  // to a std::vector<int>.  Otherwise an exception is thrown.
  // Args:
  //   r_int_vector:  An R vector of integers.
  //   subtract_one: If 'true' then subtract one from each entry in r_int_vector
  //     before returning.  This is useful for converting from R's unit offset
  //     counting system to C's zero-offset system if r_int_vector is a
  //     collection of positions in another vector.
  std::vector<int> ToIntVector(SEXP r_int_vector, bool subtract_one = false);

  // If r_int_matrix is an R matrix of integers then it is converted
  // to a std::vector<std::vector<int>>.  Otherwise an exception is
  // thrown.  Note that C++ storage is row-major, while R's storage is
  // column-major.
  //
  // If convert_to_zero_offset is true then 1 is subtracted from each
  // matrix element.  This is useful if the values are indices in R's
  // unit-offset scheme.  Setting this flag will transform the indices
  // to the equivalent location in C++'s zero-offset scheme.
  std::vector<std::vector<int>> ToIntMatrix(
      SEXP r_int_matrix,
      bool convert_to_zero_offset = false);

  // Convert an R Date object (singleton or vector) to a BOOM Date.
  Date ToBoomDate(SEXP r_Date);
  std::vector<Date> ToBoomDateVector(SEXP r_Dates);
  
  // Convert a BOOM vector, matrix, or array to its R equivalent.
  // Less type checking is needed for these functions than in the
  // other direction because we know the type of the input.
  SEXP ToRVector(const Vector &boom_vector);
  SEXP ToRMatrix(const Matrix &boom_matrix);
  SEXP ToRArray(const ConstArrayView &boom_array);
  SEXP AllocateArray(const std::vector<int> &array_dimensions);
  
  // Convert a std::vector<int> to an R vector of integers.  A common case is
  // when the first argument contains a vector of positions in the C++
  // zero-offset system for indexing arrays.  In that case you can set add_one =
  // true to add 1 to each element of ints, so that it will correspond to R's
  // unit-offset indexing scheme.
  SEXP ToRIntVector(const std::vector<int> &ints, bool add_one = false);

  // This version produces an R matrix with row names and column
  // names.  A zero-length vector indicates that no names are desired
  // for that dimension.  Otherwise the size of row_names must equal
  // the number of rows in boom_matrix, and likewise for col_names.
  SEXP ToRMatrix(const Matrix &boom_matrix,
                 const std::vector<std::string> &row_names,
                 const std::vector<std::string> &col_names);
  SEXP ToRMatrix(const LabeledMatrix &boom_labeled_matrix);

  // Convert a "scalar" string to a C++ string.
  // Args:
  //   r_string: Either a CHARSXP or a STRINGSXP.  Any other input
  //     will result in an exception being thrown.  If the input is a
  //     STRINGSXP then the first element is returned.
  std::string ToString(SEXP r_string);

  // Convert a C++ string (or vector of strings) to an R character vector.
  SEXP ToRString(const std::string &s);
  SEXP ToRStringVector(const std::vector<std::string> &string_vector);

  // A Factor object is intended to be initialized with an R factor.
  class Factor {
   public:
    explicit Factor(SEXP r_factor);

    // Corresponds to R's length(r_factor).
    int length() const;

    // Corresponds to R's length(levels(r_factor))
    int number_of_levels() const;

    // Returns the integer value of observation i.  Note that in R,
    // observation i is 1-based, while here it is zero-based.
    int operator[](int i) const;

    // Returns a BOOM::CategoricalData corresponding to observation i.
    CategoricalData to_categorical_data(int i)const;

    // The names of the factor levels.
    std::vector<std::string> labels() const { return levels_->labels(); }

    // Allocates and returns a vector of categorical data objects.
    std::vector<Ptr<CategoricalData> > vector_of_observations() const;

    const Ptr<CatKey> & key() const {return levels_;}

   private:
    std::vector<int> values_;
    Ptr<CatKey> levels_;
  };

  // A class to handle protection of R objects in an exception safe
  // way.  Define a single RMemoryProtector at the start of any
  // function where R memory needs to be allocated.  Then instead of
  // writing
  // PROTECT(my_r_object);
  // /* do stuff */
  // UNPROTECT(1);
  //
  // you write
  // RMemoryProtector protector;
  // protector.protect(my_r_object);
  //  /* do stuff */
  //
  // There is no need to call UNPROTECT, which is handled by the class
  // destructor.
  class RMemoryProtector {
   public:
    RMemoryProtector() : protection_count_(0) {}

    ~RMemoryProtector() {
      UNPROTECT(protection_count_);
    }

    // Args:
    //   r_object: An R object that needs protecting for the life of
    //     this RMemoryProtector object.
    // Returns:
    //   The protected r_object.
    SEXP protect(SEXP r_object) {
      PROTECT(r_object);
      ++protection_count_;
      return r_object;
    }

   private:
    int protection_count_;
  };

  // The job of an RErrorReporter is to reconcile the error handling mechanisms
  // of C++ (exceptions) and R (Rf_error).  When C++ exceptions are thrown,
  // stack unwinding frees memory in objects held by smart pointers that go out
  // of scope.  When Rf_error is called, the destructors that C++ relies on to
  // do the right thing are never called, and so memory leaks.  The solution is
  // to define an RErrorReporter on the first line of a function entered by
  // .Call().  If an error is encountered that should be communicated back to R
  // then write it using SetError, and then exit the function.  The
  // RErrorReporter will be the last thing destroyed on function exit, and its
  // destructor will call Rf_error with the specified error message.
  class RErrorReporter {
   public:
    RErrorReporter() : error_message_(nullptr) {}
    ~RErrorReporter();
    // If multiple error messages are passed to the error reporter,
    // only the first one is stored.
    void SetError(const std::string &error_message);

    // Returns true iff error_message_ has been set.
    bool HasError() const { return error_message_; }

   private:
    // A pointer to a string is necessary here because we need to
    // manually release the memory holding the string before calling
    // Rf_error.
    std::string *error_message_;
  };

  // A functor representing a scalar-valued function of a single Vector valued
  // argument.  The canonical use case for this class is for users to be able to
  // define a log density in R and pass it to a PosteriorSampler in BOOM.  This
  // approach will be much slower than defining everything in C++, but it is a 
  // helpful tool in exposing C++ libraries to R.
  class RVectorFunction {
   public:
    // Args:
    //   r_vector_function: An R list containing three elements, in the
    //     following order.
    //   - r_fun: The function (closure) passed in from R.  The function should
    //       take a single argument (a real-valued vector), and return a scalar.
    //   - r_env: The environment in which to evaluate r_fun.
    //   - argument_name: The name of the argument to r_fun as it appears in the
    //       function signature.
    RVectorFunction(SEXP r_vector_function);

    // The process of evaluating the function creates an object in the
    // function's environment with an ugly name.  The destructor should remove
    // this object.
    ~RVectorFunction();
    
    // Evaluate the function at x.
    double evaluate(const Vector &x);
    double operator()(const Vector &x) { return evaluate(x); }
    
   private:
    // The name of the function as it exists in R.
    std::string function_name_;

    // The name of the object to which we will be assigning numerical values.
    // Should be ugly so we don't accidentally overwrite anytihng.
    std::string argument_name_;

    // The environment in which to find the R function.
    SEXP r_env_;

    // The call we will use to get the output, something like
    // "f(argument_name_)"
    std::string call_string_;
  };

  
  // Returns true if the user has requested an interrupt.
  bool RCheckInterrupt();

}  // namespace BOOM

#endif  // BOOM_R_TOOLS_HPP_
