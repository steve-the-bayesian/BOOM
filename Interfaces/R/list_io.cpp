// Copyright 2011 Google Inc. All Rights Reserved.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA

#include <string>
#include "r_interface/list_io.hpp"
#include "r_interface/boom_r_tools.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  void RListIoManager::add_list_element(RListIoElement *element) {
    elements_.push_back(std::shared_ptr<RListIoElement>(element));
  }

  SEXP RListIoManager::prepare_to_write(int niter) {
    RMemoryProtector protector;
    SEXP ans = protector.protect(Rf_allocVector(VECSXP, elements_.size()));
    SEXP param_names = protector.protect(
        Rf_allocVector(STRSXP, elements_.size()));
    for (int i = 0; i < elements_.size(); ++i) {
      SET_VECTOR_ELT(ans, i,
                     elements_[i]->prepare_to_write(niter));
      SET_STRING_ELT(param_names, i,
                     Rf_mkChar(elements_[i]->name().c_str()));
    }
    Rf_namesgets(ans, param_names);
    return ans;
  }

  void RListIoManager::prepare_to_stream(SEXP object) {
    for (int i = 0; i < elements_.size(); ++i) {
      elements_[i]->prepare_to_stream(object);
    }
  }

  void RListIoManager::write() {
    for (int i = 0; i < elements_.size(); ++i) {
      elements_[i]->write();
    }
  }

  void RListIoManager::stream() {
    for (int i = 0; i < elements_.size(); ++i) {
      elements_[i]->stream();
    }
  }

  void RListIoManager::advance(int n) {
    for (int i = 0; i < elements_.size(); ++i) {
      elements_[i]->advance(n);
    }
  }

  //======================================================================
  RListIoElement::RListIoElement(const std::string &name) : name_(name) {}

  RListIoElement::~RListIoElement() {}

  void RListIoElement::StoreBuffer(SEXP buf) {
    rbuffer_ = buf;
    position_ = 0;
  }

  void RListIoElement::prepare_to_stream(SEXP object) {
    rbuffer_ = getListElement(object, name_);
    position_ = 0;
  }

  const std::string &RListIoElement::name()const {return name_;}

  void RListIoElement::advance(int n) {position_ += n;}

  int RListIoElement::next_position() {
    return position_++;
  }

  //======================================================================
  RealValuedRListIoElement::RealValuedRListIoElement(const std::string &name)
      : RListIoElement(name)
  {}

  SEXP RealValuedRListIoElement::prepare_to_write(int niter) {
    RMemoryProtector protector;
    SEXP buffer = protector.protect(Rf_allocVector(REALSXP, niter));
    StoreBuffer(buffer);
    return buffer;
  }

  void RealValuedRListIoElement::prepare_to_stream(SEXP object) {
    RListIoElement::prepare_to_stream(object);
    data_ = REAL(rbuffer());
  }

  void RealValuedRListIoElement::StoreBuffer(SEXP buf) {
    data_ = REAL(buf);
    RListIoElement::StoreBuffer(buf);
  }

  //======================================================================
  SEXP VectorValuedRListIoElement::prepare_to_write(int niter) {
    RMemoryProtector protector;
    SEXP buffer;
    protector.protect(buffer = SetColnames(Rf_allocMatrix(
        REALSXP, niter, dim()), element_names_));
    StoreBuffer(buffer);
    matrix_view_.reset(SubMatrix(data(), niter, dim()));
    return buffer;
  }

  void VectorValuedRListIoElement::prepare_to_stream(SEXP object) {
    RealValuedRListIoElement::prepare_to_stream(object);
    matrix_view_.reset(SubMatrix(data(),
                                 Rf_nrows(rbuffer()),
                                 Rf_ncols(rbuffer())));
  }
  
  //======================================================================
  ListValuedRListIoElement::ListValuedRListIoElement(const std::string &name)
      : RListIoElement(name)
  {}

  SEXP ListValuedRListIoElement::prepare_to_write(int niter) {
    RMemoryProtector protector;
    SEXP buffer = protector.protect(Rf_allocVector(VECSXP, niter));
    StoreBuffer(buffer);
    return buffer;
  }

  //======================================================================
  UnivariateListElement::UnivariateListElement(const Ptr<UnivParams> &prm,
                                               const std::string &name)
      : RealValuedRListIoElement(name),
        prm_(prm)
  {}

  void UnivariateListElement::write() {
    data()[next_position()] = prm_->value();
  }

  void UnivariateListElement::stream() {
    prm_->set(data()[next_position()]);
  }
  //======================================================================
  NativeUnivariateListElement::NativeUnivariateListElement(
      ScalarIoCallback *callback,
      const std::string &name,
      double *streaming_buffer)
      : RealValuedRListIoElement(name),
        streaming_buffer_(streaming_buffer),
        vector_view_(0, 0, 0)
  {
    if (callback) {
      callback_.reset(callback);
    }
  }

  void NativeUnivariateListElement::write() {
    data()[next_position()] = callback_->get_value();
  }

  void NativeUnivariateListElement::stream() {
    if(streaming_buffer_){
      *streaming_buffer_ = data()[next_position()];
    }
  }

  //======================================================================
  StandardDeviationListElement::StandardDeviationListElement(
      const Ptr<UnivParams> &variance, const std::string &name)
      : RealValuedRListIoElement(name),
        variance_(variance)
  {}

  void StandardDeviationListElement::write() {
    data()[next_position()] = sqrt(variance_->value());
  }

  void StandardDeviationListElement::stream() {
    double sd = data()[next_position()];
    variance_->set(square(sd));
  }

  //======================================================================
  VectorListElement::VectorListElement(
      const Ptr<VectorParams> &prm,
      const std::string &name,
      const std::vector<std::string> &element_names)
      : VectorValuedRListIoElement(name, element_names),
        prm_(prm)
  {}

  void VectorListElement::write() {
    CheckSize();
    matrix_view().row(next_position()) = prm_->value();
  }

  void VectorListElement::stream() {
    CheckSize();
    prm_->set(matrix_view().row(next_position()));
  }

  void VectorListElement::CheckSize() {
    if (matrix_view().ncol() != prm_->size(false)) {
      std::ostringstream err;
      err << "sizes do not match in VectorListElement::stream/write..."
          << endl
          << "buffer has space for " << matrix_view().ncol() << " elements, "
          << " but you're trying to access " << prm_->size(false)
          ;
      report_error(err.str().c_str());
    }
  }
  //======================================================================
  GlmCoefsListElement::GlmCoefsListElement(
      const Ptr<GlmCoefs> &coefs,
      const std::string &param_name,
      const std::vector<std::string> &element_names)
      : VectorListElement(coefs, param_name, element_names),
        coefs_(coefs)
  {}

  void GlmCoefsListElement::stream() {
    VectorListElement::stream();
    beta_ = coefs_->Beta();
    coefs_->set_Beta(beta_);
    coefs_->infer_sparsity();
  }

  //======================================================================
  SdVectorListElement::SdVectorListElement(const Ptr<VectorParams> &prm,
                                           const std::string &name)
      : VectorValuedRListIoElement(name),
        prm_(prm)
  {}

  void SdVectorListElement::write() {
    CheckSize();
    matrix_view().row(next_position()) = sqrt(prm_->value());
  }

  void SdVectorListElement::stream() {
    CheckSize();
    Vector sd = matrix_view().row(next_position());
    prm_->set(sd * sd);
  }

  void SdVectorListElement::CheckSize() {
    if (matrix_view().ncol() != prm_->size(false)) {
      std::ostringstream err;
      err << "sizes do not match in SdVectorListElement::stream/write..."
          << endl
          << "buffer has space for " << matrix_view().ncol() << " elements, "
          << " but you're trying to access " << prm_->size(false)
          ;
      report_error(err.str().c_str());
    }
  }

  //======================================================================
  UnivariateCollectionListElement::UnivariateCollectionListElement(
      const std::vector<Ptr<UnivParams>> &parameters,
      const std::string &name)
      : VectorValuedRListIoElement(name),
        parameters_(parameters)
  {}

  void UnivariateCollectionListElement::write() {
    CheckSize();
    int row = next_position();
    for (int i = 0; i < parameters_.size(); ++i) {
      matrix_view()(row, i) = parameters_[i]->value();
    }
  }

  void UnivariateCollectionListElement::stream() {
    CheckSize();
    int row = next_position();
    for (int i = 0; i < parameters_.size(); ++i) {
      parameters_[i]->set(matrix_view()(row, i));
    }
  }

  void UnivariateCollectionListElement::CheckSize() {
    if (matrix_view().ncol() != parameters_.size()) {
      std::ostringstream err;
      err << "The R buffer has " << matrix_view().ncol()
          << " columns, but space is needed for "
          << parameters_.size() << " parameters.";
      report_error(err.str());
    }
  }

  //======================================================================
  void SdCollectionListElement::write() {
    CheckSize();
    int row = next_position();
    for (int i = 0; i < parameters().size(); ++i) {
      matrix_view()(row, i) = sqrt(parameters()[i]->value());
    }
  }

  void SdCollectionListElement::stream() {
    CheckSize();
    int row = next_position();
    for (int i = 0; i < parameters().size(); ++i) {
      parameters()[i]->set(square(matrix_view()(row, i)));
    }
  }

  //======================================================================
  NativeVectorListElement::NativeVectorListElement(VectorIoCallback *callback,
                                                   const std::string &name,
                                                   Vector *streaming_buffer)
      : VectorValuedRListIoElement(name),
        streaming_buffer_(streaming_buffer),
        check_buffer_(true)
  {
    // Protect against a NULL callback.
    if (callback) {
      callback_.reset(callback);
    }
  }

  void NativeVectorListElement::write() {
    next_row() = callback_->get_vector();
  }

  void NativeVectorListElement::stream() {
    if (check_buffer_ && !streaming_buffer_) return;
    *streaming_buffer_ = next_row();
  }

  VectorView NativeVectorListElement::next_row() {
    return matrix_view().row(next_position());
  }

  //===========================================================================  
  GenericVectorListElement::GenericVectorListElement(
      StreamableVectorIoCallback *callback,
      const std::string &name)
      : NativeVectorListElement(callback, name, nullptr)
  {
    if (callback) {
      callback_.reset(callback);
    } else {
      callback_.reset();
    }
  }

  //======================================================================
  const std::vector<std::string> & MatrixListElementBase::row_names()const{
    return row_names_;
  }

  const std::vector<std::string> & MatrixListElementBase::col_names()const{
    return row_names_;
  }

  void MatrixListElementBase::set_row_names(
      const std::vector<std::string> &row_names){
    row_names_ = row_names;
  }

  void MatrixListElementBase::set_col_names(
      const std::vector<std::string> &col_names){
    col_names_ = col_names;
  }

  void MatrixListElementBase::set_buffer_dimnames(SEXP buffer) {
    // Set the dimnames on the buffer
    RMemoryProtector protector;
    SEXP r_dimnames = protector.protect(Rf_allocVector(VECSXP, 3));
    // The leading dimension (MCMC iteration number) does not get
    // names.
    SET_VECTOR_ELT(r_dimnames, 0, R_NilValue);

    if (!row_names_.empty()) {
      if (row_names_.size() != nrow()) {
        report_error("row names were the wrong size in MatrixListElement");
      }
      SET_VECTOR_ELT(r_dimnames, 1, CharacterVector(row_names_));
    } else {
      SET_VECTOR_ELT(r_dimnames, 1, R_NilValue);
    }

    if (!col_names_.empty()) {
      if (col_names_.size() != ncol()) {
        report_error("col names were the wrong size in MatrixListElement");
      }
      SET_VECTOR_ELT(r_dimnames, 2, CharacterVector(col_names_));
    } else {
      SET_VECTOR_ELT(r_dimnames, 2, R_NilValue);
    }
    Rf_dimnamesgets(buffer, r_dimnames);
  }

  //======================================================================
  MatrixListElement::MatrixListElement(const Ptr<MatrixParams> &m,
                                       const std::string &param_name)
      : MatrixListElementBase(param_name),
        prm_(m),
        array_view_(0, Array::index3(0, 0, 0))
  {}

  SEXP MatrixListElement::prepare_to_write(int niter) {
    int nr = prm_->nrow();
    int nc = prm_->ncol();
    RMemoryProtector protector;
    SEXP buffer = protector.protect(Rf_alloc3DArray(REALSXP, niter, nr, nc));
    set_buffer_dimnames(buffer);
    StoreBuffer(buffer);
    array_view_.reset(data(), Array::index3(niter, nr, nc));
    return buffer;
  }

  void MatrixListElement::write() {
    CheckSize();
    const Matrix &m(prm_->value());
    int iteration = next_position();
    int nr = m.nrow();
    int nc = m.ncol();
    for(int i = 0; i < nr; ++i){
      for(int j = 0; j < nc; ++j){
        array_view_(iteration, i, j) = m(i, j);
      }
    }
  }

  void MatrixListElement::prepare_to_stream(SEXP object) {
    RealValuedRListIoElement::prepare_to_stream(object);
    RMemoryProtector protector;
    SEXP r_array_dims = protector.protect(Rf_getAttrib(rbuffer(), R_DimSymbol));
    int * array_dims = INTEGER(r_array_dims);
    array_view_.reset(data(), std::vector<int>(array_dims, array_dims + 3));
  }

  void MatrixListElement::stream() {
    CheckSize();
    int iteration = next_position();
    int nr = prm_->nrow();
    int nc = prm_->ncol();
    Matrix tmp(nr, nc);
    for(int i = 0; i < nr; ++i) {
      for(int j = 0; j < nc; ++j) {
      tmp(i, j) = array_view_(iteration, i, j);
      }
    }
    prm_->set(tmp);
  }

  int MatrixListElement::nrow()const {
    return prm_->nrow();
  }

  int MatrixListElement::ncol()const {
    return prm_->ncol();
  }

  void MatrixListElement::CheckSize() {
    const std::vector<int> & dims(array_view_.dim());
    const Matrix & value(prm_->value());
    if(value.nrow() != dims[1] ||
       value.ncol() != dims[2]) {
      std::ostringstream err;
      err << "sizes do not match in MatrixListElement::stream/write..."
          << endl
          << "dimensions of buffer:    [" << dims[0] << ", " << dims[1] << ", "
          << dims[2] << "]." <<endl
          << "dimensions of parameter: [" << value.nrow() << ", "
          << value.ncol() << "].";
      report_error(err.str().c_str());
    }
  }

  //======================================================================
  HierarchicalVectorListElement::HierarchicalVectorListElement(
      const std::vector<Ptr<VectorParams>> &parameters,
      const std::string &param_name)
      : RealValuedRListIoElement(param_name),
        array_view_(0, Array::index3(0, 0, 0))
  {
    parameters_.reserve(parameters.size());
    for (int i = 0; i < parameters.size(); ++i) {
      add_vector(parameters[i]);
    }
  }

  HierarchicalVectorListElement::HierarchicalVectorListElement(
      const std::string &param_name)
      : RealValuedRListIoElement(param_name),
        array_view_(0, Array::index3(0, 0, 0))
  {}

  void HierarchicalVectorListElement::add_vector(const Ptr<VectorParams> &v) {
    if (!v) {
      report_error("Null pointer passed to HierarchicalVectorListElement");
    }
    if (!parameters_.empty()) {
      if (v->dim() != parameters_[0]->dim()) {
        report_error(
            "All parameters passed to HierarchicalVectorListElement "
            "must be the same size");
      }
    }
    parameters_.push_back(v);
  }

  SEXP HierarchicalVectorListElement::prepare_to_write(int niter) {
    int number_of_groups = parameters_.size();
    int dim = parameters_[0]->dim();
    RMemoryProtector protector;
    SEXP buffer = protector.protect(Rf_alloc3DArray(
        REALSXP, niter, number_of_groups, dim));
    set_buffer_group_names(buffer);
    StoreBuffer(buffer);
    array_view_.reset(data(), Array::index3(niter, number_of_groups, dim));
    return buffer;
  }

  void HierarchicalVectorListElement::set_buffer_group_names(SEXP buffer) {
    if (group_names_.empty()) return;
    RMemoryProtector protector;
    SEXP r_dimnames = protector.protect(Rf_allocVector(VECSXP, 3));
    // The leading dimension (MCMC iteration number) does not get
    // names.
    SET_VECTOR_ELT(r_dimnames, 0, R_NilValue);
    SET_VECTOR_ELT(r_dimnames, 1, CharacterVector(group_names_));
    SET_VECTOR_ELT(r_dimnames, 2, R_NilValue);
    Rf_dimnamesgets(buffer, r_dimnames);
  }

  void HierarchicalVectorListElement::prepare_to_stream(SEXP object) {
    RealValuedRListIoElement::prepare_to_stream(object);
    RMemoryProtector protector;
    SEXP r_array_dims = protector.protect(Rf_getAttrib(rbuffer(), R_DimSymbol));
    int * array_dims = INTEGER(r_array_dims);
    array_view_.reset(data(), std::vector<int>(array_dims, array_dims + 3));
  }

  void HierarchicalVectorListElement::write() {
    CheckSize();
    int iteration = next_position();
    int dimension = parameters_[0]->dim();
    for (int i = 0; i < parameters_.size(); ++i) {
      const Vector &value(parameters_[i]->value());
      for (int j = 0; j < dimension; ++j) {
        array_view_(iteration, i, j) = value[j];
      }
    }
  }

  void HierarchicalVectorListElement::stream() {
    CheckSize();
    int iteration = next_position();
    int dimension = parameters_[0]->dim();
    Vector values(dimension);
    for (int i = 0; i < parameters_.size(); ++i) {
      for (int j = 0; j < dimension; ++j) {
        values[j] = array_view_(iteration, i, j);
      }
      parameters_[i]->set(values);
    }
  }

  void HierarchicalVectorListElement::set_group_names(
      const std::vector<std::string> &group_names) {
    if (group_names.size() != parameters_.size()) {
      report_error("Vector of group names must be the same size as the "
                   "number of groups.");
    }
    group_names_ = group_names;
  }
  

  void HierarchicalVectorListElement::CheckSize() {
    const std::vector<int> &dims(array_view_.dim());
    if (dims[1] != parameters_.size() ||
        dims[2] != parameters_[0]->dim()) {
      std::ostringstream err;
      err << "sizes do not match in HierarchicalVectorListElement::"
          "stream/write..."
          << endl
          << "dimensions of buffer:    [" << dims[0] << ", " << dims[1] << ", "
          << dims[2] << "]." <<endl
          << "number of groups:    " << parameters_.size() << endl
          << "parameter dimension: " << parameters_[0]->dim() << "." << endl;
      report_error(err.str().c_str());
    }
  }
  //======================================================================
  PartialSpdListElement::PartialSpdListElement(const Ptr<SpdParams> &prm,
                                               const std::string &name,
                                               int which, bool
                                               report_sd)
      : RealValuedRListIoElement(name),
        prm_(prm),
        which_(which),
        report_sd_(report_sd) {}

  void PartialSpdListElement::write() {
    CheckSize();
    double variance = prm_->var()(which_, which_);
    data()[next_position()] = report_sd_ ? sqrt(variance) : variance;
  }

  void PartialSpdListElement::stream() {
    CheckSize();
    SpdMatrix Sigma = prm_->var();
    double v = data()[next_position()];
    if (report_sd_) v *= v;
    Sigma(which_, which_) = v;
    prm_->set_var(Sigma);
  }

  void PartialSpdListElement::CheckSize() {
    if (nrow(prm_->var()) <= which_) {
      std::ostringstream err;
      err << "Sizes do not match in PartialSpdListElement..."
          << endl
          << "Matrix has " << nrow(prm_->var()) << " rows, but "
          << "you're trying to access row " << which_
          << endl;
      report_error(err.str().c_str());
    }
  }
  //======================================================================

  SpdListElement::SpdListElement(const Ptr<SpdParams> &m,
                                 const std::string &param_name)
      : MatrixListElementBase(param_name),
        prm_(m),
        array_view_(0, Array::index3(0, 0, 0))
  {}

  SEXP SpdListElement::prepare_to_write(int niter) {
    int dim = prm_->dim();
    RMemoryProtector protector;
    SEXP buffer = protector.protect(Rf_alloc3DArray(REALSXP, niter, dim, dim));
    StoreBuffer(buffer);
    array_view_.reset(data(), Array::index3(niter, dim, dim));
    return buffer;
  }

  void SpdListElement::write() {
    CheckSize();
    const Matrix &m(prm_->value());
    int iteration = next_position();
    int nr = m.nrow();
    int nc = m.ncol();
    for(int i = 0; i < nr; ++i){
      for(int j = 0; j < nc; ++j){
        array_view_(iteration, i, j) = m(i, j);
      }
    }
  }

  void SpdListElement::prepare_to_stream(SEXP object) {
    RealValuedRListIoElement::prepare_to_stream(object);
    RMemoryProtector protector;
    SEXP r_array_dims = protector.protect(Rf_getAttrib(rbuffer(), R_DimSymbol));
    int * array_dims = INTEGER(r_array_dims);
    array_view_.reset(data(), std::vector<int>(array_dims, array_dims + 3));
  }

  void SpdListElement::stream() {
    CheckSize();
    int iteration = next_position();
    int nr = prm_->dim();
    int nc = prm_->dim();
    Matrix tmp(nr, nc);
    for(int i = 0; i < nr; ++i) {
      for(int j = 0; j < nc; ++j) {
      tmp(i, j) = array_view_(iteration, i, j);
      }
    }
    prm_->set(tmp);
  }

  int SpdListElement::nrow()const {
    return prm_->dim();
  }

  int SpdListElement::ncol()const {
    return prm_->dim();
  }

  void SpdListElement::CheckSize() {
    const std::vector<int> & dims(array_view_.dim());
    const Matrix & value(prm_->value());
    if(value.nrow() != dims[1] ||
       value.ncol() != dims[2]) {
      std::ostringstream err;
      err << "sizes do not match in SpdListElement::stream/write..."
          << endl
          << "dimensions of buffer:    [" << dims[0] << ", " << dims[1] << ", "
          << dims[2] << "]." <<endl
          << "dimensions of parameter: [" << value.nrow() << ", " << value.ncol()
          << "].";
      report_error(err.str().c_str());
    }
  }

  
  //======================================================================
  NativeMatrixListElement::NativeMatrixListElement(MatrixIoCallback *callback,
                                                   const std::string &name,
                                                   Matrix *streaming_buffer)
      : MatrixListElementBase(name),
        streaming_buffer_(streaming_buffer),
        array_view_(0, ArrayBase::index3(0, 0, 0)),
        check_buffer_(true)
  {
    // Protect against NULL.
    if (callback) {
      callback_.reset(callback);
    }
  }

  SEXP NativeMatrixListElement::prepare_to_write(int niter) {
    if (!callback_) {
      report_error(
          "NULL callback in NativeMatrixListElement::prepare_to_write.");
    }
    RMemoryProtector protector;
    SEXP buffer = protector.protect(
        Rf_alloc3DArray(REALSXP, niter, callback_->nrow(), callback_->ncol()));
    set_buffer_dimnames(buffer);
    StoreBuffer(buffer);
    array_view_.reset(data(),
                      Array::index3(niter,
                                    callback_->nrow(),
                                    callback_->ncol()));
    return buffer;
  }

  void NativeMatrixListElement::prepare_to_stream(SEXP object) {
    if (!streaming_buffer_) return;
    RealValuedRListIoElement::prepare_to_stream(object);
    RMemoryProtector protector;
    SEXP r_array_dims = protector.protect(Rf_getAttrib(rbuffer(), R_DimSymbol));
    int *array_dims(INTEGER(r_array_dims));
    std::vector<int> dims(array_dims, array_dims + 3);
    array_view_.reset(data(), dims);
  }

  void NativeMatrixListElement::write() {
    Matrix tmp = callback_->get_matrix();
    int niter = next_position();
    for (int i = 0; i < callback_->nrow(); ++i) {
      for (int j = 0; j < callback_->ncol(); ++j) {
        array_view_(niter, i, j) = tmp(i, j);
      }
    }
  }

  void NativeMatrixListElement::stream() {
    if (!streaming_buffer_) return;
    int niter = next_position();
    for (int i = 0; i < streaming_buffer_->nrow(); ++i) {
      for (int j = 0; j < streaming_buffer_->ncol(); ++j) {
        (*streaming_buffer_)(i, j) = array_view_(niter, i, j);
      }
    }
  }

  int NativeMatrixListElement::nrow()const {
    return callback_->nrow();
  }

  int NativeMatrixListElement::ncol()const {
    return callback_->ncol();
  }

  GenericMatrixListElement::GenericMatrixListElement(
      StreamableMatrixIoCallback *callback,
      const std::string &name)
      : NativeMatrixListElement(callback, name, nullptr)
  {
    if (callback) {
      callback_.reset(callback);
    } else {
      callback_.reset();
    }
  }

  void GenericMatrixListElement::stream() {
    if (!callback_) {
      report_error("Callback was never set.");
    }
    callback_->put_matrix(next_draw().to_matrix());
  }
  
  //======================================================================
  NativeArrayListElement::NativeArrayListElement(ArrayIoCallback *callback,
                                                 const std::string &name,
                                                 bool allow_streaming)
      : RListIoElement(name),
        callback_(callback),
        array_buffer_(NULL, std::vector<int>()),
        allow_streaming_(allow_streaming)
  {
    if (!callback) {
      report_error("NULL callback passed to NativeArrayListElement.");
    }
  }

  SEXP NativeArrayListElement::prepare_to_write(int niter) {
    std::vector<int> dims = callback_->dim();
    std::vector<int> array_dims(dims.size() + 1);
    array_dims[0] = niter;
    std::copy(dims.begin(), dims.end(), array_dims.begin() + 1);
    RMemoryProtector protector;
    SEXP r_array_dimensions =protector.protect(Rf_allocVector(
        INTSXP, array_dims.size()));
    int * rdims = INTEGER(r_array_dimensions);
    for (int i = 0; i < array_dims.size(); ++i) {
      rdims[i] = array_dims[i];
    }

    SEXP r_buffer = protector.protect(Rf_allocArray(
        REALSXP, r_array_dimensions));
    StoreBuffer(r_buffer);

    array_buffer_.reset(REAL(r_buffer), array_dims);
    array_view_index_.assign(array_dims.size(), -1);
    return r_buffer;
  }

  void NativeArrayListElement::prepare_to_stream(SEXP object) {
    if (!allow_streaming_) return;
    RListIoElement::prepare_to_stream(object);
    std::vector<int> array_dims = GetArrayDimensions(object);
    if (array_dims.empty()) {
      report_error("object is not an array in "
                   "NativeArrayListElement::prepare_to_stream.");
    }
    array_buffer_.reset(REAL(rbuffer()), array_dims);
    array_view_index_.assign(array_dims.size(), -1);
  }


  void NativeArrayListElement::write() {
    ArrayView view(next_array_view());
    callback_->write_to_array(view);
  }

  void NativeArrayListElement::stream() {
    if (!allow_streaming_) return;
    ArrayView view(next_array_view());
    callback_->read_from_array(view);
  }

  ArrayView NativeArrayListElement::next_array_view() {
    array_view_index_[0] = next_position();
    return array_buffer_.slice(array_view_index_);
  }
  //======================================================================
  RListOfMatricesListElement::RListOfMatricesListElement(
      const std::string &name,
      const std::vector<int> &rows,
      const std::vector<int> &cols,
      Callback *callback)
      : RListIoElement(name),
        rows_(rows),
        cols_(cols),
        callback_(callback)
  {
    if (rows_.size() != cols_.size()) {
      report_error("The vectors listing the number of rows and columns in "
                   "the stored matrices must be the same size.");
    }
  }
  
  SEXP RListOfMatricesListElement::prepare_to_write(int niter) {
    RMemoryProtector protector;
    int number_of_matrices = rows_.size();
    SEXP r_buffer = protector.protect(
        Rf_allocVector(VECSXP, number_of_matrices));
    views_.clear();
    for (int i = 0; i < number_of_matrices; ++i) {
      std::vector<int> array_dims = {niter, rows_[i], cols_[i]};
      SET_VECTOR_ELT(r_buffer, i, AllocateArray(array_dims));
      views_.push_back(ArrayView(REAL(VECTOR_ELT(r_buffer, i)), array_dims));
    }
    StoreBuffer(r_buffer);
    return r_buffer;
  }

  void RListOfMatricesListElement::prepare_to_stream(SEXP r_object) {
    RListIoElement::prepare_to_stream(r_object);
    SEXP r_buffer = rbuffer();
    int number_of_matrices = Rf_length(r_buffer);
    std::vector<int> array_dims = GetArrayDimensions(VECTOR_ELT(r_buffer, 0));
    int niter = array_dims[0];
    views_.clear();
    for (int i = 0; i < number_of_matrices; ++i) {
      views_.push_back(ArrayView(REAL(VECTOR_ELT(r_buffer, i)),
                                std::vector<int>{niter, rows_[i], cols_[i]}));
    }
  }

  void RListOfMatricesListElement::write() {
    int iteration = next_position();
    for (int layer = 0; layer < views_.size(); ++layer) {
      views_[layer].slice(iteration, -1, -1) = callback_->get(layer);
    }
  }

  void RListOfMatricesListElement::stream() {
    int iteration = next_position();
    for (int layer = 0; layer < views_.size(); ++layer) {
      callback_->put(layer, views_[layer].slice(iteration, -1, -1));
    }
  }

}  // namespace BOOM
