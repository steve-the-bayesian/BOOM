// Copyright 2018 Google LLC. All Rights Reserved.
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

#ifndef BOOM_R_LIST_IO_HPP_
#define BOOM_R_LIST_IO_HPP_

#include <string>
#include "LinAlg/Vector.hpp"
#include "LinAlg/VectorView.hpp"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/SubMatrix.hpp"
#include "LinAlg/Array.hpp"

#include "Models/ModelTypes.hpp"
#include "Models/ParamTypes.hpp"
#include "Models/SpdParams.hpp"
#include "Models/Glm/GlmCoefs.hpp"

#include "r_interface/boom_r_tools.hpp"

//======================================================================
// Note that the functions listed here throw exceptions.  Code that
// uses them should be wrapped in a try-block where the catch
// statement catches the exception and calls Rf_error() with an
// appropriate error message.  The functions handle_exception(), and
// handle_unknown_exception (in handle_exception.hpp), are suitable
// defaults.  These try-blocks should be present in any code called
// directly from R by .Call.
//======================================================================

namespace BOOM{
  class RListIoElement;

  // An RListIoManager manages an R list that is used to store the
  // output of an MCMC run in BOOM, and to read it back in again for
  // doing posterior computations.  The class maintains a vector of
  // RListIoElement's each of which is responsible for managing a
  // single list element.
  //
  // The basic idiom for output is
  // RListIoManager io_manager;
  // io_manager.add_list_element(new VectorListElement(...));
  // io_manager.add_list_element(new PartialSpdListElement(...));
  // int niter = 1000;
  // SEXP ans;
  // PROTECT(ans = io_manager.prepare_to_write(niter));
  // for(int i = 0; i < niter; ++i){
  //   do_an_mcmc_iteration();
  //   io_manager.write();
  // }
  // UNPROTECT(1);
  // return ans;

  // The basic idiom for streaming through an already populated list is
  // RListIoManager io_manager;
  // io_manager.add_list_element(new VectorListElement(...));
  // io_manager.add_list_element(new SpdListElement(...));
  // io_manager.prepare_to_stream();
  // int niter = 1000;
  // io_manager.advance(100);  // discard some burn-in
  // for(int i = 0; i < niter; ++i){
  //   io_manager.stream();
  //   do_something_with_the_current_value();
  // }

  class RListIoManager {
   public:
    // The class takes over ownership of 'element', and deletes it
    // when *this goes out of scope.
    void add_list_element(RListIoElement *element);

    // Returns a list with the necessary names and storage for keeping
    // track of 'niter' parameters worth of output.
    SEXP prepare_to_write(int niter);

    // Takes an existing list as an argument, and gets each component
    // in elements_ ready to stream from it.
    void prepare_to_stream(SEXP object);

    // Each managed parameter writes its value to the appropriate
    // portion of the list, and then increments its position to get
    // ready for the next write.
    void write();

    // Each managed parameter reads its next value from the list, then
    // increments its position to get ready for the next read.
    void stream();

    // Each element moves forward n steps.  This is useful for
    // discarding the first 'n' elements in an MCMC sample.
    void advance(int n);
   private:
    std::vector<std::shared_ptr<RListIoElement> > elements_;
  };

  //======================================================================
  // An RListIoelement takes care of allocating space, recording to,
  // and streaming parameters from an R list.  One instance is
  // required for each distinct parameter in the model output list.
  class RListIoElement {
   public:
    explicit RListIoElement(const std::string &name);
    virtual ~RListIoElement();

    // Allocates and returns the R object (usually a vector, matrix,
    // or array), to be stored in the list.  It is the caller's
    // responsibility to PROTECT this object, if needed.  The 'caller'
    // will almost always be the RListIoManager, which has the
    // PROTECT/UNPROTECT stuff safely hidden away so the caller won't
    // need to worry about protecting the individual list elements.
    virtual SEXP prepare_to_write(int niter) = 0;

    // Takes the list as an argument.  Finds the list element that
    // this object is supposed to manage in the given object.  Set the
    // input buffers
    virtual void prepare_to_stream(SEXP object);

    // Leaf classes keep track of the position in the output buffer,
    // and increment it whenever write() is called.
    virtual void write() = 0;

    // Leaf classes keep track of the position in the input buffer,
    // and increment it whenever stream() is called.
    virtual void stream() = 0;

    // Return the name of the component in the list.
    const std::string &name()const;

    // Move position in stream forward by n places.
    void advance(int n);
   protected:
    // StoreBuffer must be called in derived classes to pass the SEXP
    // that manages the parameter to this base class.
    virtual void StoreBuffer(SEXP buffer);
    SEXP rbuffer(){return rbuffer_;}

    // Calling next_position() returns the current position and
    // advances the counter.  If you need it more than once, be sure
    // to store it.
    int next_position();
    double *data();
   private:
    // Prohibit copy and assign
    RListIoElement(const RListIoElement &rhs);
    void operator=(const RListIoElement &rhs);

    std::string name_;
    SEXP rbuffer_;  // The R object holding the BOOM output
    int position_;  // Current position in the rbuffer
    double *data_;  // Pointer to the first element in the rbuffer
  };

  //----------------------------------------------------------------------
  // Most elements in the list will be arrays of fixed dimension
  // storing real numbers.  This class makes it easy to handle real
  // valued data.
  class RealValuedRListIoElement : public RListIoElement {
   public:
    explicit RealValuedRListIoElement(const std::string &name);
    SEXP prepare_to_write(int niter) override;
    void prepare_to_stream(SEXP object) override;
   protected:
    void StoreBuffer(SEXP buffer) override;
    double *data();
   private:
    double *data_;
  };

  //----------------------------------------------------------------------
  class ListValuedRListIoElement : public RListIoElement {
   public:
    explicit ListValuedRListIoElement(const std::string &name);
    SEXP prepare_to_write(int niter) override;
  };

  //----------------------------------------------------------------------
  // For tracking an individual diagonal element of a variance matrix.
  class PartialSpdListElement : public RealValuedRListIoElement {
   public:
    PartialSpdListElement(const Ptr<SpdParams> &prm,
                          const std::string &param_name,
                          int which,
                          bool report_sd);
    void write() override;
    void stream() override;
   private:
    void CheckSize();
    Ptr<SpdParams> prm_;
    int which_;
    bool report_sd_;
  };

  //----------------------------------------------------------------------
  // For managing UnivariateParams, stored in an R vector.
  class UnivariateListElement : public RealValuedRListIoElement {
   public:
    UnivariateListElement(const Ptr<UnivParams> &, const std::string &name);
    void write() override;
    void stream() override;
   private:
    Ptr<UnivParams> prm_;
  };

  //----------------------------------------------------------------------
  // A callback interface class for managing scalar quantities that
  // are not stored in a BOOM::Params object.  The purpose of a
  // ScalarIoCallback is to supply values for a
  // NativeUnivariateListElement to write to the vector that it
  // maintains.
  class ScalarIoCallback {
   public:
    virtual ~ScalarIoCallback(){}
    virtual double get_value()const=0;
  };

  // A callback class for saving log likelihood values.
  class LogLikelihoodCallback : public ScalarIoCallback {
   public:
    explicit LogLikelihoodCallback(LoglikeModel *model)
        : model_(model) {}
    double get_value() const override {
      return model_->log_likelihood();
    }
   private:
    LoglikeModel *model_;
  };

  // For managing scalar (double) output that is not stored in a
  // UnivParams.
  class NativeUnivariateListElement : public RealValuedRListIoElement {
   public:
    // Args:
    //   callback: A pointer to the callback object responsible for
    //     obtaining values for the list element to write().  This can
    //     be NULL if the object is being created just for streaming.
    //     The object takes ownership of the pointer, so it should not
    //     be deleted manually.
    //   name:  The name of the component in the R list.
    //   streaming_buffer: A pointer to a double that can hold the
    //     streamed value.  If streaming_buffer is NULL then this
    //     component of the list will not be streamed.
    NativeUnivariateListElement(ScalarIoCallback *callback,
                                const std::string &name,
                                double *streaming_buffer = NULL);
    SEXP prepare_to_write(int niter) override;
    void prepare_to_stream(SEXP object) override;
    void write() override;
    void stream() override;
   private:
    std::shared_ptr<ScalarIoCallback> callback_;
    double *streaming_buffer_;
    BOOM::VectorView vector_view_;
  };

  //----------------------------------------------------------------------
  // Use this class when BOOM stores a variance, but you want to
  // report a standard deviation.
  class StandardDeviationListElement : public RealValuedRListIoElement {
   public:
    StandardDeviationListElement(const Ptr<UnivParams> &prm,
                                 const std::string &name);
    void write() override;
    void stream() override;
   private:
    Ptr<UnivParams> variance_;
  };

  //----------------------------------------------------------------------
  // For managing VectorParams, stored in an R matrix.
  class VectorListElement : public RealValuedRListIoElement {
   public:
    VectorListElement(const Ptr<VectorParams> &m,
                      const std::string &param_name,
                      const std::vector<string> &element_names
                      = std::vector<string>());
    // Allocate a matrix
    SEXP prepare_to_write(int niter) override;
    void prepare_to_stream(SEXP object) override;
    void write() override;
    void stream() override;
   private:
    void CheckSize();
    Ptr<VectorParams> prm_;
    SubMatrix matrix_view_;
    std::vector<string> element_names_;
  };

  //----------------------------------------------------------------------
  // For vectors representing regression or glm coefficients.  These
  // need special attention when streaming, because include/exclude
  // indicators must be set correctly.
  class GlmCoefsListElement : public VectorListElement {
   public:
    GlmCoefsListElement(const Ptr<GlmCoefs> &m,
                        const std::string &param_name,
                        const std::vector<string> &element_names
                        = std::vector<string>());
    void stream() override;

    // If coefficient names are set prior to calling prepare_to_write()
    void set_coefficient_names(const std::vector<string> &names);
   private:
    Ptr<GlmCoefs> coefs_;

    // Workspace to use when streaming.
    Vector beta_;
    const std::vector<string> coefficient_names_;
  };

  //----------------------------------------------------------------------
  // For reporting a vector of standard deviations when the model
  // stores a vector of variances.
  class SdVectorListElement : public RealValuedRListIoElement {
   public:
    SdVectorListElement(const Ptr<VectorParams> &v,
                        const std::string &param_name);
    SEXP prepare_to_write(int niter) override;
    void prepare_to_stream(SEXP object) override;
    void write() override;
    void stream() override;
   private:
    void CheckSize();
    Ptr<VectorParams> prm_;
    SubMatrix matrix_view_;
  };

  //----------------------------------------------------------------------
  // A mix-in class for handling row and column names for list
  // elements that store MCMC draws of matrices.
  class MatrixListElementBase : public RealValuedRListIoElement {
   public:
    explicit MatrixListElementBase(const std::string &param_name)
        : RealValuedRListIoElement(param_name) {}
    virtual int nrow() const = 0;
    virtual int ncol() const = 0;
    const std::vector<std::string> & row_names()const;
    const std::vector<std::string> & col_names()const;
    void set_row_names(const std::vector<std::string> &row_names);
    void set_col_names(const std::vector<std::string> &row_names);
   protected:
    // Children of this class should call set_buffer_dimnames(buffer)
    // when they prepare_to_write().
    void set_buffer_dimnames(SEXP buffer);
   private:
    std::vector<std::string> row_names_;
    std::vector<std::string> col_names_;
  };

  //----------------------------------------------------------------------
  // For managing MatrixParams, stored in an R 3-way array.
  class MatrixListElement : public MatrixListElementBase {
   public:
    MatrixListElement(const Ptr<MatrixParams> &m,
                      const std::string &param_name);

    // Allocate an array to hold the matrix draws.
    SEXP prepare_to_write(int niter) override;
    void prepare_to_stream(SEXP object) override;
    void write() override;
    void stream() override;

    int nrow()const override;
    int ncol()const override;
   private:
    void CheckSize();
    Ptr<MatrixParams> prm_;
    ArrayView array_view_;
  };

  // For managing vectors of coefficients in a hierarchical model.
  class HierarchicalVectorListElement
      : public RealValuedRListIoElement {
   public:
    // Use this constructor if you have a list of parameter vectors
    // already collected.
    HierarchicalVectorListElement(
        const std::vector<Ptr<VectorParams>> &parameters,
        const std::string &param_name);

    // Use this constructor if you plan to add parameter vectors one
    // at a time.  This use case obviously requires holding onto the
    // list element pointer outside of an RListIoManager.
    explicit HierarchicalVectorListElement(const std::string &param_name);

    void add_vector(const Ptr<VectorParams> &vector);
    SEXP prepare_to_write(int niter) override;
    void prepare_to_stream(SEXP object) override;
    void write() override;
    void stream() override;
    void set_group_names(const std::vector<std::string> &group_names);

   private:
    void set_buffer_group_names(SEXP buffer);
    void CheckSize();
    std::vector<Ptr<VectorData>> parameters_;
    ArrayView array_view_;
    std::vector<std::string> group_names_;
  };

  //----------------------------------------------------------------------
  // Stores a collection of UnivParams objects in a matrix.
  class UnivariateCollectionListElement
      : public RealValuedRListIoElement {
   public:
    UnivariateCollectionListElement(
        const std::vector<Ptr<UnivParams>> &parameters,
        const std::string &param_name);

    SEXP prepare_to_write(int niter) override;
    void prepare_to_stream(SEXP object) override;
    void write() override;
    void stream() override;
   protected:
    void CheckSize();
    std::vector<Ptr<UnivParams>> &parameters() {return parameters_;}
    SubMatrix &matrix_view() {return matrix_view_;}
   private:
    std::vector<Ptr<UnivParams>> parameters_;
    SubMatrix matrix_view_;
  };

  // A specialization of UnivariateCollectionListElement for when the parameters
  // being stored are variances, but you want to report them as standard
  // deviations.
  class SdCollectionListElement
      : public UnivariateCollectionListElement {
   public:
    SdCollectionListElement(
        const std::vector<Ptr<UnivParams>> &variances,
        const std::string &param_name)
        : UnivariateCollectionListElement(variances, param_name) {}
    void write() override;
    void stream() override;
  };

  //----------------------------------------------------------------------
  // For managing SpdParams, stored in an R 3-way array.
  class SpdListElement : public MatrixListElementBase {
   public:
    SpdListElement(const Ptr<SpdParams> &m,
                   const std::string &param_name);

    // Allocate an array to hold the matrix draws.
    SEXP prepare_to_write(int niter) override;
    void prepare_to_stream(SEXP object) override;
    void write() override;
    void stream() override;

    int nrow()const override;
    int ncol()const override;
   private:
    void CheckSize();
    Ptr<SpdParams> prm_;
    ArrayView array_view_;
  };

  //----------------------------------------------------------------------
  // A VectorIoCallback is a base class for managing native BOOM Vec
  // objects that are not part of VectorParams.  To use it, define a
  // local class that inherits from VectorIoCallback.  The class
  // should store a pointer to the object you really care about, and
  // which can supply the two necessary member functions.  Then put
  // the callback into a NativeVectorListElement, described below.
  class VectorIoCallback{
   public:
    virtual ~VectorIoCallback(){}
    virtual int dim()const=0;
    virtual Vector get_vector()const=0;
  };

  // A NativeVectorListElement manages a native BOOM Vector that is not
  // stored in a VectorParams.
  class NativeVectorListElement : public RealValuedRListIoElement{
   public:
    // Args:
    //   callback: supplied access to the vectors that need to be
    //     recorded.  This can be NULL if the object is being created
    //     for streaming.  If it is non-NULL then this class takes
    //     ownership and deletes the callback on destruction.
    //   name:  the name of the entry in the R list.
    //   streaming_buffer: A pointer to a BOOM Vector/Vector that will
    //     receive the contents of the R list when streaming.  This
    //     can be NULL if streaming is not desired.
    NativeVectorListElement(VectorIoCallback *callback,
                         const std::string &name,
                         Vector *streaming_buffer);
    SEXP prepare_to_write(int niter) override;
    void prepare_to_stream(SEXP object) override;
    void write() override;
    void stream() override;
   private:
    std::shared_ptr<VectorIoCallback> callback_;
    Vector *streaming_buffer_;
    SubMatrix matrix_view_;
  };

  //----------------------------------------------------------------------
  // Please see the comments to VectorIoCallback, above.
  class MatrixIoCallback{
   public:
    virtual ~MatrixIoCallback(){}
    virtual int nrow()const = 0;
    virtual int ncol()const = 0;
    virtual Matrix get_matrix()const=0;
  };

  // A NativeMatrixListElement manages a BOOM Mat/Matrix that is not
  // stored in a MatrixParams.
  class NativeMatrixListElement : public MatrixListElementBase{
   public:
    // Args:
    //   callback: supplies access to the matrices that need
    //     recording.  This can be NULL if the object is being created
    //     simply for streaming.  If it is non-NULL, this class takes
    //     ownership and deletes the callback on destruction.
    //   name: the name of the component in the list.
    //   streaming_buffer: A pointer to a BOOM matrix that will
    //     receive the contents of the R list when streaming.  This
    //     can be NULL if streaming is not desired.
    //
    // Note that it is pointless to create this object if both
    // callback and streaming_buffer are NULL.
    NativeMatrixListElement(MatrixIoCallback *callback,
                            const std::string &name,
                            Matrix *streaming_buffer);
    SEXP prepare_to_write(int niter) override;
    void prepare_to_stream(SEXP object) override;
    void write() override;
    void stream() override;

    int nrow()const override;
    int ncol()const override;
   private:
    std::shared_ptr<MatrixIoCallback> callback_;
    Matrix *streaming_buffer_;
    ArrayView array_view_;
  };

  //----------------------------------------------------------------------
  // A NativeArrayListElement manages output for one or more
  // parameters where a single MCMC iteration is represented by an R
  // multidimensional array.  The array has leading dimension niter
  // (number of MCMC iterations).  The remaining dimensions are
  // specified by the dim() member function of the callback provided
  // to the constructor.
  class ArrayIoCallback {
   public:
    virtual ~ArrayIoCallback(){}

    // Returns the dimensions of the array corresponding to one MCMC
    // draw.  This will have one less element than the R object
    // holding the draws (which has a leading dimension corresponding
    // to MCMC iteration number).
    virtual std::vector<int> dim()const = 0;

    // Write the parameters to be stored.
    // Args:
    //   view: A view into an array (of dimension dim()) that holds
    //     the parameters.  The intent is for view to be a slice of
    //     the R array that will eventually be returned to the user.
    virtual void write_to_array(ArrayView &view)const = 0;

    // Read parameters from a previously stored array.
    // Args:
    //   view: A view into an array (of dimension dim()) that holds
    //     previously stored values of the parameters.  The intent is
    //     that this is a slice of the R array to which the parameters
    //     were written by write_to_array().
    virtual void read_from_array(const ArrayView &view) = 0;
  };

  // Introductory comments given above ArrayIoCallback.
  class NativeArrayListElement : public RListIoElement {
   public:
    // Args:
    //   callback: A pointer to an object descended from class
    //     ArrayIoCallback.  The NativeArrayListElement takes
    //     ownership of callback, which is deleted by the
    //     NativeArrayListElement destructor.
    //   name: The name of this object's component in the list
    //     managed by its RListIoManager.
    NativeArrayListElement(ArrayIoCallback *callback,
                           const std::string &name);
    SEXP prepare_to_write(int niter) override;
    void prepare_to_stream(SEXP object) override;
    void write() override;
    void stream() override;
   private:
    // Returns an ArrayView pointing to the next position (MCMC
    // iteration) in the buffer.  The ArrayView has one less dimension
    // than the R object, because it corresponds to a single MCMC
    // iteration.
    ArrayView next_array_view();

    std::shared_ptr<ArrayIoCallback> callback_;

    // A view into the R buffer holding the data.
    ArrayView array_buffer_;

    // An index used to subscript array_buffer_ when calling
    // next_array_view().  The leading index is the MCMC number.  All
    // other positions are -1, as detailed in the comments to
    // ArrayView::slice().
    std::vector<int> array_view_index_;
  };

}  // namespace BOOM

#endif  // BOOM_R_LIST_IO_HPP_
