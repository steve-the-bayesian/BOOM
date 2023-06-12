#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>
#include <memory>
#include <cpputil/report_error.hpp>
#include "LinAlg/Cholesky.hpp"
#include "LinAlg/EigenMap.hpp"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/Selector.hpp"
#include "LinAlg/SpdMatrix.hpp"
#include "LinAlg/CorrelationMatrix.hpp"
#include "LinAlg/Vector.hpp"
#include "LinAlg/VectorView.hpp"

namespace py = pybind11;

namespace BayesBoom {
  using namespace BOOM;

  void LinAlg_def(py::module &boom) {

    py::class_<Vector, std::unique_ptr<Vector>>(boom, "Vector")
        .def(py::init( [] (Eigen::Ref<Eigen::VectorXd> numpy_array) {
              VectorView view(numpy_array.data(), numpy_array.size(), 1);
              return std::unique_ptr<Vector>(new Vector(view));
            }),
          "Create a Vector from a numpy array.  Be sure the dtype is float!")
        .def("all_finite", &Vector::all_finite,
             "Returns true iff all elements are finite.")
        .def_property_readonly("randomize", &Vector::randomize,
             "Fill the vector with U(0, 1) random deviates.")
        .def_property_readonly("stride", &Vector::stride,
             "The distance between consecutive elements.  "
             "For a dense vector this is always 1.")
        .def("__len__", &Vector::length)
        .def_property_readonly(
            "empty",
            &Vector::empty,
            "True iff the Vector has no elements.")
        .def_property_readonly("length", &Vector::length,
                               "The number of elements in the vector.")
        .def_property_readonly("size", &Vector::length,
                               "The number of elements in the vector.")
        .def("to_numpy", [](const Vector &v) {return Eigen::VectorXd(EigenMap(v));})
        .def("__getitem__", [](const Vector &v, int i) {return v[i];}, py::is_operator())
        .def("__setitem__", [](Vector &v, int i, double value) {return v[i] = value;},
             py::is_operator())
        .def(py::pickle(
            [](const Vector &v) {
              return py::make_tuple(std::vector<double>(v));
            },
            [](const py::tuple &tup) {
              if (tup.size() != 1) {
                report_error("Invalid state for unpickling a BOOM::Vector");
              }
              Vector v(tup[0].cast<std::vector<double>>());
              return v;
            }))
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self / py::self)
        .def(py::self += py::self)
        .def(py::self *= py::self)
        .def(py::self + float())
        .def(py::self - float())
        .def(py::self * float())
        .def(py::self / float())
        .def(py::self += float())
        .def(py::self -= float())
        .def(py::self *= float())
        .def(py::self /= float())
        .def("normsq",
             &Vector::normsq,
             "The square norm (L2 norm) of the vector.  The sum of squared elements.")
        .def("__repr__",
             [](const Vector &v) {
               std::ostringstream out;
               out << v;
               return out.str();
             })
        ;
    py::implicitly_convertible<Eigen::VectorXd, Vector>();
    py::implicitly_convertible<py::array, Vector>();

    // =========================================================================
    py::class_<VectorView>(boom, "VectorView")
        .def(py::init(
            [](Vector &v, int first) {
              return VectorView(v, first);
            }),
             py::arg("v"),
             py::arg("first") = 0,
             "Create a VectorView from a boom.Vector.\n\n"
             "Args:\n"
             "  v:  The vector containing data for the view.\n"
             "  first: The first element in the view.")
        .def("__getitem__", [](const VectorView &v, int i) {return v[i];}, py::is_operator())
        .def("__setitem__", [](VectorView &v, int i, double value) {return v[i] = value;},
             py::is_operator())
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self / py::self)
        .def(py::self += py::self)
        .def(py::self *= py::self)
        .def(py::self + float())
        .def(py::self - float())
        .def(py::self * float())
        .def(py::self += float())
        .def(py::self -= float())
        .def(py::self *= float())
        .def(py::self /= float())
        ;

    // =========================================================================
    py::class_<Matrix>(boom, "Matrix")
        .def(py::init<int, int, double>(),
             py::arg("nrow") = 0,
             py::arg("ncol") = 0,
             py::arg("value") = 0.0,
             "Create a matrix with the specified number of rows and columns, "
             "with all elements set to the the given value")
        .def(py::init( [] (const Eigen::MatrixXd &numpy_array) {
              return std::unique_ptr<Matrix>(
                  new Matrix(numpy_array.rows(),
                             numpy_array.cols(),
                             numpy_array.data(),
                             false));   // byrow.  False means column-storage order.
            }),
          "Create a Matrix from a 2-D numpy array."
          )
        .def("__getitem__",
             [](const Matrix &m, py::tuple ij) {
               int i = ij[0].cast<int>();
               int j = ij[1].cast<int>();
               return m(i, j);},
             "Element access.")
        .def("__setitem__",
             [](Matrix &m, py::tuple ij, double value) {
               int i = ij[0].cast<int>();
               int j = ij[1].cast<int>();
               m(i, j) = value;
             },
             "Element assignment.")
        .def_property_readonly("nrow", &Matrix::nrow, "The number of rows in the matrix.")
        .def_property_readonly("ncol", &Matrix::ncol, "The number of columns in the matrix.")
        .def("inner",
             [](const Matrix &m) {return m.inner();},
             "If this matrix is X, return X'X as a boom.SpdMatrix.")
        .def("inv",
             &Matrix::inv,
             "Return the inverse of the matrix.  The matrix itself is unchanged.")
        .def("max_abs",
             &Matrix::max_abs,
             "The absolute value of the matrix element with the largest absolute value.")
        .def("to_numpy",
             [](const Matrix &m) {return Eigen::MatrixXd(EigenMap(m));},
             "Convert the matrix to a numpy array." )
        .def(py::pickle(
            [](const Matrix &mat) {
              int nrow = mat.nrow();
              int ncol = mat.ncol();
              return py::make_tuple(
                  nrow, ncol, std::vector<double>(vec(mat)));
            },
            [](const py::tuple &tup) {
              int nrow = tup[0].cast<int>();
              int ncol = tup[1].cast<int>();
              std::vector<double> data = tup[2].cast<std::vector<double>>();
              return Matrix(nrow, ncol, data.data());
            }))
        .def("to_numpy", [](const Vector &v) {return Eigen::VectorXd(EigenMap(v));})
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self / py::self)
        .def(py::self += py::self)
        .def(py::self + float())
        .def(py::self - float())
        .def(py::self * float())
        .def(py::self / float())
        .def(py::self * Vector())
        .def("__repr__",
             [](const Matrix &m) {
               std::ostringstream out;
               out << m;
               return out.str();
             })
        ;

    py::implicitly_convertible<py::array, Matrix>();

    // ===========================================================================
    py::class_<SpdMatrix, Matrix>(boom, "SpdMatrix")
        .def(py::init<int, double>(),
             py::arg("dim") = 0,
             py::arg("diagonal_value") = 1.0,
             "Create a symmetric positive definite matrix of the given dimension. "
             "The diagonal elements are constant and equal to 'digaonal_value'. "
             )
        .def(py::init([] (const Eigen::MatrixXd &numpy_spd) {
              return std::unique_ptr<SpdMatrix>(
                  new SpdMatrix(numpy_spd.cols(),
                                numpy_spd.data(),
                                false));
                       }),
          "Create a symmetric positive definite matrix by copying data "
          "from a 2-D numpy array.")
        .def(py::init([] (const Matrix &m) {
                        return std::unique_ptr<SpdMatrix>(
                            new SpdMatrix(m));
                      }),
            "Create a symmetric positive definite matrix from a regular "
            "Matrix object")
        .def(py::pickle(
            [](const SpdMatrix &mat) {
              return py::make_tuple(static_cast<int>(mat.nrow()),
                                    std::vector<double>(vec(mat)));
            },
            [](const py::tuple &tup) {
              int dim = tup[0].cast<int>();
              std::vector<double> data = tup[1].cast<std::vector<double>>();
              return SpdMatrix(dim, data.data());
            }))
        .def("inv",
             &Matrix::inv,
             "Return the inverse of the matrix.  The matrix itself is unchanged.")
        .def("Mdist",
             [](const SpdMatrix &S, const Vector &x, const Vector &y) {
               return S.Mdist(x, y);
             },
             "The Mahalanobis distance from x to y with respect to this object.\n"
             "Args:\n"
             "  x, y: Boom Vectors"
             "Returns:\n"
             "  The scalar distance from x to y: (x - y)^T * this * (x - y)."

             )
        .def("diag", [] (SpdMatrix &m) {return m.diag();},
             "The diagonal elements of the matrix, as a VectorView.")
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self / py::self)
        .def(py::self += py::self)
        .def(py::self + float())
        .def(py::self - float())
        .def(py::self * float())
        .def(py::self / float())
        ;

    boom.def("random_correlation_matrix",
             [](int dim, RNG &rng) {
               return SpdMatrix(random_cor_mt(rng, dim));
             },
             py::arg("dim"),
             py::arg("rng") = GlobalRng::rng,
             "A marginally uniform correlation matrix.\n"
             "Args\n:"
             "  dim:  The number of rows in the matrix.\n"
             "  rng: The random number generator.\n");

    py::implicitly_convertible<py::array, SpdMatrix>();

    py::class_<Cholesky>(boom, "Cholesky")
        .def(py::init<const Matrix &>(),
             py::arg("A"),
             "The (lower) cholesky decomposition of the matrix A.")
        .def_property_readonly("nrow", &Cholesky::nrow)
        .def_property_readonly("ncol", &Cholesky::ncol)
        .def_property_readonly("dim", &Cholesky::dim)
        .def("getL", &Cholesky::getL, py::arg("check"),
             "Extract the lower triangular matrix.\n"
             "Args:\n"
             "  check:  If True, throw an exception if the original matrix "
             "was not positive definite.\n"
             )
        .def("getLT", &Cholesky::getLT, "The transpose of L.")
        .def("solve", [](const Cholesky &chol,
                         const Matrix &mat) {
               return chol.solve(mat);
             },
             "Return Ainv * mat, where Ainv is the inverse of the "
             "decomposed matrix.")
        .def("solve", [](const Cholesky &chol,
                         const Vector &vec) {
               return chol.solve(vec);
             },
             "Return Ainv * vec, where Ainv is the inverse of the "
             "decomposed matrix.")
        .def("inv", &Cholesky::inv, "Inverse of the decomposed matrix.")
        .def_property_readonly("original_matrix",
                               &Cholesky::original_matrix,
                               "The matrix that was decomposed.")
        .def("det", &Cholesky::det, "Determinant of the original matrix.")
        .def("logdet",  &Cholesky::logdet,
             "Log of the determinant of the original matrix.\n"
             "It is computationally more stable to call chol.logdet() "
             "than log(chol.det()).")
        .def_property_readonly("pos_def", &Cholesky::is_pos_def,
                               "Is the original matrix positive definite.")
        ;

    py::class_<Selector>(boom, "Selector")
        .def(py::init<BOOM::uint, bool>(),
             py::arg("dim"),
             py::arg("all") = true,
             "Create a selector of all True or all False entries.\n"
             "Args:\n"
             "  dim:  The number of potential elements.\n"
             "  all:  If True include all elements.  If False exclude all elements.")
        .def(py::init(
            [](const std::vector<BOOM::uint> &included_positions,
               int dim) {
              return new Selector(included_positions, dim);
            }),
             py::arg("included_positions"),
             py::arg("dim"),
             "Create a Selector from a list of included positions.\n"
             "Args:\n"
             "  included_positions:  A list of integer-valued indices "
             "to mark as included.\n"
             "  dim:  The number of possible elements.\n")
        .def_property_readonly("nvars", &Selector::nvars,
                               "The number of included variables.")
        .def_property_readonly("nvars_possible", &Selector::nvars_possible,
                               "The number of available variables.")
        .def_property_readonly("nvars_excluded", &Selector::nvars_excluded,
                               "The number of excluded variables.")
        .def("add", &Selector::add, "Add the variable in position i.")
        .def("drop", &Selector::add, "Drop (remove) the variable in position i.")
        .def("flip", &Selector::add,
             "Flip the variable in position i.  If it is in, drop it.  "
             "If it is out, add it.")
        .def_property_readonly(
            "included_positions",
            &Selector::included_positions,
            "A vector of integers giving the location of included positions.")
        .def("__repr__",
             [](const Selector &inc) {
               std::ostringstream out;
               out << "A boom.Selector containing " << inc.nvars()
                   << " of " << inc.nvars_possible() << " variables.";
               if (inc.nvars() < 40) {
                 for (int i = 0; i < inc.nvars(); ++i) {
                   out << inc.expanded_index(i) << " ";
                 }
                 out << "\n";
               }
               return out.str();
             })
        ;

  }  // ends the LinAlg_def function.

}  // namespace BayesBoom
