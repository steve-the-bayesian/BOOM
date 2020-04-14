#ifndef BOOM_MATRIX_NORMAL_MODEL_HPP_
#define BOOM_MATRIX_NORMAL_MODEL_HPP_
/*
  Copyright (C) 2005-2018 Steven L. Scott

  This library is free software; you can redistribute it and/or modify it under
  the terms of the GNU Lesser General Public License as published by the Free
  Software Foundation; either version 2.1 of the License, or (at your option)
  any later version.

  This library is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
  details.

  You should have received a copy of the GNU Lesser General Public License along
  with this library; if not, write to the Free Software Foundation, Inc., 51
  Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
*/

#include "Models/Policies/ParamPolicy_3.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/MvnBase.hpp"
#include "Models/SpdParams.hpp"

namespace BOOM {

  // A matrix M follows the Matrix normal distribution MN(mu, R, C) if the
  // vector obtained by stacking the columns of M,
  //   vec(M) ~ N(vec(mu), C \otimes R).
  //
  // Let r'r = R and c'c = C.  Simulating from the matrix normal distribution is
  // done by Y = mu + r' Z c.
  class MatrixNormalModel
      : public MvnBase,
        public ParamPolicy_3<MatrixParams, SpdParams, SpdParams>,
        public IID_DataPolicy<MatrixData>,
        public PriorPolicy
  {
   public:
    // Construct based on the dimension of the random variable.
    MatrixNormalModel(int nrows, int ncols);

    // Args:
    //   mean: A matrix of the same dimension as the random variables being
    //     modeled.
    //   row_variance: The variance describing row effects.
    //   column_variance:  The variance matrix describing column effects.
    MatrixNormalModel(const Matrix &mean,
                      const SpdMatrix &row_variance,
                      const SpdMatrix &column_variance);

    MatrixNormalModel * clone() const override {return new MatrixNormalModel(*this);}

    // MvnBase interface.
    uint dim() const override {return nrow() * ncol();}
    const Vector &mu() const override;
    const SpdMatrix &Sigma() const override;
    const SpdMatrix &siginv() const override;
    double ldsi() const override {
      // The crossing of rows and columns is intentional.
      return nrow() * column_precision_logdet() + ncol() * row_precision_logdet();
    }
    // See also the matrix versions of logp and simulate, below.
    double logp(const Vector &vectorized_matrix) const override;
    Vector sim(RNG &rng = GlobalRng::rng) const override;

    // The number of rows and columns in the random variable described by this
    // model.
    int nrow() const {return mean().nrow();}
    int ncol() const {return mean().ncol();}

    // Parameter accessors
    const Matrix &mean() const {return prm1_ref().value();}
    const SpdMatrix &row_variance() const {return prm2_ref().value();}
    const SpdMatrix &row_precision() const {return prm2_ref().ivar();}
    const SpdMatrix &column_variance() const {return prm3_ref().value();}
    const SpdMatrix &column_precision() const {return prm3_ref().ivar();}

    // Parameter setters
    void set_mean(const Matrix &mean) { prm1_ref().set(mean); }
    void set_row_variance(const SpdMatrix &row_variance) {
      prm2_ref().set_var(row_variance); }
    void set_row_precision(const SpdMatrix &row_precision) {
      prm2_ref().set_ivar(row_precision); }
    void set_column_variance(const SpdMatrix &column_variance) {
      prm3_ref().set_var(column_variance); }
    void set_column_precision(const SpdMatrix &column_precision) {
      prm3_ref().set_ivar(column_precision);}

    // Parameter pointer access
    Ptr<MatrixParams> mean_prm() {return prm1();}
    const Ptr<MatrixParams> mean_prm() const {return prm1();}
    Ptr<SpdParams> row_variance_param() {return prm2();}
    const Ptr<SpdParams> row_variance_param() const {return prm2();}
    Ptr<SpdParams> column_variance_param() {return prm3();}
    const Ptr<SpdParams> column_variance_param() const {return prm3();}

    // Cholesky decompositions and log determinants of the row and column
    // variance parameters.
    double row_precision_logdet() const {return prm2_ref().ldsi();}
    Matrix row_precision_cholesky() const {
      return prm2_ref().ivar_chol(); }
    double column_precision_logdet() const {return prm3_ref().ldsi();}
    Matrix col_precision_cholesky() const {
      return prm3_ref().ivar_chol(); }

    // Mean, variance, and precision of the distribution when transformed to a
    // multivariate normal by stacking the columns of the random variable.
    Vector mvn_mean() const;
    SpdMatrix mvn_variance() const;
    SpdMatrix mvn_precision() const;

    double logp(const Matrix &y) const;
    Matrix simulate(RNG &rng = GlobalRng::rng) const;

   private:
    // Scratch space to use when implementing the MvnBase interface.
    mutable Vector mean_workspace_;
    mutable SpdMatrix variance_workspace_;
  };
}

#endif  // BOOM_MATRIX_NORMAL_MODEL_HPP_
