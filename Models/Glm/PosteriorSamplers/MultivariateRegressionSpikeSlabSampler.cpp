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

#include "Models/Glm/PosteriorSamplers/MultivariateRegressionSpikeSlabSampler.hpp"
#include "Models/PosteriorSamplers/MvnVarSampler.hpp"
#include "cpputil/lse.hpp"
#include "distributions.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  namespace {
    using MRSSS = MultivariateRegressionSpikeSlabSampler;
  }  // namespace 

  void CompositeCholesky::decompose(const Matrix &row_cholesky,
                                    const Matrix &siginv_cholesky,
                                    const SelectorMatrix &included) {
    report_error("More work is needed on decompose.  "
                 "Call decompose_simple instead.");
    if (included.nrow() != row_cholesky.nrow()
        || included.ncol() != siginv_cholesky.ncol()) {
      std::ostringstream err;
      err << "The number of rows in the inclusion matrix ("
          << included.nrow() << ") should match the dimension of "
          << "the row Cholesky matrix (" << row_cholesky.nrow()
          << ")," << std::endl
          << "and the number of columns in the inclusion matrix ("
          << included.ncol() << ") should match the dimension of the "
          << "column Cholesky matrix (" << siginv_cholesky.ncol()
          << ")." << std::endl;
      report_error(err.str());
    }
    
    if (included.all_in()) {
      chol_ = Kronecker(siginv_cholesky, row_cholesky);
      return;
    }

    int ydim = included.ncol();
    int xdim = included.nrow();
    // The rows of the selector matrix correspond to variables.  The columns
    // correspond to dimensions of Y.
    Selector in_any_column = included.row_any();
    int number_of_active_predictors = in_any_column.nvars();

    // The full precision matrix, which we don't deal with directly, has row
    // dimension (xdim * ydim).  We subset this matrix twice.  First by
    // excluding X's that are inactive for all Y's. This reduces it to
    // number_of_active_predictors * ydim, in both the row and column dimension.
    // Second by excluding X's that are active for some Y's but not others,
    // which reduces it to included.nvars().
    int active_precision_dimension = number_of_active_predictors * ydim;
    int precision_dimension = included.nvars();

    // Logically, we now want subset of rows of siginv_cholesky \otimes
    // row_cholesky, which is the Kronecker product of two lower triangular
    // matrices.  Thus the product has lower triangular blocks in addition to
    // being globally lower triangular.  After taking the subset, we would
    // compute the LQ decomposition (i.e. QR transpose).
    // 
    // Thus qr_workspace is built holding the transpose so that it can be fed to
    // QR.  In the code below we adopt notation according to the 'notional'
    // problem, and refer to a row of the logical matrix as a 'row', but we
    // actually store it as a column of qr_workspace.
    Matrix qr_workspace(active_precision_dimension,
                        precision_dimension,
                        0.0);

    // The loop over the included variables should proceed in column-major order
    // to coincide with the vec operator.  Thus the usual loops over i and j are
    // flipped.
    int next_column = 0;
    for (int j = 0; j < ydim; ++j) {
      for (int i = 0; i < xdim; ++i) {
        if (included(i, j)) {
          // If beta(i, j) is nonzero then get the corresponding rows in
          // row_cholesky and siginv_cholesky.
          int which_x = in_any_column.INDX(i);
          ConstVectorView ominv_row(row_cholesky.row(which_x));
          // ominv_row has a bunch of trailing zeros because it is a row of a
          // lower triangular matrix.  minimal_ominv_row only includes the
          // leading elements.
          ConstVectorView minimal_ominv_row(ominv_row, 0, which_x + 1);
          ConstVectorView siginv_row(siginv_cholesky.row(j));

          // Now use the rows of the two component matrices to build the row of
          // the notional matrix.
          Vector row(active_precision_dimension, 0.0);
          for (int block = 0; block <= j; ++block) {
            int start = block * number_of_active_predictors;
            // The size of the block containing possibly nonzero elements is
            // equal to which_x, because the matrix is lower triangular.  The +1
            // comes from counting from zero: the first row (row 0) has one
            // element.
            int block_size = which_x + 1;
            VectorView(row, start, block_size) =
                minimal_ominv_row * siginv_row[block];
          }
          qr_workspace.col(next_column++) = row;
        }
      }
    }
    QR qr(qr_workspace);
    chol_ = qr.getR().transpose();
    // Ensure the QR decomposition has positive diagonal elements.
    for (int i = 0; i < chol_.ncol(); ++i) {
      if (chol_(i, i) < 0) chol_.col(i) *= -1;
    }
  }
  //---------------------------------------------------------------------------
  // Like decompose, but we don't try to eliminate the X's that never appear.
  // This is computationlly less efficient, but easier to get correct.  One day
  // this will be a test case against decompose(), but until the latter can be
  // proved correct use this instead.
  void CompositeCholesky::decompose_simple(const Matrix &row_cholesky,
                                           const Matrix &siginv_cholesky,
                                           const SelectorMatrix &included) {
    if (included.all_in()) {
      chol_ = Kronecker(siginv_cholesky, row_cholesky);
    }

    int ydim = included.ncol();
    int xdim = included.nrow();
    Matrix qr_workspace(xdim * ydim, included.nvars());
    
    int column = -1;
    for (int j = 0; j < ydim; ++j) {
      for (int i = 0; i < xdim; ++i) {
        if (included(i, j)) {
          ConstVectorView ominv_row(row_cholesky.row(i));
          ConstVectorView siginv_row(siginv_cholesky.row(j));

          Vector row(xdim * ydim, 0.0);
          for (int block = 0; block <= j; ++block) {
            int start = block * xdim;
            int block_size = xdim;
            VectorView(row, start, block_size) = ominv_row * siginv_row[block];
          }
          qr_workspace.col(++column) = row;
        }
      }
    }
    QR qr(qr_workspace);
    chol_ = qr.getR().transpose();
    for (int i = 0; i < chol_.ncol(); ++i) {
      if (chol_(i, i) < 0) {
        chol_.col(i) *= -1;
      }
    }
    
  }

  //---------------------------------------------------------------------------
  
  Vector CompositeCholesky::solve(const ConstVectorView &x) const {
    if (chol_.nrow() != x.size()) {
      report_error("Argument 'x' is the wrong size.");
    }
    Vector y = x;
    Lsolve_inplace(chol_, y);
    LTsolve_inplace(chol_, y);
    return y;
  }

  double CompositeCholesky::Mdist(const Vector &x) const {
    return LTmult(chol_, x).normsq();
  }
  
  /////////////////////////////////////////////////////////////////////////////
  // TODO
  MRSSS::MultivariateRegressionSpikeSlabSampler(
        MultivariateRegressionModel *model,
        const Ptr<MatrixVariableSelectionPrior> &spike,
        const Ptr<MatrixNormalModel> &slab,
        const Ptr<WishartModel> &residual_precision_prior,
        RNG &seeding_rng) 
      : PosteriorSampler(seeding_rng),
        model_(model),
        spike_(spike),
        slab_(slab),
        residual_precision_prior_(residual_precision_prior),
        total_row_precision_cholesky_(0, 0)
  {}

  double MRSSS::logpri() const {
    double ans = spike_->logp(model_->included_coefficients());
    if (!std::isfinite(ans)) {
      return ans;
    }
    ans += residual_precision_prior_->logp(model_->Siginv());
    if (model_->included_coefficients().all_in()) {
      ans += slab_->logp(model_->Beta());
    } else if (model_->included_coefficients().all_out()) {
      // If no coefficients are included then the slab contributes nothing to
      // the answer.
    } else {
      Selector included = model_->included_coefficients().vectorize();
      Vector beta = included.select(vec(model_->Beta()));
      Vector mean = included.select(slab_->mvn_mean());
      SpdMatrix precision = included.select(slab_->mvn_precision());
      ans += dmvn(beta, mean, precision, true);
    }
    return ans;
  }
  
  void MRSSS::draw() {
    set_total_row_precision_cholesky();
    draw_inclusion_indicators();
    draw_residual_variance();
    draw_coefficients();
  }

  void MRSSS::draw_inclusion_indicators() {
    SelectorMatrix included = model_->included_coefficients();
    double current_logprob = log_model_probability(included);
    for (int i = 0; i < included.nrow(); ++i) {
      for (int j = 0; j < included.ncol(); ++j) {
        attempt_flip(rng(), included, i, j, current_logprob);
      }
    }
    model_->Beta_prm()->set_inclusion_pattern(included);
  }

  void MRSSS::attempt_flip(RNG &rng, SelectorMatrix &included, int i, int j,
                           double &current_logprob) const {
    included.flip(i, j);
    double logp = log_model_probability(included);
    double log_acceptance_prob = logp - lse2(logp, current_logprob);
    double logu = log(runif_mt(rng));
    if (logu < log_acceptance_prob) {
      // The proposal is accepted.  Update current_logprob with the current
      // value.
      current_logprob = logp;
    } else {
      // The proposal is rejected.  Flip the bit back how it was.
      included.flip(i, j);
    }
  }

  double MRSSS::log_model_probability(const SelectorMatrix &included) const {
    // Computing the prior mean involves the Cholesky of the posterior
    // precision.
    const MvRegSuf &suf(*model_->suf());

    CompositeCholesky prior_precision_cholesky(
        slab_->row_precision_cholesky(),
        model_->residual_precision_cholesky(),
        included);
    CompositeCholesky posterior_precision_cholesky(
        total_row_precision_cholesky_,
        model_->residual_precision_cholesky(),
        included);

    Vector prior_mean = included.vector_select(slab_->mean());
    Vector xty_siginv =  included.vector_select(suf.xty() * model_->Siginv());
    Vector posterior_mean = posterior_precision_cholesky.solve(
        xty_siginv + prior_precision_cholesky.solve(prior_mean));

    double ans = spike_->logp(included);
    double sum_of_squares = traceAB(suf.yty(), model_->Siginv())
        + prior_precision_cholesky.Mdist(prior_mean)
        - posterior_precision_cholesky.Mdist(posterior_mean);

    ans += .5 * (prior_precision_cholesky.logdet()
                 - posterior_precision_cholesky.logdet()
                 - sum_of_squares);
    return ans;
  }
  
  void MRSSS::draw_residual_variance() {
    // Compute centered sum of squares around posterior mean of included
    // parameters.  This draw is made conditional on B, with some elements of B
    // set to zero.
    model_->set_Siginv(MvnVarSampler::draw_precision(
        rng(), model_->suf()->n(), model_->suf()->SSE(model_->Beta()),
        *residual_precision_prior_));
  }

  void MRSSS::draw_coefficients() {
    const SelectorMatrix &included(model_->included_coefficients());
    CompositeCholesky prior_precision_cholesky(
        slab_->row_precision_cholesky(),
        model_->residual_precision_cholesky(),
        included);
    CompositeCholesky posterior_precision_cholesky(
        total_row_precision_cholesky_,
        model_->residual_precision_cholesky(),
        included);

    Vector prior_mean = included.vector_select(slab_->mean());
    Vector xty_siginv =  included.vector_select(
        model_->suf()->xty() * model_->Siginv());
    Vector posterior_mean = posterior_precision_cholesky.solve(
        xty_siginv + prior_precision_cholesky.solve(prior_mean));
    
    Vector coefficients = rmvn_precision_upper_cholesky_mt(
        rng(),
        posterior_mean,
        posterior_precision_cholesky.matrix().transpose());
    model_->set_Beta(model_->included_coefficients().expand(coefficients));
  }

  void MRSSS::set_total_row_precision_cholesky() {
    if (total_row_precision_cholesky_.nrow() == 0) {
      SpdMatrix total_row_precision =
          model_->suf()->xtx() + slab_->row_precision();
      total_row_precision_cholesky_ = total_row_precision.chol();
    }
  }
  
}  // namespace BOOM
