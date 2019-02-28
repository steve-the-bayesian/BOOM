#ifndef BOOM_MULTIVARIATE_REGRESSION_SPIKE_SLAB_SAMPLER_HPP_
#define BOOM_MULTIVARIATE_REGRESSION_SPIKE_SLAB_SAMPLER_HPP_

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

#include "Models/Glm/MultivariateRegression.hpp"
#include "Models/Glm/VariableSelectionPrior.hpp"
#include "Models/MatrixNormalModel.hpp"
#include "Models/WishartModel.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"

namespace BOOM {

  //===========================================================================
  // The variance and precision matrices for multivariate regression coefficient
  // are kronecker products, so their Cholesky decompositions are kronecker
  // products as well.
  //
  // CompositeCholesky assembles the cholesky decomposition of the matrix formed
  // by subsetting siginv_chol \otimes row_chol by the 'included' matrix.
  class CompositeCholesky {
   public:
    // Args:
    //   row_cholesky:  The cholesky decomposition of the row precision.
    //   siginv_cholesky: The cholesky decomposition of the residual precision
    //     (which could also be called the column precision).
    //   included: A matrix indicating which variables are to be included.  The
    //     row dimension of row_precision should equal
    //     included.row_or().nvars(), which is the number of variables included
    //     in at least one column of 'included.'  The row dimension of
    //     siginv_cholesky should match the number of columns in 'included.'
    CompositeCholesky(const Matrix &row_cholesky,
                      const Matrix siginv_cholesky,
                      const SelectorMatrix &included)
    {
      decompose_simple(row_cholesky, siginv_cholesky, included);
    }

    // The log determinant of the matrix represented by the cholesky
    // decomposition.  This is twice the log determinant of the cholesky matrix.
    double logdet() const { return 2 * sumlog(chol_.diag()); }

    // The lower triangular matrix.
    const Matrix &matrix() const {return chol_;}

    // If *this is the Cholesky decomposition of A, return A.inv() * x
    Vector solve(const ConstVectorView &x) const;

    // If *this is the Cholesky decomposition of A, return x' A x.
    double Mdist(const Vector &x) const;
    
    // Set chol_ from the given inputs.
    void decompose(const Matrix &row_cholesky,
                   const Matrix &siginv_cholesky,
                   const SelectorMatrix &included);

    // Set chol_ from the given inputs.
    void decompose_simple(const Matrix &row_cholesky,
                          const Matrix &siginv_cholesky,
                          const SelectorMatrix &included);

    int nrow() const {return chol_.nrow();}
    int ncol() const {return chol_.ncol();}
    
   private:
    Matrix chol_;
  };

  //===========================================================================
  // A spike and slab sampler for multivariate regression models.  This sampler
  // assumes a Wishart prior on the residual precision parameter.
  //
  // Let gamma be an xdim by ydim matrix of 0's and 1's, the same dimension as
  // the coefficient matrix beta.  Let gamma(i, j) = 1 denote that beta(i, j) is
  // nonzero and let gamma(i, j) = 0 denote that beta(i, j) = 0.
  //
  // The prior distribution factors as
  //   p(gamma) * p(Sigma | gamma) * p(beta | Sigma, gamma).
  //
  // We assume p(gamma) is independent Bernoulli, though each coefficient can
  // have a separate inclusion probability.  p(Sigma | gamma) is independent of
  // gamma, and is a standard inverse Wishart distribution.
  //
  // To comply with both the conjugate sampler and the spike and slab sampler
  // for scalar regression, p(beta | Sigma, gamma) is a subset of the matrix
  // normal model
  //
  // p(B | B0, Sigma, gamma) = MatrixNormal(B0, Omega.inv(), Sigma)
  //
  // where Omega is kappa * (alpha * X'X + (1 - alpha) * diag(X'X)) / n
  //
  // and alpha is a scalar mixing weight in [0, 1], and kappa is a positive
  // number that can be interpreted as a prior number of observations worth of
  // weight assigned to the prior mean B0.
  //
  // To incorporate gamma, vectorize B by stacking its columns, so Vec(B) ~
  // Mvn(B0_gamma, precision = (Sigma.inv \otimes Omega.inv)_gamma).
  class MultivariateRegressionSpikeSlabSampler
      : public PosteriorSampler {
   public:
    MultivariateRegressionSpikeSlabSampler(
        MultivariateRegressionModel *model,
        const Ptr<MatrixVariableSelectionPrior> &spike,
        const Ptr<MatrixNormalModel> &slab,
        const Ptr<WishartModel> &residual_precision_prior,
        RNG &seeding_rng = GlobalRng::rng);

    double logpri() const override;
    void draw() override;
    
    void draw_inclusion_indicators();
    void draw_residual_variance();
    void draw_coefficients();

    // Returns the log of the un-normalized marginal likelihood p(gamma | Y),
    // integrating out the model parameters.
    double log_model_probability(const SelectorMatrix &included) const;

    // This is normally done automatically by the draw() method.  This function
    // should be called before any of the draw_X functions are called.  It only
    // needs to be called once, and is a no-op if called again.
    void set_total_row_precision_cholesky();
    
   private:
    // Args:
    //   rng:  The random number generator.
    //   included:  The matrix indicating which variables are included in the model.
    //   i, j: The indices of the row (i) and column (j) in which to attempt a flip.
    //   current_logprob:  The value of log_model_probability evaluated at 'included.'
    //
    // Effects:
    //   included(i, j) is flipped and log_model_probability is evaluated at the
    //   new configuration.  If the flip is accepted then 'included' is changed
    //   to reflect the fip and 'current_logprob' is updated to reflect the new
    //   configuration.  If the flip is not accepted then nothing is changed.
    void attempt_flip(RNG &rng, SelectorMatrix &included, int i, int j,
                      double &current_logprob) const;

    
    MultivariateRegressionModel *model_;

    // Prior distribution over which coefficients are included or excluded.
    Ptr<MatrixVariableSelectionPrior> spike_;

    // A good prior for beta is MN(B, kappa * xtx.inverse / n, Sigma).  Scalar
    // constant multiples of the row_variance and column_variance are not
    // identified in this model, so there is no need to scale the row variance.
    Ptr<MatrixNormalModel> slab_;

    // This bit is standard.  It might be nice to refactor this class to work
    // with a more structured variance.
    Ptr<WishartModel> residual_precision_prior_;

    // total_row_precision_cholesky_ is the cholesky decomposition of X'X +
    // slab->row_precision().  It is set the first time through the sampler, and
    // then not updated later.
    //
    // TODO(stevescott): Set an observer on the prior precision if it is
    // expected to change, then update as needed.
    Matrix total_row_precision_cholesky_;
  };
  
}  // namespace BOOM

#endif  // BOOM_MULTIVARIATE_REGRESSION_SPIKE_SLAB_SAMPLER_HPP_



