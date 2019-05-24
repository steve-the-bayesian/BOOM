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

#ifndef BOOM_R_PRIOR_SPECIFICATION_HPP_
#define BOOM_R_PRIOR_SPECIFICATION_HPP_

#include "r_interface/boom_r_tools.hpp"
#include "Models/ChisqModel.hpp"
#include "Models/DoubleModel.hpp"
#include "Models/Glm/VariableSelectionPrior.hpp"
#include "Models/IndependentMvnModel.hpp"
#include "Models/IndependentMvnModelGivenScalarSigma.hpp"
#include "Models/MvnBase.hpp"
#include "Models/MvnGivenScalarSigma.hpp"

namespace BOOM{

  class MarkovModel;
  class RegressionModel;
  
  namespace RInterface{
    // Convenience classes for communicating commonly used R objects
    // to BOOM.  Each object has a corresponding R function that will
    // create the SEXP that the C++ object can use to build itself.

    // For encoding an inverse Gamma prior on a variance parameter.
    // See the R help file for SdPrior.
    class SdPrior {
     public:
      explicit SdPrior(SEXP sd_prior);
      double prior_guess()const {return prior_guess_;}
      double prior_df()const {return prior_df_;}
      double initial_value()const {return initial_value_;}
      bool fixed()const {return fixed_;}
      double upper_limit()const {return upper_limit_;}
      std::ostream & print(std::ostream &out)const;

     private:
      double prior_guess_;
      double prior_df_;
      double initial_value_;
      bool fixed_;
      double upper_limit_;
    };
    //----------------------------------------------------------------------

    // For encoding a Gaussian prior on a scalar.  See the R help file
    // for NormalPrior.
    class NormalPrior {
     public:
      explicit NormalPrior(SEXP prior);
      virtual ~NormalPrior() {}
      virtual std::ostream & print(std::ostream &out) const;
      double mu() const {return mu_;}
      double sigma() const {return sigma_;}
      double sigsq() const {return sigma_ * sigma_;}
      double initial_value() const {return initial_value_;}
      bool fixed() const {return fixed_;}

     private:
      double mu_;
      double sigma_;
      double initial_value_;
      bool fixed_;
    };

    //----------------------------------------------------------------------
    // For encoding a prior on an AR1 coefficient.  This is a Gaussian
    // prior, but users have the option of truncating the support to
    // [-1, 1] to enforce stationarity of the AR1 process.
    class Ar1CoefficientPrior : public NormalPrior {
     public:
      explicit Ar1CoefficientPrior(SEXP prior);
      bool force_stationary()const {return force_stationary_;}
      bool force_positive()const {return force_positive_;}
      std::ostream & print(std::ostream &out)const override;

     private:
      bool force_stationary_;
      bool force_positive_;
    };

    //----------------------------------------------------------------------
    // For encoding the parameters in a conditionally normal model.
    // Tyically this is the prior on mu in an normal(mu, sigsq), where
    // mu | sigsq ~ N(mu0, sigsq / sample_size).
    class ConditionalNormalPrior {
     public:
      explicit ConditionalNormalPrior(SEXP prior);
      double prior_mean()const{return mu_;}
      double sample_size()const{return sample_size_;}
      std::ostream & print(std::ostream &out)const;

     private:
      double mu_;
      double sample_size_;
    };

    //----------------------------------------------------------------------
    // A NormalInverseGammaPrior is the conjugate prior for the mean
    // and variance in a normal distribution.
    class NormalInverseGammaPrior {
     public:
      explicit NormalInverseGammaPrior(SEXP prior);
      double prior_mean_guess()const{return prior_mean_guess_;}
      double prior_mean_sample_size()const{return prior_mean_sample_size_;}
      const SdPrior &sd_prior()const{return sd_prior_;}
      std::ostream & print(std::ostream &out)const;

     private:
      double prior_mean_guess_;
      double prior_mean_sample_size_;
      SdPrior sd_prior_;
    };

    //----------------------------------------------------------------------
    // For encoding the parameters of a Dirichlet distribution.  The R
    // constructor that builds 'prior' ensures that prior_counts_ is a
    // positive length vector of positive reals.
    class DirichletPrior {
     public:
      explicit DirichletPrior(SEXP prior);
      const Vector & prior_counts()const;
      int dim()const;

     private:
      Vector prior_counts_;
    };

    //----------------------------------------------------------------------
    // For encoding a prior on the parameters of a Markov chain.  This
    // is product Dirichlet prior for the rows of the transition
    // probabilities, and an independent Dirichlet on the initial
    // state distribution.
    // TODO(stevescott): add support for fixing the initial
    //   distribution in various ways.
    class MarkovPrior {
     public:
      explicit MarkovPrior(SEXP prior);
      const Matrix & transition_counts()const {return transition_counts_;}
      const Vector & initial_state_counts()const {return initial_state_counts_;}
      int dim()const {return transition_counts_.nrow();}
      std::ostream & print(std::ostream &out)const;
      // Creates a Markov model with this as a prior.
      BOOM::MarkovModel * create_markov_model()const;

     private:
      Matrix transition_counts_;
      Vector initial_state_counts_;
    };

    //----------------------------------------------------------------------
    class BetaPrior {
     public:
      explicit BetaPrior(SEXP prior);
      double a()const{return a_;}
      double b()const{return b_;}
      double initial_value() const {return initial_value_;}
      std::ostream & print(std::ostream &out)const;

     private:
      double a_, b_;
      double initial_value_;
    };

    //----------------------------------------------------------------------
    class GammaPrior {
     public:
      explicit GammaPrior(SEXP prior);
      virtual ~GammaPrior(){}
      double a()const{return a_;}
      double b()const{return b_;}
      double initial_value()const{return initial_value_;}
      virtual std::ostream & print(std::ostream &out)const;

     private:
      double a_, b_;
      double initial_value_;
    };

    class TruncatedGammaPrior : public GammaPrior {
     public:
      explicit TruncatedGammaPrior(SEXP prior);
      double lower_truncation_point() const {return lower_truncation_point_;}
      double upper_truncation_point() const {return upper_truncation_point_;}
      std::ostream &print(std::ostream &out) const override;

     private:
      double lower_truncation_point_;
      double upper_truncation_point_;
    };

    class MvnPrior {
     public:
      explicit MvnPrior(SEXP prior);
      const Vector & mu()const{return mu_;}
      const SpdMatrix & Sigma()const{return Sigma_;}
      std::ostream & print(std::ostream &out)const;

     private:
      Vector mu_;
      SpdMatrix Sigma_;
    };

    //----------------------------------------------------------------------
    class InverseWishartPrior {
     public:
      explicit InverseWishartPrior(SEXP r_prior);
      double variance_guess_weight() const {return variance_guess_weight_;}
      const SpdMatrix & variance_guess() const {return variance_guess_;}
     private:
      double variance_guess_weight_;
      SpdMatrix variance_guess_;
    };
    //----------------------------------------------------------------------
    class NormalInverseWishartPrior {
     public:
      explicit NormalInverseWishartPrior(SEXP prior);
      const Vector & mu_guess()const{return mu_guess_;}
      double mu_guess_weight()const{return mu_guess_weight_;}
      const SpdMatrix & Sigma_guess()const{return sigma_guess_;}
      double Sigma_guess_weight()const{return sigma_guess_weight_;}
      std::ostream & print(std::ostream &out)const;

     private:
      Vector mu_guess_;
      double mu_guess_weight_;
      SpdMatrix sigma_guess_;
      double sigma_guess_weight_;
    };

    //----------------------------------------------------------------------
    class MvnIndependentSigmaPrior {
     public:
      explicit MvnIndependentSigmaPrior(SEXP prior);
      const MvnPrior & mu_prior()const{return mu_prior_;}
      const SdPrior & sigma_prior(int i)const{return sigma_priors_[i];}

     private:
      MvnPrior mu_prior_;
      std::vector<SdPrior> sigma_priors_;
    };

    //----------------------------------------------------------------------
    class MvnDiagonalPrior {
     public:
      explicit MvnDiagonalPrior(SEXP prior);
      const Vector & mean()const{return mean_;}
      const Vector & sd()const{return sd_;}

     private:
      Vector mean_;
      Vector sd_;
    };

    //----------------------------------------------------------------------
    class ScaledMatrixNormalPrior {
     public:
      explicit ScaledMatrixNormalPrior(SEXP r_prior);
      const Matrix &mean() const {return mean_;}
      double sample_size() const {return sample_size_;}
     private:
      Matrix mean_;
      double sample_size_;
    };

    //----------------------------------------------------------------------
    // A discrete prior over the integers {lo, ..., hi}.
    class DiscreteUniformPrior {
     public:
      explicit DiscreteUniformPrior(SEXP prior);
      double logp(int value) const;
      int lo() const {return lo_;}
      int hi() const {return hi_;}

     private:
      int lo_, hi_;
      double log_normalizing_constant_;
    };


    // A poisson prior, potentially truncated to the set {lo, ..., hi}.
    class PoissonPrior {
     public:
      explicit PoissonPrior(SEXP prior);
      double logp(int value) const;
      double lambda() const {return lambda_;}

     private:
      double lambda_;
      double lo_, hi_;
      double log_normalizing_constant_;
    };

    class PointMassPrior {
     public:
      explicit PointMassPrior(SEXP prior);
      double logp(int value) const;
      int location() const {return location_;}

     private:
      int location_;
    };

    //----------------------------------------------------------------------
    // This class is for handling spike and slab priors where there is no
    // residual variance parameter.  See the R help files for SpikeSlabPrior or
    // IndependentSpikeSlabPrior.
    class SpikeSlabGlmPrior {
     public:
      // Args:
      //   r_prior: An R object inheriting from SpikeSlabPriorBase.  Elements of
      //     'prior' relating to the residual variance are ignored.  If 'prior'
      //     inherits from IndependentSpikeSlabPrior then the slab will be an
      //     IndependentMvnModel.  Otherwise it will be an MvnModel.
      explicit SpikeSlabGlmPrior(SEXP r_prior);
      virtual ~SpikeSlabGlmPrior() {}
      const Vector &prior_inclusion_probabilities() {
        return spike_->prior_inclusion_probabilities();
      }
      Ptr<VariableSelectionPrior> spike() {return spike_;}
      Ptr<MvnBase> slab() {return slab_;}
      int max_flips() const {return max_flips_;}

     private:
      Ptr<VariableSelectionPrior> spike_;
      Ptr<MvnBase> slab_;
      int max_flips_;
    };
    
    //----------------------------------------------------------------------
    // beta | X, sigsq ~ N(b, sigsq * V), where
    //   V^{-1} = kappa * (a * Diag(X'X/n) + (1 - a) * X'X/n)
    //
    // The "X" must come from somewhere else, as must the 'sigsq'.
    //
    // Notation:
    //   a: diagonal_shrinkage
    //   kappa: prior_information_weight
    //   mu: prior_mean
    //   max_flips: The maximum number of in/out flips that the MCMC algorithm
    //     will try.  max_flips < 0 is a signal that the number of proposed
    //     flips is unconstrained.
    class ConditionalZellnerPrior {
     public:
      explicit ConditionalZellnerPrior(SEXP r_prior);

      const Vector &prior_inclusion_probabilities() {
        return spike_->prior_inclusion_probabilities();
      }

      Ptr<VariableSelectionPrior> spike() const {return spike_;}
      const Vector &mean() const {return prior_mean_;}
      double diagonal_shrinkage() const {return diagonal_shrinkage_;}
      double prior_information_weight() const {
        return prior_information_weight_;
      }
      int max_flips() const {return max_flips_;}
      
     private:
      Ptr<VariableSelectionPrior> spike_;
      
      Vector prior_mean_;
      double diagonal_shrinkage_;
      double prior_information_weight_;
      int max_flips_;
    };
    
    //----------------------------------------------------------------------
    // This is for the standard Zellner G prior in the regression
    // setting.  See the R help files for SpikeSlabPrior.
    class RegressionConjugateSpikeSlabPrior {
     public:
      // Args:
      //   r_prior: The R object containing the information needed to
      //     construct the prior.
      //   residual_variance: The residual variance parameter from the
      //     regression model described by the prior.
      RegressionConjugateSpikeSlabPrior(
          SEXP r_prior,
          const Ptr<UnivParams> &residual_variance);
      const Vector &prior_inclusion_probabilities() {
        return spike_->prior_inclusion_probabilities();
      }
      Ptr<VariableSelectionPrior> spike() {return spike_;}
      Ptr<MvnGivenScalarSigmaBase> slab() {return slab_;}
      Ptr<ChisqModel> siginv_prior() {return siginv_prior_;}
      int max_flips() const {return max_flips_;}
      double sigma_upper_limit() const {return sigma_upper_limit_;}

     private:
      Ptr<VariableSelectionPrior> spike_;
      Ptr<MvnGivenScalarSigmaBase> slab_;
      Ptr<ChisqModel> siginv_prior_;
      int max_flips_;
      double sigma_upper_limit_;
    };

    //----------------------------------------------------------------------
    // A version of the RegressionConjugateSpikeSlabPrior for
    // regression models with Student T errors.
     class StudentRegressionConjugateSpikeSlabPrior
         : public RegressionConjugateSpikeSlabPrior {
      public:
       StudentRegressionConjugateSpikeSlabPrior(
           SEXP r_prior, const Ptr<UnivParams> &residual_variance);
       Ptr<DoubleModel> degrees_of_freedom_prior() {return df_prior_;}

      private:
       Ptr<DoubleModel> df_prior_;
     };


    //----------------------------------------------------------------------
    // This is for the standard Zellner G prior in the regression
    // setting.  See the R help files for SpikeSlabPrior or
    // IndependentSpikeSlabPrior.
    class RegressionNonconjugateSpikeSlabPrior
        : public SpikeSlabGlmPrior {
     public:
      // Use this constructor if the prior variance is independent of
      // the residual variance.
      // Args:
      //   prior:  An R list containing the following objects
      //   - prior.inclusion.probabilities: Vector of prior inclusion
      //       probabilities.
      //   - mu:  Prior mean given inclusion.
      //   - siginv: Either a vector of prior precisions (for the
      //       independent case) or a positive definite matrix giving
      //       the posterior precision of the regression coefficients
      //       given inclusion.
      //   - prior.df: The number of observations worth of weight to
      //       be given to sigma.guess.
      //   - sigma.guess:  A guess at the residual variance
      explicit RegressionNonconjugateSpikeSlabPrior(SEXP prior);

      Ptr<ChisqModel> siginv_prior() {return siginv_prior_;}
      double sigma_upper_limit() const {return sigma_upper_limit_;}

     private:
      Ptr<ChisqModel> siginv_prior_;
      double sigma_upper_limit_;
    };

    // A unified interface for setting the prior distribution of a regression
    // model.
    // Args:
    //   model:  The model for which a posterior sampler is to be set.
    //   prior:  An R object specifying one of the following:
    //     - RegressionNonconjugateSpikeSlabPrior
    //     - RegressionConjugateSpikeSlabPrior
    //     - IndependentRegressionSpikeSlabPrior
    //     - TODO(steve): Add shrinkage regression
    // Effects:
    //   A posterior sampler is extracted from r_prior and assigned to model.
    void SetRegressionSampler(RegressionModel *model, SEXP r_prior);
    
    //----------------------------------------------------------------------
    class ArSpikeSlabPrior
        : public RegressionNonconjugateSpikeSlabPrior {
     public:
      explicit ArSpikeSlabPrior(SEXP r_prior);
      bool truncate() const {return truncate_;}

     private:
      bool truncate_;
    };

    //----------------------------------------------------------------------
    // A version of RegressionNonconjugateSpikeSlabPrior for
    // regression models with Student T errors.
    class StudentRegressionNonconjugateSpikeSlabPrior
        : public RegressionNonconjugateSpikeSlabPrior {
     public:
      explicit StudentRegressionNonconjugateSpikeSlabPrior(SEXP r_prior);
      Ptr<DoubleModel> degrees_of_freedom_prior() {return df_prior_;}

     private:
      Ptr<DoubleModel> df_prior_;
    };

    //----------------------------------------------------------------------
    // Use this class for the Clyde and Ghosh data augmentation scheme
    // for regression models.  See the R help files for
    // IndependentSpikeSlabPrior.
    class IndependentRegressionSpikeSlabPrior {
     public:
      IndependentRegressionSpikeSlabPrior(
          SEXP prior, const Ptr<UnivParams> &sigsq);
      const Vector &prior_inclusion_probabilities() {
        return spike_->prior_inclusion_probabilities();
      }
      Ptr<VariableSelectionPrior> spike() {return spike_;}
      Ptr<IndependentMvnModelGivenScalarSigma> slab() {return slab_;}
      Ptr<ChisqModel> siginv_prior() {return siginv_prior_;}
      int max_flips() const {return max_flips_;}
      double sigma_upper_limit() const {return sigma_upper_limit_;}

     private:
      Ptr<VariableSelectionPrior> spike_;
      Ptr<IndependentMvnModelGivenScalarSigma> slab_;
      Ptr<ChisqModel> siginv_prior_;
      int max_flips_;
      double sigma_upper_limit_;
    };

    //----------------------------------------------------------------------
    // A version of IndependentRegressionSpikeSlabPrior for regression
    // models with Student T errors.
    class StudentIndependentSpikeSlabPrior
        : public IndependentRegressionSpikeSlabPrior {
     public:
      StudentIndependentSpikeSlabPrior(
          SEXP prior, const Ptr<UnivParams> &sigsq);
      Ptr<DoubleModel> degrees_of_freedom_prior() {return df_prior_;}

     private:
      Ptr<DoubleModel> df_prior_;
    };

    //----------------------------------------------------------------------
    // Conjugate prior for regression coefficients.  Multivariate
    // normal given sigma^2 and X.
    class RegressionCoefficientConjugatePrior {
     public:
      explicit RegressionCoefficientConjugatePrior(SEXP prior);
      const Vector &mean() const {return mean_;}
      double sample_size() const {return sample_size_;}
      const Vector &additional_prior_precision() const {
        return additional_prior_precision_;
      }
      double diagonal_weight() const {return diagonal_weight_;}

     private:
      Vector mean_;
      double sample_size_;
      Vector additional_prior_precision_;
      double diagonal_weight_;
    };

    class UniformPrior {
     public:
      explicit UniformPrior(SEXP prior);
      double lo() const {return lo_;}
      double hi() const {return hi_;}
      double initial_value() const {return initial_value_;}
     private:
      double lo_, hi_;
      double initial_value_;
    };

    inline std::ostream & operator<<(std::ostream &out, const NormalPrior &p) {
      return p.print(out); }
    inline std::ostream & operator<<(std::ostream &out, const SdPrior &p) {
      return p.print(out); }
    inline std::ostream & operator<<(std::ostream &out, const BetaPrior &p) {
      return p.print(out); }
    inline std::ostream & operator<<(std::ostream &out, const MarkovPrior &p) {
      return p.print(out); }
    inline std::ostream & operator<<(std::ostream &out,
                                     const ConditionalNormalPrior &p) {
      return p.print(out); }
    inline std::ostream & operator<<(std::ostream &out, const MvnPrior &p) {
      return p.print(out); }

    // Creates a pointer to a DoubleModel based on the given
    // specification.  The specification must correspond to a BOOM model
    // type inheriting from DoubleModel.  Legal values for specification
    // are objects inheriting from GammaPrior, UniformPrior, BetaPrior and
    // NormalPrior.  More may be added later.
    Ptr<DoubleModel> create_double_model(SEXP specification);

    // As in create_double_model, but the model's log density is twice
    // differentiable.
    Ptr<DiffDoubleModel> create_diff_double_model(SEXP specification);

    // As in create_double_model, but the model knows how to compute
    // its mean and variance (or equivalent).
    Ptr<LocationScaleDoubleModel> create_location_scale_double_model(
        SEXP specification, bool throw_on_error);

    // Creates a pointer to a IntModel based on the given
    // specification.  The specification must correspond to a BOOM model
    // type inheriting from IntModel.  Legal values for specification
    // are objects inheriting from DiscreteUniformPrior, PoissonPrior and
    // PointMassPrior.  More may be added later.
    Ptr<IntModel> create_int_model(SEXP specification);

  }  // namespace RInterface
}  // namespace BOOM

#endif // BOOM_R_PRIOR_SPECIFICATION_HPP_
