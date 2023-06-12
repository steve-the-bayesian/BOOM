#ifndef BOOM_STATE_SPACE_MULTIVARIATE_TEST_FRAMEWORK_HPP_
#define BOOM_STATE_SPACE_MULTIVARIATE_TEST_FRAMEWORK_HPP_

#include "Models/ChisqModel.hpp"
#include "Models/MvnModel.hpp"
#include "Models/MvnGivenScalarSigma.hpp"
#include "Models/PosteriorSamplers/IndependentMvnVarSampler.hpp"
#include "Models/PosteriorSamplers/ZeroMeanGaussianConjSampler.hpp"

#include "Models/Glm/MvnGivenX.hpp"
#include "Models/Glm/RegressionModel.hpp"
#include "Models/Glm/PosteriorSamplers/RegressionSemiconjugateSampler.hpp"
#include "Models/Glm/PosteriorSamplers/IndependentRegressionModelsPosteriorSampler.hpp"

#include "Models/StateSpace/Multivariate/MultivariateStateSpaceRegressionModel.hpp"
#include "Models/StateSpace/Multivariate/StateModels/SharedLocalLevel.hpp"
#include "Models/StateSpace/Multivariate/PosteriorSamplers/SharedLocalLevelPosteriorSampler.hpp"
#include "Models/StateSpace/Multivariate/PosteriorSamplers/MvStateSpaceRegressionPosteriorSampler.hpp"
#include "Models/StateSpace/PosteriorSamplers/StateSpacePosteriorSampler.hpp"

#include "Models/StateSpace/Filters/KalmanTools.hpp"

#include "distributions.hpp"
#include "cpputil/math_utils.hpp"

namespace BoomStateSpaceTesting {
  using namespace BOOM;

  // A simulated data set, model, priors, and posterior samplers for
  // multivariate state space regression models.
  struct McmcTestFramework {
    // Args:
    //   xdim:  The number of predictor variables
    //   nseries:  The number of Y variables.
    //   nfactors:  The number of factors shared by the Y variables.
    //   sample_size: The number of time points to simulate in the training data.
    //   test_size:  The number of time points to simulate in the test data.
    //   factor_sd: The standard deviation of the innovation terms in the random
    //     walk models describing the state.
    //   residual_sd: The standard deviation of the residuals in the observation
    //     equation.  Each of the 'nseries' responses has the same SD.
    explicit McmcTestFramework(int xdim,
                               int nseries,
                               int nfactors,
                               int sample_size,
                               int test_size,
                               double residual_sd)
        : sample_size_(sample_size),
          test_size_(test_size),
          state(nfactors, sample_size + test_size),
          observation_coefficients(nseries, nfactors),
          regression_coefficients(nseries, xdim),
          predictors(sample_size + test_size, xdim),
          response(sample_size + test_size, nseries),
          model(new MultivariateStateSpaceRegressionModel(xdim, nseries))
    {
      //-----------------------------------------------------------------------
      // Simulate some fake model parameters.

      // Set up the observation coefficients, which are zero above the diagonal
      // and 1 on the diagonal.
      observation_coefficients.randomize();
      for (int i = 0; i < nfactors; ++i) {
        observation_coefficients(i, i) = 1.0;
        for (int j = i + 1; j < nfactors; ++j) {
          observation_coefficients(i, j) = 0.0;
        }
      }
      observation_coefficients *= 10;
      observation_coefficients.diag() /= 10;

      // Set up the regression coefficients and the predictors.
      regression_coefficients.randomize();
      predictors.randomize();

      build(residual_sd);
    }

    int nfactors() const {
      return observation_coefficients.ncol();
    }

    int nseries() const {
      return regression_coefficients.nrow();
    }

    int xdim() const {
      return regression_coefficients.ncol();
    }

    void build(double residual_sd) {
      simulate_state();
      simulate_response(residual_sd);
      build_model(residual_sd);
    }

    void simulate_state() {
      double factor_sd = 1.0;
      for (int factor = 0; factor < nfactors(); ++factor) {
        state(factor, 0) = rnorm() - 35;
        for (int time = 1; time < sample_size_ + test_size_; ++time) {
          state(factor, time) = state(factor, time - 1) + rnorm(0, factor_sd);
        }
      }
    }

    void simulate_response(double residual_sd) {
      for (int i = 0; i < sample_size_ + test_size_; ++i) {
        Vector yhat = observation_coefficients * state.col(i)
            + regression_coefficients * predictors.row(i);
        for (int j = 0; j < nseries(); ++j) {
          response(i, j) = yhat[j] + rnorm(0, residual_sd);
        }
      }
    }

    void build_model(double residual_sd) {
      model.reset(new MultivariateStateSpaceRegressionModel(xdim(), nseries()));
      for (int time = 0; time < sample_size_; ++time) {
        for (int series = 0; series < nseries(); ++series) {
          NEW(MultivariateTimeSeriesRegressionData, data_point)(
              response(time, series), predictors.row(time), series, time);
          model->add_data(data_point);
        }
      }
      //---------------------------------------------------------------------------
      // Define the state model.
      state_model.reset(new ConditionallyIndependentSharedLocalLevelStateModel(
          model.get(), nfactors(), nseries()));
      std::vector<Ptr<GammaModelBase>> innovation_precision_priors;
      for (int factor = 0; factor < nfactors(); ++factor) {
        innovation_precision_priors.push_back(new ChisqModel(1.0, .10));
      }
      Matrix observation_coefficient_prior_mean(nseries(), nfactors(), 0.0);

      NEW(MvnModel, slab)(Vector(nfactors(), 0.0), SpdMatrix(nfactors(), 1.0));
      NEW(VariableSelectionPrior, spike)(nfactors(), 1.0);
      std::vector<Ptr<VariableSelectionPrior>> spikes;
      std::vector<Ptr<UnivParams>> sigsq_params;
      for (int i = 0; i < nseries(); ++i) {
        spikes.push_back(spike->clone());
        sigsq_params.push_back(model->observation_model()->model(i)->Sigsq_prm());
      }

      NEW(ConditionallyIndependentSharedLocalLevelPosteriorSampler,
          state_model_sampler)(
              state_model.get(),
              std::vector<Ptr<MvnBase>>(nseries(), slab),
              spikes,
              sigsq_params);
      state_model->set_method(state_model_sampler);
      state_model->set_initial_state_mean(state.col(0));
      state_model->set_initial_state_variance(SpdMatrix(nfactors(), 1.0));
      model->add_state(state_model);

      //---------------------------------------------------------------------------
      // Set the prior and sampler for the regression model.
      for (int i = 0; i < nseries(); ++i) {
        Vector beta_prior_mean(xdim(), 0.0);
        SpdMatrix beta_precision(xdim(), 1.0);
        NEW(MvnModel, beta_prior)(beta_prior_mean, beta_precision, true);
        NEW(ChisqModel, residual_precision_prior)(1.0, square(residual_sd));
        NEW(RegressionSemiconjugateSampler, regression_sampler)(
            model->observation_model()->model(i),
            beta_prior, residual_precision_prior);
        regression_sampler->set_sigma_max(1.0);
        model->observation_model()->model(i)->set_method(regression_sampler);
      }
      NEW(IndependentRegressionModelsPosteriorSampler, observation_model_sampler)(
          model->observation_model());
      model->observation_model()->set_method(observation_model_sampler);

      NEW(MultivariateStateSpaceRegressionPosteriorSampler, sampler)(
          model.get());
      model->set_method(sampler);
    }

    // Data section
    int sample_size_;
    int test_size_;
    Matrix state;
    Matrix observation_coefficients;
    Matrix regression_coefficients;
    Matrix predictors;
    Matrix response;

    Ptr<MultivariateStateSpaceRegressionModel> model;
    Ptr<ConditionallyIndependentSharedLocalLevelStateModel> state_model;
  };

  inline SpdMatrix toSpd(const DiagonalMatrix &diag) {
    SpdMatrix ans(diag.nrow());
    ans.set_diag(diag.diag());
    return ans;
  }

  // A marginal distribution for a Kalman filter that does not
  class DenseKalmanMarginal {
   public:
    DenseKalmanMarginal(const Matrix &T, const Matrix &Z, const SpdMatrix &H,
                        const SpdMatrix &RQR)
        : T_(T),
          Z_(Z),
          H_(H),
          RQR_(RQR),
          state_mean_(T.nrow(), 0.0),
          state_variance_(T.nrow()),
          F_(H_.nrow()),
          K_(T.nrow(), H.ncol()),
          v_(H.nrow()),
          L_(T.nrow(), T.ncol())
    {}

    DenseKalmanMarginal(const MultivariateStateSpaceRegressionModel *model,
                        int time)
        : DenseKalmanMarginal(
              model->state_transition_matrix(time)->dense(),
              model->observation_coefficients(
                  time, model->observed_status(time))->dense(),
              toSpd(model->observation_variance(
                  time, model->observed_status(time))),
              model->state_variance_matrix(time)->dense())
    {}

    void set_state_mean(const Vector &mean) {
      state_mean_ = mean;
    }
    void set_state_variance(const SpdMatrix &variance) {
      state_variance_ = variance;
    }

    double update(const Vector &y, const Selector &observed) {
      return vector_kalman_update(y, state_mean_, state_variance_, K_, F_, v_,
                                  observed, Z_, H_, T_, L_, RQR_);
    }

    const Vector &state_mean() const {return state_mean_;}
    const SpdMatrix &state_variance() const {return state_variance_;}
    const Matrix &kalman_gain() const {return K_;}
    const Matrix &kalman_gain(const Selector &observed) {return K_;}
    SpdMatrix forecast_precision() const {return F_.inv();}
    const Vector &prediction_error() const {return v_;}
    Vector scaled_prediction_error() const {return F_.solve(v_);}
    double forecast_precision_log_determinant() const {
      return -1 * F_.logdet();
    }

    const Matrix &state_transition_matrix() const {return T_;}
    const Matrix &observation_coefficients() const {return Z_;}

    void set_scaled_state_error(const Vector &r) {r_ = r;}
    const Vector &scaled_state_error() const {return r_;}

   private:
    Matrix T_;
    Matrix Z_;
    SpdMatrix H_;
    SpdMatrix RQR_;

    Vector state_mean_;
    SpdMatrix state_variance_;

    SpdMatrix F_;
    Matrix K_;
    Vector v_;
    Matrix L_;

    Vector r_;
  };

  //===========================================================================
  //
  class MockKalmanFilter {
   public:
   explicit MockKalmanFilter(MultivariateStateSpaceRegressionModel *model)
       : model_(model),
         initial_state_mean_(model_->state_dimension(), 0.0),
         initial_state_variance_(model_->state_dimension(), 1.0)
    {}

    void filter() {
      ensure_size();
      for (int t = 0; t < model_->time_dimension(); ++t) {
        if (t == 0) {
          nodes_[0].set_state_mean(initial_state_mean_);
          nodes_[0].set_state_variance(initial_state_variance_);
        } else {
          nodes_[t].set_state_mean(nodes_[t-1].state_mean());
          nodes_[t].set_state_variance(nodes_[t-1].state_variance());
        }
        Vector y = model_->adjusted_observation(t);
        const Selector &observed(model_->observed_status(t));
        nodes_[t].update(y, observed);
      }
    }

    void fast_disturbance_smooth() {
      int n = model_->time_dimension();
      Vector r(model_->state_dimension(), 0.0);
      for (int t = n-1; t >= 0; --t) {
        nodes_[t].set_scaled_state_error(r);
        const Selector &observed(model_->observed_status(t));
        Vector v = nodes_[t].prediction_error();
        SpdMatrix Finv = nodes_[t].forecast_precision();
        Matrix K = nodes_[t].kalman_gain(observed);
        Vector u = Finv * v - K.Tmult(r);
        r = nodes_[t].observation_coefficients().Tmult(u)
            + nodes_[t].state_transition_matrix().Tmult(r);
      }
    }

    SpdMatrix observation_variance(int t) {
      DiagonalMatrix v_diag = model_->observation_variance(
          t, model_->observed_status(t));
      SpdMatrix ans(v_diag.nrow());
      ans.set_diag(v_diag.diag());
      return ans;
    }

    void ensure_size() {
      for (int t = nodes_.size(); t < model_->time_dimension(); ++t) {
        const Selector &observed(model_->observed_status(t));
        nodes_.push_back(DenseKalmanMarginal(
            model_->state_transition_matrix(t)->dense(),
            model_->observation_coefficients(t, observed)->dense(),
            observation_variance(t),
            model_->state_variance_matrix(t)->dense()));
      }
    }

    void set_initial_state_mean(const Vector &mean) {
      initial_state_mean_ = mean;
    }

    void set_initial_state_variance(const SpdMatrix &variance) {
      initial_state_variance_ = variance;
    }

    DenseKalmanMarginal & node(int i) {return nodes_[i];}
    const DenseKalmanMarginal & node(int i) const {return nodes_[i];}
    DenseKalmanMarginal & operator[](int i) { return nodes_[i]; }
    const DenseKalmanMarginal & operator[](int i) const { return nodes_[i]; }

   private:
    MultivariateStateSpaceRegressionModel *model_;
    std::vector<DenseKalmanMarginal> nodes_;
    Vector initial_state_mean_;
    SpdMatrix initial_state_variance_;
  };

}  // namespace BoomStateSpaceTesting


#endif //  BOOM_STATE_SPACE_MULTIVARIATE_TEST_FRAMEWORK_HPP_
