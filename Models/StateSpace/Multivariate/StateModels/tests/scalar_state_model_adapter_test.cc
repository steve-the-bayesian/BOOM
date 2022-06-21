#include "gtest/gtest.h"

#include "Models/StateSpace/tests/StateSpaceTestFramework.hpp"
#include "Models/StateSpace/Multivariate/MultivariateStateSpaceRegressionModel.hpp"
#include "Models/StateSpace/Multivariate/StateModels/ScalarStateModelAdapter.hpp"
#include "Models/StateSpace/Multivariate/PosteriorSamplers/ScalarStateModelAdapterPosteriorSampler.hpp"
#include "Models/StateSpace/Multivariate/PosteriorSamplers/MultivariateStateSpaceModelSampler.hpp"


#include "Models/StateSpace/StateModels/LocalLevelStateModel.hpp"
#include "Models/StateSpace/StateModels/LocalLinearTrend.hpp"
#include "Models/StateSpace/StateModels/SeasonalStateModel.hpp"

#include "Models/PosteriorSamplers/ZeroMeanMvnIndependenceSampler.hpp"

#include "cpputil/math_utils.hpp"
#include "distributions.hpp"

#include "test_utils/test_utils.hpp"


namespace {

  using namespace BOOM;
  using namespace BOOM::StateSpaceTesting;
  using std::endl;
  using std::cout;

  class ScalarStateModelAdapterTest : public ::testing::Test {
   protected:
    ScalarStateModelAdapterTest() {
      GlobalRng::rng.seed(8675310);
    }
  };

  //======================================================================
  TEST_F(ScalarStateModelAdapterTest, SmokeTest) {
    NEW(LocalLevelStateModel, trend)();
    NEW(SeasonalStateModel, seasonal)(4);

    int xdim = 1;
    int nseries = 12;
    MultivariateStateSpaceRegressionModel host(xdim, nseries);
    ConditionallyIndependentScalarStateModelMultivariateAdapter state(
        &host, nseries);
    state.add_state(trend);
    state.add_state(seasonal);

    EXPECT_EQ(state.state_dimension(), 1 + 3);
    EXPECT_EQ(state.state_error_dimension(), 1 + 1);

    EXPECT_EQ(state.nseries(), 12);

    Selector all(nseries, true);
    Selector none(nseries, false);
    Selector some(nseries, false);
    some.add(3);
    some.add(7);
    auto Z = state.observation_coefficients(0, all);
    Z = state.observation_coefficients(0, some);
    Z = state.observation_coefficients(0, none);

    Matrix transition{{1, 0, 0, 0}, {0, -1, -1, -1}, {0, 1, 0, 0}, {0, 0, 1, 0}};
    EXPECT_TRUE(MatrixEquals(state.state_transition_matrix(0)->dense(),
                             transition))
        << state.state_transition_matrix(0)->dense();

    std::vector<Ptr<Params>> params = state.parameter_vector();
    EXPECT_EQ(params.size(), 3);
  }

  //======================================================================
  struct AdapterTestFramework {
    // Simulate data from a local linear trend + seasonal model.
    // Have 6 series randomly present / absent.
    //
    // Args:
    //   nseries:  The number of time series to simulate.
    //   ntimes:  The number of time points to simulate.
    //   level_sd: The standard deviation of the level component of the local
    //     linear trend.
    //   slope_sd: The standard deviation of the slope component of the local
    //     linear trend.
    //   seasonal_sd: The standard deviation of the error in the seasonal state
    //     component.
    //   residual_sd:
    AdapterTestFramework(int nseries,
                         int ntimes,
                         int xdim = 1,
                         double level_sd = .1,
                         double slope_sd = .01,
                         double seasonal_sd = .1,
                         double residual_sd = .5,
                         double observation_percent = 1.0)
        : model_(new MultivariateStateSpaceRegressionModel(xdim, nseries)),
          trend_model_(new LocalLinearTrendStateModel),
          seasonal_model_(new SeasonalStateModel(4, 1)),
          observation_coefficients_(nseries)
    {
      observation_coefficients_.randomize();
      double total = observation_coefficients_.sum();
      observation_coefficients_ = observation_coefficients_ * nseries / total;

      initialize_trend_model(level_sd, slope_sd, ntimes);
      initialize_seasonal_model(seasonal_sd, ntimes);
      initialize_adapter();
      simulate_data(nseries, ntimes);
      set_posterior_sampler();
    }

    void initialize_trend_model(double level_sd, double slope_sd, int ntimes) {
      trend_model_->set_initial_state_mean(Vector(2, 0.0));
      SpdMatrix initialize_state_variance(2);
      initialize_state_variance(0, 0) = 0.1;
      initialize_state_variance(1, 1) = 0.01;
      trend_model_->set_initial_state_variance(initialize_state_variance);
      SpdMatrix Sigma(2);
      Sigma(0, 0) = square(level_sd);
      Sigma(1, 1) = square(slope_sd);
      trend_model_->set_Sigma(Sigma);
      trend_state_ = trend_model_->simulate(ntimes);
      trend_ = trend_state_.col(0);

      double prior_df = 1.0;
      int position = 0;
      NEW(ZeroMeanMvnIndependenceSampler, level_sampler)(
          trend_model_.get(),
          prior_df,
          level_sd,
          position);
      trend_model_->set_method(level_sampler);

      NEW(ZeroMeanMvnIndependenceSampler, slope_sampler)(
          trend_model_.get(),
          prior_df,
          slope_sd,
          position);
      trend_model_->set_method(slope_sampler);
    }

    void initialize_seasonal_model(double seasonal_sd, int ntimes) {
      int nseasons = 4;
      seasonal_model_->set_initial_state_mean(Vector(nseasons - 1, 0.0));
      seasonal_model_->set_initial_state_variance(SpdMatrix(nseasons - 1, 1.0));
      seasonal_model_->set_sigsq(square(seasonal_sd));
      seasonal_state_ = seasonal_model_->simulate(ntimes);
      seasonal_ = seasonal_state_.col(0);

      double prior_df = 1.0;
      NEW(ZeroMeanGaussianConjSampler, sampler)(
          seasonal_model_.get(),
          prior_df,
          seasonal_sd);
      seasonal_model_->set_method(sampler);
    }

    void simulate_data(int nseries, int ntimes) {
      Vector total_state = trend_ + seasonal_;
      simulated_data_ = Matrix(nseries, ntimes);

      for (int series = 0; series < nseries; ++series) {
        simulated_data_.row(series) = total_state * observation_coefficients_[series];
        for (int t = 0; t < ntimes; ++t) {
          simulated_data_(series, t) +=
              sqrt(model_->single_observation_variance(t, series)) * rnorm(0, 1);
        }
      }

      // Assign the data.
      model_->clear_data();
      NEW(VectorData, intercept)(Vector(1, 1.0));
      for (int time = 0; time < ntimes; ++time) {
        for (int series = 0; series < nseries; ++series) {
          NEW(DoubleData, y)(simulated_data_(series, time));
          NEW(MultivariateTimeSeriesRegressionData, data_point)(
              y, intercept, series, time);
          model_->add_data(data_point);
        }
      }
    }

    void initialize_adapter() {
      int nseries = model_->nseries();
      scalar_adapter_.reset(new Adapter(model_.get(), nseries));
      scalar_adapter_->add_state(trend_model_);
      scalar_adapter_->add_state(seasonal_model_);

      NEW(CiScalarStateAdapterPosteriorSampler, sampler)(scalar_adapter_.get());
      scalar_adapter_->set_method(sampler);

      model_->add_state(scalar_adapter_);
    }

    void set_posterior_sampler() {
      NEW(MultivariateStateSpaceModelSampler, sampler)(model_.get());
      model_->set_method(sampler);
    }

    const Matrix state() const {
      return cbind(trend_state_, seasonal_state_);
    }

    Ptr<MultivariateStateSpaceRegressionModel> model_;
    Ptr<LocalLinearTrendStateModel> trend_model_;
    double trend_level_sd_;
    double trend_slope_sd_;
    Matrix trend_state_;
    Vector trend_;

    // seasonal model with 4 seasons.
    Ptr<SeasonalStateModel> seasonal_model_;
    double sesonal_sd_;
    Matrix seasonal_state_;
    Vector seasonal_;

    using Adapter = ConditionallyIndependentScalarStateModelMultivariateAdapter;
    Ptr<Adapter> scalar_adapter_;
    Vector observation_coefficients_;

    Matrix simulated_data_;

  };

  //===========================================================================
  TEST_F(ScalarStateModelAdapterTest, Serialization) {
    int nseries = 6;
    int ntimes = 100;

    AdapterTestFramework framework(nseries, ntimes, 1, .1, .1);

    // Set some parameter values so we can tell all the parameters apart.
    for (int i = 0; i < nseries; ++i) {
      Vector beta = framework.model_->observation_model()->model(i)->Beta();
      beta.randomize();
      framework.model_->observation_model()->model(i)->set_Beta(beta);
      framework.model_->observation_model()->model(i)->set_sigsq(rgamma(3, 7));
    }

    SpdMatrix trend_Sigma = framework.trend_model_->Sigma();
    trend_Sigma.randomize();
    framework.trend_model_->set_Sigma(trend_Sigma);


    Ptr<MultivariateStateSpaceRegressionModel> model = framework.model_;

    Vector parameters = model->vectorize_params(true);
    Vector random_parameters = parameters;
    random_parameters.randomize();

    model->unvectorize_params(random_parameters);
    Vector p2 = model->vectorize_params(true);
    EXPECT_TRUE(VectorEquals(random_parameters, p2))
        << "original parameters: \n" << parameters << "\n"
        << "random parameters: \n" << random_parameters << "\n"
        << "restored parameters: \n" << p2 << "\n";
  }


  //===========================================================================
  TEST_F(ScalarStateModelAdapterTest, TestMle) {
    int nseries = 6;
    int ntimes = 100;

    AdapterTestFramework framework(nseries, ntimes, 1, .1, .1);

    framework.trend_model_->set_Sigma(SpdMatrix(2, 1.0));
    framework.seasonal_model_->set_sigsq(1.0);

    double original_loglike = framework.model_->log_likelihood();
    double loglike = framework.model_->mle();

    EXPECT_GT(loglike, original_loglike);
  }

  //======================================================================
  TEST_F(ScalarStateModelAdapterTest, TestMcmc) {
    int nseries = 6;
    int ntimes = 100;

    AdapterTestFramework framework(nseries, ntimes, 1, .1, .1);

    int burn = 50;
    int niter = 200;

    Vector log_likelihood(niter + burn);

    for (int i = 0; i < burn; ++i) {
      framework.model_->sample_posterior();
      log_likelihood[i] = framework.model_->log_likelihood();
    }

    Matrix trend_draws(niter, ntimes);
    Matrix seasonal_draws(niter, ntimes);
    Matrix observation_coefficient_draws(niter, nseries);

    for (int i = 0; i < niter; ++i) {
      framework.model_->sample_posterior();
      log_likelihood[i + burn] = framework.model_->log_likelihood();

      const Matrix &state(framework.model_->shared_state());
      EXPECT_EQ(state.nrow(), 5);
      trend_draws.row(i) = state.row(0);

      EXPECT_EQ(state.ncol(), ntimes);
      seasonal_draws.row(i) = state.row(2);

      observation_coefficient_draws.row(i) =
          framework.scalar_adapter_->observation_coefficient_slopes();
    }

    EXPECT_EQ(
        "",
        CheckStochasticProcess(trend_draws,framework.trend_, .95, .1, 0.5,
                               "trend_draws.out"));

    // TODO(steve): We don't do a great job with seasonal in this test.  Try
    // better to understand why.
    //
    // EXPECT_EQ(
    //     "",
    //     CheckStochasticProcess(seasonal_draws,framework.seasonal_, .95, .1, 0.2,
    //                            "seasonal_draws.out"));

    std::ofstream("log_likelihood.out") << log_likelihood;
    std::ofstream("trend_draws.out") << framework.trend_ << "\n" << trend_draws;
    std::ofstream("seasonal_draws.out") << framework.seasonal_ << "\n" << seasonal_draws;

    auto status = CheckMcmcMatrix(observation_coefficient_draws,
                                  framework.observation_coefficients_);
    EXPECT_TRUE(status.ok) << status;
    std::ofstream oc("observation_coefficients.out");
    oc << framework.observation_coefficients_ << "\n"
       << observation_coefficient_draws;

  }

}  // namespace
