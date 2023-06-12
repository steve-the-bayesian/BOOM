// Copyright 2018 Steven L. Scott. All Rights Reserved.
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

#include "dynamic_intercept_model_manager.h"
#include "create_dynamic_intercept_state_model.h"

#include "r_interface/prior_specification.hpp"
#include "Models/StateSpace/PosteriorSamplers/DynamicInterceptRegressionPosteriorSampler.hpp"
#include "Models/Glm/PosteriorSamplers/BregVsSampler.hpp"
#include "distributions.hpp"

#include "utils.h"

namespace BOOM {
  namespace bsts {
    namespace {
      using Manager = DynamicInterceptModelManager;
    }

    Manager::DynamicInterceptModelManager(int xdim)
        : model_(new DynamicInterceptRegressionModel(xdim)) {}

    DynamicInterceptModelManager *Manager::Create(SEXP r_data_list) {
      SEXP r_predictors = getListElement(r_data_list, "predictors");
      int xdim = Rf_ncols(r_predictors);
      return new DynamicInterceptModelManager(xdim);
    }

    DynamicInterceptRegressionModel *Manager::CreateModel(
        SEXP r_data_list,
        SEXP r_state_specification,
        SEXP r_regression_prior,
        SEXP r_options,
        RListIoManager *io_manager) {
      UnpackTimestampInfo(r_data_list);
      AddDataFromList(r_data_list);
      DynamicInterceptStateModelFactory state_model_factory(io_manager);
      state_model_factory.AddState(model_.get(), r_state_specification);
      SetDynamicRegressionStateComponentPositions(
          state_model_factory.DynamicRegressionStateModelPositions());
      // using Marginal = Kalman::ConditionalIidMarginalDistribution;
      // Marginal::set_high_dimensional_threshold_factor(
      //     Rf_asReal(getListElement(
      //         r_options, "high.dimensional.threshold.factor", true)));

      //---------------------------------------------------------------------------
      // Set the posterior samplers.
      RegressionModel *regression(model_->observation_model());
      BOOM::RInterface::RegressionConjugateSpikeSlabPrior prior(
          r_regression_prior, regression->Sigsq_prm());
      DropUnforcedCoefficients(regression,
                               prior.prior_inclusion_probabilities());
      Ptr<BregVsSampler> regression_sampler(new BregVsSampler(
          regression, prior.slab(), prior.siginv_prior(), prior.spike()));
      regression_sampler->set_sigma_upper_limit(prior.sigma_upper_limit());
      int max_flips = prior.max_flips();
      if (max_flips > 0) {
        regression_sampler->limit_model_selection(max_flips);
      }
      regression->set_method(regression_sampler);

      NEW(DynamicInterceptRegressionPosteriorSampler, sampler)(model_.get());
      model_->set_method(sampler);

      //---------------------------------------------------------------------------
      // Set the io_manager.
      io_manager->add_list_element(
          new GlmCoefsListElement(regression->coef_prm(), "coefficients"));
      io_manager->add_list_element(
          new StandardDeviationListElement(regression->Sigsq_prm(),
                                           "sigma.obs"));

      // TODO(steve):  does final state need to be sized first?
      state_model_factory.SaveFinalState(model_.get(), &final_state());

      io_manager->add_list_element(
          new NativeMatrixListElement(
              new DynamicInterceptStateContributionCallback(model_.get()),
              "state.contributions",
              nullptr));

      return model_.get();
    }

    Matrix Manager::Forecast(SEXP r_bsts_object,
                             SEXP r_prediction_data,
                             SEXP r_burn,
                             SEXP r_observed_data) {
      RListIoManager io_manager;
      SEXP r_state_specfication = getListElement(
          r_bsts_object, "state.specification");
      model_.reset(CreateModel(
          R_NilValue,
          r_state_specfication,
          R_NilValue,
          R_NilValue,
          &io_manager));
      bool refilter;
      if (Rf_isNull(r_observed_data)) {
        AddDataFromBstsObject(r_bsts_object);
        refilter = false;
      } else {
        AddDataFromList(r_observed_data);
        refilter = true;
      }
      int niter = Rf_asInteger(getListElement(r_bsts_object, "niter"));
      int burn = std::max<int>(0, Rf_asInteger(r_burn));
      io_manager.prepare_to_stream(r_bsts_object);
      io_manager.advance(burn);
      int iterations_after_burnin = niter - burn;

      if (Rf_isNull(r_prediction_data)) {
        report_error("Forecast called with NULL prediction data.");
      }
      int forecast_horizon = UnpackForecastData(r_prediction_data);
      UnpackDynamicRegressionForecastData(
          model_.get(), r_state_specfication, r_prediction_data);

      Matrix ans(iterations_after_burnin, forecast_horizon);
      for (int i = 0; i < iterations_after_burnin; ++i) {
        io_manager.stream();
        if (refilter) {
          model_->kalman_filter();
          const Kalman::MultivariateMarginalDistributionBase &marg(
              model_->get_filter().back());
          Ptr<SparseKalmanMatrix> forecast_precision = marg.sparse_forecast_precision();
          final_state() = rmvn(marg.contemporaneous_state_mean(),
                               marg.contemporaneous_state_variance(forecast_precision));
        }
        ans.row(i) = model_->simulate_forecast(
            rng(),
            forecast_predictors_,
            final_state(),
            ForecastTimestamps());
      }
      return ans;
    }

    int Manager::UnpackForecastData(SEXP r_prediction_data) {
      forecast_predictors_ = ToBoomMatrix(getListElement(
          r_prediction_data, "predictors"));
      UnpackForecastTimestamps(r_prediction_data);
      return forecast_predictors_.nrow();
    }

    void Manager::UnpackDynamicRegressionForecastData(
        DynamicInterceptRegressionModel *model,
        SEXP r_state_specification,
        SEXP r_prediction_data) {
      if (Rf_length(r_state_specification) < model->number_of_state_models()) {
        std::ostringstream err;
        err << "The number of state components in the model: ("
            << model->number_of_state_models() << ") does not match the size of "
            << "the state specification: ("
            << Rf_length(r_state_specification)
            << ") in UnpackDynamicRegressionForecastData.";
        report_error(err.str());
      }
      std::deque<int> positions(dynamic_regression_state_positions().begin(),
                                dynamic_regression_state_positions().end());
      for (int i = 0; i < model->number_of_state_models(); ++i) {
        SEXP spec = VECTOR_ELT(r_state_specification, i);
        if (Rf_inherits(spec, "DynamicRegression")) {
          Matrix predictors = ToBoomMatrix(getListElement(
              r_prediction_data, "dynamic.regression.predictors"));
          if (positions.empty()) {
            report_error("Found a previously unseen dynamic regression state "
                         "component.");
          }
          int pos = positions[0];
          positions.pop_front();
          Ptr<DynamicInterceptStateModel> state_model = model->state_model(pos);
          state_model.dcast<DynamicRegressionStateModel>()->add_forecast_data(
              predictors);
        }
      }
    }

    void Manager::AddData(const Vector &response,
                          const Matrix &predictors,
                          const Selector &response_is_observed) {
      NEW(StateSpace::TimeSeriesRegressionData, data_point)(
          response, predictors, response_is_observed);
      if (response_is_observed.nvars() == 0) {
        data_point->set_missing_status(Data::completely_missing);
      } else if (response_is_observed.nvars_excluded() > 0) {
        data_point->set_missing_status(Data::partly_missing);
      }
      model_->add_data(data_point);
    }

    void Manager::AddDataFromList(SEXP r_data_list) {
      Matrix predictors = ToBoomMatrix(getListElement(r_data_list, "predictors"));
      Vector response = ToBoomVector(getListElement(r_data_list, "response"));
      Selector observed = FindNonNA(response);

      int observation_number = 0;
      for (int t = 0; t < NumberOfTimePoints(); ++t) {
        Selector current(response.size());
        while(observation_number < response.size() &&
              TimestampMapping(observation_number) == t) {
          current.add(observation_number++);
        }
        AddData(current.select(response),
                current.select_rows(predictors),
                Selector(current.select(observed)));
      }
    }

    void Manager::AddDataFromBstsObject(SEXP r_bsts_object) {
      AddDataFromList(r_bsts_object);
    }

  }  // namespace bsts
}  // namespace BOOM
