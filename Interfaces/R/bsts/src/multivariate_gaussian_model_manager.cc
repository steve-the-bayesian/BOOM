/*
  Copyright (C) 2019 Steven L. Scott

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

#include "utils.h"
#include "multivariate_gaussian_model_manager.h"
#include "create_shared_state_model.h"

#include "LinAlg/Selector.hpp"
#include "Models/GammaModel.hpp"
#include "Models/ChisqModel.hpp"
#include "Models/Glm/IndependentRegressionModels.hpp"
#include "Models/Glm/PosteriorSamplers/IndependentRegressionModelsPosteriorSampler.hpp"

#include "Models/PosteriorSamplers/IndependentMvnVarSampler.hpp"
#include "Models/StateSpace/PosteriorSamplers/MvStateSpaceRegressionPosteriorSampler.hpp"
#include "cpputil/math_utils.hpp"

#include "r_interface/prior_specification.hpp"
#include "r_interface/boom_r_tools.hpp"
#include "r_interface/list_io.hpp"

namespace BOOM {
  namespace bsts {
    using Manager = MultivariateGaussianModelManager;
    
    Manager::MultivariateGaussianModelManager(
        int ydim, int xdim)
        : nseries_(ydim),
          predictor_dimension_(xdim)
    {}

    //---------------------------------------------------------------------------
    class MultivariateStateContributionsCallback
        : public ::BOOM::ArrayIoCallback {
     public:
      explicit MultivariateStateContributionsCallback(
          MultivariateStateSpaceRegressionModel *model)
          : model_(model) {}

      // Dimensions are state_model x series x time
      std::vector<int> dim() const override {
        return std::vector<int> {
              model_->number_of_state_models(),
              model_->nseries(),
              model_->time_dimension() };
      }

      void write_to_array(ArrayView &view) const override {
        for (int s = 0; s < model_->number_of_state_models(); ++s) {
          view.slice(s, -1, -1) = model_->state_contributions(s);
        }
      }

      // A no-op, as state contributions are a model summary, not stored by the
      // model.
      void read_from_array(const ArrayView &view) override {}
      
     private:
      MultivariateStateSpaceRegressionModel *model_;
    };
    //---------------------------------------------------------------------------
    class MultivariateSaveFullStateCallback
        : public ::BOOM::MatrixIoCallback {
     public:
      explicit MultivariateSaveFullStateCallback(
          MultivariateStateSpaceRegressionModel *model)
          : model_(model) {}

      int nrow() const override { return model_->state_dimension(); }
      int ncol() const override { return model_->time_dimension(); }
      Matrix get_matrix() const override {return model_->shared_state();}
      
     private:
      MultivariateStateSpaceRegressionModel *model_;
    };
    //---------------------------------------------------------------------------
    
    MultivariateStateSpaceRegressionModel *Manager::CreateModel(
        SEXP r_data_list_or_model_object,
        SEXP r_shared_state_specification,
        SEXP r_series_state_specification,
        SEXP r_prior,
        SEXP r_options,
        RListIoManager *io_manager) {
      CreateBareModel(r_data_list_or_model_object, r_prior, r_options, io_manager);
      SharedStateModelFactory shared_state_model_factory(nseries_, io_manager);
      shared_state_model_factory.AddState(
          model_->state_model_vector(),
          model_.get(),
          r_shared_state_specification,
          "");
      shared_state_model_factory.SaveFinalState(model_.get(), &final_state());

      if (!Rf_isNull(r_series_state_specification)) {
        // Output for the series specific state models is stored in a list named
        // "series.specific".  Its elements have names matching those of the
        // series being modeled.  series.specific[[3]] contains all the MCMC
        // draws of model parameters for series 3, as well as its series
        // specific state contributions.
        BOOM::Factor series_id(getListElement(
            r_data_list_or_model_object, "series.id", true));
        std::vector<std::string> series_names = series_id.labels();

        // Pointers to the Vector's contained in this object will be passed to
        // series specific IO managers, so they need to remain stable.  DO NOT
        // call push_back on series_specific_final_state_, or it can invalidate
        // those pointers.
        series_specific_final_state_.resize(nseries_);
        
        NEW(SubordinateModelIoElement, subordinate_model_io)("series.specific");
        io_manager->add_list_element(subordinate_model_io);
        for (int i = 0; i < nseries_; ++i) {
          SEXP r_subordinate_state_specification =
              VECTOR_ELT(r_series_state_specification, i);

          subordinate_model_io->add_subordinate_model(series_names[i]);

          if (!Rf_isNull(r_subordinate_state_specification)) {
            // Make the io list element aware that there is state to be stored
            // for series i.
            RListIoManager *subordinate_io_manager = 
                subordinate_model_io->subordinate_io_manager(i);
            StateModelFactory series_state_factory(subordinate_io_manager);
            ProxyScalarStateSpaceModel *subordinate_model =
                model_->series_specific_model(i).get();

            series_state_factory.AddState(
                subordinate_model, r_subordinate_state_specification);

            series_state_factory.SaveFinalState(
                subordinate_model,
                &series_specific_final_state_[i]);

            subordinate_io_manager->add_list_element(
                new NativeMatrixListElement(
                    new ScalarStateContributionCallback(subordinate_model),
                    "state.contributions",
                    nullptr));
          }
        }
      }

      // Save state contributions.
      io_manager->add_list_element(new NativeArrayListElement(
          new MultivariateStateContributionsCallback(model_.get()),
          "shared.state.contributions",
          false));
      
      // TODO: save prediction errors.

      // Save full state draws.
      io_manager->add_list_element(new NativeMatrixListElement(
          new MultivariateSaveFullStateCallback(model_.get()),
          "shared.state", nullptr));

      SetModelOptions(r_options);
      
      return model_.get();
    }

    // Args:
    //   r_mbsts_object: The R object returned by 'mbsts' representing the
    //     multivariate bsts model.
    //   r_prediction_data: A list containing the data needed to make the
    //     forecast.  See UnpackForecastData.  The list must contain:
    //     - A matrix named 'predictors' organized such that the first 'nseries'
    //       rows correspond to the first forecast time point, the next
    //       'nseries' to the second, and so on.
    //     - A 'timestamp info' object describing the timestamps for the
    //       forecast.  See UnpackForecastTimestamps().
    //   r_burn: An integer specifying the number of MCMC iterations in
    //     r_mbsts_object to discard as burn-in.  A negative number will be
    //     treated as zero.
    //
    // Returns:
    //   An array of dimension [niter x nseries x horizon] containing draws from
    //   the posterior predictive distribution.
    Array Manager::Forecast(SEXP r_mbsts_object,
                            SEXP r_prediction_data,
                            SEXP r_burn) {
      RListIoManager io_manager;
      Ptr<MultivariateStateSpaceRegressionModel> model = CreateModel(
          r_mbsts_object,
          getListElement(r_mbsts_object, "shared.state.specification", true),
          getListElement(r_mbsts_object, "series.state.specification", false),
          R_NilValue,
          R_NilValue,
          &io_manager);
      AddDataFromBstsObject(r_mbsts_object);
      int niter = Rf_asInteger(getListElement(r_mbsts_object, "niter", true));
      int burn = Rf_asInteger(r_burn);
      if (burn < 0) burn = 0;

      io_manager.prepare_to_stream(r_mbsts_object);
      io_manager.advance(burn);
      int iterations_after_burnin = niter - burn;

      int forecast_horizon = UnpackForecastData(r_prediction_data);
      model->observe_time_dimension(
          model->time_dimension() + forecast_horizon);

      Array ans(std::vector<int>{iterations_after_burnin,
              model_->nseries(), forecast_horizon});
      
      for (int i = 0; i < iterations_after_burnin; ++i) {
        io_manager.stream();
        ans.slice(i, -1, -1) = model_->simulate_forecast(
            GlobalRng::rng,
            forecast_predictors_,
            final_state(),
            series_specific_final_state_);
      }
      return ans;
    }
    
    MultivariateStateSpaceRegressionModel * Manager::CreateBareModel(
        SEXP r_data_list_or_model_object,
        SEXP r_prior,
        SEXP r_options,
        RListIoManager *io_manager) {
      model_.reset(new MultivariateStateSpaceRegressionModel(
          predictor_dimension_, nseries_));
      AddDataFromList(r_data_list_or_model_object);
      AssignSampler(r_prior, r_options);
      ConfigureIo(io_manager);
      return model_.get();
    }

    //---------------------------------------------------------------------------
    // For setting model state and parameters for debugging.
    void Manager::SetModelOptions(SEXP r_options) {
      if (Rf_isNull(r_options)) {
        return;
      }
      
      SEXP r_shared_state = getListElement(r_options, "fixed.shared.state");
      if (!Rf_isNull(r_shared_state)) {
        Matrix state = ToBoomMatrix(r_shared_state);
        if (state.ncol() != model_->time_dimension()) {
          state = state.transpose();
        }
        model_->permanently_set_state(state);
      } 
    }

    //---------------------------------------------------------------------------
    // Populate the model with data passed to the mbsts model fitting function.
    void Manager::AddDataFromList(SEXP r_data_list) {
      if (Rf_inherits(r_data_list, "mbsts")) {
        return AddDataFromBstsObject(r_data_list);
      } else if (!Rf_isNull(r_data_list)) {
        ConstVectorView responses = ToBoomVectorView(
            getListElement(r_data_list, "response"));
        int sample_size = responses.size();
        
        // If no predictors were supplied then create an intercept term.
        SEXP r_predictors = getListElement(r_data_list, "predictors");
        bool has_regression = !Rf_isNull(r_predictors);
        Matrix predictors = has_regression ? ToBoomMatrix(r_predictors) :
            Matrix(responses.size(), 1, 1.0);
        if (predictors.nrow() != sample_size) {
          report_error("Predictors and responses have different number "
                       "of rows.");
        }

        Factor series(getListElement(r_data_list, "series.id", true));
        if (series.length() != sample_size) {
          report_error("Series indicators and responses have different sizes.");
        }
        timestamp_info_.Unpack(r_data_list);
        AddData(responses, predictors, series);
      }
    }

    void Manager::AddDataFromBstsObject(SEXP r_bsts_object) {
      if (!Rf_inherits(r_bsts_object, "mbsts")) {
        report_error("In AddDataFromBstsObject, argument must inherit "
                     "from class 'mbsts'.");
      }
      timestamp_info_.Unpack(r_bsts_object);
      ConstVectorView responses(ToBoomVectorView(getListElement(
          r_bsts_object, "original.series")));
      Matrix predictors = ToBoomMatrix(getListElement(
          r_bsts_object, "predictors"));
      Factor series(getListElement(r_bsts_object, "series.id"));
      AddData(responses, predictors, series);
    }

    // Args:
    //   r_prior:  A list of SpikeSlabPrior objects.
    void Manager::AssignSampler(SEXP r_prior, SEXP r_options) {
      // Assign the prior.  A NULL r_prior signals that no posterior sampler is
      // needed.
      if (!Rf_isNull(r_prior)) {
        // Three samplers must be set here:
        // (0) samplers for the individual regression models in the observation
        //     model.
        // (1) the observation_model_sampler and
        // (2) the sampler for the main state space model.
        //
        // (1) and (2) are trivial.

        if (Rf_length(r_prior) != nseries_) {
          report_error("The number of elements in r_prior does not match "
                       "the number of time series.");
        }
        for (int i = 0; i < Rf_length(r_prior); ++i) {
          BOOM::RInterface::SetRegressionSampler(
              model_->observation_model()->model(i).get(),
              VECTOR_ELT(r_prior, i));
        }

        //----------------------------------------------------------------------
        // For debugging purposes the function can be called with options to fix
        // the model parameters at specific values.
        bool fixed_coefficients = false;
        bool fixed_residual_sd = false;
        if (!Rf_isNull(r_options)) {
          SEXP r_regression_coefficients = getListElement(
              r_options, "fixed.regression.coefficients");
          if (!Rf_isNull(r_regression_coefficients)) {
            Matrix coefficients = ToBoomMatrix(r_regression_coefficients);
            if (coefficients.nrow() != nseries_
                || coefficients.ncol() != predictor_dimension_) {
              report_error("supplied regression coefficients (debug) wrong size.");
            }
           for (int i = 0; i < nseries_; ++i) {
             model_->observation_model()->model(i)->set_Beta(coefficients.row(i));
           }
           fixed_coefficients = true;
          }

          SEXP r_residual_sd = getListElement(r_options, "fixed.residual.sd");
          if (!Rf_isNull(r_residual_sd)) {
            Vector residual_sd = ToBoomVector(r_residual_sd);
            for (int i = 0; i < nseries_; ++i) {
              model_->observation_model()->model(i)->set_sigsq(
                  square(residual_sd[i]));
            }
            fixed_residual_sd = true;
          }

        }
        if (fixed_residual_sd != fixed_coefficients) {
          report_error("If you fix one set of regression parameters you "
                       "must fix both.");
        }
        //----------------------------------------------------------------------

        if (!fixed_coefficients) {
          NEW(IndependentRegressionModelsPosteriorSampler,
              observation_model_sampler)(model_->observation_model());
          model_->observation_model()->set_method(observation_model_sampler);
        }
        
        // model sampler
        NEW(MultivariateStateSpaceRegressionPosteriorSampler, sampler)(
            model_.get());
        model_->set_method(sampler);
      }

    }

    //--------------------------------------------------------------------------
    class IndependentRegressionModelsCoefficientListElement
        : public MatrixValuedRListIoElement {
     public:
      IndependentRegressionModelsCoefficientListElement(
          IndependentRegressionModels *model,
          const std::string &name)
          : MatrixValuedRListIoElement(name),
            model_(model) {}

      // Rows correspond to different time series.
      int nrow() const override { return model_->ydim(); }
      int ncol() const override { return model_->xdim(); }

      void write() override {
        ArrayView view(array_view().slice(next_position(), -1, -1));
        for (int i = 0; i < nrow(); ++i) {
          const GlmCoefs &coefs(model_->model(i)->coef());
          for (int j = 0; j < ncol(); ++j) {
            view(i, j) = coefs.Beta(j);
          }
        }
      }

      void prepare_to_stream(SEXP object) override {
        MatrixValuedRListIoElement::prepare_to_stream(object);
        wsp.resize(ncol());
      }
      
      void stream() override {
        ArrayView view(array_view().slice(next_position(), -1, -1));
        for (int i = 0; i < nrow(); ++i) {
          for (int j = 0; j < ncol(); ++j) {
            wsp[j] = view(i, j);
          }
          model_->model(i)->set_Beta(wsp);
        }
      }
      
     private:
      IndependentRegressionModels *model_;
      Vector wsp;
    };

    //--------------------------------------------------------------------------
    // Store the residual standard deviations from the regression model as a
    // matrix in the model object.
    class IndependentRegressionModelsSdListElement
        : public VectorValuedRListIoElement {
     public:
      IndependentRegressionModelsSdListElement(
          IndependentRegressionModels *model,
          const std::string &name)
          : VectorValuedRListIoElement(name),
            model_(model) {}

      void write() override {
        VectorView sd(matrix_view().row(next_position()));
        for (int i = 0; i < model_->ydim(); ++i) {
          sd[i] = model_->model(i)->sigma();
        }
      }

      void stream() override {
        ConstVectorView sd(matrix_view().row(next_position()));
        for (int i = 0; i < model_->ydim(); ++i) {
          model_->model(i)->set_sigsq(square(sd[i]));
        }
      }
      
     private:
      IndependentRegressionModels *model_;
    };

    //==========================================================================
    void Manager::ConfigureIo(RListIoManager *io_manager) {
      std::vector<Ptr<UnivParams>> variance_parameters;
      std::vector<Ptr<GlmCoefs>> coefficients;
      variance_parameters.reserve(model_->nseries());
      coefficients.reserve(model_->nseries());
      IndependentRegressionModels *reg = model_->observation_model();
      for (int i = 0; i < model_->nseries(); ++i) {
        variance_parameters.push_back(reg->model(i)->Sigsq_prm());
        coefficients.push_back(reg->model(i)->coef_prm());
      }
      io_manager->add_list_element(
          new SdCollectionListElement(variance_parameters, "residual.sd"));
      io_manager->add_list_element(
          new IndependentRegressionModelsCoefficientListElement(
              model_->observation_model(),
              "regression.coefficients"));

    }

    //--------------------------------------------------------------------------
    void Manager::AddData(const ConstVectorView &responses,
                          const Matrix &predictors,
                          const Factor &series) {
      for (int i = 0; i < responses.size(); ++i) {
        NEW(TimeSeriesRegressionData, data_point)(
            responses[i],
            predictors.row(i),
            series[i],
            timestamp_info_.mapping(i));
        bool missing = BOOM::isNA(responses[i]);
        if (missing) {
          data_point->set_missing_status(Data::completely_missing);
        }
        model_->add_data(data_point);
      }
    }

    // TOOD(steve): handle predictors in the regression case.
    int Manager::UnpackForecastData(SEXP r_prediction_data) {
      forecast_predictors_ = BOOM::ToBoomMatrix(getListElement(
          r_prediction_data, "predictors"));
      UnpackForecastTimestamps(r_prediction_data);
      return forecast_predictors_.nrow() / nseries_;
    }
    
  }  // namespace bsts
}  // namespace BOOM
