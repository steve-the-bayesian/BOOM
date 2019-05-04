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
      MultivariateStateContributionsCallback(
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
      MultivariateSaveFullStateCallback(
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
        SEXP r_data_list,
        SEXP r_shared_state_specification,
        SEXP r_series_state_specification,
        SEXP r_prior,
        SEXP r_options,
        RListIoManager *io_manager) {
      CreateBareModel(r_data_list, r_prior, r_options, io_manager);
      SharedStateModelFactory shared_state_model_factory(nseries_, io_manager);
      shared_state_model_factory.AddState(
          model_->state_model_vector(),
          model_.get(),
          r_shared_state_specification,
          "");

      // Consider a new list_io element here.
      if (!Rf_isNull(r_series_state_specification)) {
        for (int i = 0; i < nseries_; ++i) {
          StateModelFactory series_state_factory(io_manager);
          std::ostringstream prefix;
          prefix << "series" << i << ".";
          series_state_factory.AddState(
              model_->series_specific_model(i).get(),
              VECTOR_ELT(r_series_state_specification, i),
              prefix.str());
        }
      }
      // TODO: save final state

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
    
    MultivariateStateSpaceRegressionModel * Manager::CreateBareModel(
        SEXP r_data_list,
        SEXP r_prior,
        SEXP r_options,
        RListIoManager *io_manager) {
      model_.reset(new MultivariateStateSpaceRegressionModel(
          predictor_dimension_, nseries_));
      AddDataFromList(r_data_list);
      AssignSampler(r_prior, r_options);
      ConfigureIo(io_manager);
      return model_.get();
    }

    //---------------------------------------------------------------------------
    void Manager::SetModelOptions(SEXP r_options) {
      if (Rf_isNull(r_options)) return;
      
      SEXP r_fixed_state = getListElement(r_options, "fixed.state");
      if (!Rf_isNull(r_fixed_state)) {
        model_->permanently_set_state(ToBoomMatrix(r_fixed_state));
      }


    }

    //---------------------------------------------------------------------------
    // Populate the model with data passed to the mbsts model fitting function.
    void Manager::AddDataFromList(SEXP r_data_list) {
      if (!Rf_isNull(r_data_list)) {
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

        Factor series(getListElement(r_data_list, "series", true));
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
      Factor series(getListElement(r_bsts_object, "series"));
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
        // (2) the sampler for the primary model.
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
      int horizon = Rf_asInteger(getListElement(
          r_prediction_data, "horizon", true));
      return horizon;
    }

    Array Manager::Forecast(SEXP r_mbsts_object,
                            SEXP r_prediction_data,
                            SEXP r_burn,
                            SEXP r_observed_data) {
      report_error("Forecast is not yet implemented.");
      return Array();
    }
    
  }  // namespace bsts
}  // namespace BOOM
