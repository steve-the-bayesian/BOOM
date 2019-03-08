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
#include "state_space_multivariate_gaussian_model_manager.h"

#include "LinAlg/Selector.hpp"
#include "Models/GammaModel.hpp"
#include "Models/ChisqModel.hpp"
#include "Models/PosteriorSamplers/IndependentMvnVarSampler.hpp"
#include "Models/StateSpace/PosteriorSamplers/MultivariateStateSpaceModelSampler.hpp"
#include "r_interface/prior_specification.hpp"

namespace BOOM {
  namespace bsts {
    using Manager = MultivariateGaussianModelManager;
    
    Manager::MultivariateGaussianModelManager(
        int ydim, int xdim)
        : ydim_(ydim),
          predictor_dimension_(xdim) {}

    MultivariateStateSpaceModel * Manager::CreateObservationModel(
        SEXP r_data_list,
        SEXP r_prior,
        SEXP r_options,
        RListIoManager *io_manager) {
      model_.reset(new MultivariateStateSpaceModel(ydim_));
      AddDataFromList(r_data_list);
      AssignSampler(r_prior);
      ConfigureIo(io_manager);
      return model_.get();
    }

    void Manager::AddDataFromList(SEXP r_data_list) {
      if (!Rf_isNull(r_data_list)) {
        Matrix responses;
        // If data was from R then use it to build the model.  This is the case
        // for model training.
        if (Rf_inherits(r_data_list, "mbsts")) {
          SEXP r_responses = getListElement(r_data_list, "original.series");
          responses = ToBoomMatrix(r_responses);
        } else {
          responses = ToBoomMatrix(getListElement(r_data_list, "response"));
        }
        SelectorMatrix response_is_observed = IsObserved(responses);
        SEXP r_predictors = getListElement(r_data_list, "predictors");
        bool regression = !Rf_isNull(r_predictors);
        Matrix predictors = regression ? ToBoomMatrix(r_predictors) :
            Matrix(responses.nrow(), 1, 1.0);
        UnpackTimestampInfo(r_data_list);

        int sample_size = responses.nrow();
        if (predictors.nrow() != sample_size) {
          report_error("Predictors and responses have different number "
                       "of rows.");
        }
        AddData(responses, predictors, IsObserved(responses));
      }
    }

    void Manager::AddDataFromBstsObject(SEXP r_bsts_object) {
      if (!Rf_inherits(r_bsts_object, "mbsts")) {
        report_error("In AddDataFromBstsObject, argument must inherit "
                     "from class 'mbsts'.");
      }
      UnpackTimestampInfo(r_bsts_object);
      Matrix responses = ToBoomMatrix(getListElement(
          r_bsts_object, "original.series"));
      Matrix predictors = ToBoomMatrix(getListElement(
          r_bsts_object, "predictors"));
      AddData(responses, predictors, IsObserved(responses));
    }
    
    void Manager::AssignSampler(SEXP r_prior) {
      // Assign the prior.  A NULL r_prior signals that no posterior sampler is
      // needed.
      if (!Rf_isNull(r_prior)) {
        // Two samplers must be set here: (1) the observation_model_sampler and
        // (2) the sampler for the primary model.
        SEXP r_residual_precision_priors = getListElement(
            r_prior, "residual.precision.priors", true);
        std::vector<Ptr<GammaModelBase>> residual_precision_priors;
        int ydim = Rf_length(r_residual_precision_priors);
        Vector sigma_max;
        for (int i = 0; i < ydim; ++i) {
          RInterface::SdPrior rpp(VECTOR_ELT(r_residual_precision_priors, i));
          sigma_max.push_back(rpp.upper_limit());
          if (sigma_max.back() < 0) {
            sigma_max.back() = infinity();
          }
          NEW(ChisqModel, residual_precision_prior_i)(
              rpp.prior_guess(), rpp.prior_df());
          residual_precision_priors.push_back(residual_precision_prior_i);
        }
        NEW(IndependentMvnVarSampler, observation_model_sampler)(
            model_->observation_model(),
            residual_precision_priors);
        observation_model_sampler->set_sigma_max(sigma_max);
        model_->observation_model()->set_method(observation_model_sampler);
        
        // model sampler
        NEW(MultivariateStateSpaceModelSampler, sampler)(model_.get());
        model_->set_method(sampler);
      }
    }
    

    
    void Manager::ConfigureIo(RListIoManager *io_manager) {
      io_manager->add_list_element(
          new SdVectorListElement(
              model_->observation_model()->Sigsq_prm(),
              "sigma.obs"));
    }


    // TODO(steve): Right now this model does not handle predictors. Fix it so
    // it does, and handle them appropriately here.
    void Manager::AddData(const Matrix &responses,
                          const Matrix &predictors,
                          const SelectorMatrix &observed) {
      for (int i = 0; i < responses.nrow(); ++i) {
        Selector observed_i = observed.row(i);
        NEW(PartiallyObservedVectorData, data_point)(
            responses.row(i), observed_i);
        if (observed_i.nvars() == 0) {
          data_point->set_missing_status(Data::completely_missing);
        } else if (observed_i.nvars() < observed_i.nvars_possible()) {
          data_point->set_missing_status(Data::partly_missing);
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
