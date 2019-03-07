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

include "state_space_multivariate_gaussian_model_manager.h"

namespace BOOM {
  namespace bsts {
    using Manager = MultivariateGaussianModelManager;
    
    Manager::MultivariateGaussianModelManager(
        int xdim)
        : predictor_dimension_(xdim) {}


    MultivariateStateSpaceModel * Manager::CreateObservationModel(
        SEXP r_data_list,
        SEXP r_prior,
        SEXP r_options,
        RListIoManager *io_manager) override {
      Matrix predictors;
      Matrix responses;
      SelectorMatrix response_is_observed;
      if (!Rf_isNull(r_data_list)) {
        // If data was from R then use it to build the model.  This is the case
        // for model training.
        if (inherit(r_data_list, "mbsts")) {
          SEXP r_responses = getListElement(r_data_list, "original.series");
          responses = ToBoomMatrix(r_responses);
          response_is_observed = IsObserved(responses);
        } else {
          responses = ToBoomMatrix(getListElement(r_data_list, "response"));
          response_is_observed = IsObserved(responses);
        }
        SEXP r_predictors = getListElement(r_data_list, "predictors");
        bool regression = !Rf_isNull(r_predictors);
        predictors = regression ? ToBoomMatrix(r_predictors) :
            Matrix(responses.nrow(), 1, 1.0);
        UnpackTimestampInfo(r_data_list);

        int xdim = predictors.ncol();
        model_.reset(new MultivariateStateSpaceModel(xdim));
        int sample_size = responses.nrow();
        if (predictors.nrow() != sample_size) {
          report_error("Predictors and responses have different number "
                       "of rows.");
        }
        for (int i = 0; i < sample_size; ++i) {
          NEW(PartiallyObservedVectorData, data_point)(
              ///////////////////////////              
        }
        
      } else {
        // If no data was passed from R then the model will probably be used for
        // forecasting, diagnostics, or some other form of analysis.
        if (predictor_dimension_ < 0) {
          report_error("If r_data_list is NULL then you must call "
                       "SetPredictorDimension before creating a model.");
        }
        model.reset_(new MultivariateStateSpaceModel(predictor_dimension_));
      }
    }
    
  }  // namespace bsts
}  // namespace BOOM
