/*
  Copyright (C) 2005-2019 Steven L. Scott

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

#include "create_shared_state_model.h"

#include <string>
#include "cpputil/report_error.hpp"
#include "cpputil/Date.hpp"

#include "r_interface/boom_r_tools.hpp"
#include "r_interface/list_io.hpp"
#include "r_interface/prior_specification.hpp"

#include "Models/ChisqModel.hpp"
#include "Models/StateSpace/MultivariateStateSpaceModelBase.hpp"
#include "Models/StateSpace/StateModels/LocalLevelStateModel.hpp"
#include "Models/StateSpace/PosteriorSamplers/MultivariateStateSpaceModelSampler.hpp"
#include "Models/StateSpace/PosteriorSamplers/SharedLocalLevelPosteriorSampler.hpp"

#include <R_ext/Print.h>

namespace BOOM {
  namespace bsts {

    void SharedStateModelFactory::AddState(
        SharedStateModelVector &state_models,
        MultivariateStateSpaceModelBase *model,
        SEXP r_shared_state_specification,
        const std::string &prefix) {
      if (!model) return;
      int number_of_state_models = Rf_length(r_shared_state_specification);
      for (int i = 0; i < number_of_state_models; ++i) {
        state_models.add_state(CreateSharedStateModel(
            model,
            VECTOR_ELT(r_shared_state_specification, i),
            prefix));
      }
      InstallPostStateListElements();
    }

    class SharedFinalStateCallback : public VectorIoCallback {
     public:
      explicit SharedFinalStateCallback(MultivariateStateSpaceModelBase *model)
          : model_(model) {}
      int dim() const override {return model_->state_dimension();}
      Vector get_vector() const override {return model_->final_state();}
     private:
      MultivariateStateSpaceModelBase *model_;
    };

    void SharedStateModelFactory::SaveFinalState(
        MultivariateStateSpaceModelBase *model,
        BOOM::Vector *final_state,
        const std::string &list_element_name) {
      if (!(model && final_state && io_manager())) return;
      final_state->resize(model->state_dimension());
      io_manager()->add_list_element(
          new NativeVectorListElement(
              new SharedFinalStateCallback(model),
              list_element_name,
              final_state));
    }

    //---------------------------------------------------------------------------
    // Stores the final state in the MultivariateStateSpaceRegressionModel as a
    // list of matrices, one for each series.  It is assumed the model has
    // series-specific state.  If not, it is the caller's responsibility to
    // avoid adding this element to the io manager.
    class SubordinateFinalStateListElement
        : public RListIoElement {
     public:
      SubordinateFinalStateListElement(
          MultivariateStateSpaceRegressionModel *model,
          std::vector<BOOM::Vector> *streaming_buffer)
          : RListIoElement("series.specific.final.state"),
            model_(model),
            streaming_buffer_(streaming_buffer)
      {}

      // Args:
      //   niter:  The number of iterations to be written.
      //
      // Returns:
      //   An R list of R matrices.
      SEXP prepare_to_write(int niter) override {
        RMemoryProtector protector;
        SEXP r_buffer = protector.protect(
            Rf_allocVector(VECSXP, model_->nseries()));
        for (int i = 0; i < model_->nseries(); ++i) {
          SEXP r_matrix = protector.protect(Rf_allocMatrix(
              REALSXP,
              niter,
              model_->series_specific_model(i)->state_dimension()));
          matrix_views_.push_back(BOOM::ToBoomMutableMatrixView(r_matrix));
        }
        StoreBuffer(r_buffer);
        return r_buffer;
      }

      // Args:
      //   r_object: A model object fit with 'mbsts'.  The object contains
      //     subordinate state that was previously written by this->write().
      void prepare_to_stream(SEXP r_object) override {
        if (!streaming_buffer_) return;
        RListIoElement::prepare_to_stream(r_object);
        SEXP r_buffer = rbuffer();
        int nseries = Rf_length(r_buffer);
        if (nseries != model_->nseries()) {
          report_error("Number of series in the final state buffer "
                       "must match model object.");
        }
        for (int i = 0; i < nseries; ++i) {
          matrix_views_.push_back(ToBoomMutableMatrixView(
              VECTOR_ELT(r_buffer, i)));
        }
      }

      void write() override {
        int row_index = next_position();
        for (int i = 0; i < model_->nseries(); ++i) {
          matrix_views_[i].row(row_index) =
              model_->series_specific_model(i)->state().last_col();
        }
      }
      
      void stream() override {
        if (streaming_buffer_) {
          int row_index = next_position();
          for (int i = 0; i < model_->nseries(); ++i) {
            (*streaming_buffer_)[i] = matrix_views_[i].row(row_index);
          }
        }
      }
      
     private:
      MultivariateStateSpaceRegressionModel *model_;
      std::vector<BOOM::SubMatrix> matrix_views_;
      std::vector<BOOM::Vector> *streaming_buffer_;
    };

    //--------------------------------------------------------------------------
    void SharedStateModelFactory::SaveSubordinateFinalState(
        MultivariateStateSpaceRegressionModel *model,
        std::vector<BOOM::Vector> *final_state,
        const std::string &list_element_name) {
      if (!model && final_state && io_manager()) return;
      if (model->has_series_specific_state()) {
        for (int i = 0; i < model->nseries(); ++i) {
          
        }
      }
      
    }
    
    Ptr<SharedStateModel>
    SharedStateModelFactory::CreateSharedStateModel(
        MultivariateStateSpaceModelBase *model,
        SEXP r_state_component,
        const std::string &prefix) {
      if (Rf_inherits(r_state_component, "SharedLocalLevel")) {
        return CreateSharedLocalLevel(r_state_component, model, prefix);
      } else {
        report_error("Unrecognized shared state model.");
      }
      return nullptr;
    }

    // ---------------------------------------------------------------------------
    // A callback class that handles I/O for the variance parameters.
    class SharedLocalLevelVarianceManager
        : public StreamableVectorIoCallback {
     public:
      explicit SharedLocalLevelVarianceManager(
          SharedLocalLevelStateModel *model)
          : model_(model)
      {}
      
      int dim() const override { return model_->number_of_factors(); }

      Vector get_vector() const override {
        Vector ans(dim());
        for (int i = 0; i < dim(); ++i) {
          ans[i] = model_->innovation_model(i)->sigma();
        }
        return ans;
      }

      void put_vector(const ConstVectorView &view) override {
        for (int i = 0; i < dim(); ++i) {
          model_->innovation_model(i)->set_sigsq(square(view[i]));
        }
      }
      
     private:
      SharedLocalLevelStateModel *model_;
    };
    // ---------------------------------------------------------------------------
    
    Ptr<SharedStateModel> SharedStateModelFactory::CreateSharedLocalLevel(
        SEXP r_state_component,
        MultivariateStateSpaceModelBase *model, 
        const std::string &prefix) {
      SEXP r_innovation_precision_priors = getListElement(
          r_state_component, "innovation.precision.priors");
      int nfactors = Rf_length(r_innovation_precision_priors);
      NEW(SharedLocalLevelStateModel, state_model)(nfactors, model, nseries_);

      // Set the initial state distribution.
      RInterface::MvnPrior initial_state_prior(getListElement(
          r_state_component, "initial.state.prior", true));
      state_model->set_initial_state_mean(initial_state_prior.mu());
      state_model->set_initial_state_variance(initial_state_prior.Sigma());
      
      // Set the prior on the innovation variances.
      std::vector<Ptr<GammaModelBase>> innovation_precision_priors;
      for (int i = 0; i < nfactors; ++i) {
        RInterface::SdPrior prior_spec(VECTOR_ELT(
            r_innovation_precision_priors, i));
        innovation_precision_priors.push_back(new ChisqModel(
            prior_spec.prior_df(),
            prior_spec.prior_guess()));
      }

      // Set the prior on the observation coefficients.
      RInterface::ScaledMatrixNormalPrior coefficient_prior(
          getListElement(r_state_component, "coefficient.prior"));

      // Set the posterior sampler for the overall state model.
      NEW(SharedLocalLevelPosteriorSampler, state_model_sampler)(
          state_model.get(),
          innovation_precision_priors,
          coefficient_prior.mean(),
          coefficient_prior.sample_size());
      state_model->set_method(state_model_sampler);

      // Set the io manager, if there is one.
      if (io_manager()) {
        io_manager()->add_list_element(new GenericVectorListElement(
            new SharedLocalLevelVarianceManager(state_model.get()),
            prefix + "shared.sigma.level"));

        io_manager()->add_list_element(new MatrixListElement(
            state_model->coefficient_model()->Beta_prm(),
            prefix + "shared.local.level.coefficients"));
      }
      return state_model;
    }
    
  }  // namespace bsts
}  // namespace BOOM
    
