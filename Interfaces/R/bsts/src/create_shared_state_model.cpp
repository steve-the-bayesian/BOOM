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
#include "Models/Glm/MvnGivenX.hpp"
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
      int nfactors = lround(Rf_asReal(getListElement(r_state_component, "size")));
      NEW(SharedLocalLevelStateModel, state_model)(nfactors, model, nseries_);

      // Set the initial state distribution.
      RInterface::MvnPrior initial_state_prior(getListElement(
          r_state_component, "initial.state.prior", true));
      state_model->set_initial_state_mean(initial_state_prior.mu());
      state_model->set_initial_state_variance(initial_state_prior.Sigma());
      
      // Set the prior on the observation coefficients.
      std::vector<Ptr<VariableSelectionPrior>> spikes;
      std::vector<Ptr<MvnBase>> slabs;
      SEXP r_coefficient_priors = getListElement(
          r_state_component, "coefficient.priors", true);
      // r_coefficient_priors is a list containing nseries
      // 'ConditionalZellnerPrior' objects.
      if (Rf_length(r_coefficient_priors) != nseries_) {
        report_error("Wrong number of coefficient priors given.");
      }
      for (int i = 0; i < nseries_; ++i) {
        RInterface::ConditionalZellnerPrior this_series_prior(VECTOR_ELT(
            r_coefficient_priors, i));
        spikes.push_back(this_series_prior.spike());
        NEW(MvnGivenXMvRegSuf, slab)(
            new VectorParams(this_series_prior.mean()),
            new UnivParams(this_series_prior.prior_information_weight()),
            Vector(),
            this_series_prior.diagonal_shrinkage(),
            state_model->coefficient_model()->suf());
        slabs.push_back(slab);
      }

      // Set the posterior sampler for the overall state model.
      NEW(SharedLocalLevelPosteriorSampler, state_model_sampler)(
          state_model.get(), slabs, spikes);
      
      state_model->set_method(state_model_sampler);

      // Set the io manager, if there is one.
      if (io_manager()) {
        io_manager()->add_list_element(new MatrixListElement(
            state_model->coefficient_model()->Beta_prm(),
            prefix + "shared.local.level.coefficients"));
      }
      return state_model;
    }
    
  }  // namespace bsts
}  // namespace BOOM
    
