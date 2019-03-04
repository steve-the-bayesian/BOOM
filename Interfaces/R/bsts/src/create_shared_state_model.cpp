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

#include "r_interface/create_shared_state_model.h"

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
        MultivariateStateSpaceModelBase *model,
        SEXP r_shared_state_specification,
        const std::string &prefix) {
      if (!model) return;
      int number_of_state_models = Rf_length(r_shared_state_specification);
      for (int i = 0; i < number_of_state_models; ++i) {
        model->add_state(CreateSharedStateModel(
            model,
            VECTOR_ELT(r_shared_state_specification, i),
            prefix));
      }
      InstallPostStateListElements();
    }

    Ptr<MultivariateStateModel>
    SharedStateModelFactory::CreateSharedStateModel(
        MultivariateStateSpaceModelBase *model,
        SEXP r_state_component,
        const std::string &prefix) {
      if (Rf_inherits(r_state_component, "")) {
        return CreateSharedLocalLevel(r_state_component, model, prefix);
      } else {
        report_error("Unrecognized shared state model.");
      }
      return nullptr;
    }


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

    Ptr<MultivariateStateModel>
    SharedStateModelFactory::CreateSharedLocalLevel(
        SEXP r_state_component,
        MultivariateStateSpaceModelBase *model, 
        const std::string &prefix) {
      SEXP r_innovation_precision_priors = getListElement(
          r_state_component, "innovation.precision.priors");
      int nfactors = Rf_length(r_innovation_precision_priors);
      NEW(SharedLocalLevelStateModel, state_model)(nfactors, model);

      // Set the initial state distribution.
      state_model->set_initial_state_mean(ToBoomVector(getListElement(
          r_state_component, "initial.state.mean")));
      state_model->set_initial_state_variance(ToBoomSpdMatrix(getListElement(
          r_state_component, "initial.state.variance")));
      
      // Set the prior.
      std::vector<Ptr<GammaModelBase>> innovation_precision_priors;
      for (int i = 0; i < nfactors; ++i) {
        RInterface::SdPrior prior_spec(VECTOR_ELT(
            r_innovation_precision_priors, i));
        innovation_precision_priors.push_back(new ChisqModel(
            prior_spec.prior_df(),
            prior_spec.prior_guess()));
      }

      NEW(SharedLocalLevelPosteriorSampler, state_model_sampler)(
          state_model.get(),
          innovation_precision_priors,
          ToBoomMatrix(getListElement(
              r_state_component, "observation.coefficient.prior.mean")),
          Rf_asReal(getListElement(
              r_state_component, "coefficient.prior.sample.size")));

      // Set the io manager, if there is one.
      if (io_manager()) {
        io_manager()->add_list_element(new GenericVectorListElement(
            new SharedLocalLevelVarianceManager(state_model.get()),
            prefix + "shared.sigma.level"));

        io_manager()->add_list_element(new MatrixListElement(
            state_model->coefficient_model()->Beta_prm(),
            prefix + "coefficients"));
      }
      return state_model;
    }
    
  }  // namespace bsts
}  // namespace BOOM
    
