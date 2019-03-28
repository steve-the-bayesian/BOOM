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

#include "create_dynamic_intercept_state_model.h"
#include "Models/StateSpace/DynamicInterceptRegression.hpp"
#include "r_interface/boom_r_tools.hpp"
#include "r_interface/list_io.hpp"
#include "r_interface/prior_specification.hpp"

#include "Models/ChisqModel.hpp"
#include "Models/GaussianModel.hpp"
#include "Models/ZeroMeanGaussianModel.hpp"
#include "Models/PosteriorSamplers/FixedSpdSampler.hpp"
#include "Models/PosteriorSamplers/FixedUnivariateSampler.hpp"
#include "Models/PosteriorSamplers/GammaPosteriorSampler.hpp"
#include "Models/PosteriorSamplers/IndependentMvnVarSampler.hpp"
#include "Models/PosteriorSamplers/ZeroMeanGaussianConjSampler.hpp"
#include "Models/PosteriorSamplers/ZeroMeanMvnIndependenceSampler.hpp"

#include "Models/StateSpace/StateModels/LocalLevelStateModel.hpp"

namespace BOOM {
  namespace bsts {
    using namespace BOOM::RInterface;
    namespace {
      using DISMF = DynamicInterceptStateModelFactory;
    }
    
    void DISMF::AddState(
        DynamicInterceptRegressionModel *model,
        SEXP r_state_specification,
        const std::string &prefix) {
      if (!model) return;
      int number_of_state_models = Rf_length(r_state_specification);
      for (int i = 0; i < number_of_state_models; ++i) {
        model->add_state(CreateStateModel(
            model,
            VECTOR_ELT(r_state_specification, i),
            prefix));
      }
      InstallPostStateListElements();
    }

    class DirmFinalStateCallback : public VectorIoCallback {
     public:
      explicit DirmFinalStateCallback(DynamicInterceptRegressionModel *model)
          : model_(model) {}
      int dim() const override {return model_->state_dimension();}
      Vector get_vector() const override {return Vector(model_->final_state());}
     private:
      DynamicInterceptRegressionModel *model_;
    };

    Ptr<DynamicInterceptStateModel> DISMF::CreateStateModel(
        DynamicInterceptRegressionModel *model,
        SEXP r_state_component,
        const std::string &prefix) {
      if (Rf_inherits(r_state_component, "LocalLevel")) {
        return CreateDynamicLocalLevel(r_state_component, prefix);
      } else {
        std::ostringstream err;
        err << "Unknown object passed where state model expected." << endl;
        std::vector<std::string> class_info = StringVector(
            Rf_getAttrib(r_state_component, R_ClassSymbol));
        if (class_info.empty()) {
          err << "Object has no class attribute." << endl;
        } else if (class_info.size() == 1) {
          err << "Object is of class " << class_info[0] << "." << endl;
        } else {
          err << "Object has class:" << endl;
          for (int i = 0; i < class_info.size(); ++i) {
            err << "     " << class_info[i] << endl;
          }
          report_error(err.str());
        }
        return nullptr;
      }
    }
    
    void DISMF::SaveFinalState(
        DynamicInterceptRegressionModel *model,
        BOOM::Vector *final_state,
        const std::string &list_element_name) {
      if (!(model && final_state && io_manager())) return;
      final_state->resize(model->state_dimension());
      io_manager()->add_list_element(
          new NativeVectorListElement(
              new DirmFinalStateCallback(model),
              list_element_name,
              final_state));
    }

    // TODO(this code is nearly identical to CreateLocalLevel in
    // create_state_model.cpp.  Find a way to share that code?
    DynamicInterceptLocalLevelStateModel * DISMF::CreateDynamicLocalLevel(
        SEXP r_state_component,
        const std::string &prefix) {
      SdPrior sigma_prior_spec(getListElement(
          r_state_component, "sigma.prior"));
      NormalPrior initial_state_prior(getListElement(
          r_state_component, "initial.state.prior"));
      DynamicInterceptLocalLevelStateModel * level(
          new DynamicInterceptLocalLevelStateModel(
              sigma_prior_spec.initial_value()));

      //----------------------------------------------------------------------
      // Set the prior for the initial state.  It is R's job to make
      // sure this is set correctly.
      level->set_initial_state_variance(square(initial_state_prior.sigma()));
      level->set_initial_state_mean(initial_state_prior.mu());

      //----------------------------------------------------------------------
      // Set the prior distribution for sigma.  The variance can be fixed,
      // or have an inverse Gamma prior.  It is R's job to document which
      // is the case.
      if (sigma_prior_spec.fixed()) {
        Ptr<FixedUnivariateSampler> sampler(
            new FixedUnivariateSampler(
                level->Sigsq_prm(),
                level->sigsq()));
      } else {
        Ptr<ZeroMeanGaussianConjSampler> sampler(
            new ZeroMeanGaussianConjSampler(level,
                                            sigma_prior_spec.prior_df(),
                                            sigma_prior_spec.prior_guess()));
        if (sigma_prior_spec.upper_limit() > 0) {
          sampler->set_sigma_upper_limit(sigma_prior_spec.upper_limit());
        }
        level->set_method(sampler);
      }

      // Add information about this parameter to the io_manager
      if (io_manager()) {
        io_manager()->add_list_element(new StandardDeviationListElement(
            level->Sigsq_prm(),
            prefix + "sigma.level"));
      }
      return level;
    }
    
  }  // namespace bsts
}  // namespace BOOM
