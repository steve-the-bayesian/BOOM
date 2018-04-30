/*
  Copyright (C) 2005-2014 Steven L. Scott

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2.1 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA
*/

#include <Models/Mixtures/ConditionalFiniteMixtureModel.hpp>
#include <Models/Glm/MultinomialLogitModel.hpp>
#include <Models/Glm/PosteriorSamplers/MultinomialLogitCompositeSpikeSlabSampler.hpp>
#include <Models/Glm/VariableSelectionPrior.hpp>
#include <Models/MvnModel.hpp>
#include <Models/IndependentMvnModel.hpp>
#include <Models/Mixtures/PosteriorSamplers/ConditionalFiniteMixtureSampler.hpp>

#include <r_interface/boom_r_tools.hpp>
#include <r_interface/create_mixture_component.hpp>
#include <r_interface/prior_specification.hpp>
#include <r_interface/print_R_timestamp.hpp>
#include <r_interface/seed_rng_from_R.hpp>

#include <distributions/rng.hpp>

#include <cpputil/report_error.hpp>

#include <R_ext/Arith.h>

namespace {
using namespace BOOM;

  class ConditionalFiniteMixtureLoglikelihoodCallback
      : public BOOM::ScalarIoCallback{
   public:
    explicit ConditionalFiniteMixtureLoglikelihoodCallback(
        BOOM::ConditionalFiniteMixtureModel *model)
        : model_(model) {}
    virtual double get_value()const{ return model_->last_loglike(); }
   private:
    BOOM::ConditionalFiniteMixtureModel *model_;
  };

  class ConditionalFiniteMixtureLogPriorCallback
      : public BOOM::ScalarIoCallback{
   public:
    explicit ConditionalFiniteMixtureLogPriorCallback(
        BOOM::ConditionalFiniteMixtureModel *model)
        : model_(model) {}
    virtual double get_value()const{ return model_->logpri(); }
   private:
    BOOM::ConditionalFiniteMixtureModel *model_;
  };

  void add_conditional_mixture_loglike_io(
      Ptr<BOOM::ConditionalFiniteMixtureModel> model,
      BOOM::RListIoManager *io_manager){
    io_manager->add_list_element(
        new NativeUnivariateListElement(
            new ConditionalFiniteMixtureLoglikelihoodCallback(model.get()),
            "log.likelihood"));

    io_manager->add_list_element(
        new NativeUnivariateListElement(
            new ConditionalFiniteMixtureLogPriorCallback(model.get()),
            "log.prior"));
  }

  //----------------------------------------------------------------------
  Ptr<MultinomialLogitModel> CreateConditionalMixingDistribution(
      SEXP rmixing_distribution_prior,
      SEXP rmixture_design_matrix) {
    Vector prior_inclusion_probabilities(
        ToBoomVector(getListElement(rmixing_distribution_prior,
                                    "prior.inclusion.probabilities")));
    int xdim = Rf_ncols(rmixture_design_matrix);
    int nchoices = 1 + prior_inclusion_probabilities.size() / xdim;

    NEW(MultinomialLogitModel, mixing_distribution)(nchoices, xdim, 0);

    Vector coefficient_prior_mean(
        ToBoomVector(getListElement(rmixing_distribution_prior, "mu")));
    Ptr<MvnBase> coefficient_prior;
    if (Rf_inherits(rmixing_distribution_prior, "SpikeSlabPrior")) {
      SpdMatrix coefficient_prior_information(
          ToBoomSpdMatrix(getListElement(
              rmixing_distribution_prior, "siginv")));
      coefficient_prior.reset(new MvnModel(
          coefficient_prior_mean,
          coefficient_prior_information));
    } else if (Rf_inherits(rmixing_distribution_prior,
                           "IndependentSpikeSlabPrior")) {
      NEW(IndependentMvnModel, tmp_coefficient_prior)(
          prior_inclusion_probabilities.size());
      tmp_coefficient_prior->set_mu(coefficient_prior_mean);
      tmp_coefficient_prior->set_sigsq(
          ToBoomVector(getListElement(rmixing_distribution_prior,
                                      "prior.variance.diagonal")));
      coefficient_prior = tmp_coefficient_prior;
    }

    NEW(VariableSelectionPrior, variable_selection_prior)(
        prior_inclusion_probabilities);
    NEW(MultinomialLogitCompositeSpikeSlabSampler, sampler)(
        mixing_distribution.get(),
        coefficient_prior,
        variable_selection_prior);
    mixing_distribution->set_method(sampler);
    return mixing_distribution;
  }

  //----------------------------------------------------------------------
  void assign_condtional_mixture_data(
      Ptr<ConditionalFiniteMixtureModel> model,
      SEXP rmixture_component_list,
      SEXP rmixture_design_matrix,
      SEXP rknown_source) {
    std::vector<std::vector<Ptr<Data> > > subjects(
        BOOM::RInterface::ExtractCompositeDataFromMixtureComponentList(
            rmixture_component_list));
    Matrix mixture_design_matrix(ToBoomMatrix(rmixture_design_matrix));
    int state_space_size = model->number_of_mixture_components();

    std::vector<int> known_data_source;
    bool have_known_source = !Rf_isNull(rknown_source);
    if (have_known_source) {
      known_data_source =
          BOOM::RInterface::UnpackKnownDataSource(rknown_source);
      if (known_data_source.size() != nrow(mixture_design_matrix)) {
        report_error("The vector of known data sources must either be NULL, "
                     "or else have the same dimension as the "
                     "mixture design matrix.");
      }
    }

    int counter = 0;
    int nrow_design = nrow(mixture_design_matrix);
    for (int i = 0; i < subjects.size(); ++i) {
      for (int j = 0; j < subjects[i].size(); ++j) {
        if (counter >= nrow_design) {
          report_error("Mixture data had more observations than mixture "
                       "design matrix.");
        }
        NEW(ConditionalMixtureData, dp)(
            subjects[i][j],
            Ptr<VectorData>(new VectorData(
                mixture_design_matrix.row(counter))),
            state_space_size,
            have_known_source ? known_data_source[counter] : -1);
        ++counter;
        model->add_conditional_mixture_data(dp);
      }
    }
  }

}  // namespace

extern "C" {
  using BOOM::Ptr;

  SEXP boom_rinterface_fit_conditional_mixture_(
      SEXP rmixture_component_list,
      SEXP rmixing_distribution_prior,
      SEXP rmixture_design_matrix,
      SEXP rniter,
      SEXP rping,
      SEXP rknown_source,
      SEXP rseed) {
    try {
      BOOM::RListIoManager io_manager;
      BOOM::RInterface::seed_rng_from_R(rseed);
      Ptr<MultinomialLogitModel> mixing_distribution =
          CreateConditionalMixingDistribution(rmixing_distribution_prior,
                                              rmixture_design_matrix);

      std::vector<BOOM::Ptr<BOOM::MixtureComponent> > mixture_components =
          BOOM::RInterface::UnpackCompositeMixtureComponents(
              rmixture_component_list,
              mixing_distribution->Nchoices(),
              &io_manager);

      NEW(ConditionalFiniteMixtureModel, model)(
          mixture_components, mixing_distribution);

      add_conditional_mixture_loglike_io(model, &io_manager);
      NEW(BOOM::ConditionalFiniteMixtureSampler, sampler)(model.get());

      model->set_method(sampler);
      assign_condtional_mixture_data(model,
                                     rmixture_component_list,
                                     rmixture_design_matrix,
                                     rknown_source);

      int niter = Rf_asInteger(rniter);
      int ping = Rf_asInteger(rping);
      SEXP ans;
      PROTECT(ans = io_manager.prepare_to_write(niter));

      Matrix class_membership_probabilities(model->dat().size(),
                                            model->number_of_mixture_components(),
                                            0.0);
      for (int i = 0; i < niter; ++i) {
        BOOM::print_R_timestamp(i, ping);
        R_CheckUserInterrupt();
        model->sample_posterior();
        io_manager.write();
        class_membership_probabilities += model->class_membership_probabilities();
      }

      class_membership_probabilities /= niter;
      SEXP rclass_membership_probabilities;
      PROTECT(rclass_membership_probabilities =
              BOOM::ToRMatrix(class_membership_probabilities));
      ans = BOOM::appendListElement(ans, rclass_membership_probabilities,
                                    "state.probabilities");
      // ans is unprotected at this point, but since the next line
      // calls UNPROTECT there's no point in protecting it.  Be sure
      // to PROTECT it if code gets added below this point.
      UNPROTECT(2);
      return ans;
    } catch(std::exception &e) {
      ostringstream err;
      err << "Caught an exception with the following error message:" << endl
          << e.what() << endl;
      Rf_error(err.str().c_str());
    } catch(...) {
      ostringstream err;
      err << "Caught an unknown exception in "
          << "boom_rinterface_fit_finite_mixture_";
      Rf_error(err.str().c_str());
    }
    return R_NilValue;
  }

}  // extern "C"
