/*
  Copyright (C) 2005-2012 Steven L. Scott

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

#include <Models/FiniteMixtureModel.hpp>
#include <Models/MultinomialModel.hpp>
#include <Models/DirichletModel.hpp>
#include <Models/PosteriorSamplers/MultinomialDirichletSampler.hpp>
#include <Models/PosteriorSamplers/FiniteMixturePosteriorSampler.hpp>

#include <distributions/rng.hpp>
#include <cpputil/report_error.hpp>

#include <r_interface/boom_r_tools.hpp>
#include <r_interface/create_mixture_component.hpp>
#include <r_interface/prior_specification.hpp>
#include <r_interface/print_R_timestamp.hpp>
#include <r_interface/seed_rng_from_R.hpp>

#include <R_ext/Arith.h>

namespace {
  using namespace BOOM;
  class FiniteMixtureLoglikelihoodCallback : public BOOM::ScalarIoCallback{
   public:
    explicit FiniteMixtureLoglikelihoodCallback(BOOM::FiniteMixtureModel *model)
        : model_(model) {}
    virtual double get_value()const{ return model_->last_loglike(); }
   private:
    BOOM::FiniteMixtureModel *model_;
  };

  class FiniteMixtureLogPriorCallback : public BOOM::ScalarIoCallback{
   public:
    explicit FiniteMixtureLogPriorCallback(BOOM::FiniteMixtureModel *model)
        : model_(model) {}
    virtual double get_value()const{ return model_->logpri(); }
   private:
    BOOM::FiniteMixtureModel *model_;
  };

  void add_loglike_io(Ptr<BOOM::FiniteMixtureModel> model,
                      BOOM::RListIoManager *io_manager){
    io_manager->add_list_element(
        new NativeUnivariateListElement(
            new FiniteMixtureLoglikelihoodCallback(model.get()),
            "log.likelihood"));

    io_manager->add_list_element(
        new NativeUnivariateListElement(
            new FiniteMixtureLogPriorCallback(model.get()),
            "log.prior"));
  }

  // Create the mixing distribution for the FiniteMixtureModel based
  // on the prior distribution passed from R.  The mixing distribution
  // will be registered with the io_manager.
  BOOM::Ptr<BOOM::MultinomialModel> create_mixing_distribution(
      SEXP rmixing_weight_prior, RListIoManager * io_manager){

    BOOM::RInterface::DirichletPrior mixing_weight_prior_spec(
        rmixing_weight_prior);
    NEW(BOOM::MultinomialModel, mixing_distribution)(
        mixing_weight_prior_spec.dim());
    io_manager->add_list_element(
        new BOOM::VectorListElement(mixing_distribution->Pi_prm(),
                                    "mixing.weights"));
    NEW(BOOM::DirichletModel, mixing_weight_prior)(
        mixing_weight_prior_spec.prior_counts());
    NEW(BOOM::MultinomialDirichletSampler, mixing_weight_sampler)(
        mixing_distribution.get(), mixing_weight_prior);
    mixing_distribution->set_method(mixing_weight_sampler);
    return mixing_distribution;
  }

  // Unpack the data from the rmixture_component_list and assign it to
  // the model.
  void assign_data(Ptr<FiniteMixtureModel> model,
                   SEXP rmixture_component_list){
    std::vector<std::vector<Ptr<Data> > > subjects =
        BOOM::RInterface::ExtractCompositeDataFromMixtureComponentList(
            rmixture_component_list);
    for(int i = 0; i < subjects.size(); ++i){
      for(int j = 0; j < subjects[i].size(); ++j){
        model->add_data(subjects[i][j]);}}
  }

}  // namespace

extern "C" {
  using BOOM::Ptr;

  SEXP boom_rinterface_fit_finite_mixture_(
      SEXP rmixture_component_list,
      SEXP rmixing_weight_prior,
      SEXP rniter,
      SEXP rping,
      SEXP rknown_source,
      SEXP rseed){
    try{
      BOOM::RListIoManager io_manager;
      BOOM::RInterface::seed_rng_from_R(rseed);

      Ptr<MultinomialModel> mixing_distribution =
          create_mixing_distribution(rmixing_weight_prior, &io_manager);

      std::vector<BOOM::Ptr<BOOM::MixtureComponent> > mixture_components =
          BOOM::RInterface::UnpackCompositeMixtureComponents(
              rmixture_component_list,
              mixing_distribution->dim(),
              &io_manager);

      NEW(BOOM::FiniteMixtureModel, model)(
          mixture_components, mixing_distribution);

      add_loglike_io(model, &io_manager);
      NEW(BOOM::FiniteMixturePosteriorSampler, sampler)(model.get());
      model->set_method(sampler);

      assign_data(model, rmixture_component_list);
      if(!Rf_isNull(rknown_source)){
        std::vector<int> known_data_source =
            BOOM::RInterface::UnpackKnownDataSource(rknown_source);
        model->set_data_source(known_data_source);
      }

      int niter = Rf_asInteger(rniter);
      int ping = Rf_asInteger(rping);
      SEXP ans;
      PROTECT(ans = io_manager.prepare_to_write(niter));
      Matrix class_membership_probabilities(
          model->dat().size(),
          model->number_of_mixture_components(),
          0.0);

      for(int i = 0; i < niter; ++i){
        BOOM::print_R_timestamp(i, ping);
        // TODO(stevescott): There is a potentially large resource leak
        // here, because the BOOM objects will not be freed.
        R_CheckUserInterrupt();
        model->sample_posterior();
        io_manager.write();
        class_membership_probabilities += model->class_membership_probability();
      }

      // Divide the class_membership_probabilities by the number of
      // mcmc iterations (thus averaging them over the life of the
      // MCMC algorithm.  Copy the class membership probabilities to
      // an R matrix, and place them in the list returned to the user.
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
}
