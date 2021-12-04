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

#include <Models/Mixtures/DirichletProcessMvnModel.hpp>
#include <Models/Mixtures/PosteriorSamplers/DirichletProcessMvnCollapsedGibbsSampler.hpp>

#include <distributions/rng.hpp>
#include <cpputil/report_error.hpp>

#include <r_interface/boom_r_tools.hpp>
#include <r_interface/create_mixture_component.hpp>
#include <r_interface/handle_exception.hpp>
#include <r_interface/prior_specification.hpp>
#include <r_interface/print_R_timestamp.hpp>
#include <r_interface/seed_rng_from_R.hpp>

#include <R_ext/Arith.h>

namespace {
  using namespace BOOM;

  Ptr<DirichletProcessMvnModel> create_model(
      SEXP r_data,
      SEXP r_mean_base_measure,
      SEXP r_variance_base_measure,
      SEXP r_concentration_parameter) {
    ConstSubMatrix data = ToBoomMatrixView(r_data);
    double concentration_parameter = Rf_asReal(r_concentration_parameter);
    NEW(DirichletProcessMvnModel, model)(data.ncol(), concentration_parameter);

    BOOM::RInterface::MvnGivenSigmaMatrixPrior mean_prior_spec(r_mean_base_measure);
    BOOM::RInterface::InverseWishartPrior variance_prior_spec(r_variance_base_measure);
    NEW(DirichletProcessMvnCollapsedGibbsSampler, sampler)(
        model.get(),
        mean_prior_spec.boom(),
        variance_prior_spec.boom(),
        GlobalRng::rng);

    model->set_method(sampler);
    for (long i = 0; i < data.nrow(); ++i) {
      model->add_data(new VectorData(data.row(i)));
    }
    return model;
  }

  // A container to hold DirichletProcessMvn parameter draws and other other
  // model artifacts as they are drawn, and which can package them up for return
  // values later.
  class DpMvnParamStorage {
   public:
    DpMvnParamStorage() {}

    void store(const DirichletProcessMvnModel &model) {
      int nclusters = model.number_of_clusters();
      int dim = model.dim();
      Matrix model_means(nclusters, dim);
      Array model_variances(std::vector<int>{nclusters, dim, dim});

      for (int s = 0; s < nclusters; ++s) {
        model_means.row(s) = model.cluster(s).mu();
        model_variances.slice(s, -1, -1) = model.cluster(s).Sigma();
      }
      means_.push_back(model_means);
      variances_.push_back(model_variances);
      loglike_.push_back(model.log_likelihood());
    }

    // Create R instances of the objects created during the MCMC run, and return
    // these in an R list.
    SEXP package_parameters() const {
      RMemoryProtector protector;

      // The draws of means and variances, arranged by cluster size.  If the
      // model produced draws with 4, 7, and 9 clusters, then sorted_means[4],
      // [7], and [9] will be populated with arrays of dimension [4, dim], [7,
      // dim] and [9, dim].
      std::map<int, std::vector<Matrix>> sorted_means;
      std::map<int, std::vector<Array>> sorted_variances;
      std::map<int, std::vector<int>> iteration_numbers;

      long niter = means_.size();
      int dim = means_[0].ncol();

      for (long i = 0; i < niter; ++i) {
        int iteration_dim = means_[i].nrow();
        sorted_means[iteration_dim].push_back(means_[i]);
        sorted_variances[iteration_dim].push_back(variances_[i]);
        iteration_numbers[iteration_dim].push_back(i);
      }
      // The output format is means[[cluster_size]]

      std::vector<SEXP> per_cluster_size_results;
      std::vector<std::string> result_names;

      for (const auto &el : sorted_means) {
        int nclusters = el.first;
        const std::vector<Matrix> &means(el.second);
        int ndraws = means.size();

        Array means_array(std::vector<int>{ndraws, nclusters, dim});
        Array variance_array(std::vector<int>{ndraws, nclusters, dim, dim});
        for (size_t draw = 0; draw < means.size(); ++draw) {
          auto dims = means_array.slice(draw, -1, -1).dim();
          means_array.slice(draw, -1, -1) = sorted_means[nclusters][draw];
          variance_array.slice(draw, -1, -1, -1) = sorted_variances[nclusters][draw];
        }

        std::vector<SEXP> loading_dock = {
          protector.protect(ToRArray(means_array)),
          protector.protect(ToRArray(variance_array)),
          protector.protect(ToRIntVector(iteration_numbers[nclusters]))
        };

        per_cluster_size_results.push_back(
            protector.protect(CreateList(
                loading_dock, std::vector<std::string>{
                  "mean", "variance", "iteration"})));

        std::ostringstream name_translator;
        name_translator << nclusters;
        result_names.push_back(name_translator.str());
      }

      return CreateList(per_cluster_size_results, result_names);
    }

    SEXP package() const {
      RMemoryProtector protector;
      std::vector<SEXP> elements;
      elements.push_back(protector.protect(package_parameters()));
      elements.push_back(protector.protect(ToRVector(Vector(loglike_))));
      std::vector<std::string> names = {"parameters", "log.likelihood"};
      return CreateList(elements, names);
    }

   private:
    std::vector<Matrix> means_;
    std::vector<Array> variances_;
    std::vector<double> loglike_;
  };

}  // namespace

extern "C" {
  using BOOM::Ptr;

  SEXP boom_rinterface_fit_dirichlet_process_mvn_(
      SEXP r_data,
      SEXP r_mean_base_measure,
      SEXP r_variance_base_measure,
      SEXP r_concentration_parameter,
      SEXP rniter,
      SEXP rping,
      SEXP rseed) {
    BOOM::RErrorReporter error_reporter;
    BOOM::RMemoryProtector protector;
    try{
      BOOM::RInterface::seed_rng_from_R(rseed);

      Ptr<DirichletProcessMvnModel> model = create_model(
          r_data,
          r_mean_base_measure,
          r_variance_base_measure,
          r_concentration_parameter);

      int niter = Rf_asInteger(rniter);
      int ping = Rf_asInteger(rping);

      DpMvnParamStorage param_storage;

      for(int i = 0; i < niter; ++i){
        BOOM::print_R_timestamp(i, ping);
        R_CheckUserInterrupt();
        model->sample_posterior();
        param_storage.store(*model);
      }
      return param_storage.package();
    } catch(std::exception &e) {
      BOOM::RInterface::handle_exception(e);
    } catch(...) {
      BOOM::RInterface::handle_unknown_exception();
    }
    return R_NilValue;
  }
}
