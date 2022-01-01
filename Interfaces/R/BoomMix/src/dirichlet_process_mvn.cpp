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
#include <Models/Mixtures/identify_permutation.hpp>

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
    DpMvnParamStorage()
        : iteration_counter_(0)
    {}

    // Store the the current state of the model.
    void store(const DirichletProcessMvnModel &model) {
      int nclusters = model.number_of_clusters();
      int dim = model.dim();
      Matrix model_means(nclusters, dim);
      Array model_variances(std::vector<int>{nclusters, dim, dim});
      for (int s = 0; s < nclusters; ++s) {
        model_means.row(s) = model.cluster(s).mu();
        model_variances.slice(s, -1, -1) = model.cluster(s).Sigma();
      }

      means_[nclusters].push_back(model_means);
      variances_[nclusters].push_back(model_variances);
      cluster_labels_[nclusters].push_back(model.cluster_indicators());
      iterations_[nclusters].push_back(iteration_counter_++);
      loglike_.push_back(model.log_likelihood());
    }

    // Create R instances of the objects created during the MCMC run, and return
    // these in an R list.
    SEXP package_parameters() {

      remove_label_switching();

      RMemoryProtector protector;
      int dim = means_.begin()->second[0].ncol();
      int nobs = cluster_labels_.begin()->second[0].size();

      std::vector<SEXP> per_cluster_size_results;
      std::vector<std::string> result_names;

      for (const auto &el : means_) {
        int nclusters = el.first;
        const std::vector<Matrix> &means(el.second);
        int ndraws = means.size();

        Array means_array(std::vector<int>{ndraws, nclusters, dim});
        Array variance_array(std::vector<int>{ndraws, nclusters, dim, dim});
        Matrix cluster_labels_matrix(ndraws, nobs);
        for (size_t draw = 0; draw < ndraws; ++draw) {
          means_array.slice(draw, -1, -1) = means_[nclusters][draw];
          variance_array.slice(draw, -1, -1, -1) = variances_[nclusters][draw];
          cluster_labels_matrix.row(draw) = Vector(cluster_labels_[nclusters][draw]);
        }

        std::vector<SEXP> loading_dock = {
          protector.protect(ToRArray(means_array)),
          protector.protect(ToRArray(variance_array)),
          protector.protect(ToRMatrix(cluster_labels_matrix)),
          protector.protect(ToRIntVector(iterations_[nclusters]))
        };

        per_cluster_size_results.push_back(
            protector.protect(CreateList(
                loading_dock, std::vector<std::string>{
                  "mean", "variance", "cluster.labels", "iteration"})));

        std::ostringstream name_translator;
        name_translator << nclusters;
        result_names.push_back(name_translator.str());
      }

      return CreateList(per_cluster_size_results, result_names);
    }

    // Package the state of the MCMC run for return to R.
    SEXP package() {
      RMemoryProtector protector;
      std::vector<SEXP> elements;
      elements.push_back(protector.protect(package_parameters()));
      elements.push_back(protector.protect(ToRVector(Vector(loglike_))));

      std::vector<std::string> names = {"parameters", "log.likelihood"};
      return CreateList(elements, names);
    }

    // Choose a permutation of the state labels at each MCMC iteration and apply
    // the chosen permutation to the model parameters and state labels.
    void remove_label_switching() {
      for (const auto &it : cluster_labels_) {
        int nclusters = it.first;
        std::vector<std::vector<int>> permutation =
            identify_permutation_from_labels(
                cluster_labels_[nclusters]);
        int niter = permutation.size();
        for (int i = 0; i < niter; ++i) {
          const std::vector<int> &perm(permutation[i]);

          Matrix means = means_[nclusters][i];
          for (int j = 0; j < nclusters; ++j) {
            means.row(perm[j]) = means_[nclusters][i].row(j);
          }
          means_[nclusters][i] = means;

          Array variances = variances_[nclusters][i];
          for (int j = 0; j < nclusters; ++j) {
            variances.slice(perm[j], -1, -1) = variances_[nclusters][i].slice(j, -1, -1);
          }
          variances_[nclusters][i] = variances;

          int nobs = cluster_labels_[nclusters][i].size();
          for (int j = 0; j < nobs; ++j) {
            cluster_labels_[nclusters][i][j] = perm[cluster_labels_[nclusters][i][j]];
          }
        }
      }
    }

   private:
    int iteration_counter_;

    // means_[nclusters][iteration][cluster][dim]
    std::map<int, std::vector<Matrix>> means_;

    // variances_[nclusters][iteration][cluster][dim1][dim2]
    std::map<int, std::vector<Array>> variances_;

    // cluster_labels_[nclusters][iteration][observation]
    std::map<int, std::vector<std::vector<int>>> cluster_labels_;

    // iterations_[nclusters][draw_number_within_cluster]
    std::map<int, std::vector<int>> iterations_;

    // log likelihood of each iteration, regardless of the number of clusters to
    // which it corresponds.
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
