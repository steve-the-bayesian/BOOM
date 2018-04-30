/*
  Copyright (C) 2005-2013 Steven L. Scott

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

#include <r_interface/boom_r_tools.hpp>
#include <r_interface/handle_exception.hpp>
#include <r_interface/list_io.hpp>
#include <r_interface/print_R_timestamp.hpp>
#include <r_interface/prior_specification.hpp>
#include <r_interface/seed_rng_from_R.hpp>

#include <LinAlg/Array.hpp>

#include <Models/Bart/GaussianBartModel.hpp>
#include <Models/Bart/PosteriorSamplers/GaussianBartPosteriorSampler.hpp>
#include <Models/Bart/PoissonBartModel.hpp>
#include <Models/Bart/PosteriorSamplers/PoissonBartPosteriorSampler.hpp>
#include <Models/Bart/ProbitBartModel.hpp>
#include <Models/Bart/PosteriorSamplers/ProbitBartPosteriorSampler.hpp>
#include <Models/Bart/LogitBartModel.hpp>
#include <Models/Bart/PosteriorSamplers/LogitBartPosteriorSampler.hpp>

#include <Samplers/MoveAccounting.hpp>

namespace BOOM {
  using namespace Bart;

  struct BartModelAndSampler {
    Ptr<BartModelBase> model;
    Ptr<BartPosteriorSamplerBase> sampler;
  };

  //======================================================================
  // A class for implementing RListIO for Bart trees.  Streams an
  // ensemble of trees.
  class TreeEnsembleListElement : public ListValuedRListIoElement {
   public:
    TreeEnsembleListElement(Ptr<BartModelBase> model, const string &name);
    virtual void write();
    virtual void stream();
   private:
    Ptr<BartModelBase> model_;
  };

  //----------------------------------------------------------------------
  TreeEnsembleListElement::TreeEnsembleListElement(
      Ptr<BartModelBase> model,
      const string &name)
      : ListValuedRListIoElement(name),
        model_(model)
  {}

  void TreeEnsembleListElement::write() {
    // Build a list from the vector of trees.  The number of trees can
    // differ from one iteration to the next.
    int number_of_nodes = 0;
    for (int i = 0; i < model_->number_of_trees(); ++i) {
      number_of_nodes += model_->tree(i)->number_of_nodes();
    }
    Matrix tree_ensemble_matrix(number_of_nodes, 3);
    int start = 0;
    for (int i = 0; i < model_->number_of_trees(); ++i) {
      int end = start + model_->tree(i)->number_of_nodes();
      SubMatrix tree_submatrix(tree_ensemble_matrix,
                               start, end - 1, 0, 2);
      tree_submatrix = model_->tree(i)->to_matrix();
      start = end;
    }
    SET_VECTOR_ELT(rbuffer(),
                   next_position(),
                   ToRMatrix(tree_ensemble_matrix));
  }

  // The number of trees can vary from one iteration to the next.
  void TreeEnsembleListElement::stream() {
    BOOM::Matrix tree_ensemble_matrix = ToBoomMatrix(VECTOR_ELT(
        rbuffer(), next_position()));
    int number_of_trees = 0;
    const ConstVectorView &parent_id(tree_ensemble_matrix.col(0));
    for (int i = 0; i < parent_id.size(); ++i) {
      number_of_trees += (lround(parent_id[i]) == -1);
    }
    model_->set_number_of_trees(number_of_trees);
    int start = 0;
    for (int i = 0; i < number_of_trees; ++i) {
      int end = start + 1;
      while (end < parent_id.size() && lround(parent_id[end]) != -1) {
        ++end;
      }
      SubMatrix tree_submatrix(tree_ensemble_matrix, start, end - 1, 0, 2);
      model_->rebuild_tree(i, tree_submatrix);
      start = end;
    }
  }

  //======================================================================
  // A callback class responsible for recording the MCMC draws of the
  // residuals in a GaussianBartModel.
  class ResidualRecorderCallback : public VectorIoCallback {
   public:
    explicit ResidualRecorderCallback(Ptr<GaussianBartPosteriorSampler> sampler)
        : sampler_(sampler.get())
    {}

    virtual int dim()const {
      return sampler_->residuals().size();
    }

    virtual Vector get_vector()const {
      const std::vector<const Bart::GaussianResidualRegressionData *> data(
          sampler_->residuals());
      Vector ans(data.size());
      for (int i = 0; i < ans.size(); ++i) {
        ans[i] = data[i]->residual();
      }
      return ans;
    }
   private:
    const GaussianBartPosteriorSampler *sampler_;
  };

  SEXP SerializeVariableSummary(const Bart::VariableSummary &variable_summary) {
    SerializedVariableSummary serialized = variable_summary.serialize();
    SEXP r_integers;
    PROTECT(r_integers = Rf_allocVector(INTSXP, 3));
    int *integers = INTEGER(r_integers);
    integers[0] = serialized.variable_number;
    integers[1] = serialized.is_continuous;
    integers[2] = static_cast<int>(serialized.strategy);
    SEXP ans;
    PROTECT(ans = Rf_allocVector(VECSXP, 2));
    SET_VECTOR_ELT(ans, 0, r_integers);
    SET_VECTOR_ELT(ans, 1, ToRVector(serialized.data));
    UNPROTECT(2);
    return ans;
  }

  SerializedVariableSummary GetSerializedVariableSummary(SEXP r_list) {
    int *integers = INTEGER(VECTOR_ELT(r_list, 0));
    SerializedVariableSummary ans;
    ans.finalized = true;
    ans.variable_number = integers[0];
    ans.is_continuous = integers[1];
    ans.strategy = Bart::ContinuousCutpointStrategy(integers[2]);
    ans.data = ToBoomVector(VECTOR_ELT(r_list, 1));
    return ans;
  }

  // Returns a list containing an R representation of each variable
  // summary in the model.
  SEXP WriteVariableSummaries(Ptr<BartModelBase> model) {
    int number_of_variables = model->number_of_variables();
    SEXP r_variable_summaries;
    PROTECT(r_variable_summaries = Rf_allocVector(VECSXP, number_of_variables));
    for (int i = 0; i < number_of_variables; ++i) {
      SEXP r_variable_summary;
      PROTECT(r_variable_summary =
              SerializeVariableSummary(model->variable_summary(i)));
      SET_VECTOR_ELT(r_variable_summaries, i, r_variable_summary);
      UNPROTECT(1);
    }
    UNPROTECT(1);
    return r_variable_summaries;
  }

}  // namespace BOOM

namespace {
  using namespace BOOM;
  //----------------------------------------------------------------------
  // A utility function to convert the strategy string to an enum.
  Bart::ContinuousCutpointStrategy GetStrategy(
      SEXP r_continuous_distribution_strategy) {
    std::string continuous_distribution_strategy_string(
        CHAR(STRING_ELT(r_continuous_distribution_strategy, 0)));

    Bart::ContinuousCutpointStrategy continuous_distribution_strategy =
        Bart::UNIFORM_CONTINUOUS;

    if (continuous_distribution_strategy_string ==
        "uniform.continuous") {
      continuous_distribution_strategy = Bart::UNIFORM_CONTINUOUS;
    } else if (continuous_distribution_strategy_string ==
               "uniform.discrete") {
      continuous_distribution_strategy = Bart::UNIFORM_DISCRETE;
    } else {
      ostringstream err;
      err << "Unknown continuous.distribution.strategy passed to"
          "BoomBart:  " << continuous_distribution_strategy_string
          << endl;
      report_error(err.str());
    }
    return continuous_distribution_strategy;
  }

  void SetVariableSummaries(Ptr<BartModelBase> model,
                            SEXP r_variable_summaries) {
    int number_of_variables = Rf_length(r_variable_summaries);
    std::vector<Bart::SerializedVariableSummary> serialized_summaries;
    for (int i = 0; i < number_of_variables; ++i) {
      serialized_summaries.push_back(
          GetSerializedVariableSummary(
              VECTOR_ELT(r_variable_summaries, i)));
    }
    model->set_variable_summaries(serialized_summaries);
  }

  //----------------------------------------------------------------------
  // The primary model factory for use post MCMC.
  Ptr<BartModelBase> ReCreateBartModel(SEXP r_bart_object,
                                       RListIoManager *io_manager) {
    std::string family = GetStringFromList(r_bart_object, "family");
    Ptr<BartModelBase> model;
    // Actual number of trees will be inferred when streaming
    // r_bart_object, and can vary each iteration.
    int number_of_trees = 1;
    if (family == "gaussian") {
      NEW(GaussianBartModel, gaussian_model)(number_of_trees);
      model = gaussian_model;
      io_manager->add_list_element(
          new StandardDeviationListElement(gaussian_model->Sigsq_prm(),
                                           "sigma"));
      // Residuals are streamed in the MCMC method, but not streamed
      // here.
    } else if (family == "poisson") {
      model.reset(new PoissonBartModel(number_of_trees));
    } else if (family == "probit") {
      model.reset(new ProbitBartModel(number_of_trees));
    } else if (family == "logit") {
      model.reset(new LogitBartModel(number_of_trees));
    }
    io_manager->add_list_element(new TreeEnsembleListElement( model, "trees"));
    io_manager->prepare_to_stream(r_bart_object);

    SetVariableSummaries(model, getListElement(
        r_bart_object, "variable.summaries"));
    return model;
  }

  //----------------------------------------------------------------------
  class BartTreeCountPriorImpl{
   public:
    virtual ~BartTreeCountPriorImpl() {}
    virtual double evaluate_log_prior(int number_of_trees) const = 0;
  };

  class BartTreeCountPrior {
   public:
    explicit BartTreeCountPrior(BartTreeCountPriorImpl *impl)
        : impl_(impl) {}
    double operator()(int number_of_trees) const {
      return impl_->evaluate_log_prior(number_of_trees);
    }
   private:
    std::shared_ptr<BartTreeCountPriorImpl> impl_;
  };

  //----------------------------------------------------------------------
  class BartDiscreteUniformTreePrior : public BartTreeCountPriorImpl {
   public:
    BartDiscreteUniformTreePrior(int lo, int hi)
        : lo_(lo), hi_(hi) {
      if (hi < lo) {
        report_error("Illegal arguments to BartDiscreteUniformTreePrior "
                     "constructor.  'hi' must be >= 'lo'. ");
      }
      log_density_value_ = -log1p(hi_ - lo_);
    }

    virtual double evaluate_log_prior(int number_of_trees) const {
      if (number_of_trees >= lo_ && number_of_trees <= hi_) {
        return log_density_value_;
      } else {
        return negative_infinity();
      }
    }

   private:
    int lo_;
    int hi_;
    double log_density_value_;
  };

  //----------------------------------------------------------------------
  class BartPoissonTreePrior : public BartTreeCountPriorImpl {
   public:
    explicit BartPoissonTreePrior(double mean)
        : mean_(mean),
          lo_(0),
          hi_(std::numeric_limits<int>::max()),
          log_support_probability_(0)
    {
      if (mean <= 0) {
        report_error("Illegal argument to BartPoissonTreePrior constructor.  "
                     "'mean' must be positive.");
      }
    }

    void truncate(int lo, int hi) {
      if (hi < lo) {
        report_error("Illegal arguments to BartPoissonTreePrior::truncate().  "
                     "'hi' must be >= 'lo'.");
      }
      lo_ = lo;
      hi_ = hi;

      double plo = ppois(lo - 1, mean_, true, false);
      double phi = ppois(hi + 1, mean_, false, false);
      log_support_probability_ = log(1 - plo - phi);
    }

    virtual double evaluate_log_prior(int number_of_trees) const {
      if (number_of_trees >= lo_ && number_of_trees <= hi_) {
        return dpois(number_of_trees, mean_, true) - log_support_probability_;
      } else {
        return negative_infinity();
      }
    }

   private:
    double mean_;
    int lo_;
    int hi_;
    // The log_support_probability_ is log(1 - ppois(number_of_trees < lo,
    // mean) - ppois(number_of_trees > hi, mean))
    double log_support_probability_;
  };

  class BartPointMassTreePrior : public BartTreeCountPriorImpl {
   public:
    explicit BartPointMassTreePrior(int location) : location_(location) {
      if (location < 0) {
        report_error("Illegal argument to BartPointMassTreePrior constructor.  "
                     "'location' must be non-negative.");
      }
    }

    virtual double evaluate_log_prior(int number_of_trees) const {
      return (number_of_trees == location_) ? 0.0 : negative_infinity();
    }

   private:
    int location_;
  };

  //----------------------------------------------------------------------
  // Factory function for creating the prior for number of trees.
  BartTreeCountPrior  CreatePriorOnNumberOfTrees(SEXP r_number_of_trees_prior) {
    if (Rf_inherits(r_number_of_trees_prior, "PointMassPrior")) {
      double location = Rf_asReal(getListElement(
          r_number_of_trees_prior, "location"));
      BartTreeCountPrior ans(new BartPointMassTreePrior(lround(location)));
      return ans;
    } else if (Rf_inherits(r_number_of_trees_prior, "PoissonPrior")) {
      double mean = Rf_asReal(getListElement(
          r_number_of_trees_prior, "mean"));
      double lower_limit = Rf_asReal(getListElement(
          r_number_of_trees_prior, "lower.limit"));
      double upper_limit = Rf_asReal(getListElement(
          r_number_of_trees_prior, "upper.limit"));
      BartPoissonTreePrior * impl = new BartPoissonTreePrior(mean);
      if (lower_limit > 0 || upper_limit < std::numeric_limits<int>::max()) {
        impl->truncate(lround(lower_limit), lround(upper_limit));
      }
      return BartTreeCountPrior(impl);
    } else if (Rf_inherits(r_number_of_trees_prior, "DiscreteUniformPrior")) {
      double lower_limit = Rf_asReal(getListElement(
          r_number_of_trees_prior, "lower.limit"));
      double upper_limit = Rf_asReal(getListElement(
          r_number_of_trees_prior, "upper.limit"));
      return BartTreeCountPrior(new BartDiscreteUniformTreePrior(
          lround(lower_limit), lround(upper_limit)));
    } else {
      report_error("Unrecognized class in CreatePriorOnNumberOfTrees.");
      return BartTreeCountPrior(NULL);
    }
  }

  class TreeSizeDistributionCallback : public VectorIoCallback {
   public:
    explicit TreeSizeDistributionCallback(Ptr<BartModelBase> model)
        : model_(model) {}

    virtual int dim() const {
      return 8;
    }

    virtual Vector get_vector() const {
      Vector size_distribution(8);
      int number_of_trees = model_->number_of_trees();
      std::vector<int> number_of_nodes(number_of_trees);
      for (int i = 0; i < number_of_trees; ++i) {
        number_of_nodes[i] = model_->tree(i)->number_of_nodes();
      }
      std::sort(number_of_nodes.begin(), number_of_nodes.end());
      size_distribution[0] = number_of_trees;
      size_distribution[1] = number_of_nodes.front();
      size_distribution[2] = GetQuantile(number_of_nodes, .1);
      size_distribution[3] = GetQuantile(number_of_nodes, .25);
      size_distribution[4] = GetQuantile(number_of_nodes, .5);
      size_distribution[5] = GetQuantile(number_of_nodes, .75);
      size_distribution[6] = GetQuantile(number_of_nodes, .9);
      size_distribution[7] = number_of_nodes.back();
      return size_distribution;
    }

    double GetQuantile(const std::vector<int> &numbers_of_nodes,
                       double quantile) const {
      int number_of_trees = numbers_of_nodes.size();
      double pos = quantile * (number_of_trees - 1);
      double lo = numbers_of_nodes[lround(floor(pos))];
      double hi = numbers_of_nodes[lround(ceil(pos))];
      return .5 * (hi + lo);
    }

   private:
    Ptr<BartModelBase> model_;
  };


  //----------------------------------------------------------------------
  // The primary model factory for use with MCMC.
  BartModelAndSampler CreateBartModel(
      RListIoManager *io_manager,
      SEXP r_number_of_trees,
      SEXP r_design_matrix,
      SEXP r_response,
      SEXP r_family,
      SEXP r_tree_prior,
      SEXP r_discrete_distribution_limit,
      SEXP r_continuous_distribution_strategy) {
    BartModelAndSampler ans;
    Ptr<BartModelBase> model;
    Matrix design_matrix(ToBoomMatrix(r_design_matrix));
    int number_of_trees = Rf_asInteger(r_number_of_trees);

    std::string family = ToString(r_family);

    double tree_prior_alpha = Rf_asReal(getListElement(
        r_tree_prior, "root.split.probability"));
    double tree_prior_beta = Rf_asReal(getListElement(
        r_tree_prior, "split.decay.rate"));
    double total_prediction_sd = Rf_asReal(getListElement(
        r_tree_prior, "total.prediction.sd"));

    std::function<double(int)> log_prior_on_number_of_trees =
        CreatePriorOnNumberOfTrees(getListElement(
            r_tree_prior, "number.of.trees.prior"));

    int discrete_distribution_limit = Rf_asInteger(
        r_discrete_distribution_limit);
    Bart::ContinuousCutpointStrategy cutpoint_strategy =
        GetStrategy(r_continuous_distribution_strategy);

    if (family == "gaussian") {
      //------------------------------------------------------------
      BOOM::Vector response_vector(ToBoomVector(r_response));
      NEW(GaussianBartModel, gaussian_bart_model)(
          number_of_trees,
          response_vector,
          design_matrix);
      model = gaussian_bart_model;
      model->finalize_data(discrete_distribution_limit, cutpoint_strategy);
      RInterface::SdPrior sigma_prior_spec(
          getListElement(r_tree_prior, "sigma.prior"));

      Ptr<GaussianBartPosteriorSampler> sampler(
          new GaussianBartPosteriorSampler(
              gaussian_bart_model.get(),
              sigma_prior_spec.prior_guess(),
              sigma_prior_spec.prior_df(),
              total_prediction_sd,
              tree_prior_alpha,
              tree_prior_beta,
              log_prior_on_number_of_trees));
      ans.sampler = sampler;
      model->set_method(sampler);
      sampler->check_residuals();
      io_manager->add_list_element(
          new StandardDeviationListElement(
              gaussian_bart_model->Sigsq_prm(),
              "sigma"));
      io_manager->add_list_element(
          new NativeVectorListElement(
              new ResidualRecorderCallback(sampler),
              "residuals",
              NULL));
    } else if (family == "poisson") {
      //------------------------------------------------------------
      int *response_array_begin = INTEGER(r_response);
      int *response_array_end = response_array_begin + LENGTH(r_response);
      std::vector<int> response_vector(response_array_begin,
                                       response_array_end);
      NEW(PoissonBartModel, poisson_bart_model)(
          number_of_trees,
          response_vector,
          design_matrix);
      model = poisson_bart_model;
      model->finalize_data();
      NEW(PoissonBartPosteriorSampler, sampler)(
          poisson_bart_model.get(),
          total_prediction_sd,
          tree_prior_alpha,
          tree_prior_beta,
          log_prior_on_number_of_trees);
      ans.sampler = sampler;
      poisson_bart_model->set_method(sampler);
      sampler->check_residuals();
    } else if (family == "probit") {
      //------------------------------------------------------------
      NEW(ProbitBartModel, probit_model)(number_of_trees);
      model = probit_model;
      Matrix response_matrix(ToBoomMatrix(r_response));
      for (int i = 0; i < nrow(response_matrix); ++i) {
        int successes = lround(response_matrix(i, 0));
        int trials = lround(response_matrix(i, 1));
        ConstVectorView predictors(design_matrix.row(i));
        NEW(BinomialRegressionData, dp)(successes, trials, predictors);
        probit_model->add_data(dp);
      }
      model->finalize_data(discrete_distribution_limit, cutpoint_strategy);
      NEW(ProbitBartPosteriorSampler, sampler)(
          probit_model.get(),
          total_prediction_sd,
          tree_prior_alpha,
          tree_prior_beta,
          log_prior_on_number_of_trees);
      ans.sampler = sampler;
      model->set_method(sampler);
      sampler->check_residuals();
    } else if (family == "logit") {
      //------------------------------------------------------------
      NEW(LogitBartModel, logit_model)(number_of_trees);
      model = logit_model;
      Matrix response_matrix(ToBoomMatrix(r_response));
      for (int i = 0; i < nrow(response_matrix); ++i) {
        int successes = lround(response_matrix(i, 0));
        int trials = lround(response_matrix(i, 1));
        ConstVectorView predictors(design_matrix.row(i));
        NEW(BinomialRegressionData, dp)(successes, trials, predictors);
        logit_model->add_data(dp);
      }
      model->finalize_data(discrete_distribution_limit, cutpoint_strategy);
      NEW(LogitBartPosteriorSampler, sampler)(
          logit_model.get(),
          total_prediction_sd,
          tree_prior_alpha,
          tree_prior_beta,
          log_prior_on_number_of_trees);
      ans.sampler = sampler;
      model->set_method(sampler);
      sampler->check_residuals();
    } else {
      // Should never get here.  Errors like this should be caught in
      // R.
      report_error("An unknonwn family was passed to BoomBart.");
    }

    io_manager->add_list_element(new TreeEnsembleListElement(model, "trees"));
    io_manager->add_list_element(new NativeVectorListElement(
        new TreeSizeDistributionCallback(model),
        "tree.size.distribution",
        NULL));

    ans.model = model;
    return ans;
  }

}  // namespace

extern "C" {
  SEXP boom_bart_wrapper_(SEXP r_number_of_trees,
                          SEXP r_design_matrix,
                          SEXP r_response,
                          SEXP r_family,
                          SEXP r_tree_prior,
                          SEXP r_discrete_distribution_limit,
                          SEXP r_continuous_distribution_strategy,
                          SEXP r_niter,
                          SEXP r_ping,
                          SEXP r_seed) {
    try {
      BOOM::RInterface::seed_rng_from_R(r_seed);
      RListIoManager io_manager;

      BartModelAndSampler model_and_sampler =
          CreateBartModel(
              &io_manager,
              r_number_of_trees,
              r_design_matrix,
              r_response,
              r_family,
              r_tree_prior,
              r_discrete_distribution_limit,
              r_continuous_distribution_strategy);

      BOOM::Ptr<BOOM::BartModelBase> model = model_and_sampler.model;
      BOOM::Ptr<BOOM::BartPosteriorSamplerBase>
          sampler = model_and_sampler.sampler;

      int niter = Rf_asInteger(r_niter);
      SEXP ans;
      PROTECT(ans = io_manager.prepare_to_write(niter));
      int ping = Rf_asInteger(r_ping);

      for (int i = 0; i < niter; ++i) {
        R_CheckUserInterrupt();
        print_R_timestamp(i, ping);
        model->sample_posterior();
        io_manager.write();
      }

      SEXP r_variable_summaries;
      PROTECT(r_variable_summaries = WriteVariableSummaries(model));

      SEXP r_MH_performance;
      PROTECT(r_MH_performance = ToRMatrix(sampler->move_accounting().to_matrix()));

      std::vector<SEXP> list_elements;
      list_elements.push_back(r_variable_summaries);
      list_elements.push_back(r_MH_performance);
      std::vector<std::string> element_names;
      element_names.push_back("variable.summaries");
      element_names.push_back("MH.performance");
      UNPROTECT(3);
      return appendListElements(ans, list_elements, element_names);
    } catch (std::exception &e) {
      BOOM::RInterface::handle_exception(e);
    } catch (...) {
      BOOM::RInterface::handle_unknown_exception();
    }
    return R_NilValue;
  }

  //======================================================================

  // The primary interface into the predict method.
  // Args:
  //   r_object:  The model object returnd by BoomBart.
  //   r_newdata_model_matrix: An R matrix of observations at which to
  //     make predictions.
  //   r_burn:  The number of MCMC observations to discard as burn-in.
  //   r_seed:  The seed to use for the BOOM random number generator.
  // Returns:
  //   An R matrix containing draws from the posterior predictive
  //   distribution of either future oservations or function values.
  //   The number of rows is the number of MCMC observations, minus
  //   burn-in.  The number of columns is the number of rows in
  //   newdata.
  SEXP boom_bart_prediction_wrapper_(
      SEXP r_object,
      SEXP r_newdata_model_matrix,
      SEXP r_burn,
      SEXP r_thin) {
    RListIoManager io_manager;

    BOOM::Ptr<BOOM::BartModelBase> model =
        ReCreateBartModel(r_object, &io_manager);
    int burn = std::max<int>(0, Rf_asInteger(r_burn));
    int thin = std::max<int>(1, Rf_asInteger(r_thin));
    int number_of_mcmc_draws =
        Rf_length(BOOM::getListElement(r_object, "trees"));

    int number_of_prediction_draws =
        (number_of_mcmc_draws - burn) / thin;

    BOOM::Matrix new_observations(
        BOOM::ToBoomMatrix(r_newdata_model_matrix));

    BOOM::Matrix predictions(number_of_prediction_draws,
                             nrow(new_observations));

    for (int iteration = 0; iteration < burn; ++iteration) {
      io_manager.stream();
    }
    for (int iteration = burn, prediction_row = 0;
         iteration < number_of_mcmc_draws; ++iteration) {
      io_manager.stream();
      if (iteration % thin == 0) {
        for (int i = 0; i < nrow(new_observations); ++i) {
          const BOOM::ConstVectorView x(new_observations.row(i));
          predictions(prediction_row, i) = model->predict(x);
        }
        ++prediction_row;
      }
    }
    return BOOM::ToRMatrix(predictions);
  }

  //======================================================================

  // This is the interface to the partial dependence plot.  It returns
  // the matrix of draws to be plotted.
  SEXP boom_bart_partial_dependence_plot_wrapper_(
      SEXP r_object,
      SEXP r_which_variable_integer,
      SEXP r_newdata_model_matrix,
      SEXP r_burn,
      SEXP r_thin) {
    RListIoManager io_manager;
    BOOM::Ptr<BOOM::BartModelBase> model =
        ReCreateBartModel(r_object, &io_manager);

    int number_of_mcmc_draws = Rf_length(BOOM::getListElement(
        r_object, "trees"));
    int burn = Rf_asInteger(r_burn);
    int thin = std::max<int>(1, Rf_asInteger(r_thin));
    int which_variable = Rf_asInteger(r_which_variable_integer);
    // The R user is thinking in terms of a 1-based counting system.
    // Convert to 0-based.
    --which_variable;

    Bart::SerializedVariableSummary serialized =
        model->variable_summary(which_variable).serialize();
    Vector x_range;
    if (serialized.is_continuous) {
      switch (serialized.strategy) {
        case UNIFORM_CONTINUOUS: {
          x_range.resize(100);
          double hi = serialized.data[1];
          double lo = serialized.data[0];
          double dx = (hi - lo) / x_range.size();
          x_range[0] = lo;
          for (int i = 1; i < x_range.size(); ++i) {
            x_range[i] = x_range[i-1] + dx;
          }
          break;
        }
        case UNIFORM_DISCRETE:
          x_range = serialized.data;
          break;
        default:
          report_error("Unknown strategy enum in partial dependence plot.");
          break;
      }
    } else {
      x_range = serialized.data;
    }

    BOOM::Matrix design_matrix(ToBoomMatrix(r_newdata_model_matrix));
    int number_of_observations = nrow(design_matrix);
    int number_of_useful_iterations = (number_of_mcmc_draws - burn) / thin;

    BOOM::Matrix predictions(number_of_useful_iterations, x_range.size());

    // std::vector<int> array_dims(3);
    // array_dims[0] = number_of_useful_iterations;
    // array_dims[1] = x_range.size();
    // array_dims[2] = number_of_observations;
    // BOOM::Array debug_predictions(array_dims);

    // Iterate over iteration, data, and values of x
    for (int iteration = 0; iteration < burn; ++iteration) {
      io_manager.stream();
    }
    for (int iteration = burn, prediction_row = -1;
         iteration < number_of_mcmc_draws; ++iteration) {
      io_manager.stream();
      if (iteration % thin == 0) {
        ++prediction_row;
        for (int obs = 0; obs < number_of_observations; ++obs) {
          Vector x = design_matrix.row(obs);
          for (int x_index = 0; x_index < x_range.size(); ++x_index) {
            x[which_variable] = x_range[x_index];
            double pred = model->predict(x);
            predictions(prediction_row, x_index) += pred;
          }
        }
      }
    }
    predictions /= number_of_observations;

    SEXP ans;
    PROTECT(ans = Rf_allocVector(VECSXP, 2));
    SET_VECTOR_ELT(ans, 0, BOOM::ToRMatrix(predictions));
    SET_VECTOR_ELT(ans, 1, BOOM::ToRVector(x_range));
    std::vector<std::string> names(2);
    names[0] = "draws";
    names[1] = "x";
    BOOM::setListNames(ans, names);
    UNPROTECT(1);
    return ans;
  }

}  // extern "C"
