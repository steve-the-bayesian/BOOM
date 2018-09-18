// Copyright 2018 Steven L. Scott. All Rights Reserved.

#include "Models/Nnet/GaussianFeedForwardNeuralNetwork.hpp"
#include "Models/Nnet/PosteriorSamplers/GaussianFeedForwardPosteriorSampler.hpp"

#include "Models/Glm/PosteriorSamplers/BregVsSampler.hpp"
#include "Models/ChisqModel.hpp"
#include "Models/MvnModel.hpp"
#include "Models/MvnGivenScalarSigma.hpp"

#include "Models/Glm/PosteriorSamplers/BinomialLogitCompositeSpikeSlabSampler.hpp"


#include "r_interface/boom_r_tools.hpp"
#include "r_interface/handle_exception.hpp"
#include "r_interface/list_io.hpp"
#include "r_interface/print_R_timestamp.hpp"
#include "r_interface/prior_specification.hpp"
#include "r_interface/seed_rng_from_R.hpp"
#include "utils.h"

#include "cpputil/Ptr.hpp"

namespace {
  using namespace BOOM;
  using namespace BOOM::RInterface;
  using namespace BOOM::Nnet;

  void SetHiddenLayerPriors(Ptr<HiddenLayer> layer, SEXP r_layer) {
    SEXP r_prior = getListElement(r_layer, "prior");
    Ptr<MvnBase> slab;
    Ptr<VariableSelectionPrior> spike;
    bool allow_model_selection = false;
    int max_flips = -1;
    if (Rf_inherits(r_prior, "MvnPrior")) {
      MvnPrior prior_spec(r_prior);
      slab = new MvnModel(prior_spec.mu(), prior_spec.Sigma());
      spike = new VariableSelectionPrior(prior_spec.mu().size(), true);
      
    } else if (Rf_inherits(r_prior, "SpikeSlabGlmPrior")) {
      SpikeSlabGlmPrior prior_spec(r_prior);
      slab = prior_spec.slab();
      spike = prior_spec.spike();
      allow_model_selection = true;
      max_flips = prior_spec.max_flips();
    } else {
      report_error("Unrecognized object passed as a prior distribution for "
                   "hidden layer parameters.");
    }

    for (int i = 0; i < layer->output_dimension(); ++i) {
      NEW(BinomialLogitCompositeSpikeSlabSampler, sampler)(
          layer->logistic_regression(i).get(),
          slab, spike, 5, 3, 10);
      if (max_flips > 0) {
        sampler->limit_model_selection(max_flips);
      }
      sampler->allow_model_selection(allow_model_selection);
      layer->logistic_regression(i)->set_method(sampler);
    }
  }

  class HiddenLayerParametersCallback
      : public RListOfMatricesListElement::Callback {
   public:
    HiddenLayerParametersCallback(GaussianFeedForwardNeuralNetwork *model)
        : model_(model) {}
    Matrix get(int layer) override {
      Ptr<HiddenLayer> lyr = model_->hidden_layer(layer);
      Matrix ans(lyr->input_dimension(), lyr->output_dimension());
      for (int i = 0; i < lyr->output_dimension(); ++i) {
        ans.col(i) = lyr->logistic_regression(i)->Beta();
      }
      return ans;
    }
    void put(int layer, const ConstArrayView &values) override {
      Ptr<HiddenLayer> lyr = model_->hidden_layer(layer);
      for (int i = 0; i < lyr->output_dimension(); ++i) {
        lyr->logistic_regression(i)->set_Beta(
            values.vector_slice(-1, i));
      }
    }
    
   private:
    GaussianFeedForwardNeuralNetwork *model_;
  };
  
  void SetHiddenLayerIo(Ptr<GaussianFeedForwardNeuralNetwork> model,
                        RListIoManager *io_manager) {
    std::vector<int> rows;
    std::vector<int> cols;
    for (int i = 0; i < model->number_of_hidden_layers(); ++i) {
      Ptr<HiddenLayer> layer = model->hidden_layer(i);
      rows.push_back(layer->input_dimension());
      cols.push_back(layer->output_dimension());
    }
    io_manager->add_list_element(
        new RListOfMatricesListElement(
            "hidden.layers", rows, cols,
            new HiddenLayerParametersCallback(model.get())));
  }

  void SetTerminalLayerIo(Ptr<GaussianFeedForwardNeuralNetwork> model,
                          RListIoManager *io_manager) {
    io_manager->add_list_element(new GlmCoefsListElement(
        model->terminal_layer()->coef_prm(),
        "terminal.layer.coefficients"));
    io_manager->add_list_element(new StandardDeviationListElement(
        model->terminal_layer()->Sigsq_prm(),
        "residual.sd"));
  }
  
  void SetTerminalLayerPrior(Ptr<GaussianFeedForwardNeuralNetwork> model,
                             SEXP r_prior) {
    if (Rf_inherits(r_prior, "SpikeSlabPrior")) {
      RegressionConjugateSpikeSlabPrior prior_spec(
          r_prior, model->terminal_layer()->Sigsq_prm());
      NEW(BregVsSampler, sampler)(model->terminal_layer().get(),
                                  prior_spec.slab(),
                                  prior_spec.siginv_prior(),
                                  prior_spec.spike());
      sampler->limit_model_selection(prior_spec.max_flips());
      if (std::isfinite(prior_spec.sigma_upper_limit()) &&
          prior_spec.sigma_upper_limit() > 0) {
        sampler->set_sigma_upper_limit(prior_spec.sigma_upper_limit());
      }
      model->terminal_layer()->set_method(sampler);
    } else {
      report_error("Unrecognized object passed in place of prior distribution "
                   "for terminal layer.");
    }
  }
  
  Ptr<GaussianFeedForwardNeuralNetwork> SpecifyNnetModel(
      SEXP r_predictors,
      SEXP r_response,
      SEXP r_layers,
      SEXP r_prior,
      RListIoManager *io_manager) {
    ConstSubMatrix predictors(ToBoomMatrixView(r_predictors));
    ConstVectorView response(ToBoomVectorView(r_response));

    NEW(GaussianFeedForwardNeuralNetwork, model)();
    if (predictors.nrow() != response.size()) {
      std::ostringstream err;
      err << "Length of response (" << response.size()
          << ") does not match the number of predictor rows("
          << predictors.nrow() << ").";
      report_error(err.str());
    }

    for (int i = 0; i < response.size(); ++i) {
      NEW(RegressionData, data_point)(response[i], predictors.row(i));
      model->add_data(data_point);
    }  

    int number_of_layers = Rf_length(r_layers);
    int input_dimension = predictors.ncol();
    for (int i = 0; i < number_of_layers; ++i) {
      SEXP r_layer = VECTOR_ELT(r_layers, i);
      if (!Rf_inherits(r_layer, "HiddenLayerSpecification")) {
        report_error("Unknonwn object passed where HiddenLayerSpecification "
                     "expected.");
      }
      int output_dimension = Rf_asInteger(getListElement(
          r_layer, "number.of.nodes"));
      NEW(HiddenLayer, layer)(input_dimension, output_dimension);
      SetHiddenLayerPriors(layer, r_layer);
      model->add_layer(layer);
      input_dimension = output_dimension;
    }
    //    model->finalize_network_structure();
    SetHiddenLayerIo(model, io_manager);
    SetTerminalLayerPrior(model, r_prior);
    SetTerminalLayerIo(model, io_manager);
    return model;
  }

}  // namespace

extern "C" {
  using namespace BOOM;  // NOLINT

  SEXP analysis_common_r_do_feedforward(SEXP r_predictors,
                                        SEXP r_response,
                                        SEXP r_layers,
                                        SEXP r_prior,
                                        SEXP r_niter,
                                        SEXP r_ping,
                                        SEXP r_seed) {
    RErrorReporter error_reporter;
    RMemoryProtector protector;
    try {
      seed_rng_from_R(r_seed);
      RListIoManager io_manager;
      Ptr<Model> model = SpecifyNnetModel(
          r_predictors,
          r_response,
          r_layers,
          r_prior,
          &io_manager);

      int niter = Rf_asInteger(r_niter);
      int ping = Rf_asInteger(r_ping);
      SEXP ans = protector.protect(io_manager.prepare_to_write(niter));
      for (int i = 0; i < niter; ++i) {
        if (RCheckInterrupt()) {
          error_reporter.SetError("Canceled by user.");
          return R_NilValue;
        }
        print_R_timestamp(i, ping);
        model->sample_posterior();
        io_manager.write();
      }
      return ans;
    } catch(std::exception &e) {
      handle_exception(e);
    } catch (...) {
      handle_unknown_exception();
    }
    return R_NilValue;
  }

}  // extern "C"
