// Copyright 2011 Google Inc. All Rights Reserved.
// Author: stevescott@google.com (Steve Scott)

#include <r_interface/create_state_model.hpp>
#include <string>
#include <cpputil/report_error.hpp>
#include <cpputil/Date.hpp>

#include <r_interface/boom_r_tools.hpp>
#include <r_interface/list_io.hpp>
#include <r_interface/prior_specification.hpp>

#include <Models/ChisqModel.hpp>
#include <Models/GaussianModel.hpp>
#include <Models/ZeroMeanGaussianModel.hpp>
#include <Models/PosteriorSamplers/FixedSpdSampler.hpp>
#include <Models/PosteriorSamplers/FixedUnivariateSampler.hpp>
#include <Models/PosteriorSamplers/GammaPosteriorSampler.hpp>
#include <Models/PosteriorSamplers/IndependentMvnVarSampler.hpp>
#include <Models/PosteriorSamplers/ZeroMeanGaussianConjSampler.hpp>
#include <Models/PosteriorSamplers/ZeroMeanMvnIndependenceSampler.hpp>

#include <Models/StateSpace/PosteriorSamplers/DynamicRegressionPosteriorSampler.hpp>
#include <Models/StateSpace/PosteriorSamplers/StudentLocalLinearTrendPosteriorSampler.hpp>
#include <Models/StateSpace/StateModels/ArStateModel.hpp>
#include <Models/StateSpace/StateModels/DynamicRegressionStateModel.hpp>
#include <Models/StateSpace/StateModels/Holiday.hpp>
#include <Models/StateSpace/StateModels/LocalLevelStateModel.hpp>
#include <Models/StateSpace/StateModels/LocalLinearTrend.hpp>
#include <Models/StateSpace/StateModels/SemilocalLinearTrend.hpp>
#include <Models/StateSpace/StateModels/RandomWalkHolidayStateModel.hpp>
#include <Models/StateSpace/StateModels/SeasonalStateModel.hpp>
#include <Models/StateSpace/StateModels/StateModel.hpp>
#include <Models/StateSpace/StateModels/StudentLocalLinearTrend.hpp>
#include <Models/StateSpace/StateModels/TrigStateModel.hpp>

#include <Models/TimeSeries/NonzeroMeanAr1Model.hpp>
#include <Models/TimeSeries/PosteriorSamplers/NonzeroMeanAr1Sampler.hpp>
#include <Models/TimeSeries/PosteriorSamplers/ArPosteriorSampler.hpp>
#include <Models/TimeSeries/PosteriorSamplers/ArSpikeSlabSampler.hpp>

namespace BOOM{
  namespace RInterface{

    StateModelFactory::StateModelFactory(RListIoManager * io_manager,
                                         StateSpaceModelBase * model)
        : io_manager_(io_manager),
          model_(model)
    {}

    void StateModelFactory::AddState(SEXP r_state_specification_list,
                                     const std::string &prefix) {
      if (!model_) return;
      CallbackVector callbacks;
      int number_of_state_models = Rf_length(r_state_specification_list);
      for (int i = 0; i < number_of_state_models; ++i) {
        model_->add_state(
            CreateStateModel(VECTOR_ELT(r_state_specification_list, i),
                             prefix,
                             &callbacks));
      }
      for (int i = 0; i < callbacks.size(); ++i) {
        callbacks[i](model_);
      }
    }

    // A factory function that unpacks information from an R object
    // created by AddXXX (where XXX is the name of a type of state
    // model), and use it to build the appropriate BOOM StateModel.  The
    // specific R function associated with each method is noted in the
    // comments to the worker functions that implement each specific
    // type.
    // Args:
    //   list_arg:  The R object created by AddXXX
    //   prefix: An optional prefix to be prepended to the name of the
    //     state component in the io_manager.
    //   callbacks: A vector of callbacks to be executed once all
    //     state has been added.  Each callback will be passed a
    //     pointer to the StateSpaceModelBase that is receiving the
    //     state models being created here.  This is useful for
    //     storing elements of state that won't be computable until
    //     after all state models have been added.
    // Returns:
    //   A BOOM smart pointer to the appropriately typed StateModel.
    Ptr<StateModel> StateModelFactory::CreateStateModel(
        SEXP list_arg, const std::string &prefix, CallbackVector * callbacks) {
      if (Rf_inherits(list_arg, "LocalLinearTrend")) {
        return CreateLocalLinearTrend(list_arg, prefix);
      } else if (Rf_inherits(list_arg, "Seasonal")) {
        return CreateSeasonal(list_arg, prefix);
      } else if (Rf_inherits(list_arg, "SemilocalLinearTrend")) {
        return CreateSemilocalLinearTrend(list_arg, prefix);
      } else if (Rf_inherits(list_arg, "LocalLevel")) {
        return CreateLocalLevel(list_arg, prefix);
      } else if (Rf_inherits(list_arg, "Holiday")) {
        return CreateRandomWalkHolidayStateModel(list_arg, prefix);
      } else if (Rf_inherits(list_arg, "DynamicRegression")) {
        return CreateDynamicRegressionStateModel(list_arg, prefix, callbacks);
      } else if (Rf_inherits(list_arg, "AutoAr")) {
        // AutoAr also inherits from ArProcess, so this case must be
        // handled before ArProcess.
        return CreateAutoArStateModel(list_arg, prefix);
      } else if (Rf_inherits(list_arg, "ArProcess")) {
        return CreateArStateModel(list_arg, prefix);
      } else if (Rf_inherits(list_arg, "StudentLocalLinearTrend")) {
        return CreateStudentLocalLinearTrend(list_arg, prefix);
      } else if (Rf_inherits(list_arg, "Trig")) {
        return CreateTrigStateModel(list_arg, prefix);
      }

      // Should never get here
      report_error("Unknown state model type.");

      Ptr<StateModel> keep_compiler_quiet;
      return keep_compiler_quiet;
    }

    // A callback class for recording the final state that the
    // StateSpaceModelBase sampled in an MCMC iteration.
    class FinalStateCallback : public VectorIoCallback {
     public:
      explicit FinalStateCallback(StateSpaceModelBase *model)
          : model_(model) {}
      virtual int dim() const {return model_->state_dimension();}
      virtual Vector get_vector() const { return model_->final_state();}
     private:
      StateSpaceModelBase * model_;
    };

    void StateModelFactory::SaveFinalState(
        Vector * final_state,
        const std::string & list_element_name) {
      if (!model_) return;
      if (final_state) {
        final_state->resize(model_->state_dimension());
      }
      if (io_manager_) {
        io_manager_->add_list_element(
            new NativeVectorListElement(
                new BOOM::RInterface::FinalStateCallback(model_),
                list_element_name,
                final_state));
      }
    }

    //======================================================================
    LocalLevelStateModel * StateModelFactory::CreateLocalLevel(
        SEXP list_arg, const std::string &prefix) {

      SdPrior sigma_prior_spec(getListElement(list_arg, "sigma.prior"));
      NormalPrior initial_state_prior(getListElement(
          list_arg, "initial.state.prior"));

      LocalLevelStateModel * level(
          new LocalLevelStateModel(sigma_prior_spec.initial_value()));

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
      if (io_manager_) {
        io_manager_->add_list_element(new StandardDeviationListElement(
            level->Sigsq_prm(),
            prefix + "sigma.level"));
      }
      return level;
    }

    //======================================================================
    // See comments to CreateStateModel.  This function expects a
    // list_arg created by R's AddLocalLinearTrend.
    LocalLinearTrendStateModel * StateModelFactory::CreateLocalLinearTrend(
        SEXP list_arg, const std::string &prefix) {

      LocalLinearTrendStateModel * local_linear_trend(
          new LocalLinearTrendStateModel);

      SdPrior level_sigma_prior_spec(
          getListElement(list_arg, "level.sigma.prior"));
      SdPrior slope_sigma_prior_spec(
          getListElement(list_arg, "slope.sigma.prior"));

      //----------------------------------------------------------------------
      // Set the prior for the initial state.
      NormalPrior level_initial_value_prior_spec(
          getListElement(list_arg, "initial.level.prior"));
      NormalPrior slope_initial_value_prior_spec(
          getListElement(list_arg, "initial.slope.prior"));

      Vector initial_state_mean(2);
      initial_state_mean[0] = level_initial_value_prior_spec.mu();
      initial_state_mean[1] = slope_initial_value_prior_spec.mu();
      local_linear_trend->set_initial_state_mean(initial_state_mean);

      SpdMatrix initial_state_variance(2);
      initial_state_variance(0, 0) =
          square(level_initial_value_prior_spec.sigma());
      initial_state_variance(1, 1) =
          square(slope_initial_value_prior_spec.sigma());
      local_linear_trend->set_initial_state_variance(initial_state_variance);

      // Set initial values of model parameters
      SpdMatrix Sigma = local_linear_trend->Sigma();
      Sigma(0, 0) = square(level_sigma_prior_spec.initial_value());
      Sigma(1, 1) = square(slope_sigma_prior_spec.initial_value());
      Sigma(0, 1) = 0;
      Sigma(1, 0) = 0;
      local_linear_trend->set_Sigma(Sigma);

      //----------------------------------------------------------------------
      // Set prior distribution for level_sigma.
      int pos = 0;
      if (level_sigma_prior_spec.fixed()) {
        Ptr<FixedSpdSampler> sampler(
            new FixedSpdSampler(local_linear_trend->Sigma_prm(),
                                square(level_sigma_prior_spec.initial_value()),
                                pos));
        local_linear_trend->set_method(sampler);
      } else {
        Ptr<ZeroMeanMvnIndependenceSampler> sampler(
            new ZeroMeanMvnIndependenceSampler(
                local_linear_trend,
                level_sigma_prior_spec.prior_df(),
                level_sigma_prior_spec.prior_guess(),
                pos));
        if (level_sigma_prior_spec.upper_limit() > 0) {
          sampler->set_sigma_upper_limit(level_sigma_prior_spec.upper_limit());
        }
        local_linear_trend->set_method(sampler);
      }

      //----------------------------------------------------------------------
      // Set prior distribution for slope_sigma.
      pos = 1;
      if (slope_sigma_prior_spec.fixed()) {
        Ptr<FixedSpdSampler> sampler(
            new FixedSpdSampler(local_linear_trend->Sigma_prm(),
                                square(slope_sigma_prior_spec.initial_value()),
                                pos));
        local_linear_trend->set_method(sampler);
      } else {
        Ptr<ZeroMeanMvnIndependenceSampler> sampler(
            new ZeroMeanMvnIndependenceSampler(
                local_linear_trend,
                slope_sigma_prior_spec.prior_df(),
                slope_sigma_prior_spec.prior_guess(),
                pos));
        if (slope_sigma_prior_spec.upper_limit() > 0) {
          sampler->set_sigma_upper_limit(slope_sigma_prior_spec.upper_limit());
        }
        local_linear_trend->set_method(sampler);
      }

      //----------------------------------------------------------------------
      // Now that the priors are all set, the last thing to do is to add
      // the model parameters to the io_manager.
      if (io_manager_) {
        io_manager_->add_list_element(
            new PartialSpdListElement(
                local_linear_trend->Sigma_prm(),
                prefix + "sigma.trend.level",
                0,
                true));

        io_manager_->add_list_element(
            new PartialSpdListElement(
                local_linear_trend->Sigma_prm(),
                prefix + "sigma.trend.slope",
                1,
                true));
      }
      return local_linear_trend;
    }

    //======================================================================
    // Two callback classes for recording the latent weights in the
    // StudentLocalLinearTrend state model.

    // Record weights for the level.
    class StudentLocalLinearTrendLevelWeightCallback
        : public VectorIoCallback {
     public:
      StudentLocalLinearTrendLevelWeightCallback(
          StudentLocalLinearTrendStateModel *model) : model_(model) {}
      virtual int dim() const { return model_->latent_level_weights().size(); }
      virtual Vector get_vector() const {
        return model_->latent_level_weights();
      }
     private:
      StudentLocalLinearTrendStateModel *model_;
    };

    class StudentLocalLinearTrendSlopeWeightCallback
        : public VectorIoCallback {
     public:
      StudentLocalLinearTrendSlopeWeightCallback(
          StudentLocalLinearTrendStateModel *model) : model_(model) {}
      virtual int dim() const { return model_->latent_level_weights().size(); }
      virtual Vector get_vector() const {
        return model_->latent_slope_weights();
      }
     private:
      StudentLocalLinearTrendStateModel *model_;
    };

    StudentLocalLinearTrendStateModel *
    StateModelFactory::CreateStudentLocalLinearTrend(
        SEXP list_arg, const std::string &prefix) {

      StudentLocalLinearTrendStateModel * robust_local_linear_trend(
          new StudentLocalLinearTrendStateModel(1, 10, 1, 10));

      //----------------------------------------------------------------------
      // Unpack the prior and create the posterior sampler.
      SdPrior level_sigma_prior_spec(
          getListElement(list_arg, "level.sigma.prior"));
      NEW(ChisqModel, level_sigma_prior)(
          level_sigma_prior_spec.prior_df(),
          level_sigma_prior_spec.prior_guess());
      SdPrior slope_sigma_prior_spec(
          getListElement(list_arg, "slope.sigma.prior"));
      NEW(ChisqModel, slope_sigma_prior)(
          slope_sigma_prior_spec.prior_df(),
          slope_sigma_prior_spec.prior_guess());
      Ptr<DoubleModel> level_nu_prior(create_double_model(
          getListElement(list_arg, "level.nu.prior")));
      Ptr<DoubleModel> slope_nu_prior(create_double_model(
          getListElement(list_arg, "slope.nu.prior")));

      NEW(StudentLocalLinearTrendPosteriorSampler, sampler)(
          robust_local_linear_trend,
          level_sigma_prior,
          level_nu_prior,
          slope_sigma_prior,
          slope_nu_prior);
      sampler->set_sigma_slope_upper_limit(
          slope_sigma_prior_spec.upper_limit());
      sampler->set_sigma_level_upper_limit(
          level_sigma_prior_spec.upper_limit());
      robust_local_linear_trend->set_method(sampler);

      //----------------------------------------------------------------------
      // Set the prior for the initial state.
      NormalPrior level_initial_value_prior_spec(
          getListElement(list_arg, "initial.level.prior"));
      NormalPrior slope_initial_value_prior_spec(
          getListElement(list_arg, "initial.slope.prior"));

      Vector initial_state_mean(2);
      initial_state_mean[0] = level_initial_value_prior_spec.mu();
      initial_state_mean[1] = slope_initial_value_prior_spec.mu();
      robust_local_linear_trend->set_initial_state_mean(initial_state_mean);

      SpdMatrix initial_state_variance(2);
      initial_state_variance(0, 0) =
          square(level_initial_value_prior_spec.sigma());
      initial_state_variance(1, 1) =
          square(slope_initial_value_prior_spec.sigma());
      robust_local_linear_trend->set_initial_state_variance(
          initial_state_variance);

      //----------------------------------------------------------------------
      // Add parameters to io_manager.
      if (io_manager_) {
        io_manager_->add_list_element(
            new StandardDeviationListElement(
                robust_local_linear_trend->SigsqLevel_prm(),
                prefix + "sigma.trend.level"));
        io_manager_->add_list_element(
            new StandardDeviationListElement(
                robust_local_linear_trend->SigsqSlope_prm(),
                prefix + "sigma.trend.slope"));
        io_manager_->add_list_element(
            new UnivariateListElement(
                robust_local_linear_trend->NuLevel_prm(),
                prefix + "nu.trend.level"));
        io_manager_->add_list_element(
            new UnivariateListElement(
                robust_local_linear_trend->NuSlope_prm(),
                prefix + "nu.trend.slope"));

        bool save_weights = Rf_asInteger(getListElement(
            list_arg, "save.weights"));
        if (save_weights) {
          io_manager_->add_list_element(
              new NativeVectorListElement(
                  new StudentLocalLinearTrendLevelWeightCallback(
                      robust_local_linear_trend),
                  prefix + "trend.level.weights",
                  NULL));

          io_manager_->add_list_element(
              new NativeVectorListElement(
                  new StudentLocalLinearTrendSlopeWeightCallback(
                      robust_local_linear_trend),
                  prefix + "trend.slope.weights",
                  NULL));
        }
      }

      return robust_local_linear_trend;
    }
    //======================================================================
    TrigStateModel *StateModelFactory::CreateTrigStateModel(
        SEXP list_arg, const std::string &prefix) {
      double period = Rf_asReal(getListElement(list_arg, "period"));
      Vector frequencies = ToBoomVector(getListElement(
          list_arg, "frequencies"));
      TrigStateModel * trig_state_model(
          new TrigStateModel(period, frequencies));

      //-------------- set the prior and the posterior sampler.
      SdPrior sigma_prior(getListElement(list_arg, "sigma.prior"));
      int dimension = trig_state_model->dim();
      NEW(ChisqModel, single_siginv_prior)(
          sigma_prior.prior_df(),
          sigma_prior.prior_guess());
      std::vector<Ptr<GammaModelBase>> priors(dimension, single_siginv_prior);
      double sigma_upper_limit = sigma_prior.upper_limit();
      if (sigma_upper_limit < 0) {
        sigma_upper_limit = infinity();
      }
      Vector sd_max_values(dimension, sigma_upper_limit);
      NEW(IndependentMvnVarSampler, sampler)(
          trig_state_model,
          priors,
          sd_max_values);
      trig_state_model->set_method(sampler);

      //-------------- set the prior for the initial state
      MvnPrior initial_prior(getListElement(list_arg, "initial.state.prior"));
      trig_state_model->set_initial_state_mean(initial_prior.mu());
      trig_state_model->set_initial_state_variance(initial_prior.Sigma());

      //-------------- adjust the io manager.
      if (io_manager_) {
        io_manager_->add_list_element(
            new SdVectorListElement(trig_state_model->Sigsq_prm(),
                                    prefix + "trig.coefficient.sd"));
      }
      return trig_state_model;
    }
    //======================================================================
    SemilocalLinearTrendStateModel *
    StateModelFactory::CreateSemilocalLinearTrend(
        SEXP list_arg, const std::string &prefix) {

      SdPrior level_sigma_prior_spec(getListElement(
          list_arg, "level.sigma.prior"));
      NEW(ZeroMeanGaussianModel, level)(level_sigma_prior_spec.initial_value());

      NormalPrior slope_mean_prior_spec(getListElement(
          list_arg, "slope.mean.prior"));
      Ar1CoefficientPrior slope_ar1_prior_spec(getListElement(
          list_arg, "slope.ar1.prior"));
      SdPrior slope_sd_prior_spec(getListElement(
          list_arg, "slope.sigma.prior"));

      NEW(NonzeroMeanAr1Model, slope)(slope_mean_prior_spec.initial_value(),
                                      slope_ar1_prior_spec.initial_value(),
                                      slope_sd_prior_spec.initial_value());

      SemilocalLinearTrendStateModel *trend
          = new SemilocalLinearTrendStateModel(level, slope);

      // Create the prior for level model.  This prior is simple,
      // because it is for a random walk.
      if (!level_sigma_prior_spec.fixed()) {
        NEW(ZeroMeanGaussianConjSampler, level_sampler)(
            level.get(),
            level_sigma_prior_spec.prior_df(),
            level_sigma_prior_spec.prior_guess());

        if (level_sigma_prior_spec.upper_limit() > 0) {
          level_sampler->set_sigma_upper_limit(
              level_sigma_prior_spec.upper_limit());
        }
        trend->set_method(level_sampler);
      }

      // Now create the prior for the slope model.  The prior has three
      // components: a prior for the long run mean of the slope, a prior
      // for the slope's AR coefficient, and a prior for the standard
      // deviation of the AR1 process.
      NEW(GaussianModel, slope_mean_prior)(slope_mean_prior_spec.mu(),
                                           slope_mean_prior_spec.sigma());

      NEW(GaussianModel, slope_ar_prior)(slope_ar1_prior_spec.mu(),
                                         slope_ar1_prior_spec.sigma());
      NEW(ChisqModel, slope_sigma_prior)(slope_sd_prior_spec.prior_df(),
                                         slope_sd_prior_spec.prior_guess());

      // The components have been created, so we can create the overall
      // prior now.
      NEW(NonzeroMeanAr1Sampler, slope_sampler)(slope.get(),
                                                slope_mean_prior,
                                                slope_ar_prior,
                                                slope_sigma_prior);
      // Optional features of the slope prior...
      // Set an upper limit for sigma, if desired.
      if (slope_sd_prior_spec.upper_limit() > 0) {
        slope_sampler->set_sigma_upper_limit(slope_sd_prior_spec.upper_limit());
      }

      // Force the slope model to be stationarity, if desired.
      if (slope_ar1_prior_spec.force_stationary()) {
        slope_sampler->force_stationary();
      }

      if (slope_ar1_prior_spec.force_positive()) {
        slope_sampler->force_ar1_positive();
      }

      // The slope prior is built and configured.  Pack it in to the
      // trend model.  Note that it goes in the trend model, not the
      // slope model, because it is the trend model's "sample_posterior"
      // method that will be called.
      trend->set_method(slope_sampler);

      NormalPrior level_initial_value_prior(getListElement(
          list_arg, "initial.level.prior"));
      NormalPrior slope_initial_value_prior(getListElement(
          list_arg, "initial.slope.prior"));

      // Finally, the last task is to set the prior for the initial
      // value of the state
      trend->set_initial_level_mean(level_initial_value_prior.mu());
      trend->set_initial_slope_mean(slope_initial_value_prior.mu());
      trend->set_initial_level_sd(level_initial_value_prior.sigma());
      trend->set_initial_slope_sd(slope_initial_value_prior.sigma());

      if (io_manager_) {
        io_manager_->add_list_element(
            new StandardDeviationListElement(
                level->Sigsq_prm(),
                prefix + "trend.level.sd"));

        io_manager_->add_list_element(
            new UnivariateListElement(
                slope->Mu_prm(),
                prefix + "trend.slope.mean"));
        io_manager_->add_list_element(
            new UnivariateListElement(
                slope->Phi_prm(),
                prefix + "trend.slope.ar.coefficient"));
        io_manager_->add_list_element(
            new StandardDeviationListElement(
                slope->Sigsq_prm(),
                prefix + "trend.slope.sd"));
      }
      return trend;
    }

    //======================================================================
    // See comments to CreateStateModel.  This function expects a
    // list_arg created by R's AddSeasonal.
    SeasonalStateModel * StateModelFactory::CreateSeasonal(
        SEXP list_arg, const std::string &prefix) {

      int nseasons = Rf_asInteger(getListElement(
          list_arg, "nseasons"));
      int season_duration = Rf_asInteger(getListElement(
          list_arg, "season.duration"));
      SdPrior sigma_prior_spec(getListElement(
          list_arg, "sigma.prior"));

      SeasonalStateModel * seasonal(
          new SeasonalStateModel(nseasons, season_duration));
      seasonal->set_sigsq(square(sigma_prior_spec.initial_value()));

      // Set prior distribution for initial state.
      SEXP r_initial_state_prior(getListElement(
          list_arg, "initial.state.prior"));
      if (Rf_inherits(r_initial_state_prior, "NormalPrior")) {
        NormalPrior initial_value_prior_spec(r_initial_state_prior);
        seasonal->set_initial_state_variance(
            square(initial_value_prior_spec.sigma()));
      } else if (Rf_inherits(r_initial_state_prior, "MvnDiagonalPrior")) {
        MvnDiagonalPrior initial_value_prior_spec(r_initial_state_prior);
        seasonal->set_initial_state_mean(
            initial_value_prior_spec.mean());
        SpdMatrix variance(initial_value_prior_spec.sd().size());
        variance.set_diag(pow(initial_value_prior_spec.sd(), 2));
        seasonal->set_initial_state_variance(variance);
      }

      // Set prior distribution for variance parameter
      if (sigma_prior_spec.fixed()) {
        Ptr<FixedUnivariateSampler> sampler(
            new FixedUnivariateSampler(
                seasonal->Sigsq_prm(),
                seasonal->sigsq()));
        seasonal->set_method(sampler);
      } else {
        Ptr<ZeroMeanGaussianConjSampler> sampler(
            new ZeroMeanGaussianConjSampler(seasonal,
                                            sigma_prior_spec.prior_df(),
                                            sigma_prior_spec.prior_guess()));

        if (sigma_prior_spec.upper_limit() > 0) {
          sampler->set_sigma_upper_limit(sigma_prior_spec.upper_limit());
        }
        seasonal->set_method(sampler);
      }

      std::ostringstream parameter_name;
      parameter_name  <<  "sigma.seasonal" << "." << nseasons;
      if (season_duration > 1) parameter_name << "." << season_duration;

      // Add information about this parameter to the io_manager
      if (io_manager_) {
        io_manager_->add_list_element(new StandardDeviationListElement(
            seasonal->Sigsq_prm(),
            prefix + parameter_name.str()));
      }
      return seasonal;
    }

    //======================================================================
    // Creates a holiday state model.
    // Args:
    //   list_arg: An R object inheriting from class "Holiday".  This
    //     is a list with a named element "holidays".
    //   prefix: An optional prefix to be prepended to the name of the
    //     state component in the io_manager.
    RandomWalkHolidayStateModel *
    StateModelFactory::CreateRandomWalkHolidayStateModel(
        SEXP list_arg, const std::string &prefix) {

      std::string holiday_name = GetStringFromList(list_arg, "name");
      int days_before = Rf_asInteger(getListElement(list_arg, "days.before"));
      int days_after = Rf_asInteger(getListElement(list_arg, "days.after"));
      Holiday *holiday;
      if (Rf_inherits(list_arg, "NamedHoliday")) {
        holiday = CreateNamedHoliday(holiday_name, days_before, days_after);
      } else if (Rf_inherits(list_arg, "FixedDateHoliday")) {
        MonthNames month_name = str2month(GetStringFromList(list_arg, "month"));
        int holiday_day = Rf_asInteger(getListElement(list_arg, "day"));
        holiday = new FixedDateHoliday(
            month_name, holiday_day, days_before, days_after);
      } else if (Rf_inherits(list_arg, "NthWeekdayInMonthHoliday")) {
        MonthNames month_name = str2month(GetStringFromList(list_arg, "month"));
        DayNames day_name = str2day(GetStringFromList(list_arg, "day.of.week"));
        int which_week = Rf_asInteger(getListElement(list_arg, "which.week"));
        if (which_week > 0) {
          holiday = new NthWeekdayInMonthHoliday(
              which_week, day_name, month_name, days_before, days_after);
        } else {
          holiday = new LastWeekdayInMonthHoliday(
              day_name, month_name, days_before, days_after);
        }
      } else {
        report_error("Unknown type of holiday state model");
        return NULL;
      }
      int month_of_time0 =
          Rf_asInteger(getListElement(list_arg, "time0.month"));
      int day_of_time0 = Rf_asInteger(getListElement(list_arg, "time0.day"));
      int year_of_time0 = Rf_asInteger(getListElement(list_arg, "time0.year"));
      Date time0(month_of_time0, day_of_time0, year_of_time0);
      SdPrior sigma_prior_spec(getListElement(
          list_arg, "sigma.prior"));
      NormalPrior initial_value_prior_spec(getListElement(
          list_arg, "initial.state.prior"));

      RandomWalkHolidayStateModel * holiday_model
          = new RandomWalkHolidayStateModel(holiday, time0);
      holiday_model->set_sigsq(square(sigma_prior_spec.initial_value()));

      //------------------------------------------------------------
      // Set prior distribution for initial state
      Vector initial_state_mean(holiday_model->state_dimension(), 0.0);
      SpdMatrix initial_state_variance(holiday_model->state_dimension());
      initial_state_variance.set_diag(square(initial_value_prior_spec.sigma()));
      holiday_model->set_initial_state_mean(initial_state_mean);
      holiday_model->set_initial_state_variance(initial_state_variance);

      //------------------------------------------------------------
      // Set prior distribution for innovation variance parameter
      if (sigma_prior_spec.fixed()) {
        Ptr<FixedUnivariateSampler> sampler(
            new FixedUnivariateSampler(
                holiday_model->Sigsq_prm(),
                holiday_model->sigsq()));
        holiday_model->set_method(sampler);
      } else {
        Ptr<ZeroMeanGaussianConjSampler> sampler(
            new ZeroMeanGaussianConjSampler(
                holiday_model,
                sigma_prior_spec.prior_df(),
                sigma_prior_spec.prior_guess()));
        holiday_model->set_method(sampler);
      }

      std::ostringstream parameter_name;
      parameter_name  <<  "sigma." << holiday_name;
      // Add information about this parameter to the io_manager
      if (io_manager_) {
        io_manager_->add_list_element(new StandardDeviationListElement(
            holiday_model->Sigsq_prm(),
            prefix + parameter_name.str()));
      }
      return holiday_model;
    }

    //======================================================================
    ArStateModel * StateModelFactory::CreateArStateModel(
        SEXP list_arg, const std::string & prefix) {

      SdPrior sigma_prior_spec(getListElement(list_arg, "sigma.prior"));
      int number_of_lags = Rf_asInteger(getListElement(list_arg, "lags"));
      ArStateModel *state_model(new ArStateModel(number_of_lags));

      NEW(ChisqModel, siginv_prior)(sigma_prior_spec.prior_df(),
                                    sigma_prior_spec.prior_guess());

      NEW(ArPosteriorSampler, sampler)(state_model,
                                       siginv_prior);
      if(sigma_prior_spec.upper_limit() > 0) {
        sampler->set_sigma_upper_limit(sigma_prior_spec.upper_limit());
      }
      state_model->set_method(sampler);

      if (io_manager_) {
        std::ostringstream phi_parameter_name;
        phi_parameter_name << prefix << "AR" << number_of_lags
                           << ".coefficients";
        io_manager_->add_list_element(new GlmCoefsListElement(
            state_model->Phi_prm(),
            phi_parameter_name.str()));

        std::ostringstream sigma_parameter_name;
        sigma_parameter_name << prefix << "AR" << number_of_lags << ".sigma";
        io_manager_->add_list_element(new StandardDeviationListElement(
            state_model->Sigsq_prm(),
            sigma_parameter_name.str()));
      }
      return state_model;
    }
    //======================================================================
    ArStateModel * StateModelFactory::CreateAutoArStateModel(
        SEXP list_arg, const std::string & prefix) {
      int number_of_lags = Rf_asInteger(getListElement(list_arg, "lags"));
      ArStateModel *state_model(new ArStateModel(number_of_lags));

      ArSpikeSlabPrior prior_spec(getListElement(list_arg, "prior"));

      NEW(ArSpikeSlabSampler, sampler)(state_model,
                                       prior_spec.slab(),
                                       prior_spec.spike(),
                                       prior_spec.siginv_prior(),
                                       prior_spec.truncate());
      if (prior_spec.max_flips() > 0) {
        sampler->limit_model_selection(prior_spec.max_flips());
      }

      if (prior_spec.sigma_upper_limit() > 0) {
        sampler->set_sigma_upper_limit(prior_spec.sigma_upper_limit());
      }

      state_model->set_method(sampler);

      if (io_manager_) {
        std::ostringstream phi_parameter_name;
        phi_parameter_name << prefix << "AR" << number_of_lags
                           << ".coefficients";
        std::vector<std::string> column_names;
        for (int i = 0; i < number_of_lags; ++i) {
          ostringstream column_name;
          column_name << "lag." << i + 1;
          column_names.push_back(column_name.str());
        }
        io_manager_->add_list_element(new GlmCoefsListElement(
            state_model->Phi_prm(),
            phi_parameter_name.str(),
            column_names));

        std::ostringstream sigma_parameter_name;
        sigma_parameter_name << prefix << "AR" << number_of_lags << ".sigma";
        io_manager_->add_list_element(new StandardDeviationListElement(
            state_model->Sigsq_prm(),
            sigma_parameter_name.str()));
      }
      return state_model;
    }

    //======================================================================
    // This is a callback designed to be used with a
    // NativeMatrixListElement in an R io_manager.  When this callback
    // is invoked it grabs the dynamic regression component of the
    // model's current state.  It is needed so that the dynamic
    // regression can store the dynamic regression coefficients at
    // each iteration.
    class DynamicRegressionStateCallback : public BOOM::MatrixIoCallback {
     public:
      DynamicRegressionStateCallback(BOOM::StateSpaceModelBase *model,
                                     BOOM::DynamicRegressionStateModel
                                         *state_model,
                                     int model_position)
          : model_(model),
            state_model_(state_model),
            model_position_(model_position) {}

      // There is one row for each dynamic regression coefficient.
      virtual int nrow() const {return state_model_->state_dimension(); }
      virtual int ncol() const {return model_->time_dimension();}
      virtual BOOM::Matrix get_matrix() const {
        return model_->full_state_subcomponent(model_position_);
      }

     private:
      BOOM::StateSpaceModelBase * model_;
      BOOM::DynamicRegressionStateModel * state_model_;
      int model_position_;
    };

    //======================================================================
    // A callback for adding dynamic regression coefficients to the
    // io_manager for a StateSpaceModel.
    class RecordDynamicRegressionCallback {
     public:
      // Args:
      //   model: The state space model that may or may not have a dynamic
      //     regression state component to be recorded.
      //   io_manager: The io_manager in charge of building the list
      //     containing the dynamic regression coefficients.
      RecordDynamicRegressionCallback(
          DynamicRegressionStateModel *dynamic_regression,
          RListIoManager* io_manager)
          : dynamic_regression_(dynamic_regression),
            io_manager_(io_manager) {}

      // Adds an element to the io_manager that records the
      // coefficients of the dynamic regression model.
      void operator()(StateSpaceModelBase* model) {
        if (io_manager_) {
          std::string list_element_name = "dynamic.regression.coefficients";
          int model_position = compute_model_position(model);
          BOOM::NativeMatrixListElement * state_recorder(
              new NativeMatrixListElement(
                  new DynamicRegressionStateCallback(
                      model,
                      dynamic_regression_,
                      model_position),
                  list_element_name.c_str(),
                  NULL));
          state_recorder->set_row_names(dynamic_regression_->xnames());
          io_manager_->add_list_element(state_recorder);
        }
      }

      // Compute the position of the dynamic regression state model in
      // the list of state components.
      int compute_model_position(StateSpaceModelBase *model) const {
        for (int i = 0; i < model->nstate(); ++i) {
          if(model->state_model(i).get() == dynamic_regression_) {
            return i;
          }
        }
        report_error("Could not determine the position of "
                     "DynamicRegressionStateModel.");
        return -1;
      }

     private:
      DynamicRegressionStateModel *dynamic_regression_;
      RListIoManager * io_manager_;
    };

    //======================================================================
    //
    DynamicRegressionStateModel *
    StateModelFactory::CreateDynamicRegressionStateModel(
        SEXP list_arg, const std::string &prefix, CallbackVector *callbacks) {

      SEXP r_design_matrix(getListElement(list_arg, "predictors"));
      Matrix X = ToBoomMatrix(r_design_matrix);
      // Get colnames for X.  The R code should ensure that X has them.
      std::vector<std::string> xnames =
          StringVector(Rf_GetColNames(r_design_matrix));
      if (xnames.empty()) {
        xnames.reserve(ncol(X));
        for (int i = 0; i < ncol(X); ++i) {
          std::ostringstream name_maker;
          name_maker << "V" << i+1;
          xnames.push_back(name_maker.str());
        }
      }

      DynamicRegressionStateModel * dynamic_regression(
          new DynamicRegressionStateModel(X));
      dynamic_regression->set_xnames(xnames);

      Ptr<DoubleModel> sigma_mean_prior =
          create_double_model(getListElement(
              list_arg, "sigma.mean.prior"));
      Ptr<DoubleModel> shrinkage_parameter_prior =
          create_double_model(getListElement(
              list_arg, "shrinkage.parameter.prior"));

      NEW(GammaModel, siginv_prior)(1, 1);
      NEW(GammaPosteriorSampler, hyperparameter_sampler)(
          siginv_prior.get(),
          sigma_mean_prior,
          shrinkage_parameter_prior);
      siginv_prior->set_method(hyperparameter_sampler);

      NEW(DynamicRegressionPosteriorSampler, sampler)(
          dynamic_regression, siginv_prior);
      dynamic_regression->set_method(sampler);

      if (io_manager_) {
        // Store the standard deviations for each variable.
        for (int i = 0; i < ncol(X); ++i) {
          std::ostringstream vname;
          vname << prefix << xnames[i] << ".sigma";
          io_manager_->add_list_element(new StandardDeviationListElement(
              dynamic_regression->Sigsq_prm(i),
              vname.str()));
        }

        // Store the hyperparameters describing the model for 1.0 /
        // sigma^2.
        io_manager_->add_list_element(new UnivariateListElement(
            siginv_prior->Alpha_prm(),
            prefix + "siginv_shape_hyperparameter"));

        io_manager_->add_list_element(new UnivariateListElement(
            siginv_prior->Beta_prm(),
            prefix + "siginv_scale_hyperparameter"));

        if (callbacks) {
          // We need to add a component to the io_manager so that it will
          // record the state of the dynamic regression coefficients.
          // This should be done in a callback so that the returned object
          // has all the model parameters grouped together.  The
          // RecordDynamicRegressionCallback will be invoked after all the
          // components of state have been created.
          RecordDynamicRegressionCallback callback(
              dynamic_regression, io_manager_);
          callbacks->push_back(callback);
        }
      }
      return dynamic_regression;
    }

  }  // namespace RInterface
}  // namespace BOOM
