// Copyright 2011 Google Inc. All Rights Reserved.
// Author: stevescott@google.com (Steve Scott)

#include <memory>

#include "cpputil/report_error.hpp"

#include "Models/BetaModel.hpp"
#include "Models/ChisqModel.hpp"
#include "Models/CompositeModel.hpp"
#include "Models/GammaModel.hpp"
#include "Models/GaussianModel.hpp"
#include "Models/GaussianModelGivenSigma.hpp"
#include "Models/Glm/BinomialLogitModel.hpp"
#include "Models/Glm/PosteriorSamplers/BinomialLogitCompositeSpikeSlabSampler.hpp"
#include "Models/Glm/PosteriorSamplers/BregVsSampler.hpp"
#include "Models/Glm/RegressionModel.hpp"
#include "Models/Glm/VariableSelectionPrior.hpp"
#include "Models/IndependentMvnModel.hpp"
#include "Models/MultinomialModel.hpp"
#include "Models/MvnModel.hpp"
#include "Models/PoissonModel.hpp"
#include "Models/PosteriorSamplers/BetaBinomialSampler.hpp"
#include "Models/PosteriorSamplers/CompositeModelSampler.hpp"
#include "Models/PosteriorSamplers/GaussianConjSampler.hpp"
#include "Models/PosteriorSamplers/IndependentMvnConjSampler.hpp"
#include "Models/PosteriorSamplers/MarkovConjSampler.hpp"
#include "Models/PosteriorSamplers/MultinomialDirichletSampler.hpp"
#include "Models/PosteriorSamplers/MvnConjSampler.hpp"
#include "Models/PosteriorSamplers/PoissonGammaSampler.hpp"
#include "Models/PosteriorSamplers/ZeroInflatedLognormalPosteriorSampler.hpp"
#include "Models/PosteriorSamplers/ZeroInflatedPoissonSampler.hpp"
#include "Models/ProductDirichletModel.hpp"
#include "Models/ZeroInflatedLognormalModel.hpp"
#include "Models/ZeroInflatedPoissonModel.hpp"

#include "r_interface/create_mixture_component.hpp"
#include "r_interface/prior_specification.hpp"

namespace BOOM {
  namespace RInterface {

    //======================================================================
    // An abstract class to manage the building of individual mixture
    // component elements.
    class MixtureComponentBuilder {
     public:
      // Args:
      //   rmixture_component: An R list describing the mixture
      //     component to be created.  Each type of mixture component
      //     corresponds to a different derived type of
      //     MixtureComponentBuilder.  All instances of
      //     rmixture_component must have an element named 'name'
      //     intended to label the variable they are modeling.
      explicit MixtureComponentBuilder(SEXP rmixture_component)
          : rmixture_component_(rmixture_component),
            component_name_prefix_(""),
            state_number_(-1)
      {}

      virtual ~MixtureComponentBuilder() {}

      // Virtual constructor for creating the right type of
      // MixtureComponentBuilder based on the type (in R) of the
      // rmixture_component passed as an argument.
      static MixtureComponentBuilder * Create(SEXP rmixture_component);

      // Create a MixtureComponent of the appropriate type.  This is the
      // primary job the MixtureComponentBuilder was created to perform.
      // Args:
      //   rmixture_component: The R object describing the mixture
      //     component to be built.
      //   io_manager: The object in charge of maintaining the (R-) list
      //     containing Monte Carlo draws from the posterior.
      // Returns:
      //   A BOOM pointer to a mixture component of the appropriate
      //   type.  The parameters of the mixture component will be
      //   recorded by the io_manager.
      virtual Ptr<MixtureComponent> Build(
          SEXP rmixture_component,
          BOOM::RListIoManager *io_manager) const = 0;

      // Create a string containing the name of the mixture component's
      // parameters in the associated io_manager.  There are two
      // strategies for name creation: numeric and text names.  With the
      // numeric strategy, the names will be something like
      // my.variable.mu.0, my.variable.mu.1, my.variable.sigma.0,
      // my.variable.sigma.1.  With text naming, the names will be
      // something like good.my.variable.mu, bad.my.variable.mu,
      // good.my.variable.sigma, bad.my.variable.sigma.
      //
      // Before the Build method is called, one of set_state_number(),
      // or set_component_name_prefix() must be called to indicate which
      // of the strategies to use, and what the prefix or suffix should
      // be.
      std::string create_name(const std::string & parameter_name)const {
        std::ostringstream name_stream;
        std::string component_name
            = GetStringFromList(rmixture_component_, "name");
        if (state_number_ > -1) {
          if (!component_name.empty()) {
            name_stream << component_name << ".";
          }
          name_stream << parameter_name << "." << state_number_;
        } else if (!component_name_prefix_.empty()) {
          name_stream << component_name_prefix_ << ".";
          if (!component_name.empty()) {
            name_stream << component_name << ".";
          }
          name_stream << parameter_name;
        } else {
          report_error("Either state_number_ or "
                       "component_name_prefix_ must be set");
        }
        return name_stream.str();
      }

      // Indicates that the MixtureComponentBuilder should use the
      // numeric naming strategy (see 'create_name' above), and
      // specifies the index of the next mixture component to be
      // created.
      void set_state_number(int n) {
        if (n < -1) n = -1;
        state_number_ = n;
        component_name_prefix_ = "";
      }

      // Indicates that the MixtureComponentBuilder should use the text
      // naming strategy (see 'create_name' above), and specifies the
      // prefix of the next mixture component to be created.
      void set_component_name_prefix(const std::string &prefix) {
        state_number_ = -1;
        component_name_prefix_ = prefix;
      }

     private:
      SEXP rmixture_component_;
      std::string component_name_prefix_;
      int state_number_;
    };

    //======================================================================
    // Creates a mixture component of the appropriate type based on the
    // input from R, using the numeric naming scheme.
    Ptr<MixtureComponent> CreateMixtureComponent(
        SEXP mixture_component,
        int state_number,
        RListIoManager *io_manager) {
      std::unique_ptr<MixtureComponentBuilder> builder(
          MixtureComponentBuilder::Create(mixture_component));
      builder->set_state_number(state_number);
      return builder->Build(mixture_component, io_manager);
    }

    //======================================================================
    // Creates a mixture component of the appropriate type based on the
    // input from R, using the text naming scheme.
    Ptr<MixtureComponent> CreateNamedMixtureComponent(
        SEXP mixture_component,
        const std::string & component_name_prefix,
        RListIoManager *io_manager) {
      std::unique_ptr<MixtureComponentBuilder> builder(
          MixtureComponentBuilder::Create(mixture_component));
      builder->set_component_name_prefix(component_name_prefix);
      return builder->Build(mixture_component, io_manager);
    }

    //----------------------------------------------------------------------
    class NormalMixtureComponentBuilder : public MixtureComponentBuilder {
     public:
      explicit NormalMixtureComponentBuilder(SEXP rmixture_component)
          : MixtureComponentBuilder(rmixture_component)
      {}

      virtual  Ptr<MixtureComponent> Build(
          SEXP rmixture_component,
          RListIoManager *io_manager) const {
        Ptr<GaussianModel> model(new GaussianModel(0, 1));

        // Use the intermediate C++ objects defined in
        // prior_specfication.hpp to unpack the priors passed in by R.
        SEXP prior = getListElement(rmixture_component, "prior");
        double mu_prior_guess = Rf_asReal(getListElement(prior, "mu.guess"));
        double mu_prior_guess_weight = Rf_asReal(
            getListElement(prior, "mu.guess.weight"));
        BOOM::RInterface::SdPrior sd_prior(getListElement(
            prior, "sigma.prior"));
        Ptr<BOOM::GaussianModelGivenSigma> mu_prior(
            new GaussianModelGivenSigma(
                model->Sigsq_prm(),
                mu_prior_guess,
                mu_prior_guess_weight));
        Ptr<ChisqModel> sigma_prior(new ChisqModel(
            sd_prior.prior_df(),
            sd_prior.prior_guess()));
        Ptr<GaussianConjSampler> sampler(new GaussianConjSampler(
            model.get(),
            mu_prior,
            sigma_prior));
        model->set_method(sampler);
        io_manager->add_list_element(
            new BOOM::UnivariateListElement(
                model->Mu_prm(), create_name("mu")));
        io_manager->add_list_element(
            new BOOM::StandardDeviationListElement(
                model->Sigsq_prm(), create_name("sigma")));

        return model;
      }
    };

    //----------------------------------------------------------------------
    class PoissonMixtureComponentBuilder : public MixtureComponentBuilder {
     public:
      explicit PoissonMixtureComponentBuilder(SEXP rmixture_component)
          : MixtureComponentBuilder(rmixture_component) {}

      virtual Ptr<MixtureComponent> Build(
          SEXP rmixture_component,
          RListIoManager *io_manager)const {
        GammaPrior prior_spec(getListElement(rmixture_component, "prior"));
        NEW(PoissonModel, model)(1.0);
        NEW(GammaModel, prior)(prior_spec.a(), prior_spec.b());
        NEW(PoissonGammaSampler, sampler)(model.get(), prior);
        model->set_method(sampler);

        io_manager->add_list_element(new UnivariateListElement(
            model->Lam(), create_name("lambda")));
        return model;
      }
    };

    //----------------------------------------------------------------------
    class ZeroInflatedPoissonMixtureComponentBuilder
        : public MixtureComponentBuilder {
     public:
      explicit ZeroInflatedPoissonMixtureComponentBuilder(
          SEXP rmixture_component)
          : MixtureComponentBuilder(rmixture_component) {}

      virtual Ptr<MixtureComponent> Build(
          SEXP rmixture_component,
          RListIoManager *io_manager)const {

        GammaPrior lambda_prior_spec(getListElement(
            rmixture_component, "gamma.prior"));
        BetaPrior zero_probability_prior_spec(getListElement(
            rmixture_component, "beta.prior"));

        NEW(ZeroInflatedPoissonModel, model)(1, .5);
        NEW(GammaModel, lambda_prior)(
            lambda_prior_spec.a(), lambda_prior_spec.b());
        NEW(BetaModel, zero_probability_prior)(
            zero_probability_prior_spec.a(),
            zero_probability_prior_spec.b());

        NEW(ZeroInflatedPoissonSampler, sampler)(
            model.get(), lambda_prior, zero_probability_prior);
        model->set_method(sampler);

        io_manager->add_list_element(new UnivariateListElement(
            model->Lambda_prm(), create_name("lambda")));
        io_manager->add_list_element(new UnivariateListElement(
            model->ZeroProbability_prm(),
            create_name("zero.probability")));
        return model;
      }
    };

    //----------------------------------------------------------------------
    class RegressionMixtureComponentBuilder : public MixtureComponentBuilder {
     public:
      explicit RegressionMixtureComponentBuilder(SEXP rmixture_component)
          : MixtureComponentBuilder(rmixture_component)
      {}

      virtual Ptr<MixtureComponent> Build(SEXP rmixture_component,
                                          RListIoManager *io_manager) const {
        SEXP r_prior = getListElement(rmixture_component, "prior");
        RegressionNonconjugateSpikeSlabPrior temporary_prior(r_prior);
        int beta_dimension = temporary_prior.slab()->mu().size();
        NEW(RegressionModel, model)(beta_dimension);
        RegressionConjugateSpikeSlabPrior prior(r_prior, model->Sigsq_prm());
        model->set_Beta(prior.slab()->mu());
        model->set_sigsq(prior.siginv_prior()->sigsq());
        NEW(BregVsSampler, sampler)(model.get(),
                                    prior.slab(),
                                    prior.siginv_prior(),
                                    prior.spike());
        if (prior.max_flips() > 0) {
          sampler->limit_model_selection(prior.max_flips());
        }
        model->set_method(sampler);
        io_manager->add_list_element(
            new GlmCoefsListElement(
                model->coef_prm(), create_name("beta")));
        io_manager->add_list_element(
            new StandardDeviationListElement(
                model->Sigsq_prm(), create_name("sigma")));
        return model;
      }
    };

    //----------------------------------------------------------------------
    class LogitMixtureComponentBuilder : public MixtureComponentBuilder {
     public:
      explicit LogitMixtureComponentBuilder(SEXP rmixture_component)
          : MixtureComponentBuilder(rmixture_component)
      {}

      virtual Ptr<MixtureComponent> Build(SEXP rmixture_component,
                                          RListIoManager *io_manager) const {
        SpikeSlabGlmPrior prior(getListElement(rmixture_component, "prior"));
        NEW(BinomialLogitModel, model)(prior.slab()->mu());
        int clt_threshold = 5;
        int proposal_degrees_of_freedom = 3;
        int max_tim_chunk_size = 10;
        int max_rwm_chunk_size = 1;
        double rwm_proposal_variance_scale_factor = .025;
        NEW(BinomialLogitCompositeSpikeSlabSampler, sampler)(
            model.get(),
            prior.slab(),
            prior.spike(),
            clt_threshold,
            proposal_degrees_of_freedom,
            max_tim_chunk_size,
            max_rwm_chunk_size,
            rwm_proposal_variance_scale_factor);
        if (prior.max_flips() > 0) {
          sampler->limit_model_selection(prior.max_flips());
        }
        sampler->reassign_data_each_time(true);
        model->set_method(sampler);
        io_manager->add_list_element(new GlmCoefsListElement(
            model->coef_prm(), create_name("beta")));

        return model;
      }
    };

    //----------------------------------------------------------------------
    class ZeroInflatedLognormalMixtureComponentBuilder
        : public MixtureComponentBuilder {
     public:
      explicit ZeroInflatedLognormalMixtureComponentBuilder(
          SEXP rmixture_component)
          : MixtureComponentBuilder(rmixture_component)
      {}

      virtual Ptr<MixtureComponent> Build(SEXP rmixture_component,
                                          RListIoManager *io_manager) const {
        Ptr<ZeroInflatedLognormalModel> model(
            new ZeroInflatedLognormalModel);

        io_manager->add_list_element(
            new BOOM::UnivariateListElement(
                model->Gaussian_model()->Mu_prm(),
                create_name("mu")));
        io_manager->add_list_element(
            new BOOM::StandardDeviationListElement(
                model->Gaussian_model()->Sigsq_prm(),
                create_name("sigma")));
        io_manager->add_list_element(
            new BOOM::UnivariateListElement(
                model->Binomial_model()->Prob_prm(),
                create_name("nonzero.prob")));

        BOOM::RInterface::BetaPrior beta_prior(
            getListElement(rmixture_component, "beta.prior"));
        BOOM::RInterface::NormalInverseGammaPrior normal_prior(
            getListElement(rmixture_component, "normal.inverse.gamma.prior"));

        NEW(GaussianModelGivenSigma, mu_prior)(
            model->Gaussian_model()->Sigsq_prm(),
            normal_prior.prior_mean_guess(),
            normal_prior.prior_mean_sample_size());
        NEW(ChisqModel, siginv_prior)(
            normal_prior.sd_prior().prior_df(),
            normal_prior.sd_prior().prior_guess());
        NEW(GaussianConjSampler, gaussian_sampler)(
            model->Gaussian_model().get(),
            mu_prior,
            siginv_prior);
        model->Gaussian_model()->set_method(gaussian_sampler);

        NEW(BetaModel, positive_probability_prior)(
            beta_prior.a(), beta_prior.b());
        NEW(BetaBinomialSampler, positive_probability_sampler)(
            model->Binomial_model().get(),
            positive_probability_prior);
        model->Binomial_model()->set_method(positive_probability_sampler);

        NEW(ZeroInflatedLognormalPosteriorSampler, sampler)(
            model.get());
        model->set_method(sampler);

        return model;
      }
    };

    //----------------------------------------------------------------------
    class MvnMixtureComponentBuilder : public MixtureComponentBuilder {
     public:
      explicit MvnMixtureComponentBuilder(SEXP rmixture_component)
          : MixtureComponentBuilder(rmixture_component) {}

      virtual Ptr<MixtureComponent> Build(SEXP rmixture_component,
                                          RListIoManager *io_manager)const{
        NormalInverseWishartPrior prior_spec(
            getListElement(rmixture_component, "prior"));

        NEW(MvnModel, model)(prior_spec.mu_guess(),
                             prior_spec.Sigma_guess(),
                             false);

        NEW(MvnConjSampler, sampler)(model.get(),
                                     prior_spec.mu_guess(),
                                     prior_spec.mu_guess_weight(),
                                     prior_spec.Sigma_guess(),
                                     prior_spec.Sigma_guess_weight());
        model->set_method(sampler);

        io_manager->add_list_element(new VectorListElement(
            model->Mu_prm(), create_name("mu")));
        io_manager->add_list_element(new SpdListElement(
            model->Sigma_prm(), create_name("Sigma")));

        return model;
      }
    };

    //----------------------------------------------------------------------
    class IndependentMvnMixtureComponentBuilder
        : public MixtureComponentBuilder {
     public:
      explicit IndependentMvnMixtureComponentBuilder(SEXP rmixture_component)
          : MixtureComponentBuilder(rmixture_component) {}

      virtual Ptr<MixtureComponent> Build(SEXP rmixture_component,
                                          RListIoManager *io_manager) const {
        Vector prior_mean_guess(ToBoomVector(getListElement(
            rmixture_component, "prior.mean.guess")));

        Vector prior_mean_sample_size(ToBoomVector(getListElement(
            rmixture_component, "prior.mean.sample.size")));

        Vector prior_sd_guess(ToBoomVector(getListElement(
            rmixture_component, "prior.sd.guess")));

        Vector prior_sd_sample_size(ToBoomVector(getListElement(
            rmixture_component, "prior.sd.sample.size")));

        Vector sigma_upper_limit(ToBoomVector(getListElement(
            rmixture_component, "sigma.upper.limit")));
        for (int i = 0; i < sigma_upper_limit.size(); ++i) {
          if (sigma_upper_limit[i] == R_PosInf) {
            sigma_upper_limit[i] = infinity();
          }
        }

        NEW(IndependentMvnModel, model)(prior_mean_guess.size());

        NEW(IndependentMvnConjSampler, sampler)(model.get(),
                                                prior_mean_guess,
                                                prior_mean_sample_size,
                                                prior_sd_guess,
                                                prior_sd_sample_size,
                                                sigma_upper_limit);
        model->set_method(sampler);

        io_manager->add_list_element(new VectorListElement(
            model->Mu_prm(), create_name("mu")));

        io_manager->add_list_element(new SdVectorListElement(
            model->Sigsq_prm(), create_name("sigma")));

        return model;
      }
    };

    //----------------------------------------------------------------------
    class MultinomialMixtureComponentBuilder : public MixtureComponentBuilder {
     public:
      explicit MultinomialMixtureComponentBuilder(SEXP rmixture_component)
          : MixtureComponentBuilder(rmixture_component) {}

      virtual Ptr<MixtureComponent> Build(
          SEXP rmixture_component,
          RListIoManager *io_manager)const {
        DirichletPrior prior_spec(getListElement(rmixture_component, "prior"));
        NEW(MultinomialModel, model)(prior_spec.dim());
        NEW(DirichletModel, prior)(prior_spec.prior_counts());
        NEW(MultinomialDirichletSampler, sampler)(model.get(), prior);
        model->set_method(sampler);

        io_manager->add_list_element(new VectorListElement(
            model->Pi_prm(),
            create_name("prob"),
            StringVector(getListElement(rmixture_component, "levels"))));
        return model;
      }
    };

    class MarkovMixtureComponentBuilder : public MixtureComponentBuilder {
     public:
      explicit MarkovMixtureComponentBuilder(SEXP rmixture_component)
          : MixtureComponentBuilder(rmixture_component) {}

      virtual Ptr<MixtureComponent> Build(
          SEXP rmixture_component,
          RListIoManager *io_manager)const {
        MarkovPrior prior_spec(getListElement(rmixture_component, "prior"));
        NEW(MarkovModel, model)(prior_spec.dim());
        NEW(ProductDirichletModel, transition_matrix_prior)(
            prior_spec.transition_counts());
        NEW(DirichletModel, initial_distribution_prior)(
            prior_spec.initial_state_counts());
        NEW(MarkovConjSampler, sampler)(model.get(),
                                        transition_matrix_prior,
                                        initial_distribution_prior);
        model->set_method(sampler);

        MatrixListElement * transition_probability_list_element =
            new MatrixListElement(model->Q_prm(),
                                  create_name("trans"));
        std::vector<std::string> factor_names = StringVector(
            getListElement(rmixture_component, "levels"));
        transition_probability_list_element->set_row_names(factor_names);
        transition_probability_list_element->set_col_names(factor_names);
        io_manager->add_list_element(transition_probability_list_element);

        io_manager->add_list_element(new VectorListElement(
            model->Pi0_prm(),
            create_name("initial.distribution"),
            factor_names));

        return model;
      }
    };

    //======================================================================
    MixtureComponentBuilder * MixtureComponentBuilder::Create(
        SEXP mixture_component) {
      if (Rf_inherits(mixture_component, "NormalMixtureComponent")) {
        return new NormalMixtureComponentBuilder(mixture_component);
      } else if (Rf_inherits(mixture_component, "PoissonMixtureComponent")) {
        return new PoissonMixtureComponentBuilder(mixture_component);
      } else if (Rf_inherits(
          mixture_component, "RegressionMixtureComponent")) {
        return new RegressionMixtureComponentBuilder(mixture_component);
      } else if (Rf_inherits(
          mixture_component, "LogitMixtureComponent")) {
        return new LogitMixtureComponentBuilder(mixture_component);
      } else if (Rf_inherits(
          mixture_component, "ZeroInflatedLognormalMixtureComponent")) {
        return new ZeroInflatedLognormalMixtureComponentBuilder(
            mixture_component);
      } else if (Rf_inherits(mixture_component, "MvnMixtureComponent")) {
        return new MvnMixtureComponentBuilder(mixture_component);
      } else if (Rf_inherits(mixture_component,
                             "IndependentMvnMixtureComponent")) {
        return new IndependentMvnMixtureComponentBuilder(mixture_component);
      } else if (Rf_inherits(mixture_component,
                             "MultinomialMixtureComponent")) {
        return new MultinomialMixtureComponentBuilder(mixture_component);
      } else if (Rf_inherits(mixture_component, "MarkovMixtureComponent")) {
        return new MarkovMixtureComponentBuilder(mixture_component);
      } else if (Rf_inherits(mixture_component,
                             "ZeroInflatedPoissonMixtureComponent")) {
        return new ZeroInflatedPoissonMixtureComponentBuilder(
            mixture_component);
      }
      report_error("Unknown mixture component");
      return NULL;
    }

    //======================================================================
    // Creates a vector of MixtureComponent's based on R data
    // structures.  The result is suitable for passing into the
    // constructor for a hidden Markov model or finite mixture model.
    // Args:
    //   rmixture_components:  An R list of class "CompositeMixtureComponent".
    //   state_space_size:  Size of the state space for the latent Markov chain.
    //   io_manager: A BOOM RListIoManager responsible for recording the
    //     MCMC output.
    // Returns:
    //   A vector of MixtureComponents of length state_space_size,
    //   suitable for passing to the HiddenMarkovModel constructor.
    //   Each element will have a PosteriorSampler assigned, but no
    //   data.  Each MixtureComponent has an actual type of
    //   BOOM::CompositeModel, which inherits from
    //   BOOM::MixtureComponent.  The io_manager will create entries for
    //   the parameters of each mixture component with names based on
    //   the numeric system described in 'create_name' above.
    std::vector<Ptr<MixtureComponent> > UnpackCompositeMixtureComponents(
        SEXP rmixture_components,
        int state_space_size,
        RListIoManager *io_manager) {
      int number_of_composite_elements = Rf_length(rmixture_components);

      std::vector<Ptr<MixtureComponent> > ans;
      ans.reserve(state_space_size);
      for (int s = 0; s < state_space_size; ++s) {
        std::vector<Ptr<MixtureComponent> > composite_elements;
        composite_elements.reserve(number_of_composite_elements);
        for (int m = 0; m < number_of_composite_elements; ++m) {
          composite_elements.push_back(
              CreateMixtureComponent(
                  VECTOR_ELT(rmixture_components, m),
                  s,
                  io_manager));
        }
        Ptr<CompositeModel> component =
            new BOOM::CompositeModel(composite_elements);
        Ptr<BOOM::CompositeModelSampler> composite_model_sampler(
            new BOOM::CompositeModelSampler(component.get()));
        component->set_method(composite_model_sampler);
        ans.push_back(component);
      }
      return ans;
    }

    // Creates a vector of MixtureComponent's based on R data
    // structures.  The result is suitable for passing into the
    // constructor for a hidden Markov model or finite mixture model.
    // Args:
    //   rmixture_components:  An R list of class "CompositeMixtureComponent".
    //   state_space_size:  Size of the state space for the latent Markov chain.
    //   io_manager: A BOOM RListIoManager responsible for recording the
    //     MCMC output.
    // Returns:
    //   A vector of MixtureComponents of length state_space_size,
    //   suitable for passing to the HiddenMarkovModel constructor.
    //   Each element will have a PosteriorSampler assigned, but no
    //   data.  Each MixtureComponent has an actual type of
    //   BOOM::CompositeModel, which inherits from
    //   BOOM::MixtureComponent.  The io_manager will create entries for
    //   the parameters of each mixture component with names based on
    //   the textual strategy described in 'create_name' above.
    std::map<std::string, Ptr<MixtureComponent> >
    UnpackNamedCompositeMixtureComponents(
        SEXP rmixture_components,
        const std::vector<std::string> & mixture_component_names,
        RListIoManager *io_manager) {
      int number_of_composite_elements = Rf_length(rmixture_components);
      std::map<std::string, Ptr<MixtureComponent> > ans;
      int state_space_size = mixture_component_names.size();
      for (int s = 0; s < state_space_size; ++s) {
        std::vector<Ptr<MixtureComponent> > composite_elements;
        composite_elements.reserve(number_of_composite_elements);
        for (int m = 0; m < number_of_composite_elements; ++m) {
          composite_elements.push_back(
              CreateNamedMixtureComponent(
                  VECTOR_ELT(rmixture_components, m),
                  mixture_component_names[s],
                  io_manager));
        }
        Ptr<CompositeModel> component =
            new BOOM::CompositeModel(composite_elements);
        Ptr<BOOM::CompositeModelSampler> composite_model_sampler(
            new BOOM::CompositeModelSampler(component.get()));
        component->set_method(composite_model_sampler);
        ans[mixture_component_names[s]] = component;
      }
      return ans;
    }

  }  // namespace RInterface
}  // namespace BOOM
