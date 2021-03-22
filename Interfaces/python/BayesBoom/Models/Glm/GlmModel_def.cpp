#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Models/ModelTypes.hpp"
#include "Models/Glm/Glm.hpp"
#include "Models/Glm/GlmCoefs.hpp"
#include "Models/Glm/RegressionModel.hpp"
#include "Models/Glm/TRegression.hpp"
#include "Models/Glm/LoglinearModel.hpp"
#include "Models/Glm/BinomialLogitModel.hpp"
#include "Models/Glm/PoissonRegressionModel.hpp"
#include "Models/Glm/VariableSelectionPrior.hpp"

#include "Models/Glm/PosteriorSamplers/BregVsSampler.hpp"
#include "Models/Glm/PosteriorSamplers/TRegressionSpikeSlabSampler.hpp"
#include "Models/Glm/PosteriorSamplers/BinomialLogitSpikeSlabSampler.hpp"
#include "Models/Glm/PosteriorSamplers/PoissonRegressionSpikeSlabSampler.hpp"

#include "cpputil/Ptr.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;
  using BOOM::uint;

  void GlmModel_def(py::module &boom) {

    py::class_<GlmCoefs,
               VectorParams,
               Ptr<GlmCoefs>>(boom, "GlmCoefs")
        .def_property_readonly(
            "inc",
            [](const GlmCoefs &coefs) {return coefs.inc();},
            "The Selector object indicating which variables are included "
            "or excluded.")
        .def_property_readonly(
            "included_coefficients",
            &GlmCoefs::included_coefficients,
            "The vector of coefficients corresponding to the subset of "
            "included variables.")
        .def("set_sparse_coefficients",
             [](GlmCoefs &coefs,
                const Vector &nonzero_values,
                const std::vector<uint> &nonzero_positions) {
               coefs.set_sparse_coefficients(nonzero_values, nonzero_positions);
             },
             "Set a specific set of coefficients to nonzero values, setting all "
             "others to zero.")
        ;

    // A single data point for a regression model.
    py::class_<RegressionData,
               Ptr<RegressionData>>(boom, "RegressionData")
        .def(py::init<double, Vector>(),
             py::arg("y"),
             py::arg("x"),
             "Args:\n\n"
             "  y: Response"
             "  x: BOOM::Vector of predictors.")
        ;

    py::class_<MvRegData, Ptr<MvRegData>>(boom, "MvRegData")
        .def(py::init<Vector, Vector>(),
             py::arg("y"),
             py::arg("x"),
             "Args:\n\n"
             "  y: BayesBoom.Vector of responses."
             "  x: BayesBoom.Vector of predictors.")
        ;

    // Base class for generalized linear models: regression, logistic
    // regression, Poisson regression, etc.
    py::class_<GlmModel,
               Model,
               Ptr<GlmModel>>(boom, "GlmModel")
        .def_property_readonly(
            "coef",
            [] (GlmModel &model) {return model.coef();},
            "The model coefficient object.")
        .def_property_readonly("xdim", &GlmModel::xdim)
        .def("add_all", &GlmModel::add_all)
        .def("drop_all", &GlmModel::drop_all)
        .def("drop_all_but_intercept", &GlmModel::drop_all_but_intercept)
        .def("add", &GlmModel::add, "Add the variable in the specified position.")
        .def("drop", &GlmModel::add, "Drop the variable in the specified position.")
        .def("flip", &GlmModel::add, "Flip the variable in the specified position.")
        .def_property_readonly(
            "coef",
            [](const GlmModel &m) {return m.coef();},
            "The object containing the model coefficients.")
        ;

    py::class_<RegSuf,
               Ptr<RegSuf>>(
                   boom, "RegSuf", py::multiple_inheritance())
        .def_property_readonly(
            "sample_mean",
            [](const RegSuf &s) { return s.ybar(); },
            "The sample mean of the training data.")
        .def_property_readonly(
            "sample_variance",
            [](const RegSuf &s) {
              Vector beta(s.size());
              beta[0] = s.ybar();
              return s.relative_sse(beta) / (s.n() - 1.0);
            },
            "Sample variance of the training data.")
        ;

    py::class_<RegressionModel,
               GlmModel,
               PriorPolicy,
               Ptr<RegressionModel>>(
                   boom, "RegressionModel", py::multiple_inheritance())
        .def(py::init<Matrix, Vector, bool>(),
             py::arg("X"),
             py::arg("y"),
             py::arg("start_at_mle") = false,
             "Create a regression model from a data set.\n\n"
             "Args:"
             "  X:  boom.Matrix of predictor variables.\n"
             "  y:  boom.Vector of responses.\n"
             "  start_at_mle: If True then the model parameters will be initialized "
             "to the maximum likelihood estimate (which will be undefined if X is "
             "less than full rank).  If False then model parameters begin at "
             "default levels.")
        .def_property_readonly(
            "suf",
            [](const RegressionModel &m) {return m.suf();},
            "RegSuf object containing the sufficient statistics for the model.")
        .def_property_readonly(
            "Sigsq_prm",
            [](RegressionModel &m) {
              return m.Sigsq_prm();
            },
            "The parameter object representing the residual variance.  boom.UnivParams")
        .def_property_readonly(
            "coefficients",
            [](RegressionModel &m) {
              return m.coef();
            },
            "The parameter object representing the model coefficients.  boom.GlmCoefs")
        .def("set_coefficients", [](RegressionModel &m, const Vector &coefficients) {
          m.set_Beta(coefficients);
        })
        .def_property_readonly(
            "sigma",
            [](const RegressionModel &m){
              return m.sigma();
            },
            "The residual standard deviation.")
        .def_property_readonly(
            "suf",
            [](const RegressionModel &m) {
              return m.suf();
            },
            "The sufficient statistics for the regression model.")
        .def("log_likelihood",
             [](const RegressionModel& m) {
               return m.log_likelihood();
             })
        ;

    py::class_<TRegressionModel,
               GlmModel,
               PriorPolicy,
               Ptr<TRegressionModel>>(
                   boom, "TRegressionModel", py::multiple_inheritance())
        .def(py::init(
            [] (int xdim) {
              return new TRegressionModel(xdim);
            }),
             py::arg("xdim"),
             "Args:\n\n"
             "  xdim:  Dimension of the predictor vector.\n")
        .def(py::init(
            [] (const Vector &coef, double sigma, double nu) {
              return new TRegressionModel(coef, sigma, nu);
            }),
             py::arg("coefficients"),
             py::arg("residual_sd"),
             py::arg("residual_df"),
             "Args:\n\n"
             "  coefficients:  Coefficient vector\n"
             "  residual_sd:  residual standard deviation\n"
             "  residual_df:  tail thickness parameter\n")
        .def(py::init(
            [] (const Matrix &predictors, const Vector &response) {
              return new TRegressionModel(predictors, response);
            }),
             py::arg("predictors"),
             py::arg("response"),
             "Args:\n\n"
             "  predictors:  A boom.Matrix containing the predictor variables"
             ".\n"
             "  response:  A boom.Vector containing the response variable.\n")
        .def_property_readonly(
            "Sigsq_prm",
            [](TRegressionModel &m) {
              return m.Sigsq_prm();
            },
            "The parameter object representing the residual variance.  boom.UnivParams")
        .def_property_readonly(
            "coefficients",
            [](TRegressionModel &m) {
              return m.coef();
            },
            "The parameter object representing the model coefficients.  boom.GlmCoefs")
        .def("set_coefficients", [](TRegressionModel &m, const Vector &coefficients) {
          m.set_Beta(coefficients);
        })
        .def_property_readonly(
            "residual_sd", [] (const TRegressionModel &m) {return m.sigma();
            })
        .def("set_residual_sd", [] (TRegressionModel &m, double sigma) {
          m.set_sigsq(sigma * sigma);
        })
        .def_property_readonly("residual_df", [] (const TRegressionModel &m) {
          return m.nu();
        })
        .def("set_residual_df", [] (TRegressionModel &m, double nu) {
          m.set_nu(nu);
        })
        ;

    py::class_<BinomialLogitModel,
               GlmModel,
               PriorPolicy,
               Ptr<BinomialLogitModel>>(
                   boom, "BinomialLogitModel", py::multiple_inheritance())
        .def(py::init([](int xdim, bool include_all) {
          return new BinomialLogitModel(xdim, include_all);
        }),
          py::arg("xdim"),
          py::arg("include_all") = true,
          "Args:\n\n"
          "  xdim:  Dimension of the predictor vector.\n"
          "  include_all:  Include all the predictors initially.  If False "
          "then only the intercept starts out included.\n"
          )
        ;

    py::class_<PoissonRegressionModel,
               GlmModel,
               PriorPolicy,
               Ptr<PoissonRegressionModel>>(
                   boom, "PoissonRegressionModel", py::multiple_inheritance())
        .def(py::init([](int xdim) {
          return new PoissonRegressionModel(xdim);
        }),
          py::arg("xdim"),
          "Args:\n\n"
          "  xdim:  Dimension of the predictor vector.\n"
          )
        ;

    py::class_<LoglinearModel,
               PriorPolicy,
               Ptr<LoglinearModel>>(
                   boom, "LoglinearModel", py::multiple_inheritance())
        .def(py::init(
            []() {
              return new LoglinearModel;
            }),
             "An empty LoglinearModel.  The first time this model calls \n"
             "add_data main effects will be added for each variable in \n"
             "the added data point. ")
        .def("add_data",
             [](LoglinearModel &model, const Matrix &integer_codes) {
             },
             py::arg("integer_codes"),
             "Args: \n"
             "  integer_codes:  A Matrix containing codes for the data to \n"
             "    be modeled.  Each variable is coded from 0 to nlevels - 1.\n"
             "    A value less than 0, or a NaN is interpreted as missing. \n")
        .def_property_readonly(
            "nvars", &LoglinearModel::nvars,
            "The number of variables being modeled.")
        .def("add_interaction",
             [](LoglinearModel &model,
                const std::vector<int> &interaction) {
               model.add_interaction(interaction);
             },
             py::arg("interaction"),
             "Args:\n"
             "  interaction:  A list of integer valued indices identifying\n"
             "    the variables in the interaction.  The indices should be in\n"
             "    ascending order.")
        .def("logp",
             [](const LoglinearModel &model,
                const std::vector<int> data_values) {
               return model.logp(data_values);
             },
             "Args:\n"
             "  data_values: The integer codes identifying the levels of a \n"
             "    single data point.\n\n"
             "Returns:\n"
             "   The un-normalized log density of the input data.  ")
        .def("initialize_missing_data",
             &LoglinearModel::initialize_missing_data,
             "Fill any missing values with uniform draws across the \n"
             "level-range of the appropriate variable.")
        .def("impute_missing_data",
             [](LoglinearModel &model, RNG &rng) {
               model.impute_missing_data(rng);
             },
             "Args:\n"
             "  rng: A Boom random number generator.\n\n"
             "Perform one Gibbs sampling pass over the missing data.  Each \n"
             "missing observation is imputed given all other missing and \n"
             "observed data, and model parameters.")
        ;


    //===========================================================================
    // Priors and posterior samplers.
    //===========================================================================
    py::class_<VariableSelectionPrior,
               Ptr<VariableSelectionPrior>>(boom, "VariableSelectionPrior")
        .def(py::init<const Vector &>(),
             py::arg("prior_inclusion_probabilities"),
             "Create a VariableSelectionPrior from a vector of prior inclusion probabilities.\n\n"
             "Args:\n\n"
             "  prior_inclusion_probabilities: boom.Vector containing the prior "
             "probability that each variable is to be included. "
             )
        ;

    py::class_<BregVsSampler,
               PosteriorSampler,
               Ptr<BregVsSampler>>(boom, "BregVsSampler")
        .def(py::init(
            [](Ptr<RegressionModel> model,
               const Ptr<MvnGivenScalarSigma> &slab,
               const Ptr<GammaModelBase> &residual_precision_prior,
               const Ptr<VariableSelectionPrior> &spike,
               RNG &seeding_rng) {
              return new BregVsSampler(model.get(), slab, residual_precision_prior,
                                       spike, seeding_rng);
            }),
             py::arg("model"),
             py::arg("slab"),
             py::arg("residual_precision_prior"),
             py::arg("spike"),
             py::arg("seeding_rng") = BOOM::GlobalRng::rng,
             "Create a BregVsSampler -- a spike and slab sampler for regression "
             "models.\n\n"
             "Args:\n"
             "  model:  The model to be sampled.\n"
             "  slab:  The conditional normal prior on the regression "
             "coefficients.\n"
             "  residual_precision_prior:   Prior distribution on the residual "
             "precision (reciprocal of the residual variance).\n"
             "  spike:  Prior distribution over the vector of inclusion "
             "indicators.\n"
             "  seeding_rng:  The random number generator used to set the seed "
             "of the RNG owned by this sampler."
             )
        ;

    py::class_<TRegressionSpikeSlabSampler,
               PosteriorSampler,
               Ptr<TRegressionSpikeSlabSampler>>(
                   boom, "TRegressionSpikeSlabSampler")
        .def(py::init(
            [] (TRegressionModel *model,
                MvnBase *coefficient_slab_prior,
                VariableSelectionPrior *coefficient_spike_prior,
                GammaModelBase *siginv_prior,
                DoubleModel *nu_prior,
                RNG &seeding_rng) {
              return new TRegressionSpikeSlabSampler(
                  model, coefficient_slab_prior, coefficient_spike_prior,
                  siginv_prior, nu_prior, seeding_rng);
            }),
            py::arg("model"),
            py::arg("slab"),
            py::arg("spike"),
            py::arg("residual_precision_prior"),
            py::arg("tail_thickness_prior"),
            py::arg("rng"),
            "Args:\n\n"
            "  model: The boom.TRegressionModel that the sampler will simulate "
            "for.\n"
            "  slab: A boom.MvnBase prior for the conditional distribution of "
            "the regression coefficients given inclusion.\n"
            "  spike:  A boom.VariableSelectionPrior describing which variables"
            " are included in the model.\n"
            "  residual_precision_prior: A boom.GammaModelBase giving the "
            "prior distribution on the residual precision parameter (one "
            "over the residual variance).\n"
            "  tail_thickness_prior: A boom.DoubleModel with support on "
            "a subset of the positive real line.\n"
             "  rng: A boom.RNG random number generator.\n")
        .def("set_sigma_upper_limit",
             [](TRegressionSpikeSlabSampler *sampler, double upper_limit) {
               sampler->set_sigma_upper_limit(upper_limit);
             },
             "Args:\n\n"
             "  upper_limit: The upper truncation point for the residual "
             "'standard deviation' parameter.  Anything other than a positive "
             "real is treated as infinity.")
        .def("limit_model_selection",
             [](TRegressionSpikeSlabSampler *sampler, int max_flips) {
               sampler->limit_model_selection(max_flips);
             },
             "Args:\n\n"
             "  max_flips:  At most this many model exploration steps will be "
             "attempted each iteration.\n"
             "")
        ;

    py::class_<BinomialLogitSpikeSlabSampler,
               PosteriorSampler,
               Ptr<BinomialLogitSpikeSlabSampler>>(
                   boom, "BinomialLogitSpikeSlabSampler")
        .def(py::init([](BinomialLogitModel *model,
                         MvnBase *slab,
                         VariableSelectionPrior *spike,
                         int clt_threshold,
                         RNG &seeding_rng) {
          return new BinomialLogitSpikeSlabSampler(
              model, slab, spike, clt_threshold, seeding_rng);
        }),
             py::arg("model"),
             py::arg("slab"),
             py::arg("spike"),
             py::arg("clt_threshold") = 5,
             py::arg("seeding_rng") = BOOM::GlobalRng::rng,
             "Args:\n\n"
             "  model:  The boom.BinomialLogitModel to be sampled.\n"
             "  slab: A boom.MvnBase prior for the conditional distribution of "
             "the regression coefficients given inclusion.\n"
             "  spike:  A boom.VariableSelectionPrior describing which variables"
             " are included in the model.\n"
             "  clt_threshold:  See below.\n"
             "  seeding_rng:  The random number generator used to set the seed "
             "of the RNG owned by this sampler.\n\n"
             "When imputing latent data, if the number of trials is below the"
             " 'clt_threshold' each Bernoulli trial will be imputed separately."
             "  If the number of trials exceeds 'clt_threshold' then the "
             "moments of the latent data will be imputed instead. \n"
             )
        .def("limit_model_selection",
             [](BinomialLogitSpikeSlabSampler *sampler, int max_flips) {
               sampler->limit_model_selection(max_flips);
             },
             "Args:\n\n"
             "  max_flips:  At most this many model exploration steps will be "
             "attempted each iteration.\n"
             "")
        ;

    py::class_<PoissonRegressionSpikeSlabSampler,
               PosteriorSampler,
               Ptr<PoissonRegressionSpikeSlabSampler>>(
                   boom, "PoissonRegressionSpikeSlabSampler")
        .def(py::init([](PoissonRegressionModel *model,
                         MvnBase *slab,
                         VariableSelectionPrior *spike,
                         int num_threads,
                         RNG &seeding_rng) {
          return new PoissonRegressionSpikeSlabSampler(
              model, slab, spike, num_threads, seeding_rng);
        }),
             py::arg("model"),
             py::arg("slab"),
             py::arg("spike"),
             py::arg("num_threads") = 1,
             py::arg("seeding_rng") = BOOM::GlobalRng::rng,
             "Args:\n\n"
             "  model:  The boom.PoissonRegressionModel to be sampled.\n"
             "  slab: A boom.MvnBase prior for the conditional distribution of "
             "the regression coefficients given inclusion.\n"
             "  spike:  A boom.VariableSelectionPrior describing which variables"
             " are included in the model.\n"
             "  num_threads:  The number of threads to use when imputing "
             "latent data.\n"
             "  seeding_rng:  The random number generator used to set the seed "
             "of the RNG owned by this sampler.\n"
             )
        .def("limit_model_selection",
             [](PoissonRegressionSpikeSlabSampler *sampler, int max_flips) {
               sampler->limit_model_selection(max_flips);
             },
             "Args:\n\n"
             "  max_flips:  At most this many model exploration steps will be "
             "attempted each iteration.\n"
             "")
        ;


  }  // module definition

}  // namespace BayesBoom
