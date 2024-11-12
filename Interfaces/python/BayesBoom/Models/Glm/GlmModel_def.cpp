#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>

#include "Models/ModelTypes.hpp"
#include "Models/Glm/Glm.hpp"
#include "Models/Glm/GlmCoefs.hpp"
#include "Models/Glm/RegressionModel.hpp"
#include "Models/Glm/IndependentRegressionModels.hpp"
#include "Models/Glm/TRegression.hpp"
#include "Models/Glm/LoglinearModel.hpp"
#include "Models/Glm/BinomialLogitModel.hpp"
#include "Models/Glm/PoissonRegressionModel.hpp"
#include "Models/Glm/RegressionSlabPrior.hpp"
#include "Models/Glm/VariableSelectionPrior.hpp"

#include "Models/Glm/PosteriorSamplers/BregVsSampler.hpp"
#include "Models/Glm/PosteriorSamplers/RegressionConjSampler.hpp"
#include "Models/Glm/PosteriorSamplers/IndependentRegressionModelsPosteriorSampler.hpp"
#include "Models/Glm/PosteriorSamplers/TRegressionSpikeSlabSampler.hpp"
#include "Models/Glm/PosteriorSamplers/BinomialLogitSpikeSlabSampler.hpp"
#include "Models/Glm/PosteriorSamplers/PoissonRegressionSpikeSlabSampler.hpp"
#include "Models/Glm/PosteriorSamplers/BigAssSpikeSlabSampler.hpp"

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
             "Set a specific set of coefficients to nonzero values, "
             "setting all others to zero.")
        .def("add_all",
             [](GlmCoefs &coefs) {
               coefs.add_all();
             },
             "Allow all model coefficients to be nonzero.")
        .def("drop_all",
             [](GlmCoefs &coefs) {
               coefs.drop_all();
             },
             "Force all model coefficients to zero.")
        .def("add",
             [](GlmCoefs &coef, int i) {
               coef.add(i);
             },
             py::arg("i"),
             "Allow coefficient i to be nonzero.")
        .def("drop",
             [](GlmCoefs &coef, int i) {
               coef.drop(i);
             },
             py::arg("i"),
             "Force coefficient i to be zero.")
        .def("flip",
             [](GlmCoefs &coef, int i) {
               coef.flip(i);
             },
             py::arg("i"),
             "Flip the include/exclude status of coefficient i.")
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
        .def("__repr__",
             [](const RegressionData &dp) {
               std::ostringstream out;
               out << dp;
               return out.str();
             })
        ;

    py::class_<BinomialRegressionData,
               Ptr<BinomialRegressionData>>(boom, "BinomialRegressionData")
        .def(py::init(
            [](double y, double n, const Vector &x) {
              return new BinomialRegressionData(y, n, x);
            }),
             py::arg("y"),
             py::arg("n"),
             py::arg("x"),
             "Args:\n\n"
             "  y: Success count.  0 <= y <= n\n"
             "  n: Trial count.  n >= 0\n"
             "  x: Vector of predictors\n")
        .def("__repr__",
             [](const BinomialRegressionData &dp) {
               std::ostringstream out;
               out << dp;
               return out.str();
             })
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
        .def_property_readonly("xdim", &GlmModel::xdim)
        .def("add_all", &GlmModel::add_all)
        .def("drop_all", &GlmModel::drop_all)
        .def("drop_all_but_intercept", &GlmModel::drop_all_but_intercept)
        .def("add", &GlmModel::add,
             "Add the variable in the specified position.")
        .def("drop", &GlmModel::add,
             "Drop the variable in the specified position.")
        .def("flip", &GlmModel::add,
             "Flip the variable in the specified position.")
        .def_property(
            "coef",
            [](GlmModel &model) {return
                  Ptr<GlmCoefs>(&model.coef());},
            [](GlmModel &model, GlmCoefs &beta) {
              model.coef().set_sparse_coefficients(
                  beta.included_coefficients(),
                  beta.inc().included_positions());
            },
            "The object containing the model coefficients.")
        .def_property_readonly(
            "Beta",
            [](const GlmModel &m) {return m.Beta();},
            "A BayesBoom.Vector containing the model coefficients, including "
            "any 0's if sparse modeling is being used.")
        ;

    py::class_<RegSuf,
               Ptr<RegSuf>>(
                   boom, "RegSuf", py::multiple_inheritance())
        .def(py::init(
            [](int dim) {
              return new NeRegSuf(dim);
            }),
             py::arg("dim"),
             "Args:\n\n"
             "  dim:  The dimension of the predictor variable ('x').\n")
        .def(py::init(
            [](const Matrix &X, const Vector &y) {
              return new NeRegSuf(X, y);
            }),
             py::arg("X"),
             py::arg("y"),
             "Args:\n\n"
             "  X:  The predictor matrix.  An explicit column of 1's is needed"
             " if an intercept term is desired.\n"
             "  y:  The response vector.  The lenght must match the number of "
             "rows in X.\n")
        .def(py::init(
            [](const SpdMatrix &xtx, const Vector &xty, double sample_sd,
               double sample_size, double ybar, const Vector &xbar) {

              // E(X^2) = sigma^2 + mu^2
              double yty = (sample_size - 1) * sample_sd * sample_sd
                  + sample_size * ybar * ybar;

              return new NeRegSuf(xtx, xty, yty, sample_size, ybar, xbar);
            }),
             py::arg("xtx"),
             py::arg("xty"),
             py::arg("sample_sd"),
             py::arg("sample_size"),
             py::arg("ybar"),
             py::arg("xbar"),
             "Args:\n\n"
             "  xtx:  The cross product matrix X'X, where X is the matrix of "
             "predictors.\n"
             "  xty:  The X'y matrix, where y is the matrix of responses.\n"
             "  sample_sd:  The sample standard deviation of the responses.\n"
             "  sample_size:  The number of observations contained in the "
             "sufficient statistics.\n"
             "  ybar:  The mean of the response variable."
             "  xbar:  The mean of each column of the predictor matrix X.\n")
        .def_property_readonly(
            "xtx", [](const RegSuf &suf) {return suf.xtx();})
        .def_property_readonly(
            "xty", [](const RegSuf &suf) {return suf.xty();})
        .def_property_readonly(
            "yty", [](const RegSuf &suf) {return suf.yty();})
        .def_property_readonly(
            "n", [](const RegSuf &suf) {return suf.n();})
        .def_property_readonly(
            "sample_size", [](const RegSuf &suf) {return suf.n();})
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
             "  start_at_mle: If True then the model parameters will be "
             "initialized to the maximum likelihood estimate (which will "
             "be undefined if X is less than full rank).  If False then "
             "model parameters begin at default levels.")
        .def(py::init(
            [](const Ptr<RegSuf> &suf) {
              return new RegressionModel(suf);
            }),
             py::arg("suf"),
             "Args:\n\n"
             "  suf:  An object of class boom.RegSuf")
        .def_property_readonly(
            "suf",
            [](const RegressionModel &m) {return m.suf();},
            "RegSuf object containing the sufficient statistics for the model.")
        .def_property_readonly(
            "Sigsq_prm",
            [](RegressionModel &m) {
              return m.Sigsq_prm();
            },
            "The parameter object representing the residual variance.  "
            "boom.UnivParams")
        // .def_property(
        //     "coefficients",
        //     [](RegressionModel &m) {
        //       return m.coef();
        //     },
        //     "The parameter object representing the model coefficients.  "
        //     "boom.GlmCoefs")
        .def("set_coefficients", [](RegressionModel &m,
                                    const Vector &coefficients) {
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

    py::class_<IndependentRegressionModels,
               PriorPolicy,
               PosteriorModeModel,
               Ptr<IndependentRegressionModels>>(
                   boom,
                   "IndependentRegressionModels",
                   py::multiple_inheritance())
        .def(py::init(
            [](int xdim, int ydim) {
              return new IndependentRegressionModels(xdim, ydim);
            }),
             py::arg("xdim"),
             py::arg("ydim"),
             "Args:\n\n"
             "  xdim:  Dimension of the predictors (X).\n"
             "  ydim:  Dimension of the output/response (Y).\n")
        .def("model",
             [](IndependentRegressionModels *model, int which_model) {
               return model->model(which_model);
             },
             py::arg("which_model"),
             "Args:\n\n"
             "  which_model:  The index (0, ... ydim-1) of the "
             "component model.\n")
        ;

    py::class_<IndependentGlms<TRegressionModel>,
               PriorPolicy,
               Ptr<IndependentGlms<TRegressionModel>>>(
                   boom,
                   "IndependentStudentRegressions",
                   py::multiple_inheritance())
        .def(py::init(
            [](int xdim, int ydim) {
              return new IndependentGlms<TRegressionModel>(
                  xdim, ydim);
            }),
             py::arg("xdim"),
             py::arg("ydim"),
             "Args:\n\n"
             "  xdim:  Dimension of the predictors (X).\n"
             "  ydim:  Dimension of the output/response (Y).\n")
        .def("model",
             [](IndependentGlms<TRegressionModel> *model, int which_model) {
               return model->model(which_model);
             },
             py::arg("which_model"),
             "Args:\n\n"
             "  which_model:  The index (0, ... ydim-1) of the "
             "component model.\n")
        ;

    py::class_<IndependentGlms<CompleteDataStudentRegressionModel>,
               PriorPolicy,
               Ptr<IndependentGlms<CompleteDataStudentRegressionModel>>>(
                   boom,
                   "IndependentStudentCompleteDataRegressions",
                   py::multiple_inheritance())
        .def(py::init(
            [](int xdim, int ydim) {
              return new IndependentGlms<CompleteDataStudentRegressionModel>(
                  xdim, ydim);
            }),
             py::arg("xdim"),
             py::arg("ydim"),
             "Args:\n\n"
             "  xdim:  Dimension of the predictors (X).\n"
             "  ydim:  Dimension of the output/response (Y).\n")
        .def("model",
             [](IndependentGlms<CompleteDataStudentRegressionModel> *model, int which_model) {
               return model->model(which_model);
             },
             py::arg("which_model"),
             "Args:\n\n"
             "  which_model:  The index (0, ... ydim-1) of the "
             "component model.\n")
        ;

    py::class_<RegressionSlabPrior,
               MvnBase,
               Ptr<RegressionSlabPrior>>(
                   boom, "RegressionSlabPrior", py::multiple_inheritance())
        .def(py::init(
            [] (const SpdMatrix &xtx,
                const Ptr<UnivParams> &sigsq_param,
                double sample_mean,
                double sample_size,
                double prior_sample_size,
                double diagonal_shrinkage) {
              return new RegressionSlabPrior(
                  xtx, sigsq_param, sample_size, sample_size,
                  prior_sample_size, diagonal_shrinkage);
            }),
             py::arg("xtx"),
             py::arg("sigsq_param"),
             py::arg("sample_mean"),
             py::arg("sample_size"),
             py::arg("prior_sample_size"),
             py::arg("diagonal_shrinkage"),
             "Args:\n\n"
             "  xtx:  The cross product matrix from the regression model.\n"
             "  sigsq_param:  The residual variance parameter object from "
             "the regression model.\n"
             "  sample_mean:  The mean of the response variable.\n"
             "  sample_size:  The number of observations in the regression "
             "problem.\n"
             "  prior_sample_size:  The number of observations worth of weight "
             "to assign the prior.\n"
             "  diagonal_shrinkage:  The xtx matrix is averaged with its own "
             "diagonal to protect against the possibility that xtx is less "
             "than full rank.  The diagonal_shrinkage parameter is the weight "
             "(between 0 and 1) assigned to the diagonal in this averaging "
             "procedure.\n" )
        ;

    py::class_<BigRegressionModel,
               GlmModel,
               PriorPolicy,
               Ptr<BigRegressionModel>>(
                   boom, "BigRegressionModel", py::multiple_inheritance())
        .def(py::init(
            [](uint xdim, int subordinate_model_max_dim, bool force_intercept) {
              return new BigRegressionModel(
                  xdim, subordinate_model_max_dim, force_intercept);
                  }),
             py::arg("xdim"),
             py::arg("subordinate_model_max_dim") = 500,
             py::arg("use_threads") = true,
             "Args:\n\n"
             "  xdim:  Dimension of the predictor vector.\n"
             "  subordinate_model_max_dim:  The largest dimension of each "
             "subordinate model (the model used to do the initial screen).\n"
             "  use_threads:  If True then C++11 threads are used to implement"
             " the initial screen.  If False then no threads are used.  This "
             "argument is primarily used for debugging.")
        .def("stream_data_for_initial_screen",
             [](BigRegressionModel &model,
                const RegressionData &data_point) {
               model.stream_data_for_initial_screen(data_point);
             },
             py::arg("data_point"),
             "Args:\n\n"
             "   data_point:  A data point (of class boom.RegressionData).\n"
             "\n"
             "Pass the data to the subordinate models stored inside the "
             "BigRegressionModel. The data are not stored in raw form, but "
             "added to the subordinate models's sufficient statistics.\n")
        .def("stream_data_for_restricted_model",
             [](BigRegressionModel &model, const RegressionData &data_point) {
               model.stream_data_for_restricted_model(data_point);
             },
             py::arg("data_point"),
             "Args:\n\n"
             "   data_point:  A data point (of class boom.RegressionData).\n"
             "\n"
             "Pass the data to the restricted model stored inside the  "
             "BigRegressionModel. The data are not stored in raw form, but "
             "added to the restricted models's sufficient statistics.\n")
        .def_property_readonly(
            "Sigsq_prm",
            [](BigRegressionModel &m) {
              return m.Sigsq_prm();
            },
            "The parameter object representing the residual variance.  "
            "boom.UnivParams")
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
            "The parameter object representing the residual variance.  "
            "boom.UnivParams")
        .def_property_readonly(
            "coefficients",
            [](TRegressionModel &m) {
              return m.coef();
            },
            "The parameter object representing the model coefficients.  "
            "boom.GlmCoefs")
        .def("set_coefficients", [](TRegressionModel &m,
                                    const Vector &coefficients) {
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

    py::class_<CompleteDataStudentRegressionModel,
               TRegressionModel,
               Ptr<CompleteDataStudentRegressionModel>>(
                   boom,
                   "CompleteDataStudentRegressionModel",
                   py::multiple_inheritance())
        .def(py::init(
            [] (int xdim) {
              return new CompleteDataStudentRegressionModel(xdim);
            }),
             py::arg("xdim"),
             "Args:\n\n"
             "  xdim:  Dimension of the predictor vector.\n")
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
        .def("add_data",
             [](BinomialLogitModel &model,
                const Ptr<BinomialRegressionData> &data_point) {
               model.add_data(data_point);
             },
             py::arg("data_point"),
             "Args:\n\n"
             "  data_point: An object of class BinomialRegressionData "
             "containing the data for a single observation.\n")
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
             "    A value less than 0, or a nan is interpreted as missing. \n")
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
                const std::vector<int> &data_values) {
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


    //=========================================================================
    // Priors and posterior samplers.
    //=========================================================================
    py::class_<VariableSelectionPrior,
               Ptr<VariableSelectionPrior>>(boom, "VariableSelectionPrior")
        .def(py::init<const Vector &>(),
             py::arg("prior_inclusion_probabilities"),
             "Create a VariableSelectionPrior from a vector of prior "
             "inclusion probabilities.\n\n"
             "Args:\n\n"
             "  prior_inclusion_probabilities: boom.Vector containing the "
             "prior probability that each variable is to be included. "
             )
        ;

    py::class_<RegressionConjugateSampler,
               PosteriorSampler,
               Ptr<RegressionConjugateSampler>>(
                   boom, "RegressionConjugateSampler")
        .def(py::init(
            [](RegressionModel *model,
               MvnGivenScalarSigmaBase *coefficient_prior,
               GammaModelBase *residual_precision_prior,
               RNG &seeding_rng) {
              return new RegressionConjugateSampler(
                  model,
                  coefficient_prior,
                  residual_precision_prior,
                  seeding_rng);
            }),
             py::arg("model"),
             py::arg("coefficient_prior"),
             py::arg("residual_precision_prior"),
             py::arg("seeding_rng") = BOOM::GlobalRng::rng,
             "Args:\n\n"
             "  model:  The regression model to be sampled.\n"
             "  coefficient_prior:  A conditionally Gaussian prior for the "
             "regression coefficients.\n"
             "  residual_precision_prior:  Prior distribution for the "
             "residual precision parameter.\n"
             "  seeding_rng:  The random number generator used to set the seed "
             "of the RNG owned by this sampler."
             )
        .def("set_sigma_upper_limit",
             [](RegressionConjugateSampler *sampler,
                double sigma_max) {
               sampler->set_sigma_upper_limit(sigma_max);
             },
             py::arg("sigma_max"),
             "Truncate the support of the residual standard deviation "
             "parameter to (0, sigma_max).\n")
        ;

    py::class_<IndependentRegressionModelsPosteriorSampler,
               PosteriorSampler,
               Ptr<IndependentRegressionModelsPosteriorSampler>>(
                   boom, "IndependentRegressionModelsPosteriorSampler")
        .def(py::init(
            [](IndependentRegressionModels *model,
               RNG &seeding_rng) {
              return new IndependentRegressionModelsPosteriorSampler(
                  model, seeding_rng);
            }),
             py::arg("model"),
             py::arg("seeding_rng") = BOOM::GlobalRng::rng,
             "Args:\n\n"
             "  model:  The model to be sampled.\n"
             "  seeding_rng:  The RNG used to initialize the RNG owned by "
             "this object.\n")
        ;

    py::class_<IndependentGlmsPosteriorSampler<TRegressionModel>,
               PosteriorSampler,
               Ptr<IndependentGlmsPosteriorSampler<TRegressionModel>>>(
                   boom,
                   "IndependentStudentRegressionsPosteriorSampler",
                   py::multiple_inheritance())
        .def(py::init(
            [](IndependentGlms<TRegressionModel> *model,
               RNG &seeding_rng) {
              return new IndependentGlmsPosteriorSampler<
                TRegressionModel>(model, seeding_rng);
            }),
             py::arg("model"),
             py::arg("seeding_rng") = BOOM::GlobalRng::rng,
             "Args:\n\n"
             "  model:  The model to be sampled.\n"
             "  seeding_rng:  The RNG used to initialize the RNG owned by "
             "this object.\n")
        ;

    py::class_<IndependentGlmsPosteriorSampler<CompleteDataStudentRegressionModel>,
               PosteriorSampler,
               Ptr<IndependentGlmsPosteriorSampler<CompleteDataStudentRegressionModel>>>(
                   boom,
                   "IndependentCompleteDataStudentRegressionsPosteriorSampler",
                   py::multiple_inheritance())
        .def(py::init(
            [](IndependentGlms<CompleteDataStudentRegressionModel> *model,
               RNG &seeding_rng) {
              return new IndependentGlmsPosteriorSampler<
                CompleteDataStudentRegressionModel>(model, seeding_rng);
            }),
             py::arg("model"),
             py::arg("seeding_rng") = BOOM::GlobalRng::rng,
             "Args:\n\n"
             "  model:  The model to be sampled.\n"
             "  seeding_rng:  The RNG used to initialize the RNG owned by "
             "this object.\n")
        ;

    py::class_<BregVsSampler,
               PosteriorSampler,
               Ptr<BregVsSampler>>(boom, "BregVsSampler")
        .def(py::init(
            [](RegressionModel *model,
               MvnGivenScalarSigma *slab,
               GammaModelBase *residual_precision_prior,
               VariableSelectionPrior *spike,
               RNG &seeding_rng) {
              return new BregVsSampler(
                  model,
                  Ptr<MvnGivenScalarSigma>(slab),
                  Ptr<GammaModelBase>(residual_precision_prior),
                  Ptr<VariableSelectionPrior>(spike),
                  seeding_rng);
            }),
             py::arg("model"),
             py::arg("slab"),
             py::arg("residual_precision_prior"),
             py::arg("spike"),
             py::arg("seeding_rng") = BOOM::GlobalRng::rng,
             "Create a BregVsSampler -- a spike and slab sampler for "
             "regression models.\n\n"
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

    py::class_<TRegressionSampler,
               PosteriorSampler,
               Ptr<TRegressionSampler>>(
                   boom,
                   "TRegressionSampler",
                   py::multiple_inheritance())
        .def(py::init(
            [](TRegressionModel *model,
               MvnBase *coefficient_prior,
               GammaModelBase *siginv_prior,
               DoubleModel *tail_thickness_prior,
               RNG &seeding_rng) {
              return new TRegressionSampler(
                  model,
                  coefficient_prior,
                  siginv_prior,
                  tail_thickness_prior,
                  seeding_rng);
            }),
             py::arg("model"),
             py::arg("coefficient_prior"),
             py::arg("siginv_prior"),
             py::arg("tail_thickness_prior"),
             py::arg("seeding_rng") = GlobalRng::rng,
            "Args:\n\n"
             "  model: The boom.TRegressionModel that the sampler will simulate"
             " for.\n"
             "  coefficient_prior:  A boom.MvnBase object giving the prior "
             "distribution on the regression coefficients\n"
            "  residual_precision_prior: A boom.GammaModelBase giving the "
            "prior distribution on the residual precision parameter (one "
            "over the residual variance).\n"
            "  tail_thickness_prior: A boom.DoubleModel with support on "
            "a subset of the positive real line.\n"
             "  seeding_rng:  The random number generator used to set the seed "
             "of the RNG owned by this sampler.")
        .def("set_sigma_upper_limit",
             [](TRegressionSampler *sampler,
                double sigma_max) {
               sampler->set_sigma_upper_limit(sigma_max);
             },
             py::arg("sigma_max"),
             "Truncate the support of the residual standard deviation "
             "parameter to (0, sigma_max).\n\n"
             "Args:\n\n"
             "  sigma_max:  Any non-negative value, including zero and "
             "infinity() is allowed.")
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
            py::arg("seeding_rng"),
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
             "  seeding_rng:  The random number generator used to set the seed "
             "of the RNG owned by this sampler."
             )
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
             "  spike:  A boom.VariableSelectionPrior describing which "
             "variables are included in the model.\n"
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
        .def("set_prior_inclusion_probability",
             [](BinomialLogitSpikeSlabSampler *sampler, double prob) {
               if (prob < 0 || prob > 1) {
                 report_error("The 'prob' parameter must be between 0 and 1.");
               }
               int xdim = sampler->xdim();
               NEW(VariableSelectionPrior, spike)(xdim, prob);
               sampler->set_spike(spike);
             },
             py::arg("prob"),
             "Set all prior inclusion probabilities to the same value.\n\n"
             "Args:\n\n"
             "  prob:  The new value for all the prior inclusion probabilities."
             "  This parameter must satisfy 0 \le prob \le 1. \n")
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
             "  spike:  A boom.VariableSelectionPrior describing which "
             "variables are included in the model.\n"
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

    py::class_<BigAssSpikeSlabSampler,
               PosteriorSampler,
               Ptr<BigAssSpikeSlabSampler>>(
                       boom, "BigAssSpikeSlabSampler")
        .def(py::init([](BigRegressionModel *model,
                         VariableSelectionPrior *global_spike,
                         RegressionSlabPrior *slab_prototype,
                         GammaModelBase *residual_precision_prior,
                         RNG &seeding_rng){
                        return new BigAssSpikeSlabSampler(
                            model, global_spike, slab_prototype,
                            residual_precision_prior, seeding_rng);
                      }),
             py::arg("model"),
             py::arg("global_spike"),
             py::arg("slab_prototype"),
             py::arg("residual_precision_prior"),
             py::arg("seeding_rng_") = BOOM::GlobalRng::rng,
             "Args:\n\n"
             "  model:  The boom.BigRegressionModel to be sampled.\n"
             "  global_spike:  A boom.VariableSelectionPrior describing which "
             "variables are included in the model.\n"
             "  slab_prototype: A boom.RegressionSlabPrior prior for the "
             "conditional distribution of "
             "the regression coefficients given inclusion.\n"
             "  seeding_rng:  The random number generator used to set the seed "
             "of the RNG owned by this sampler.\n"
             )
        .def("initial_screen",
             [](BigAssSpikeSlabSampler &sampler, int niter, double threshold,
                bool use_threads) {
               sampler.initial_screen(niter, threshold, use_threads);
             },
             py::arg("niter"),
             py::arg("threshold"),
             py::arg("use_threads") = true,
             "Args:\n\n"
             "  niter:  The number of MCMC iterations to use in the initial "
             "screen.\n"
             "  threshold: The variables whose 'marginal inclusion "
             " probabilities' exceed 'threshold' become candidates "
             "in the next round.\n"
             "  use_threads: If 'True' then C++11 threads will be used to run "
             "the MCMC algorithms for the subordinate models.  If 'False' then "
             "the code path for doing the MCMC will not use threads.\n")
        .def("draw", [](BigAssSpikeSlabSampler &sampler) {sampler.draw();})
        ;

  }  // module definition

}  // namespace BayesBoom
