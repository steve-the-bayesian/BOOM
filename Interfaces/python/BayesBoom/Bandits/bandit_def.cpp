#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <functional>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "Bandits/BinomialBandit.hpp"
#include "Bandits/LogitBandit.hpp"
#include "Bandits/LogitBanditExternalValue.hpp"
#include "Bandits/LinearBanditEncoder.hpp"
#include "Bandits/bandit_functions.hpp"

#include "Models/BinomialModel.hpp"
#include "Models/BetaModel.hpp"
#include "Models/MvnBase.hpp"
#include "Models/Glm/BinomialLogitModel.hpp"

#include "Models/PosteriorSamplers/BetaBinomialSampler.hpp"
#include "Models/Glm/PosteriorSamplers/BinomialLogitAuxmixSampler.hpp"

// #include "boom_functional.hpp"

#include "stats/DataTable.hpp"
#include "stats/Encoders.hpp"

#include "distributions/rng.hpp"

#include "cpputil/Ptr.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;

  void Bandit_def(py::module &boom) {

    // =========================================================================
    // ArmMap: bijective mapping between arm indices and factor-level
    // combinations.
    py::class_<ArmMap, Ptr<ArmMap>>(boom, "ArmMap")
        .def(py::init(
            [](const ExperimentStructure &xp) {
              return new ArmMap(xp);
            }),
             py::arg("experiment_structure"),
             "Args:\n\n"
             "  experiment_structure: An ExperimentStructure describing the "
             "factors and levels in the experiment.  The number of arms equals "
             "the product of the number of levels for each factor.\n")
        .def_property_readonly("number_of_arms", &ArmMap::number_of_arms,
             "The total number of arms in the experiment.")
        .def_property_readonly("number_of_factors", &ArmMap::number_of_factors,
             "The number of experimental factors.")
        .def_property_readonly("factor_names", &ArmMap::factor_names,
             "The names of the experimental factors.")
        .def("integer_factor_levels",
             [](const ArmMap &am, int arm) {
               return am.integer_factor_levels(arm);
             },
             py::arg("arm"),
             "Args:\n"
             "  arm: The arm index (0-based).\n\n"
             "Returns the integer factor level indices for the given arm.")
        .def_property_readonly(
            "factor_level_matrix",
            [](const ArmMap &am) {
              Matrix ans(am.number_of_arms(), am.number_of_factors());
              for (int i = 0; i < am.number_of_arms(); ++i) {
                const std::vector<int> &levels(am.integer_factor_levels(i));
                for (int j = 0; j < am.number_of_factors(); ++j) {
                  ans(i, j) = levels[j];
                }
              }
              return ans;
            },
            "The full matrix of arm mappings.")
        .def("factor_level_names",
             [](const ArmMap &am,
                int arm) {
               return am.factor_level_names(arm);
             },
             py::arg("arm"),
             "Args:\n"
             "  arm: The arm index (0-based).\n\n"
             "Returns the string factor level names for the given arm.")
        .def("__repr__",
             [](const ArmMap &am) { return am.to_string(); })
        ;

    // =========================================================================
    // ExperimentArmEncoder: encodes one experimental factor as effects-coded
    // predictors for a LinearBanditEncoder.
    py::class_<ExperimentArmEncoder,
               MainEffectEncoder,
               Ptr<ExperimentArmEncoder>>(boom, "ExperimentArmEncoder")
        .def(py::init(
            [](const std::string &variable_name,
               const Ptr<ArmMap> &arm_map,
               const std::string &baseline_level) {
              return new ExperimentArmEncoder(
                  variable_name, arm_map, baseline_level);
            }),
             py::arg("variable_name"),
             py::arg("arm_map"),
             py::arg("baseline_level") = "",
             "Args:\n\n"
             "  variable_name: The name of the experimental factor this "
             "encoder handles.  Must match a factor name in arm_map.\n"
             "  arm_map: The ArmMap describing the experiment.\n"
             "  baseline_level: The reference level for effects coding.  "
             "If empty, the last level is used as the baseline.\n")
        .def_property_readonly("dim", &ExperimentArmEncoder::dim,
             "The number of encoded predictor columns for this factor "
             "(nlevels - 1 for effects coding).")
        ;

    // =========================================================================
    // LinearBanditEncoder: combines per-arm and context encoders into a single
    // predictor vector suitable for a generalized linear model.
    py::class_<LinearBanditEncoder, Ptr<LinearBanditEncoder>>(
        boom, "LinearBanditEncoder")
        .def(py::init(
            [](const Ptr<ArmMap> &arm_map,
               const Ptr<DatasetEncoder> &dataset_encoder) {
              return new LinearBanditEncoder(arm_map, dataset_encoder);
            }),
             py::arg("arm_map"),
             py::arg("dataset_encoder"),
             "Args:\n\n"
             "  arm_map: The ArmMap describing the experiment arms.\n"
             "  dataset_encoder: A DatasetEncoder combining the arm encoders "
             "(ExperimentArmEncoder objects) and any context encoders.\n")
        .def_property_readonly("number_of_arms",
             &LinearBanditEncoder::number_of_arms,
             "The number of arms in the experiment.")
        .def("encode_dataset",
             [](LinearBanditEncoder &enc,
                const DataTable &input_data) {
               return enc.encode_dataset(input_data);
             },
             py::arg("input"),
             "Encode the input data from a past run.  Past action variables "
             "are assumed present in the data.  The column names of any "
             "ExperimentArmEncoder's must be present in the column names of "
             "the data table, and any table values must match those expected "
             "by the encoders.")
        .def("encode_row",
             [](LinearBanditEncoder &enc,
                int arm,
                const MixedMultivariateData &context) {
               return enc.encode_row(arm, context);
             },
             py::arg("arm"),
             py::arg("context"),
             "Encode the arm and context into a single predictor vector.\n\n"
             "Args:\n"
             "  arm: The arm index (0-based).\n"
             "  context: A MixedMultivariateData with the context variables "
             "for this observation.\n\n"
             "Returns:\n"
             "  A boom.Vector suitable as input to the logistic regression "
             "model.\n")
        .def_property_readonly(
            "encoded_variable_names",
            [](const LinearBanditEncoder &enc) {
              return enc.encoded_variable_names();
            })
        ;

    // =========================================================================
    // GenericBanditBase: abstract base class for all bandit types.
    py::class_<GenericBanditBase, Ptr<GenericBanditBase>>(
        boom, "GenericBanditBase")
        .def_property_readonly("number_of_arms",
             &GenericBanditBase::number_of_arms,
             "The number of arms in the bandit.")
        ;

    // =========================================================================
    // BinomialBandit: multi-armed bandit for independent success/failure arms.
    //
    // Each arm maintains an independent BinomialModel with a Beta conjugate
    // prior.  Thompson sampling is performed by calling update_posterior(),
    // which draws from each arm's posterior and accumulates statistics about
    // which arm is most likely to be optimal.
    py::class_<BinomialBandit, GenericBanditBase, Ptr<BinomialBandit>>(
        boom, "BinomialBandit")
        .def(py::init(
            [](const std::vector<Ptr<BinomialModel>> &models) {
              return new BinomialBandit(models);
            }),
             py::arg("models"),
             "Args:\n\n"
             "  models: A list of BinomialModel objects, one per arm.  Each "
             "model should have a BetaBinomialSampler set via "
             "model.set_method() before update_posterior() is called.\n")
        .def("observe_data",
             [](BinomialBandit &bandit,
                int arm,
                int num_successes,
                int num_trials) {
               bandit.observe_data(arm, num_successes, num_trials);
             },
             py::arg("arm"),
             py::arg("num_successes"),
             py::arg("num_trials"),
             "Record an observed batch outcome for the specified arm.\n\n"
             "Args:\n"
             "  arm: The arm index (0-based).\n"
             "  num_successes: Number of successes in this batch.\n"
             "  num_trials: Number of trials in this batch.\n")
        .def("update_posterior",
             [](BinomialBandit &bandit, int ndraws) {
               bandit.update_posterior(ndraws);
             },
             py::arg("ndraws"),
             "Draw samples from the posterior distribution of each arm's "
             "success probability and compute optimal arm probabilities.\n\n"
             "Args:\n"
             "  ndraws: Number of posterior samples to draw.\n")
        .def("value",
             [](const BinomialBandit &bandit, int arm) {
               return bandit.value(arm);
             },
             py::arg("arm"),
             "Return the current posterior mean success probability for the "
             "specified arm.\n\n"
             "Args:\n"
             "  arm: The arm index (0-based).\n")
        .def_property_readonly(
            "optimal_arm_probabilities",
            [](const BinomialBandit &bandit) {
              return bandit.optimal_arm_probabilities();
            },
            "A Vector giving the probability that each arm is optimal, "
            "estimated from the most recent call to update_posterior().  "
            "Element i is the fraction of posterior draws in which arm i "
            "had the highest success probability.  Requires a prior call to "
            "update_posterior().")
        .def_property_readonly(
            "value_remaining_distribution",
            [](const BinomialBandit &bandit) {
              return bandit.value_remaining_distribution();
            },
            "A Vector of length ndraws giving the value-remaining statistic "
            "for each posterior draw from the most recent update_posterior() "
            "call.  Element i is the success probability of the best arm "
            "minus the success probability of arm 0 in that draw.  Requires "
            "a prior call to update_posterior().")
        ;

    // =========================================================================
    // LogitBandit: contextual multi-armed bandit using logistic regression.
    //
    // Models the arm-specific success probability as a logistic regression on
    // the arm indicator variables and optional context covariates.  Thompson
    // sampling is performed by drawing from the posterior distribution of the
    // regression coefficients.
    py::class_<LogitBandit, GenericBanditBase, Ptr<LogitBandit>>(
        boom, "LogitBandit")
        .def(py::init(
            [](const Ptr<BinomialLogitModel> &model,
               const Ptr<LinearBanditEncoder> &encoder) {
              return new LogitBandit(model, encoder);
            }),
             py::arg("model"),
             py::arg("encoder"),
             "Args:\n\n"
             "  model: A BinomialLogitModel whose xdim matches the output "
             "dimension of encoder.  A BinomialLogitAuxmixSampler (or other "
             "suitable sampler) must be attached via model.set_method() before "
             "update_posterior() is called.\n"
             "  encoder: A LinearBanditEncoder that maps (arm, context) pairs "
             "to predictor vectors for the logistic regression model.\n")
        .def("observe_data",
             [](LogitBandit &bandit,
                int arm,
                int num_successes,
                int num_trials,
                const MixedMultivariateData &context) {
               bandit.observe_data(arm, num_successes, num_trials, context);
             },
             py::arg("arm"),
             py::arg("num_successes"),
             py::arg("num_trials"),
             py::arg("context"),
             "Record an observed outcome for the specified arm and context.\n\n"
             "Args:\n"
             "  arm: The arm index (0-based).\n"
             "  num_successes: Number of successes observed.\n"
             "  num_trials: Number of trials observed.\n"
             "  context: A MixedMultivariateData with the context variables "
             "for this observation.  Use an empty MixedMultivariateData() when "
             "there are no context variables.\n")
        .def("value",
             [](const LogitBandit &bandit,
                int arm,
                const MixedMultivariateData &context) {
               return bandit.value(arm, context);
             },
             py::arg("arm"),
             py::arg("context"),
             "Return the predicted success probability for the given arm and "
             "context given the current model parameters.\n\n"
             "Args:\n"
             "  arm: The arm index (0-based).\n"
             "  context: The context data for this subject.\n")
        .def("update_posterior",
             [](LogitBandit &bandit, int ndraws) {
               bandit.update_posterior(ndraws);
             },
             py::arg("ndraws"),
             "Draw samples from the posterior distribution of the logistic "
             "regression coefficients.\n\n"
             "Args:\n"
             "  ndraws: Number of posterior samples to draw.\n")
        .def_property_readonly(
            "ndraws",
            &LogitBandit::ndraws,
             "The number of posterior draws from the most recent call to "
             "update_posterior().")
        .def_property_readonly(
            "coefficient_draws",
            [](const LogitBandit &bandit) {return bandit.draws();},
            "The matrix of MCMC draws of the model coefficients.  Row 'i' "
            "is the coefficient vector for MCMC draw i.")
        .def("set_coefficient_draws",
             [](LogitBandit &bandit, const Matrix &draws) {bandit.set_draws(draws);},
             py::arg("draws"),
             "Args:\n"
             "  draws: A boom.Matrix, with each row containing a posterior "
             "draw of the model coefficients.  It is an error to call this "
             "function unless the number of columns in 'draws' matches the "
             "number of predictors in the model.")
        .def("set_log_likelihood",
             [](LogitBandit &bandit, const Vector &log_likelihood){
               bandit.set_log_likelihood(log_likelihood);
             },
             "Args: \n"
             "  log_likelihood:  A Vector of log likelihood values from "
             "a previous MCMC run.\n")
        .def_property_readonly(
            "log_likelihood",
            [](const LogitBandit &bandit) {return bandit.log_likelihood();},
            "The log likelihood of each draw in the simulated model "
            "coefficients.")
        .def("arm_predictors",
             [](const LogitBandit &bandit,
                const MixedMultivariateData &context) {
               return bandit.arm_predictors(context);
             },
             py::arg("context"),
             "Return the predictor matrix for all arms given the context.\n\n"
             "Args:\n"
             "  context: The context data for this subject.\n\n"
             "Returns:\n"
             "  A boom.Matrix with one row per arm and one column per "
             "predictor variable.\n")
        .def("optimal_arm_probabilities",
             [](const LogitBandit &bandit,
                const MixedMultivariateData &context,
                RNG &rng) {
               return bandit.optimal_arm_probabilities(context, rng);
             },
             py::arg("context"),
             py::arg("rng") = BOOM::GlobalRng::rng,
             "Computes probability that each arm is "
             "optimal for the given context, using the posterior draws from "
             "the most recent update_posterior() call.\n\n"
             "Args:\n"
             "  context: The context data for this subject.\n"
             "  rng: Optional boom random number generator.\n\n"
             "Returns:\n"
             "  A boom.Vector of probabilities, one per arm, summing to 1.\n")
        .def("thompson",
             [](const LogitBandit &bandit,
                const MixedMultivariateData &context,
                RNG &rng) { return bandit.thompson(context, rng); },
             py::arg("context"),
             py::arg("rng") = GlobalRng::rng,
             "Return one draw of Thompson sampling for the bandit.  This does "
             "not update the posterior distribution.  It samples one set of "
             "model parameters from the set of posterior draws, calls "
             "'optimal_arm_probabilities' assuming that draw is the true set "
             "of parameters, and returns the values of the chosen arm.")
        .def_property_readonly(
            "last_thompson_row",
            [](const LogitBandit &bandit) {return bandit.last_thompson_row();})
        .def_property_readonly(
            "last_thompson_arm",
            [](const LogitBandit &bandit) {return bandit.last_thompson_arm();})
        .def("value_remaining_distribution",
             [](const LogitBandit &bandit,
                const MixedMultivariateData &context,
                RNG &rng) {
               return bandit.value_remaining_distribution(context, rng);
             },
             py::arg("context"),
             py::arg("rng") = BOOM::GlobalRng::rng,
             "Compute the distribution of value remaining given context.\n\n"
             "Args:\n"
             "  context: The context data for this subject.\n"
             "  rng: Optional boom random number generator.\n\n"
             "Returns:\n"
             "  A boom.Vector of length ndraws, where each element is the "
             "difference between the best arm's predicted success probability "
             "and arm 0's predicted success probability in that posterior "
             "draw.\n")
        ;

    py::class_<LogitBanditExternalValue,
               LogitBandit,
               Ptr<LogitBanditExternalValue>>(boom, "LogitBanditExternalValue")
        .def(py::init(
            [](const Ptr<BinomialLogitModel> &model,
               const Ptr<LinearBanditEncoder> &encoder,
               const LogitBanditExternalValue::ValueFunctionType &value_function) {
              return new LogitBanditExternalValue(model, encoder, value_function);
            }),
             py::arg("model"),
             py::arg("encoder"),
             py::arg("value_function"),
             "Args:\n"
             "  model: A Boom.BinomialLogitModel describing the probability "
             "of success.\n"
             "  encoder:  A Boom.LinearBanditEncoder describing the 'model "
             "formula' for the bandit.\n"
             "  value_function:  A functor taking a 'prob' and an 'args' "
             "argument giving the value of the bandit under a specified success"
             " probability assuming the arms are defined with the given levels."
             "  The order of the arguments is the order of the variables "
             "defined in the ExperimentStructure object contained in the "
             "encoder.\n")
        .def("value",
             [](const LogitBanditExternalValue &bandit,
                int arm,
                const MixedMultivariateData &context) {
               return bandit.value(arm, context);
             },
             py::arg("arm"),
             py::arg("context"),
             "Args:\n"
             "  arm: the integer valued index of the arm being evaluated.\n"
             "  context:  The context data for the user.\n\n"
             "Returns:\n"
             "  The value of the given user under the specified arm.\n")
        .def("optimal_arm_probabilities",
             [](const LogitBanditExternalValue &bandit,
                const MixedMultivariateData &context,
                RNG &rng) {
               return bandit.optimal_arm_probabilities(context, rng);
             },
             py::arg("context"),
             py::arg("rng") = BOOM::GlobalRng::rng,
             "Computes probability that each arm is "
             "optimal for the given context, using the posterior draws from "
             "the most recent update_posterior() call.\n\n"
             "Args:\n"
             "  context: The context data for this subject.\n"
             "  rng: Optional boom random number generator.\n\n"
             "Returns:\n"
             "  A boom.Vector of probabilities, one per arm, summing to 1.\n")
        .def("value_remaining_distribution",
             [](const LogitBanditExternalValue &bandit,
                const MixedMultivariateData &context,
                RNG &rng) {
               return bandit.value_remaining_distribution(context, rng);
             },
             py::arg("context"),
             py::arg("rng") = GlobalRng::rng,
             "Compute the distribution of value remaining given context.\n\n"
             "Args:\n"
             "  context: The context data for this subject.\n"
             "  rng: Optional boom random number generator.\n\n"
             "Returns:\n"
             "  A boom.Vector of length ndraws, where each element is the "
             "difference between the best arm's predicted success probability "
             "and arm 0's predicted success probability in that posterior "
             "draw.\n")
        ;

  }  // Bandit_def

}  // namespace BayesBoom
