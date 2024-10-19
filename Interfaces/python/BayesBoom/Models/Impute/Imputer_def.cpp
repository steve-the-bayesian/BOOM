#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Models/ModelTypes.hpp"

#include "Models/PosteriorSamplers/MultinomialDirichletSampler.hpp"

#include "Models/Glm/Glm.hpp"
#include "Models/Glm/GlmCoefs.hpp"
#include "Models/Glm/PosteriorSamplers/MultivariateRegressionSampler.hpp"

#include "Models/Impute/MvRegCopulaDataImputer.hpp"
#include "Models/Impute/MixedDataImputer.hpp"
#include "Models/Impute/MixedDataImputerWithErrorCorrection.hpp"

#include "cpputil/Ptr.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;

  void Imputation_def(py::module &boom) {
    py::class_<MvRegCopulaDataImputer,
               Ptr<MvRegCopulaDataImputer>>(boom, "MvRegCopulaDataImputer")
        .def(py::init(
            [](int num_clusters,
               const std::vector<Vector> &atoms,
               int xdim,
               BOOM::RNG &seeding_rng) {
              return new MvRegCopulaDataImputer(num_clusters, atoms,
                                                xdim, seeding_rng);
            }),
             py::arg("num_clusters"),
             py::arg("atoms"),
             py::arg("xdim"),
             py::arg("seeding_rng") = BOOM::GlobalRng::rng,
             "Args:\n"
             "  num_clusters:  The number of clusters in the patern "
             "matching model \n"
             "    that handles data errors. \n"
             "  atoms: A collection of Vectors (or 1-d numpy arrays) "
             "containing \n"
             "    values that will receive special modeling treatment.  "
             "One entry is\n"
             "    needed for each variable.  An entry can be the empty "
             "Vector. \n"
             "  xdim:  The dimension of the predictor variable.\n"
             "  seeding_rng:  A boom random number generator used to seed the\n"
             "    RNG in this object.")
        .def("add_data",
             [](MvRegCopulaDataImputer &imputer,
                const Ptr<MvRegData> &data_point) {
               imputer.add_data(data_point);
             },
             py::arg("data_point"),
             "Add a data point to the training data set.\n\n"
             "Args:\n"
             "  data_point:  Object of type boom.MvRegData.  The y variable \n"
             "    should indicate missing values with nan.\n")
        .def_property_readonly("xdim", &MvRegCopulaDataImputer::xdim,
                               "dimension of the predictor variable")
        .def_property_readonly("ydim", &MvRegCopulaDataImputer::ydim,
                               "dimension of the numeric data")
        .def_property_readonly(
            "coefficients",
            [](MvRegCopulaDataImputer &imputer) {
              Matrix ans = imputer.regression()->Beta();
              return ans;
            },
            "The matrix of regression coefficients.  Rows correspond to "
            "Y (output).\n"
            "Columns correspond to X (input).  Coefficients represent the \n"
            "relationship between X and the copula transform of Y.")
        .def_property_readonly(
            "residual_variance",
            [](MvRegCopulaDataImputer &imputer) {
              SpdMatrix ans = imputer.regression()->Sigma();
              return ans;
            },
            "The residual variance matrix on the transformed (copula) scale.")
        .def_property_readonly(
            "nclusters",
            &MvRegCopulaDataImputer::nclusters,
            "The number of clusters in the error pattern matching model.")
        .def_property_readonly("imputed_data",
                               &MvRegCopulaDataImputer::imputed_data,
                               "The numeric portion of the imputed data set.")
        .def_property_readonly("atoms",
                               &MvRegCopulaDataImputer::atoms,
                               "The atoms for each y variable.")
        .def("atom_probs",
             [](const MvRegCopulaDataImputer &imputer, int cluster,
                int variable_index) {
               return imputer.atom_probs(cluster, variable_index);
             },
             "The marginal probability that each atom is the 'truth'.")
        .def("atom_error_probs",
             [](const MvRegCopulaDataImputer &imputer, int cluster,
                int variable_index) {
               return imputer.atom_error_probs(cluster, variable_index);
             },
             "The marginal probability that each atom is the 'truth'.")
        .def("set_default_priors",
             &MvRegCopulaDataImputer::set_default_priors,
             "Set default priors on everything.")
        .def("set_default_regression_prior",
             &MvRegCopulaDataImputer::set_default_regression_prior,
             "Set a 'nearly flat' prior on the regression coefficients and "
             "residual variance.")
        .def("set_default_prior_for_mixing_weights",
             &MvRegCopulaDataImputer::set_default_prior_for_mixing_weights)
        .def("set_atom_prior",
             [](MvRegCopulaDataImputer &imputer,
                const Vector &prior_counts,
                int variable_index) {
               imputer.set_atom_prior(prior_counts, variable_index);
             },
             "Args:\n"
             "  prior_counts: Vector of prior counts indicating the "
             "likelihood \n"
             "    that each atom is the true value.  Negative counts "
             "indicate\n"
             "    an a-priori assertion that the level cannot be the "
             "true value.\n"
             "    The size of the vector must be one larger than the "
             "number of \n"
             "    atoms, with the final element corresponding to the "
             "continuous atom.\n"
             "  variable_index:  Index of the variable to which the prior "
             "refers.\n")
        .def("set_atom_error_prior",
             [](MvRegCopulaDataImputer &imputer,
                const Matrix &prior_counts,
                int variable_index) {
               imputer.set_atom_error_prior(prior_counts, variable_index);
             },
             "TODO: docstring")
        .def("sample_posterior",
             [](MvRegCopulaDataImputer &imputer) {
               imputer.sample_posterior();
             },
             "Take one draw from the posterior distribution")
        .def("impute_data_set",
             [](MvRegCopulaDataImputer &imputer,
                const std::vector<Ptr<MvRegData>> &data) {
               return imputer.impute_data_set(data);
             },
             "Return a Boom.Matrix containing the imputed draws.")
        .def("set_residual_variance",
             [](MvRegCopulaDataImputer &imputer, const Matrix &Sigma) {
               imputer.regression()->set_Sigma(SpdMatrix(Sigma));
             })
        .def("set_coefficients",
             [](MvRegCopulaDataImputer &imputer, const Matrix &Beta) {
               imputer.regression()->set_Beta(Beta);
             })
        .def("set_atom_probs",
             &MvRegCopulaDataImputer::set_atom_probs)
        .def("set_atom_error_probs",
             &MvRegCopulaDataImputer::set_atom_error_probs)
        .def_property_readonly(
            "empirical_distributions",
            &MvRegCopulaDataImputer::empirical_distributions,
            "The approximate numerical distribution of each numeric variable")
        .def("set_empirical_distributions",
             &MvRegCopulaDataImputer::set_empirical_distributions,
             "Restore the empirical distributions from serialized state.")
        .def("setup_worker_pool",
             &MvRegCopulaDataImputer::setup_worker_pool,
             py::arg("nworkers"),
             "Set up a worker pool to train with 'nworkers' threads.")
        ;

    //==========================================================================
    py::class_<MixedDataImputer, Ptr<MixedDataImputer>>(
        boom, "MixedDataImputer")
        .def(py::init(
            [](int num_clusters,
               const DataTable &table,
               const std::vector<Vector> &atoms,
               RNG &rng = GlobalRng::rng) {
              return new MixedDataImputer(num_clusters, table, atoms, rng);
            }),
             py::arg("num_clusters"),
             py::arg("table"),
             py::arg("atoms"),
             py::arg("rng") = GlobalRng::rng,
             "Args:\n"
             "  num_clusters:  The number of clusters in the patern matching\n"
             "    model that handles data errors. \n"
             "  atoms: A collection of Vectors (or 1-d numpy arrays) "
             "containing\n"
             "    values that will receive special modeling treatment.  One \n"
             "    entry is needed for each numeric variable.  An entry can be\n"
             "    the empty Vector. \n"
             "  seeding_rng:  A boom random number generator used to seed the\n"
             "    RNG in this object.")
        .def("set_mixing_weight_prior",
             [](MixedDataImputer &imp, const Vector &prior_counts) {
               NEW(MultinomialDirichletSampler, sampler)(
                   imp.mixing_distribution().get(),
                   prior_counts,
                   imp.rng());
               imp.mixing_distribution()->clear_methods();
               imp.mixing_distribution()->set_method(sampler);
             },
             py::arg("prior_counts"),
             "Set a Diriclhet prior on the weights of the mixing"
             "distribution.\n\n"
             "Args:\n"
             "  prior_counts:  A vector of positive numbers with length "
             "matching the number of clusters.")
        .def("set_regression_prior",
             [](MixedDataImputer &imp,
                const Matrix &coefficient_mean,
                double coefficient_weight,
                const SpdMatrix &residual_variance_guess,
                double variance_weight) {
               NEW(MultivariateRegressionSampler, sampler)(
                   imp.numeric_data_model().get(),
                   coefficient_mean,
                   coefficient_weight,
                   variance_weight,
                   residual_variance_guess,
                   imp.rng());
               imp.numeric_data_model()->clear_methods();
               imp.numeric_data_model()->set_method(sampler);
             },
             py::arg("coefficient_mean"),
             py::arg("coefficient_weight"),
             py::arg("residual_variance_guess"),
             py::arg("residual_variance_weight"),
             "Set a conjugate matrix-normal Wishart prior on the parameters\n"
             "of the regression model describing the numeric data. \n\n"
             "Args:\n"
             "  coefficient_mean:  An xdim-by-ydim matrix.  Each column \n"
             "    gives the prior mean for the coefficients relating that \n"
             "    scalar y variable to x.\n"
             "  coefficient_weight:  The number of observations worth of \n"
             "    weight assigned to 'coefficient_mean'.\n"
             "  residual_variance_guess:  A ydim-by-ydim matrix containing \n"
             "    a guess at the residual variance matrix.\n"
             "  residual_variance_weight: The number of observations of \n"
             "    weight assigned to 'residual_variance_guess'.  This number\n"
             "    must be ydim or larger for the prior to be proper.")
        .def("set_atom_prior",
             [](MixedDataImputer &imp,
                const Vector &counts,
                int which_numeric_variable) {
               for (int i = 0; i < imp.nclusters(); ++i) {
                 imp.row_model(i)->numeric_model(
                     which_numeric_variable)->set_conjugate_prior(counts);
               }
             },
             "Args:\n"
             "  prior_counts: Vector of prior counts indicating the "
             "likelihood \n"
             "    that each atom is the true value.  Negative counts "
             "indicate\n"
             "    an a-priori assertion that the level cannot be the "
             "true value.\n"
             "    The size of the vector must be one larger than the "
             "number of \n"
             "    atoms, with the final element corresponding to the "
             "continuous atom.\n"
             "  which_numeric_variable:  Index of the numeric variable the \n"
             "    prior describes.  For purposes of this function numeric \n"
             "    variables are indexed 0, 1, 2, ... regardless of whether \n"
             "    there are intervening non-numeric variables.\n")
        .def("set_level_prior",
             [](MixedDataImputer &imp, const Vector &counts, int which_cat) {
               for (int i = 0; i < imp.nclusters(); ++i) {
                 imp.row_model(i)->categorical_model(which_cat)->
                     set_conjugate_prior(counts);
               }
             },
             py::arg("counts"),
             py::arg("which_cat"),
             "Set a constrained Diriclhet prior on the 'true' levels of a \n"
             "specific categorical variable.\n\n"
             "Args:\n"
             "  counts: A vector of prior counts for each level.  A \n"
             "    non-positive value indicates a-priori assurance that the \n"
             "    corressponding probability is zero.\n"
             "  which_cat:  The index of the categorical variable, ignoring \n"
             "    any intervening numeric variables." )
        .def("sample_posterior",
             &MixedDataImputerWithErrorCorrection::sample_posterior,
             "Take one MCMC draw from the posterior distribution.")
        .def("impute_data_set",
             [](MixedDataImputer &imputer, DataTable &table, int burn) {
               std::vector<Ptr<MixedImputation::CompleteData>> rows;
               for (int i = 0; i < table.nrow(); ++i) {
                 rows.push_back(
                     new MixedImputation::CompleteData(table.row(i)));
               }
               for (int i = 0; i < burn; ++i) {
                 imputer.impute_data_set(rows);
               }
               imputer.impute_data_set(rows);
               DataTable output_table;
               for (size_t i = 0; i < rows.size(); ++i) {
                 output_table.append_row(rows[i]->to_mixed_multivariate_data());
               }
               return output_table;
             },
             py::arg("table"),
             py::arg("burn"),
             "Args:\n"
             "  table: A boom.DataTable object containing the data to be"
             " imputed.\n"
             "  burn: The number of burn-in imputations to be discarded \n"
             "    before the final imputation is drawn.\n")
        .def_property_readonly(
            "ybar", [](MixedDataImputer &imputer) {return imputer.ybar();},
            "Sample mean of the numeric variables, returned as a "
            "Boom.boom.Vector.")
        .def_property_readonly(
            "nclusters", &MixedDataImputer::nclusters,
            "The number of clusters in the mixture portion of the model.")
        .def_property_readonly(
            "xdim", &MixedDataImputer::xdim,
            "Predictor dimension in the multivariate regression relating the\n"
            "numeric and categorical data.  The dimension of the categorical\n"
            "data after a dummy variable expansion.")
        .def_property_readonly(
            "ydim", &MixedDataImputer::ydim, "Number of numeric columns.")
        .def("sample_posterior",
             &MixedDataImputer::sample_posterior,
             "Take one MCMC draw from the posterior distribution.")
        .def_property_readonly(
            "coefficients",
            [](MixedDataImputer &imputer) {
              Matrix ans = imputer.numeric_data_model()->Beta();
              return ans;
            },
            "The matrix of regression coefficients.  Rows correspond to "
            "Y (output).\n"
            "Columns correspond to X (input).  Coefficients represent the \n"
            "relationship between X and the copula transform of Y.")
        .def("set_coefficients",
             [](MixedDataImputer &imputer, const Matrix &coefficients) {
               imputer.numeric_data_model()->set_Beta(coefficients);
             },
             py::arg("coefficients"),
             "Set the coefficients of the multivariate regression model.\n"
             "\n"
             "Args:\n"
             "  coefficients: an (xdim x ydim) matrix of coefficients.")
        .def_property_readonly(
            "residual_variance",
            [](MixedDataImputer &imputer) {
              SpdMatrix ans = imputer.numeric_data_model()->Sigma();
              return ans;
            },
            "The residual variance matrix on the transformed (copula) scale.")
        .def("set_residual_variance",
             [](MixedDataImputer &imputer, const SpdMatrix &residual_variance) {
               imputer.numeric_data_model()->set_Sigma(residual_variance);
             },
             py::arg("residual_variance"),
             "Args:\n"
             "  residual_variance:  The residual variance matrix for the "
             "multivariate regression model.")
        .def_property_readonly(
            "mixing_weights",
            [](MixedDataImputer &imputer) {
              return imputer.mixing_distribution()->pi();
            },
            "The mixing weights of the row-level mixture. ")
        .def("atom_probs",
             [](MixedDataImputer &imputer, int cluster, int numeric_index) {
               return imputer.row_model(cluster)->numeric_model(
                   numeric_index)->atom_probs();
             },
             py::arg("cluster"),
             py::arg("numeric_index"),
             "The atom probabilities for a particular combination of \n"
             "mixture component and numeric variable.\n\n"
             "Args:\n"
             "  cluster: Index of the mixture component.\n"
             "  numeric_index: The index of the desired numeric variable.  \n"
             "    This index counts from 0 and ignores any intervening \n"
             "    categorical variables. ")
        .def("set_atom_probs",
             [](MixedDataImputer &imputer, int cluster, int numeric_index,
                const Vector &atom_probs) {
               imputer.row_model(cluster)->numeric_model(
                   numeric_index)->set_atom_probs(atom_probs);
             })
        .def("level_probs",
             [](MixedDataImputer &imputer, int cluster, int cat_index) {
               return imputer.row_model(cluster)->categorical_model(
                   cat_index)->level_probs();
             },
             py::arg("cluster"),
             py::arg("cat_index"),
             "The level probabilities for a particular combination of \n"
             "mixture component and categorical variable.\n\n"
             "Args:\n"
             "  cluster: Index of the mixture component.\n"
             "  cat_index: The index of the desired numeric variable.  \n"
             "    This index counts from 0 and ignores any intervening \n"
             "    categorical variables. ")
        .def("set_level_probs",
             [](MixedDataImputer &imputer, int cluster, int cat_index,
                const Vector &level_probs) {
               return imputer.row_model(cluster)->categorical_model(
                   cat_index)->set_level_probs(level_probs);
             })
    ;

    //==========================================================================
    py::class_<MixedDataImputerWithErrorCorrection,
               Ptr<MixedDataImputerWithErrorCorrection>>(
                   boom, "MixedDataImputerWithErrorCorrection")
        .def(py::init(
            [](int num_clusters,
               const DataTable &table,
               const std::vector<Vector> &atoms,
               RNG &rng = GlobalRng::rng) {
              return new MixedDataImputerWithErrorCorrection(
                  num_clusters, table, atoms, rng);
            }),
             py::arg("num_clusters"),
             py::arg("table"),
             py::arg("atoms"),
             py::arg("rng") = GlobalRng::rng,
             "Args:\n"
             "  num_clusters:  The number of clusters in the patern matching\n"
             "    model that handles data errors. \n"
             "  atoms: A collection of Vectors (or 1-d numpy arrays) "
             "containing\n"
             "    values that will receive special modeling treatment.  One \n"
             "    entry is needed for each numeric variable.  An entry can be\n"
             "    the empty Vector. \n"
             "  seeding_rng:  A boom random number generator used to seed the\n"
             "    RNG in this object.")
        .def("set_mixing_weight_prior",
             [](MixedDataImputerWithErrorCorrection &imp,
                const Vector &prior_counts) {
               NEW(MultinomialDirichletSampler, sampler)(
                   imp.mixing_distribution().get(),
                   prior_counts,
                   imp.rng());
               imp.mixing_distribution()->clear_methods();
               imp.mixing_distribution()->set_method(sampler);
             },
             py::arg("prior_counts"),
             "Set a Diriclhet prior on the weights of the mixing"
             "distribution.\n\n"
             "Args:\n"
             "  prior_counts:  A vector of positive numbers with length "
             "matching the number of clusters.")
        .def("set_regression_prior",
             [](MixedDataImputerWithErrorCorrection &imp,
                const Matrix &coefficient_mean,
                double coefficient_weight,
                const SpdMatrix &residual_variance_guess,
                double variance_weight) {
               NEW(MultivariateRegressionSampler, sampler)(
                   imp.numeric_data_model().get(),
                   coefficient_mean,
                   coefficient_weight,
                   variance_weight,
                   residual_variance_guess,
                   imp.rng());
               imp.numeric_data_model()->clear_methods();
               imp.numeric_data_model()->set_method(sampler);
             },
             py::arg("coefficient_mean"),
             py::arg("coefficient_weight"),
             py::arg("residual_variance_guess"),
             py::arg("residual_variance_weight"),
             "Set a conjugate matrix-normal Wishart prior on the parameters\n"
             "of the regression model describing the numeric data. \n\n"
             "Args:\n"
             "  coefficient_mean:  An xdim-by-ydim matrix.  Each column \n"
             "    gives the prior mean for the coefficients relating that \n"
             "    scalar y variable to x.\n"
             "  coefficient_weight:  The number of observations worth of \n"
             "    weight assigned to 'coefficient_mean'.\n"
             "  residual_variance_guess:  A ydim-by-ydim matrix containing \n"
             "    a guess at the residual variance matrix.\n"
             "  residual_variance_weight: The number of observations of \n"
             "    weight assigned to 'residual_variance_guess'.  This number \n"
             "    must be ydim or larger for the prior to be proper.")
        .def("set_atom_prior",
             [](MixedDataImputerWithErrorCorrection &imp,
                const Vector &counts,
                int which_numeric_variable) {
               for (int i = 0; i < imp.nclusters(); ++i) {
                 imp.row_model(i)->numeric_model(which_numeric_variable)->
                     set_conjugate_prior_for_true_categories(counts);
               }
             },
             "Args:\n"
             "  prior_counts: Vector of prior counts indicating the "
             "likelihood \n"
             "    that each atom is the true value.  Negative counts indicate\n"
             "    an a-priori assertion that the level cannot be the "
             "true value.\n"
             "    The size of the vector must be one larger than the number "
             "of \n"
             "    atoms, with the final element corresponding to the "
             "continuous atom.\n"
             "  which_numeric_variable:  Index of the numeric variable the \n"
             "    prior describes.  For purposes of this function numeric \n"
             "    variables are indexed 0, 1, 2, ... regardless of whether \n"
             "    there are intervening non-numeric variables.\n")

        .def("set_atom_error_prior",
             [](MixedDataImputerWithErrorCorrection &imp,
                const Matrix &counts,
                int which_numeric_variable) {
               for (int i = 0; i < imp.nclusters(); ++i) {
                 imp.row_model(i)->numeric_model(which_numeric_variable)->
                     set_conjugate_prior_for_observation_categories(counts);
               }
             },
             "Set a constrained Diriclhet prior (i.e. zeros are possible) on\n"
             "the conditional distribution of the observed atom conditional \n"
             "on the true atom.\n\n"
             "Args:\n"
             "  counts:  If there are k atoms then this matrix has k+1 rows\n"
             "    and k+2 columns, where columns are viewed as conditional\n"
             "    given rows..  The first k elements refer to explicit\n"
             "    atoms.  The first implicit atom is the numeric part of the\n"
             "    model.  The final column indicates the observation is\n"
             "    missing.\n" )

        .def("set_level_prior",
             [](MixedDataImputerWithErrorCorrection &imp,
                const Vector &counts, int which_cat) {
               for (int i = 0; i < imp.nclusters(); ++i) {
                 imp.row_model(i)->categorical_model(which_cat)->
                     set_conjugate_prior_for_levels(counts);
               }
             },
             py::arg("counts"),
             py::arg("which_cat"),
             "Set a constrained Diriclhet prior on the 'true' levels of a \n"
             "specific categorical variable.\n\n"
             "Args:\n"
             "  counts: A vector of prior counts for each level.  A \n"
             "    non-positive value indicates a-priori assurance that the \n"
             "    corressponding probability is zero.\n"
             "  which_cat:  The index of the categorical variable, ignoring \n"
             "    any intervening numeric variables." )

        .def("set_level_observation_prior",
             [](MixedDataImputerWithErrorCorrection &imp,
                const Matrix &counts, int which_cat) {
               for (int i = 0; i < imp.nclusters(); ++i) {
                 imp.row_model(i)->categorical_model(which_cat)->
                     set_conjugate_prior_for_observations(counts);
               }
             },
             py::arg("counts"),
             py::arg("which_cat"),
             "Set a constrained Diriclhet prior on the 'true' levels of a \n"
             "specific categorical variable.\n\n"
             "Args:\n"
             "  counts: A matrix of prior counts for each level.  Each row \n"
             "    of the matrix models the conditional distribution of \n"
             "    observed levels given true levels.  Anon-positive value \n"
             "    indicates a-priori assurance that the corressponding \n"
             "    probability is zero.  Each row must have at least one \n"
             "    positive entry.\n"
             "  which_cat:  The index of the categorical variable, ignoring \n"
             "    any intervening numeric variables." )

        .def("sample_posterior",
             &MixedDataImputerWithErrorCorrection::sample_posterior,
             "Take one MCMC draw from the posterior distribution.")
        .def("impute_data_set",
             [](MixedDataImputerWithErrorCorrection &imputer,
                DataTable &table, int burn) {
               std::vector<Ptr<MixedImputation::CompleteData>> rows;
               for (int i = 0; i < table.nrow(); ++i) {
                 rows.push_back(
                     new MixedImputation::CompleteData(table.row(i)));
               }
               for (int i = 0; i < burn; ++i) {
                 imputer.impute_data_set(rows);
               }
               imputer.impute_data_set(rows);
               DataTable output_table;
               for (size_t i = 0; i < rows.size(); ++i) {
                 output_table.append_row(rows[i]->to_mixed_multivariate_data());
               }
               return output_table;
             },
             py::arg("table"),
             py::arg("burn"),
             "Args:\n"
             "  table: A boom.DataTable object containing the data to be"
             " imputed.\n"
             "  burn: The number of burn-in imputations to be discarded \n"
             "    before the final imputation is drawn.\n")
        .def_property_readonly(
            "nclusters", &MixedDataImputerWithErrorCorrection::nclusters,
            "The number of clusters in the mixture portion of the model.")
        .def_property_readonly(
            "xdim", &MixedDataImputerWithErrorCorrection::xdim,
            "Predictor dimension in the multivariate regression relating the\n"
            "numeric and categorical data.  The dimension of the categorical\n"
            "data after a dummy variable expansion.")
        .def_property_readonly(
            "ydim", &MixedDataImputerWithErrorCorrection::ydim,
            "Number of numeric columns.")
        .def_property_readonly(
            "coefficients",
            [](MixedDataImputerWithErrorCorrection &imputer) {
              Matrix ans = imputer.numeric_data_model()->Beta();
              return ans;
            },
            "The matrix of regression coefficients.  Rows correspond to "
            "Y (output).\n"
            "Columns correspond to X (input).  Coefficients represent the \n"
            "relationship between X and the copula transform of Y.")
        .def("set_coefficients",
             [](MixedDataImputerWithErrorCorrection &imputer,
                const Matrix &coefficients) {
               imputer.numeric_data_model()->set_Beta(coefficients);
             },
             py::arg("coefficients"),
             "Set the coefficients of the multivariate regression model.\n"
             "\n"
             "Args:\n"
             "  coefficients: an (xdim x ydim) matrix of coefficients.")
        .def_property_readonly(
            "residual_variance",
            [](MixedDataImputerWithErrorCorrection &imputer) {
              SpdMatrix ans = imputer.numeric_data_model()->Sigma();
              return ans;
            },
            "The residual variance matrix on the transformed (copula) scale.")
        .def("set_residual_variance",
             [](MixedDataImputerWithErrorCorrection &imputer,
                const SpdMatrix &residual_variance) {
               imputer.numeric_data_model()->set_Sigma(residual_variance);
             },
             py::arg("residual_variance"),
             "Args:\n"
             "  residual_variance:  The residual variance matrix for the "
             "multivariate regression model.")
        .def_property_readonly(
            "mixing_weights",
            [](MixedDataImputerWithErrorCorrection &imputer) {
              return imputer.mixing_distribution()->pi();
            },
            "The mixing weights of the row-level mixture. ")
        .def("atom_probs",
             [](MixedDataImputerWithErrorCorrection &imputer,
                int cluster, int numeric_index) {
               return imputer.row_model(cluster)->numeric_model(
                   numeric_index)->atom_probs();
             },
             py::arg("cluster"),
             py::arg("numeric_index"),
             "The atom probabilities for a particular combination of \n"
             "mixture component and numeric variable.\n\n"
             "Args:\n"
             "  cluster: Index of the mixture component.\n"
             "  numeric_index: The index of the desired numeric variable.  \n"
             "    This index counts from 0 and ignores any intervening \n"
             "    categorical variables. ")
        .def("set_atom_probs",
             [](MixedDataImputerWithErrorCorrection &imputer,
                int cluster, int numeric_index,
                const Vector &atom_probs) {
               imputer.row_model(cluster)->numeric_model(
                   numeric_index)->set_atom_probs(atom_probs);
             })
        .def("atom_error_probs",
             [](MixedDataImputerWithErrorCorrection &imputer,
                int cluster, int numeric_index) {
               return imputer.row_model(cluster)->numeric_model(
                   numeric_index)->atom_error_probs();
             },
             py::arg("cluster"),
             py::arg("numeric_index"),
             "The atom error probabilities for a particular combination of \n"
             "mixture component and numeric variable.\n\n"
             "Args:\n"
             "  cluster: Index of the mixture component.\n"
             "  numeric_index: The index of the desired numeric variable.  \n"
             "    This index counts from 0 and ignores any intervening \n"
             "    categorical variables. ")
        .def("set_atom_error_probs",
             [](MixedDataImputerWithErrorCorrection &imputer,
                int cluster, int numeric_index,
                const Matrix &atom_error_probs) {
               imputer.row_model(cluster)->numeric_model(
                   numeric_index)->set_atom_error_probs(
                       atom_error_probs);
             })
        .def("level_probs",
             [](MixedDataImputerWithErrorCorrection &imputer,
                int cluster, int cat_index) {
               return imputer.row_model(cluster)->categorical_model(
                   cat_index)->level_probs();
             },
             py::arg("cluster"),
             py::arg("cat_index"),
             "The level probabilities for a particular combination of \n"
             "mixture component and categorical variable.\n\n"
             "Args:\n"
             "  cluster: Index of the mixture component.\n"
             "  cat_index: The index of the desired numeric variable.  \n"
             "    This index counts from 0 and ignores any intervening \n"
             "    categorical variables. ")
        .def("set_level_probs",
             [](MixedDataImputerWithErrorCorrection &imputer,
                int cluster, int cat_index,
                const Vector &level_probs) {
               return imputer.row_model(cluster)->categorical_model(
                   cat_index)->set_level_probs(level_probs);
             })
        .def("level_observation_probs",
             [](MixedDataImputerWithErrorCorrection &imputer,
                int cluster, int cat_index) {
               return imputer.row_model(cluster)->categorical_model(
                   cat_index)->level_observation_probs();
             },
             py::arg("cluster"),
             py::arg("cat_index"),
             "The level observation probabilities for a particular \n"
             "combination of mixture component and categorical variable.\n\n"
             "Args:\n"
             "  cluster: Index of the mixture component.\n"
             "  cat_index: The index of the desired numeric variable.  \n"
             "    This index counts from 0 and ignores any intervening \n"
             "    categorical variables. ")
        .def("set_level_observation_probs",
             [](MixedDataImputerWithErrorCorrection &imputer,
                int cluster, int cat_index,
                const Matrix &level_observation_probs) {
               return imputer.row_model(cluster)->categorical_model(
                   cat_index)->set_level_observation_probs(
                       level_observation_probs);
             })
        ;

  }  // module boom

}  // namespace BayesBoom
