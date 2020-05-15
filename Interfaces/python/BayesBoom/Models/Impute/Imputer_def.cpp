#include <pybind11/pybind11.h>

#include "Models/ModelTypes.hpp"
#include "Models/Glm/Glm.hpp"
#include "Models/Glm/GlmCoefs.hpp"

#include "Models/Impute/MvRegCopulaDataImputer.hpp"

#include "cpputil/Ptr.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;

  py::class_<MvRegCopulaDataImputer,
             Ptr<MvRegCopulaDataImputer>>(boom, "MvRegCopulaDataImputer")
  .def(py::init(
      [](int num_clusters,
         const std::vector<Vector> &atoms,
         int xdim,
         BOOM::RNG &seeding_rng) {
        return new MvRegCopulaDataImputer(num_clusters, atoms, xdim, seeding_rng);
      },
      py::arg("num_clusters"),
      py::arg("atoms"),
      py::arg("xdim"),
      py::arg("seeding_rng") = BOOM::GlobalRng::rng,
      "Args:\n"
      "  num_clusters:  The number of clusters in the patern matching model \n"
      "    that handles data errors. \n"
      "  atoms: A collection of Vectors (or 1-d numpy arrays) containing \n"
      "    values that will receive special modeling treatment.  One entry is\n"
      "    needed for each variable.  An entry can be the empty Vector. \n"
      "  xdim:  The dimension of the predictor variable.\n"
      "  seeding_rng:  A boom random number generator used to seed the \n"
      "    RNG in this object."))
  .def("add_data",
       [](MvRegCopulaDataImputer &imputer,
          cont Ptr<MvRegData> &data_point) {
         imputer.add_data(data_point);
       },
       py::arg("data_point"),
       "Add a data point to the training data set.\n\n"
       "Args:\n"
       "  data_point:  Object of type boom.MvRegData.  The y variable \n"
       "    should indicate missing values with NaN.\n")
  .def("sample_posterior",
       &MvRegCopulaDataImputer::sample_posterior,
       "Take one draw from the posterior distribution")
  .def_property_readonly(
      "coefficients",
      [](MvRegCopulaDataImputer &imputer) {
        Matrix ans = imputer.regression()->Beta();
        return ans;
      },
      "The matrix of regression coefficients.  Rows correspond to Y (output).\n"
      "Columns correspond to X (input).  Coefficients represent the \n"
      "relationship between X and the copula transform of Y.")
  def_property_readonly(
      "residual_variance",
      [](MvRegCopulaDataImputer &imputer) {
        SpdMatrix ans = imputer.regression()->Sigma();
        return ans;
      },
      "The residual variance matrix on the transformed (copula) scale.")
  def_property_readonly(
      "nclusters",
      &MvRegCopulaDataImputer::nclusters,
      "The number of clusters in the error pattern matching model.")
  def_property_readonly(
      "atom_probs",
      [](const MvRegCopulaDataImputer &imputer, int cluster, int variable_index) {
        return imputer.atom_probs(cluster, variable_index);
      },
      "The marginal probability that each atom is the 'truth'.")
  .def("set_default_regression_prior",
       [](MvRegCopulaDataImputer &imputer, int xdim, int ydim) {
         Ptr<MultivariateRegressionModel> reg = imputer.regression();
         NEW(MultivariateRegressionSampler, regression_sampler)(
             reg.get(),
             Matrix(xdim, ydim, 0.0),
             1.0,
             ydim + 1,
             SpdMatrix(ydim, 1.0));
         reg->set_method(regression_sampler);
       },
       "Set a 'nearly flat' prior on the regression coefficients and residual "
       "variance.")
  .def("set_default_prior_for_mixing_weights",
       &Imputer::set_default_prior_for_mixing_weights)
  .def("set_atom_prior",
       [](MvRegCopulaDataImputer &imputer, const Vector &prior_counts) {
         imputer.set_atom_prior(prior_counts);
       },
       "TODO: docstring")
  .def("set_atom_error_prior",
       [](MvRegCopulaDataImputer &imputer, const Matrix &prior_counts) {
         imputer.set_atom_error_prior(prior_counts);
       },
       "TODO: docstring")
  ;
}  // namespace BayesBoom
