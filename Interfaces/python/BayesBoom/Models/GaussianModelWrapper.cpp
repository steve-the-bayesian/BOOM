#include <pybind11/pybind11.h>

#include "Models/GaussianModelBase.hpp"
#include "Models/GaussianModel.hpp"
#include "Models/GaussianModelGivenSigma.hpp"
#include "Models/PosteriorSamplers/GaussianConjSampler.hpp"

#include "cpputil/Ptr.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;

  // Define the classes related to Gaussian model.
  void GaussianModel_def(py::module &boom) {

    py::class_<GaussianModelBase,
               Model,
               Ptr<GaussianModelBase>>(boom, "GaussianModelBase")
        .def_property_readonly("mean", &GaussianModelBase::mu)
        .def_property_readonly("sd", &GaussianModelBase::sigma)
        .def_property_readonly("variance", &GaussianModelBase::sigsq)
        ;

    py::class_<GaussianModel,
               GaussianModelBase,
               Ptr<GaussianModel>>(boom, "GaussianModel")
        .def(py::init<double, double>(),
             py::arg("mean") = 0.0, py::arg("sd") = 1.0,
             "Args:\n"
             "\n"
             "  mean:  Mean of the distribution.\n"
             "  sd:  Standard deviation of the distribution.\n")
        .def("set_mean_sd", &GaussianModel::set_params,
             py::arg("mean"),
             py::arg("sd"))
        .def("mle", &GaussianModel::mle)
        .def("log_likelihood", &GaussianModel::Loglike)
        .def("set_data", [](GaussianModel &model, const Vector &data) {
            for (const auto &el: data) {
              NEW(DoubleData, data_point)(el);
              model.add_data(data_point);
            }
          },
          py::arg("data"),
          "Assign the data in the supplied vector to the model.  \n"
          "Args:\n"
          "  data: a boom.Vector containing the data values."
          )
        .def("sample_posterior", &GaussianModel::sample_posterior)
        .def_property_readonly("mean_parameter",
                               [] (const GaussianModel &model) {
                                 return model.Mu_prm();
                               },
                               "The parameter object representing the model variance.")
        .def_property_readonly("sigsq_parameter",
                               [] (const GaussianModel &model) {
                                 return model.Sigsq_prm();
                               },
                               "The parameter object representing the model variance.")
        .def("__repr__", [](const Ptr<GaussianModel> &model) {
            std::ostringstream out;
            out << "A BOOM Gaussian model with mean " << model->mu()
                << " and standard deviation " << model->sigma()
                << ".";
            return out.str();
          })
        ;

    py::class_<GaussianModelGivenSigma,
               GaussianModelBase,
               Ptr<GaussianModelGivenSigma>>(boom, "GaussianModelGivenSigma")
        .def(py::init<
             const Ptr<UnivParams> &,
             double,
             double
             >(),
             py::arg("scaling_variance"),
             py::arg("mean") = 1.0,
             py::arg("sample_size") = 1.0,
             "A Gaussian model conditional on an external variance parameter\n."
             "The model is theta ~ N(mu, sigsq / kappa)"
             "\n"
             "Args:\n"
             "  scaling_variance: The 'sigsq' in the model definition.\n"
             "  mean: The mean of the distribution: 'mu' in the model "
             "definition.\n"
             "  sample_size: The number of observations worth of weight to "
             "place on the mean.  This is 'kappa' in the model definition."
             )
        ;

    py::class_<GaussianConjSampler,
               PosteriorSampler,
               Ptr<GaussianConjSampler>>(boom, "GaussianConjugateSampler")
        .def(py::init([] (GaussianModel *model,
                          const Ptr<GaussianModelGivenSigma> &mean,
                          const Ptr<GammaModelBase> &precision,
                          RNG &seeding_rng = BOOM::GlobalRng::rng) {
                        return new GaussianConjSampler(
                            model, mean, precision, seeding_rng);
                      }
                      ),
             py::arg("model"),
             py::arg("mean_prior"),
             py::arg("precision_prior"),
             py::arg("rng") = GlobalRng::rng,
             "Create a GaussianConjugateSampler for a GaussianModel\n"
             "\n"
             "Args:\n"
             "  model: the model to be managed by this sampler.\n"
             "  mean_prior: A prior distribution for the mean of 'model'.\n"
             "  precision_prior: A prior_distribution for the precision (reciprocal variance)")
        .def("draw", &GaussianConjSampler::draw,
             "Simulate one draw from the posterior distribution.  "
             "Updated paramater draws are stored in the model.")
        ;


  }  // Module

}  // namespace BOOM
