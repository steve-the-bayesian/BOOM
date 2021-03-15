#include <pybind11/pybind11.h>

#include "Models/ChisqModel.hpp"
#include "Models/GaussianModelBase.hpp"
#include "Models/GaussianModel.hpp"
#include "Models/GaussianModelGivenSigma.hpp"
#include "Models/ZeroMeanGaussianModel.hpp"
#include "Models/PosteriorSamplers/GaussianConjSampler.hpp"
#include "Models/PosteriorSamplers/ZeroMeanGaussianConjSampler.hpp"

#include "cpputil/Ptr.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;

  // Define the classes related to Gaussian model.
  void GaussianModel_def(py::module &boom) {

    py::class_<GaussianModelBase,
               DoubleModel,
               Ptr<GaussianModelBase>>(boom, "GaussianModelBase")
        .def_property_readonly("mean", &GaussianModelBase::mu)
        .def_property_readonly("sd", &GaussianModelBase::sigma)
        .def_property_readonly("variance", &GaussianModelBase::sigsq)
        .def_property_readonly("mu", &GaussianModelBase::mu)
        .def_property_readonly("sigma", &GaussianModelBase::sigma)
        .def_property_readonly("sigsq", &GaussianModelBase::sigsq)
        ;

    py::class_<GaussianModel,
               GaussianModelBase,
               PriorPolicy,
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
               PriorPolicy,
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
        .def(py::init([] (Ptr<GaussianModel> model,
                          const Ptr<GaussianModelGivenSigma> &mean,
                          const Ptr<GammaModelBase> &precision,
                          RNG &seeding_rng = BOOM::GlobalRng::rng) {
                        return new GaussianConjSampler(
                            model.get(), mean, precision, seeding_rng);
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
             "  precision_prior: A prior_distribution for the precision \n"
             "    (reciprocal variance)")

        .def("draw", &GaussianConjSampler::draw,
             "Simulate one draw from the posterior distribution.  "
             "Updated paramater draws are stored in the model.")
        ;


    py::class_<ZeroMeanGaussianModel,
               GaussianModelBase,
               PriorPolicy,
               Ptr<ZeroMeanGaussianModel>>(boom, "ZeroMeanGaussianModel")
        .def(py::init<double>(),
             py::arg("sigma") = 1.0,
             "Args:\n"
             "  sigma:  Standard deviation of the distribution.")
        .def("set_sigma",
             [](ZeroMeanGaussianModel &model, double sigma) {
               model.set_sigsq(sigma * sigma);
             })
        .def("set_sigsq",
             [](ZeroMeanGaussianModel &model, double sigsq) {
               model.set_sigsq(sigsq);
             })
        .def("set_data",
             [](ZeroMeanGaussianModel &model, const Vector &data) {
               for (const auto &el: data) {
                 NEW(DoubleData, data_point)(el);
                 model.add_data(data_point);
               }
             },
             py::arg("data"),
             "Assign the data in the supplied vector to the model.  \n"
             "Args:\n"
             "  data: a boom.Vector containing the data values.")
        ;

    py::class_<ZeroMeanGaussianConjSampler,
               PosteriorSampler,
               Ptr<ZeroMeanGaussianConjSampler>>(boom, "ZeroMeanGaussianConjSampler")
        .def(py::init(
            [] (ZeroMeanGaussianModel &model,
                GammaModelBase &siginv_prior,
                RNG &seeding_rng) {
              return new ZeroMeanGaussianConjSampler(
                  &model, Ptr<GammaModelBase>(&siginv_prior), seeding_rng);
            }),
             py::arg("model"),
             py::arg("siginv_prior"),
             py::arg("seeding_rng") = BOOM::GlobalRng::rng,
             "Create a ZeroMeanGaussianConjSampler -- The conjugate sampler "
             "for a ZeroMeanGaussianModel.\n"
             "\n\nArgs:\n"
             "  model:  The model to be sampled.\n"
             "  siginv_prior:  GammaModelBase: Prior distribution for the "
             "precision of the innovation errors.\n"
             "  rng:  Random number generator used to seed the RNG of this "
             "sampler."
             )
        .def("set_sigma_upper_limit",
             &ZeroMeanGaussianConjSampler::set_sigma_upper_limit,
             py::arg("upper_limit") = BOOM::infinity(),
             "Truncate the support for the standard deviation, so that it "
             "does not go above 'upper_limit'.\n"
             )
        .def_property_readonly(
            "sigma_prior_guess", [](ZeroMeanGaussianConjSampler &sampler) {
              return sampler.sigma_prior_guess();
            })
        .def_property_readonly(
            "sigma_prior_sample_size", [](ZeroMeanGaussianConjSampler &sampler) {
              return sampler.sigma_prior_sample_size();
            })
        ;


  }  // Module

}  // namespace BOOM
