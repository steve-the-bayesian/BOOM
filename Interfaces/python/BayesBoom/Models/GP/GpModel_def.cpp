#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include <sstream>

#include "Models/GP/GpMeanFunction.hpp"
#include "Models/GP/kernels.hpp"
#include "Models/GP/GaussianProcessRegressionModel.hpp"
#include "Models/GP/HierarchicalGpRegressionModel.hpp"

#include "Models/GP/PosteriorSamplers/GaussianProcessRegressionPosteriorSampler.hpp"
#include "Models/GP/PosteriorSamplers/HierarchicalGpPosteriorSampler.hpp"
#include "Models/GP/PosteriorSamplers/MahalanobisKernelSampler.hpp"
#include "Models/GP/PosteriorSamplers/LinearMeanFunctionSampler.hpp"

#include "cpputil/Ptr.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;
  using BOOM::uint;

  void GpModel_def(py::module &boom) {

    //==========================================================================
    py::class_<FunctionParams,
               Params,
               Ptr<FunctionParams>>(boom, "FunctionParams")
        .def("__call__",
             [](const FunctionParams &fun,
                const Vector &x1) {
               return fun(x1);
             },
             py::is_operator(),
             py::arg("x"),
             "Evaluate the function at the given argument, "
             "returning a scalar.\n")
        .def("__call__",
             [](const FunctionParams &fun,
                const Matrix &X) {
               return fun(X);
             },
             py::is_operator(),
             py::arg("X"),
             "Evaluate the function at each row of the given argument, "
             "returning a boom.Vector.\n")
        ;

    //==========================================================================
    py::class_<ZeroFunction,
               FunctionParams,
               Ptr<ZeroFunction>>(boom, "ZeroFunction")
        .def(py::init([](){ return new ZeroFunction; }))
        ;

    //==========================================================================
    py::class_<LinearMeanFunction,
               FunctionParams,
               Ptr<LinearMeanFunction>>(boom, "LinearMeanFunction")
        .def(py::init([](const Vector &coefficients) {
          return new LinearMeanFunction(new GlmCoefs(coefficients));
        }),
          "Args:\n\n"
          "   coefficients:  The coefficients for the linear mean function.\n")

        .def_property_readonly(
            "coefficients",
            [](const LinearMeanFunction &fun) {
              return fun.coef()->Beta();
            })
        ;
    //==========================================================================
    py::class_<KernelParams,
               Params,
               Ptr<KernelParams>>(boom, "KernelParams")
        .def("__call__",
             [](const KernelParams &kernel,
                const Vector &x1,
                const Vector &x2) {
               return kernel(x1, x2);
             },
             py::is_operator(),
             py::arg("x1"),
             py::arg("x2"),
             "Evaluate the kernel function at the given arguments, "
             "returning a scalar.\n")
        .def("__call__",
             [](const KernelParams &kernel,
                const Matrix &X) {
               return kernel(X);
             },
             py::is_operator(),
             py::arg("X"),
             "Evaluate the kernel at every pair of rows in X, "
             "returning a boom.SpdMatrix.\n")
        ;

    //==========================================================================
    py::class_<RadialBasisFunction,
               KernelParams,
               Ptr<RadialBasisFunction>>(boom, "RadialBasisFunction")
        .def(py::init(
            [](double scale) {
              return new RadialBasisFunction(scale);
            }),
            py::arg("scale"),
            "Args:\n\n"
            "  scale: The size of a 'standard deviation' over which "
            "the kernel should reach.\n")
        ;

    //==========================================================================
    py::class_<MahalanobisKernel,
               KernelParams,
               Ptr<MahalanobisKernel>>(boom, "MahalanobisKernel")
        .def(py::init(
            [](int dim, double scale) {
              return new MahalanobisKernel(dim, scale);
            }),
             py::arg("dim"),
             py::arg("scale"),
             "Args:\n\n"
             "  dim:  The dimension of the Vectors that the kernel accepts.\n"
             "  scale:  The scale factor.\n")
        .def(py::init(
            [](const Matrix &X, double scale, double diagonal_shrinkage) {
              return new MahalanobisKernel(X, scale, diagonal_shrinkage);
            }),
             py::arg("X"),
             py::arg("scale") = 1.0,
             py::arg("diagonal_shrinkage") = 0.05,
             "Args:\n\n"
             "  X:  The boom.Matrix of predictors on which to base the distance "
             "metric.\n"
             "  scale:  The scale factor multiplying X'X / n.\n"
             "  diagonal_shrinkage:  A number between 0 and 1 indicating how "
             "much the X'X matrix should be shrunk towards its diagonal.\n")
        .def_property_readonly(
            "scale",
            [](const MahalanobisKernel &kernel) {
              return kernel.scale();
            })
        ;

    //==========================================================================
    // A Gaussian process regression model with a specified prior mean function
    // and kernel.
    py::class_<GaussianProcessRegressionModel,
               PriorPolicy,
               Ptr<GaussianProcessRegressionModel>>(
                   boom, "GaussianProcessRegressionModel",
                   py::multiple_inheritance())
        .def(py::init(
            [](FunctionParams &mean_function,
               KernelParams &kernel,
               double residual_sd) {
              return new GaussianProcessRegressionModel(
                  Ptr<FunctionParams>(&mean_function),
                  Ptr<KernelParams>(&kernel),
                  new UnivParams(square(residual_sd)));
            }),
             py::arg("mean_function"),
             py::arg("kernel"),
             py::arg("residual_sd"),
             "Args:\n\n"
             "  mean_function:  An object of class boom.FunctionParams "
             "giving the prior mean of the Gaussian process.\n"
             "  kernel: An objet of class boom.KernelParams giving the kernel "
             "(variance) of the Gaussian process.\n"
             "  residual_sd:  the residual standard deviation of the "
             "Gaussian process.\n")
        .def_property_readonly(
            "kernel",
            [](GaussianProcessRegressionModel *model) {
              return model->kernel_param();
            })
        .def_property_readonly(
            "mean_function",
            [](GaussianProcessRegressionModel *model) {
              return model->mean_param();
            })
        .def_property_readonly(
            "sigsq_param",
            [](GaussianProcessRegressionModel *model) {
              return model->sigsq_param();
            })
        .def_property_readonly(
            "sigma",
            [](const GaussianProcessRegressionModel *model) {
              return model->sigma();})
        .def_property_readonly(
            "sigsq",
            [](const GaussianProcessRegressionModel *model) {
              return model->sigsq();})
        .def_property_readonly(
            "residual_variance",
            [](const GaussianProcessRegressionModel *model) {
              return model->sigsq();})
        .def_property_readonly(
            "residual_sd",
            [](const GaussianProcessRegressionModel *model) {
              return model->sigma();})
        .def("add_data",
             [](GaussianProcessRegressionModel *model,
                const Matrix &X,
                const Vector &y) {
               if (X.nrow() <= 0) {
                 report_error("X must have at least one row.");
               }
               if (X.nrow() != y.size()) {
                 report_error("The number of rows in X must "
                              "match the length of y.");
               }

               for (int i = 0; i < X.nrow(); ++i) {
                 model->add_data(new RegressionData(y[i], X.row(i)));
               }
             })
        .def("add_data",
             [](GaussianProcessRegressionModel *model,
                const Vector &y,
                const Matrix &X) {
               if (X.nrow() <= 0) {
                 report_error("X must have at least one row.");
               }
               if (X.nrow() != y.size()) {
                 report_error("The number of rows in X must "
                              "match the length of y.");
               }

               for (int i = 0; i < X.nrow(); ++i) {
                 model->add_data(new RegressionData(y[i], X.row(i)));
               }
             })
        .def("predict",
             [](GaussianProcessRegressionModel *model,
                const Vector &x) {
               return model->predict(x);
             })
        .def("predict_distribution",
             [](GaussianProcessRegressionModel *model,
                const Matrix &X) {
               return model->predict_distribution(X);
             })
        ;

    //==========================================================================
    py::class_<HierarchicalRegressionData,
               RegressionData,
               Ptr<HierarchicalRegressionData>>(boom, "HierarchicalRegressionData")
        .def(py::init(
            [](double y, const Vector &x, const std::string &group) {
              return new HierarchicalRegressionData(y, x, group);
            }),
             py::arg("y"),
             py::arg("x"),
             py::arg("group"),
             "Regression data that is part of a hierarchical model.\n"
             "Args:\n\n"
             "  y:  The response variable.\n"
             "  x:  A boom.Vector containing the predictors.\n"
             "  group:  The position in the hierarchy containing the observation.\n"
             )
        .def_property_readonly(
            "group",
            [](const HierarchicalRegressionData &data_point) {
              return data_point.group();
            })
        ;

    //==========================================================================
    py::class_<HierarchicalGpRegressionModel,
               PriorPolicy,
               Ptr<HierarchicalGpRegressionModel>>(
                   boom, "HierarchicalGpRegressionModel")
        .def(py::init(
            [](const Ptr<GaussianProcessRegressionModel> &mean_function_model) {
              return new HierarchicalGpRegressionModel(mean_function_model);
            }),
             py::arg("mean_function_model"),
             "Args:\n\n"
             "  mean_function_model: The GaussianProcessRegressionModel to use "
             "as the prior mean function.\n")
        .def("add_model",
             [](HierarchicalGpRegressionModel &model,
                GaussianProcessRegressionModel &data_model,
                const std::string &group_id) {
               Ptr<GaussianProcessRegressionModel> data_model_ptr(&data_model);
               model.add_model(data_model_ptr, group_id);
             },
             py::arg("data_model"),
             py::arg("group"),
             "Args:\n\n"
             "  data_model: the boom.GaussianProcessRegressionModel object "
             "responsible for modeling the specified group in the hierarchy.\n"
             "  group:  The name of the hierarchy group to be modeled.\n")
        .def("add_data",
             [](HierarchicalGpRegressionModel &model,
                const Vector &response,
                const Matrix &predictors,
                const std::vector<std::string> &group) {
               size_t sample_size = response.size();
               if (predictors.nrow() != sample_size) {
                 std::ostringstream err;
                 err << "The number of rows in 'predictors' ("
                     << predictors.nrow()
                     << ") did not match the length of 'response' ("
                     << sample_size << ").\n";
                 report_error(err.str());
               }
               if (group.size() != sample_size) {
                 std::ostringstream err;
                 err << "The length of 'group' ("
                     << group.size()
                     << ") must match the length of 'response' ("
                     << sample_size
                     << ").\n";
                 report_error(err.str());
               }
               for (size_t i = 0; i < sample_size; ++i) {
                 NEW(HierarchicalRegressionData, data_point)(
                     response[i], predictors.row(i), group[i]);
                 model.add_data(data_point);
               }
             },
             py::arg("response"),
             py::arg("predictors"),
             py::arg("group"),
             "Args:\n\n"
             "  response:  A boom.Vector of responses.\n"
             "  predictors:  A boom.Matrix of predictor variables.\n"
             "  group:  A sequqnce of strings, giving the group id of "
             "each observation.\n"
             " The number of rows in 'predictors' must match the lengths of "
             "'response' and 'group'.\n")
        .def("data_model",
             [](HierarchicalGpRegressionModel &model, const std::string &group) {
               return model.data_model(group);
             },
             py::arg("group"),
             "Args:\n\n"
             "  group:  The group id of the desired model.\n"
             "Returns:\n"
             "  The GaussianProcessRegressionModel describing the "
             "requested group.\n")
        .def_property_readonly(
            "prior",
            [](HierarchicalGpRegressionModel &model) {
              return model.prior();
            },
            "Returns:\n"
            "  The GaussianProcessRegressionModel describing the "
            "prior mean function.\n")
        ;

    //===========================================================================
    py::class_<GP::ParameterSampler, Ptr<GP::ParameterSampler>>(
        boom, "GpParameterSampler")
        .def("draw",
             [](GP::ParameterSampler &sampler, RNG &rng) {
               sampler.draw(rng);
             },
             py::arg("rng"),
             "Args:\n\n",
             "  rng:  A boom.RNG random number generator.\n")
        .def("logpri",
             [](const GP::ParameterSampler &sampler) {
               return sampler.logpri();
             })
        ;

    //===========================================================================
    py::class_<GP::NullSampler,
               GP::ParameterSampler,
               Ptr<GP::NullSampler>>(boom, "GpNullSampler")
        .def(py::init(
            []() {
              return new GP::NullSampler;
            }),
             "A NullSampler object can be used as a placeholder for "
             "FunctionParams or KernelParams objects that have no "
             "unknown parameters.\n"
             )
        ;

    //===========================================================================
    py::class_<LinearMeanFunctionSampler,
               GP::ParameterSampler,
               Ptr<LinearMeanFunctionSampler>>(
                   boom, "LinearMeanFunctionSampler")
        .def(py::init(
            [](LinearMeanFunction &mean_function,
               GaussianProcessRegressionModel &model,
               MvnBase &prior) {
              return new LinearMeanFunctionSampler(
                  &mean_function, &model, Ptr<MvnBase>(&prior));
            }),
             py::arg("mean_function"),
             py::arg("model"),
             py::arg("prior"),
             "Args:\n\n"
             "   mean_function:  The LinearMeanFunction object whose coefficients "
             "are to be sampled.\n"
             "   model:  The model that owns the 'mean_function'.\n"
             "   prior:  An MvnBase object giving the prior distribution for "
             "the coefficients of the mean function.\n")
        ;

    //===========================================================================
    py::class_<MahalanobisKernelSampler,
               GP::ParameterSampler,
               Ptr<MahalanobisKernelSampler>>(
                   boom, "MahalanobisKernelSampler")
        .def(py::init(
            [](MahalanobisKernel *kernel,
               GaussianProcessRegressionModel *model,
               DoubleModel *prior) {
              return new MahalanobisKernelSampler(kernel, model, Ptr<DoubleModel>(prior));
            }),
             py::arg("kernel"),
             py::arg("model"),
             py::arg("prior"),
             "A ParameterSampler object for sampling the 'scale' parameter in "
             "a MahalanobisKernel.\n\n"
             "Args:\n\n"
             "  kernel:  The MahalanobisKernel object to be sampled.\n"
             "  model:  The model that owns 'kernel'.\n"
             "  prior:  A boom.DoubleModel giving the prior distribution "
             "on the kernel's 'scale' parameter.\n")
        ;

    //===========================================================================
    py::class_<GaussianProcessRegressionPosteriorSampler,
               PosteriorSampler,
               Ptr<GaussianProcessRegressionPosteriorSampler>>(
                   boom, "GaussianProcessRegressionPosteriorSampler")
        .def(py::init(
            [](GaussianProcessRegressionModel *model,
               GP::ParameterSampler &mean_function_sampler,
               GP::ParameterSampler &kernel_sampler,
               GammaModelBase &residual_variance_prior,
               RNG &seeding_rng) {
              return new GaussianProcessRegressionPosteriorSampler(
                  model,
                  Ptr<GP::ParameterSampler>(&mean_function_sampler),
                  Ptr<GP::ParameterSampler>(&kernel_sampler),
                  Ptr<GammaModelBase>(&residual_variance_prior),
                  seeding_rng);
            }),
             py::arg("model"),
             py::arg("mean_function_sampler"),
             py::arg("kernel_sampler"),
             py::arg("residual_variance_prior"),
             py::arg("seeding_rng") = GlobalRng::rng,
             "Args:\n\n"
             "   model:  The model to be posterior sampled.\n"
             "   mean_function_sampler:  A boom.GpParameterSampler for drawing the parameters of the mean function.\n"
             "   kernel_sampler:  A boom.GpParameterSampler for drawing the parameters of the kernel.\n"
             "   seeding_rng:  A random number generator used to see the RNG "
             "in this sampler.\n")
        ;

    py::class_<HierarchicalGpPosteriorSampler,
               PosteriorSampler,
               Ptr<HierarchicalGpPosteriorSampler>>(
                   boom, "HierarchicalGpPosteriorSampler")
        .def(py::init(
            [](HierarchicalGpRegressionModel &model, RNG &seeding_rng) {
              return new HierarchicalGpPosteriorSampler(&model, seeding_rng);
            }),
             py::arg("model"),
             py::arg("rng") = GlobalRng::rng,
             "Args:\n\n"
             "   model:  The model to be posterior sampled.  All subcomponents "
             "(the prior and all data_model components) must have posterior "
             "samplers assigned to them.\n"
             "   seeding_rng:  A random number generator used to see the RNG "
             "in this sampler.\n")
        ;

  }  // GpModel_def

}  // namespace BayesBoom
