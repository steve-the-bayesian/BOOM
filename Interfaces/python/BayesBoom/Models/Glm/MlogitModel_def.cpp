#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>

#include "cpputil/Ptr.hpp"

#include "Models/ModelTypes.hpp"
#include "Models/Glm/Glm.hpp"
#include "Models/Glm/GlmCoefs.hpp"
#include "Models/Glm/VariableSelectionPrior.hpp"
#include "Models/Glm/MultinomialLogitModel.hpp"
#include "Models/Glm/PosteriorSamplers/MultinomialLogitCompositeSpikeSlabSampler.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;
  using BOOM::uint;

  void MultinomialLogitModel_def(py::module &boom) {

    py::class_<ChoiceData,
               CategoricalData,
               Ptr<ChoiceData>>(
                   boom, "ChoiceData", py::multiple_inheritance())
        .def(py::init(
            [](const CategoricalData &response,
               const Vector &subject_predictors,
               const std::vector<Vector> &choice_predictors) {
              NEW(VectorData, subject)(subject_predictors);
              std::vector<Ptr<VectorData>> choice;
              for (int i = 0; i < choice_predictors.size(); ++i) {
                choice.push_back(new VectorData(choice_predictors[i]));
              }
              return new ChoiceData(response, subject, choice);
            }),
             py::arg("response"),
             py::arg("subject_predictors"),
             py::arg("choice_predictors"),
             "Args:\n\n"
             "  response:  The chosen value.\n"
             "  subject_predictors: A vector of predictors describing subject "
             "level characteristics.  Should contain an explicit intercept if "
             "an intercept term is desired for the model.  If empty then no "
             "subject-level predictors will be considered. \n"
             "  choice_predictors: Characteristics of the object being chosen."
             "  May be empty if all data is subject-level data.\n"
             )
        ;


    py::class_<MultinomialLogitModel,
               PriorPolicy,
               Ptr<MultinomialLogitModel>>(
                   boom, "MultinomialLogitModel", py::multiple_inheritance())
        .def(py::init<uint, uint, uint>(),
             py::arg("nchoices"),
             py::arg("subject_xdim"),
             py::arg("choice_xdim"),
             "Create a MultinomialLogitModel.\n\n"
             "Args:"
             "  nchoices:  The number of potential outcomes.\n"
             "  subject_xdim:  The dimension of the predictor variables "
             "related to the subject making the choice.\n"
             "  choice_xdim: The dimension of predictor variables related "
             "to the choice being made.\n\n"
             "For example, a 25 year old female is buying a car based on the "
             "car's gas mileage, horsepower, and number of seats.  "
             "'subject_xdim=2' because Age and Sex are aspects of the subject."
             "  choice_xdim=3 because there are three predictors related to "
             "the item being chosen.\n")
        .def(py::init(
            [](const std::vector<std::string> &responses,
               const Matrix &subject_predictors,
               const std::vector<Matrix> &choice_predictors) {
              std::vector<Ptr<CategoricalData>> boom_responses =
                  create_categorical_data(responses);

              int nchoices = boom_responses[0]->nlevels();
              int subject_xdim = subject_predictors.ncol();
              int choice_xdim = choice_predictors.empty() ?
                  0 : choice_predictors[0].ncol();
              NEW(MultinomialLogitModel, model)(
                  nchoices, subject_xdim, choice_xdim);

              for (int i = 0; i < boom_responses.size(); ++i) {
                std::vector<Ptr<VectorData>> row_level_choice_predictors;
                if (choice_xdim > 0) {
                  for (int m = 0; m < nchoices; ++m){
                    row_level_choice_predictors.push_back(new VectorData(
                        choice_predictors[m].row(i)));
                  }
                }
                NEW(ChoiceData, data_point)(
                    *boom_responses[i],
                    Ptr<VectorData>(new VectorData(subject_predictors.row(i))),
                    row_level_choice_predictors);
                model->add_data(data_point);
              }
              return model;
            }),
             py::arg("responses"),
             py::arg("subject_predictors"),
             py::arg("choice_predictors"),
             "Args:\n\n"
             "  responses:  A list or numpy array of strings.\n"
             "  subject_predictors:  A Matrix of predictor variables for "
             "the subject making the choice.\n"
             "  choice_predictors:  A dict, keyed by response levels, "
             "containing predictors describing the object begin chosen.\n")
        .def("beta_size",
             [](const MultinomialLogitModel &model, bool include_zeros) {
               return model.beta_size(include_zeros);
             },
             py::arg("include_zeros") = false,
             "Args:\n\n"
             "  include_zeros:  If True then account for the space needed to "
             "include the baseline category.  In a typical setting the "
             "coefficients specific to the baseline category are set to zero "
             "for identifiability.\n")
        .def_property_readonly(
            "nchoices",
            [](MultinomialLogitModel &m) {return m.Nchoices();})
        .def_property_readonly(
            "xtx",
            [](MultinomialLogitModel &m) {
              SpdMatrix xtx(m.beta_size(false), 0.0);
              for (long i = 0; i < m.sample_size(); ++i) {
                const ChoiceData *dp = m.dat()[i].get();
                xtx.add_inner(dp->X(false));
              }
              return xtx;
            },
            "The cross product matrix from the training data.")
        .def_property_readonly(
            "sample_size",
            [](const MultinomialLogitModel &model) {
              return model.sample_size();
            })
        .def_property_readonly(
            "coefficients",
            [](MultinomialLogitModel &m) {
              return m.coef();
            },
            "The parameter object representing the model coefficients.  "
            "boom.GlmCoefs")
        .def_property_readonly(
            "log_likelihood",
            [](const MultinomialLogitModel &model) {
              return model.log_likelihood();
            })
        ;


    py::class_<MultinomialLogitCompositeSpikeSlabSampler,
               PosteriorSampler,
               Ptr<MultinomialLogitCompositeSpikeSlabSampler>>(
                   boom, "MultinomialLogitCompositeSpikeSlabSampler")
        .def(py::init(
            [](MultinomialLogitModel *model,
               MvnBase *coefficient_prior,
               VariableSelectionPrior *inclusion_prior,
               double t_degrees_of_freedom,
               double rwm_variance_scale_factor,
               int nthreads,
               int max_chunk_size,
               bool check_initial_condition,
               RNG &seeding_rng) {
              return new MultinomialLogitCompositeSpikeSlabSampler(
                  model, coefficient_prior, inclusion_prior,
                  t_degrees_of_freedom, rwm_variance_scale_factor,
                  nthreads, max_chunk_size,
                  check_initial_condition, seeding_rng);
            }),
             py::arg("model"),
             py::arg("coefficient_prior"),
             py::arg("inclusion_prior"),
             py::arg("t_degrees_of_freedom") = 4,
             py::arg("rwm_variance_scale_factor") = 1,
             py::arg("nthreads") = 1,
             py::arg("max_chunk_size") = 10,
             py::arg("check_initial_condition") = true,
             py::arg("seeding_rng") = GlobalRng::rng,
             "Args:\n\n"
             "  model:  The model to be posterior sampled.\n"
             "  coefficient_prior: The conditional prior distribution on all "
             "the coefficients, given inclusion.\n"
             "  inclusion_prior: The prior probability of inclusion for the "
             "coefficients.\n"
             "  t_degrees_of_freedom: The tail thickness parameter for the "
             "proposal distribution when making Metropolis Hastings "
             "proposals.  If t_degrees_of_freedom is <= 0 then a Normal "
             "proposal will be used.\n"
             "  rwm_variance_scale_factor: A constant amount (> 0) by which "
             "to scale the proposal variance in random walk Metropolis "
             "proposals.  Smaller values will yield more frequent "
             "acceptances but smaller moves conditional on acceptance. \n"
             "  nthreads: The number of threads to use for data augmentation "
             "when making a data augmentation move.  If the sample size "
             "is small then a single thread is the fastest option.  Once "
             "you hit a hundred thousand observations or so then it makes "
             "sense to set nthreads to the maximum number of hardware "
             "threads.\n"
             "  max_chunk_size: The largest 'chunk' of coefficients that will "
             "be proposed by a Metropolis Hastings proposal.\n"
             "  check_initial_condition: Passed to MLVS.  Throws an exception "
             "if the initial state of model has zero support under the "
             "prior.  This is mainly an issue when variables are forced "
             "in or out of the model with the inclusion_prior, but those "
             "variables are initially excluded (or included) by the model.\n")
        ;

  }  // module definition

}  // namespace BayesBoom
