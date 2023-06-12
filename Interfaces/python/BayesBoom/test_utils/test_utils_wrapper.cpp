#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "test_utils/test_utils.hpp"

#include "LinAlg/Vector.hpp"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/SpdMatrix.hpp"


namespace py = pybind11;

namespace BayesBoom {
  using namespace BOOM;

  void test_utils_def(py::module &boom) {

    boom.def("check_mcmc_vector",
             [](const Vector &v, double truth, double confidence){
               return CheckMcmcVector(v, truth, confidence);
             },
             py::arg("v"),
             py::arg("truth"),
             py::arg("confidence") = .95,
             "Check to see if a vector of Monte Carlo draws covers a known value.\n\n"
             "Args:\n"
             "  draws:  The vector of Monte Carlo draws to check.\n"
             "  truth:  The true value against which 'draws' will be checked.\n"
             "  confidence: The probability content of the credibility interval used to\n"
             "    check coverage.\n\n"
             "Returns:\n"
             "  A central credibility interval with probability content 'confidence' is\n"
             "  constructed from 'draws'.  The boolean return indicates whether this\n"
             "  interval covers 'truth'.\n");

    boom.def("two_sample_ks",
             [](const Vector &data1,
                const Vector &data2,
                double significance) {
               return TwoSampleKs(data1, data2, significance);
             },
             py::arg("data1"),
             py::arg("data2"),
             py::arg("significance: float") = 0.05,
             "Performs a 2-sample Kolmogorov Smirnoff test that the two sets of draws are\n"
             "from the same distribution.\n\n"
             "Args:\n"
             "  data1, data2: Sets of draws thought to be from the same distribution.\n"
             "  significance:  The significance level of the KS test.\n\n"
             "Returns:\n"
             "  If the null hypothesis cannot be rejected at the given significance level\n"
             "  then this function returns 'true'.  If the null is rejected then 'false'\n"
             "  is returned.  In other words, 'true' indicates that 'data1' and 'data2'\n"
             "  are a match.\n");

    boom.def("check_stochastic_process",
             [](const Matrix &draws,
                const Vector &truth,
                double confidence,
                double sd_ratio_threshold,
                double coverage_fraction) {
               return CheckStochasticProcess(draws,
                                             truth,
                                             confidence,
                                             sd_ratio_threshold,
                                             coverage_fraction,
                                             "");
             },
             py::arg("draws"),
             py::arg("truth"),
             py::arg("confidence") = .95,
             py::arg("sd_ratio_threshold") = .1,
             py::arg("coverage_fraction") = .5,
             "A check similar to CheckMcmcMatrix, but designed for stochastic processes\n"
             "or other functions exhibiting serial correlation, which can mess up the\n"
             "multiple comparisons adjustments used by CheckMcmcMatrix.\n\n"
             "Args:\n"
             "  draws: A matrix of Monte Carlo draws to be checked.  Each row is a draw\n"
             "    and each column is a variable.\n"
             "  truth: A vector of true values against which draws will be compared.\n"
             "    truth.size() must match ncol(draws).\n"
             "  confidence: The confidence associated with the marginal posterior\n"
             "    intervals used to determine coverage.\n"
             "  sd_ratio_threshold: One of the testing diagnostics compares the standard\n"
             "    deviation of the centered draws to the standard deviation of the true\n"
             "    function.  If that ratio is less than this threshold the diagnostic is\n"
             "    passed.\n"
             "  coverage_fraction: The fraction of marginal posterior intervals that must\n"
             "    cover their true values at the specified confidence level in order for\n"
             "    the check to pass. \n"
             "\n"
             "Details:\n"
             "  Half the marginal confidence intervals should cover, and the residual\n"
             "  standard deviation should be small relative to the standard deviation of\n"
             "  truth.\n\n"
             "Returns:\n"
             "  An error message.  An empty message means the test passed.\n");

  }  // test_utils_def

}  // namespace BayesBoom
