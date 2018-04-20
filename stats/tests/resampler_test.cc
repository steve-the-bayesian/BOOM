#include "gtest/gtest.h"
#include "cpputil/seq.hpp"
#include "LinAlg/Vector.hpp"
#include "stats/Resampler.hpp"
#include "test_utils/test_utils.hpp"
#include "stats/ChiSquareTest.hpp"
#include "stats/FreqDist.hpp"

namespace {
  using namespace BOOM;
  using std::endl;

  class ResamplerTest : public ::testing::Test {
   protected:
    ResamplerTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(ResamplerTest, NormalUse) {
    Vector weights = {1, 2, 3};
    Resampler sam(weights);
    EXPECT_EQ(3, sam.dimension());

    int sample_size = 10000;
    std::vector<int> samples = sam(sample_size);
    EXPECT_EQ(sample_size, samples.size());
    OneWayChiSquareTest chisq(FrequencyDistribution(samples),
                              Vector{1.0/6, 2.0/6, 3.0/6});
    EXPECT_GE(chisq.p_value(), .05);
    EXPECT_EQ(3, sam.dimension());

    // Now reset sam and try sampling with data.
    Vector probs(7, 1.0 / 7);
    sam.set_probs(probs, true);
    EXPECT_EQ(7, sam.dimension());
    // Seven unique values...
    std::vector<int> data = {8, 6, 7, 5, 3, 0, 9};

    samples = sam(data, sample_size * 3);
    EXPECT_EQ(samples.size(), sample_size * 3);
    OneWayChiSquareTest chisq_uniform(FrequencyDistribution(samples, false),
                                      probs);
    EXPECT_GT(chisq_uniform.p_value(), .05)
        << "Chisq test results: " << chisq_uniform << endl
        << "Frequency distribution: " << FrequencyDistribution(samples, false);
  }

  TEST_F(ResamplerTest, ZeroWeights) {
    Vector weights = {.5, 0, 0, .5};
    Resampler sam(weights);
    int sample_size = 10000;
    std::vector<int> values = seq<int>(0, 3);
    std::vector<int> samples = sam(values, sample_size);

    EXPECT_EQ(samples.size(), sample_size);
    FrequencyDistribution freq(samples, true);
    EXPECT_GT(freq.counts()[0], 4000);
    EXPECT_EQ(freq.counts()[1], 0);
    EXPECT_EQ(freq.counts()[2], 0);
    EXPECT_GT(freq.counts()[3], 4000);
  }
  
}  // namespace
