#include "gtest/gtest.h"
#include "numopt/ClassAssigner.hpp"
#include "LinAlg/Matrix.hpp"
#include "test_utils/test_utils.hpp"
#include "distributions.hpp"

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  template <class T>
  std::string print_vector(const std::vector<T> &stuff) {
    std::ostringstream out;
    for (size_t i = 0; i < stuff.size(); ++i) {
      out << stuff[i];
      if (i + 1  < stuff.size()) {
        out << ", ";
      }
    }
    return out.str();
  }


  class ClassAssignerTest : public ::testing::Test {
   protected:
    ClassAssignerTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  // Suppose users are generated from one of three groups with probabilities
  // (.3, .4, .3).  A single binomial trial is conduc
  TEST_F(ClassAssignerTest, SmallExample) {

    int sample_size = 1000;
    Vector class_probs = {.3, .4, .3};
    std::vector<int> true_class_values = rmulti_vector_mt(
        GlobalRng::rng, sample_size, class_probs);

    // 
    Vector binomial_probs = {.4, .45, .5};
    int n = 1;
    
    std::vector<int> y(sample_size);
    Matrix posteriors(sample_size, class_probs.size());
    for (int i = 0; i < sample_size; ++i) {
      y[i] = rbinom_mt(GlobalRng::rng,
                       n,
                       binomial_probs[true_class_values[i]]);
      double total = 0;
      for (int j = 0; j < posteriors.ncol(); ++j) {
        posteriors(i, j) =
            class_probs[j] * dbinom(y[i], n, binomial_probs[j]);
        total += posteriors(i, j);
      }
      posteriors.row(i) /= total;
    }

    ClassAssigner assigner;
    std::vector<int> assignment = assigner.assign(
        posteriors,
        class_probs,
        GlobalRng::rng);
    
  }

}  // namespace
