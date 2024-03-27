#include "gtest/gtest.h"
#include "numopt/ClassAssigner.hpp"
#include "LinAlg/Matrix.hpp"
#include "test_utils/test_utils.hpp"
#include "distributions.hpp"
#include "stats/kl_divergence.hpp"

#include "cpputil/report_error.hpp"

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

    void simulate_data(
        int sample_size,
        const Vector &class_probs,
        int binomial_n,
        const Vector &binomial_probs,
        RNG &rng) {

      if (class_probs.size() != binomial_probs.size()) {
        report_error("class_probs and binomial_probs must be the same size.");
      }
      class_probs_ = class_probs;
      posteriors_.resize(sample_size, class_probs.size());
      true_class_.resize(sample_size);
      for (int i = 0; i < sample_size; ++i) {
        true_class_[i] = rmulti_mt(rng, class_probs);
        int y = rbinom_mt(rng, binomial_n, binomial_probs[true_class_[i]]);
        for (size_t j = 0; j < class_probs.size(); ++j) {
          posteriors_(i, j) = class_probs[j] * dbinom(
              y, binomial_n, binomial_probs[j]);
        }
        double total = posteriors_.row(i).sum();
        posteriors_.row(i) /= total;
      }
    }

    Matrix posteriors_;
    Vector class_probs_;
    std::vector<int> true_class_;
  };

  // With a very weak likelihood most marginal posteriors will look like the
  // prior.  In this case the KL divergence between the assignment and the
  // target should be less than the given bound on KL.
  TEST_F(ClassAssignerTest, SmallExample) {
    simulate_data(1000,
                  Vector{.3, .4, .3},
                  1,
                  Vector{.4, .45, .5},
                  GlobalRng::rng);
    ClassAssigner assigner;
    assigner.set_max_kl(.01);
    std::vector<int> assignment = assigner.assign(
        posteriors_,
        class_probs_,
        GlobalRng::rng);

    EXPECT_LE(assigner.kl(), .01);
  }

  // With a highly informative likelihood and a weak bound on KL, the assignment
  // should yield (mostly) MAP estimates.
  TEST_F(ClassAssignerTest, StrongInformation) {
    simulate_data(1000,
                  Vector{.3, .4, .3},
                  100,
                  Vector{.2, .5, .8},
                  GlobalRng::rng);
    ClassAssigner assigner;
    //    assigner.set_max_kl(1);
    std::vector<int> assignment = assigner.assign(
        posteriors_,
        class_probs_,
        GlobalRng::rng);

    std::cout << "kl = " << assigner.kl() << ".\n";

    EXPECT_EQ(assignment.size(), 1000);
    int mistakes = 0;
    for (size_t i = 0; i < assignment.size(); ++i) {
      if (assignment[i] != true_class_[i]) {
        ++mistakes;
      }
    }
    EXPECT_LE(mistakes, 10);
    cout << "mistakes = " << mistakes << ".\n";
  }

}  // namespace
