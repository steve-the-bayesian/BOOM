#include "gtest/gtest.h"
#include "numopt/MarkovDecisionProcess.hpp"
#include "LinAlg/Matrix.hpp"
#include "test_utils/test_utils.hpp"

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


  class MarkovDecisionProcessTest : public ::testing::Test {
   protected:
    MarkovDecisionProcessTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  // Imagine a setting where you can be nice or be mean.  Being nice gets you a
  // reward of 1.  Being mean gets you a reward of 10.
  //
  // The states of the game are continue and end.  Being nice means there is a
  // high probability of continuing the game.  Being mean means there is a high
  // probability of ending the game.
  TEST_F(MarkovDecisionProcessTest, SmallExample) {

    Array transition_probabilities({2,2,2});
    Array rewards({2,2,2});

    // state 0 is play, state 1 is quit.
    // action 0 is nice, action 1 is mean.

    // If you play nice in state 0 you get a reward of 1.
    rewards(0, 0, 0) = 1.0;
    rewards(0, 0, 1) = 1.0;

    // If you play mean in state 0 you get a reward of 10.
    rewards(0, 1, 0) = 10.0;
    rewards(0, 1, 1) = 10.0;

    // If you start off in state 1 there will be no reward.
    rewards(1, 0, 0) = 0.0;
    rewards(1, 0, 1) = 0.0;
    rewards(1, 1, 0) = 0.0;
    rewards(1, 1, 1) = 0.0;

    // If you play nice, there's a good chance the game will go on.
    transition_probabilities(0, 0, 0) = .99;
    transition_probabilities(0, 0, 1) = .01;

    // If you play mean, there's a good chance the game will be quit.
    transition_probabilities(0, 1, 0) = .01;
    transition_probabilities(0, 1, 1) = .99;

    // Once the game has been quit you never play again.
    transition_probabilities(1, 0, 0) = 0.0;
    transition_probabilities(1, 0, 1) = 1.0;
    transition_probabilities(1, 1, 0) = 0.0;
    transition_probabilities(1, 1, 1) = 1.0;

    MarkovDecisionProcess mdp(transition_probabilities, rewards);
    Vector value = mdp.value_iteration(1000, .001);

    // If the future is not heavily discounted, it pays to be nice so we can
    // play the game for a long time.
    Vector policy = mdp.optimal_policy(1000, .999);
    EXPECT_EQ(policy[0], 0);

    // If the future is heavily discounted we should take the money and run.
    policy = mdp.optimal_policy(1000, .001);
    EXPECT_EQ(policy[0], 1);
  }

}  // namespace
