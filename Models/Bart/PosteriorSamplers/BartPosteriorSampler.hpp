// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2013 Steven L. Scott

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2.1 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA
*/

#ifndef BART_POSTERIOR_SAMPLER_BASE_HPP_
#define BART_POSTERIOR_SAMPLER_BASE_HPP_

#include "Models/Bart/Bart.hpp"
#include "Models/GaussianModel.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "Samplers/MoveAccounting.hpp"
#include "cpputil/math_utils.hpp"

namespace BOOM {

  //======================================================================
  // Contains the parameters common to all Bart priors.
  struct BartPriorParameters {
    BartPriorParameters() {}
    BartPriorParameters(double total_prediction_sd,
                        double prior_tree_depth_alpha,
                        double prior_tree_depth_beta)
        : total_prediction_sd(total_prediction_sd),
          prior_tree_depth_alpha(prior_tree_depth_alpha),
          prior_tree_depth_beta(prior_tree_depth_beta) {}
    double total_prediction_sd;
    double prior_tree_depth_alpha;
    double prior_tree_depth_beta;
  };

  //======================================================================
  // Prior distribution specifying an exact number of trees to use in
  // the ensemble.
  class PointMassPrior {
   public:
    explicit PointMassPrior(int n) : number_of_trees_(n) {}
    double operator()(int n) const {
      if (n == number_of_trees_) {
        return 0;
      } else {
        return negative_infinity();
      }
    }

   private:
    int number_of_trees_;
  };

  //======================================================================
  // This is the base class for PosteriorSampler classes for drawing
  // from concrete Bart models.  This class handles moves that modify
  // the tree structure, and tree birth/death moves, so that derived
  // classes can focus on the details of data augmentation (for
  // non-Gaussian models).
  //
  // All the models share some common structure in their prior
  // distribution.  There is p(tree) * p(variable | tree) * p(cutpoint
  // | variable, tree, ancestors) * p(mean | tree).  The middle two
  // distributions are often ignored because they are uniform.
  //
  // The prior probability that a node at depth 'd' splits into
  // children at depth (d + 1) is a / (1 + d)^b.  Given a split, a
  // variable is chosen uniformly from the set of available variables,
  // and a cutpoint uniformly from the set of available cutpoints.
  // Note that 'available' is influenced by a node's position in the
  // tree, because splits made by ancestors will make some splits
  // logically impossible, and impossible splits are not 'available.'
  // For example, descendants cannot split on the same dummy variable
  // as an ancestor.
  //
  // The conditional prior on the mean parameters at the leaves is
  // N(0, total_prediction_sd^2 / number_of_trees).  Derived classes may
  // require priors on other model components (e.g. sigma^2 for the
  // Gassian case).
  class BartPosteriorSamplerBase : public PosteriorSampler {
   public:
    // Args:
    //   model:  The Bart model for which this is a prior.
    //   total_prediction_sd: The standard deviation of the
    //     predictions that will come out of the model, across
    //     predictor space.  This is used to infer the standard
    //     deviation of the individual mean parameters in each tree,
    //     which is total_prediction_sd / sqrt(number_of_trees).
    //   prior_tree_depth_alpha:  The probability of a split at the root.
    //   prior_tree_depth_beta: The exponent controlling how the
    //     probability of a split diminishes as a function of depth
    //     (minimal number of steps to the root).  psplit(d) =
    //     prior_tree_depth_alpha / (1 + depth)^prior_tree_depth_beta.
    //   log_prior_on_number_of_trees: The log of the prior
    //     probability function over the number of trees.  Should
    //     return negative_infinity if the probability is zero.  At
    //     construction time, model->number_of_trees() must assume a
    //     value with positive probability (finite log probability)
    //     under this distribution.
    BartPosteriorSamplerBase(
        BartModelBase *model, double total_prediction_sd,
        double prior_tree_depth_alpha, double prior_tree_depth_beta,
        const std::function<double(int)> &log_prior_on_number_of_trees,
        RNG &seeding_rng = GlobalRng::rng);

    // The destructor should clear pointers to the
    // ResidualRegressionData owned by this class but observed by the
    // trees in the model.
    ~BartPosteriorSamplerBase() override;

    // Sets the vector of move probabilities to the uniform
    // distribution.
    void set_default_move_probabilities();

    // Sets the vector of move probabilities to move_probs, which must
    // have the same number of elements as the TreeStructureMoveType
    // enum.
    void set_move_probabilities(const Vector &move_probs);

    // I implemented a skeleton version of logpri() to get this class
    // to compile.  It does not return anything useful, and will throw
    // an exception if called.
    double logpri() const override;

    // The draw method includes a call to check_residuals, to ensure
    // they have been created and placed where they need to be, and
    // calls to modify tree.
    void draw() override;

    // Returns a draw of the mean parameter for the given leaf,
    // conditional on the tree structure and the data assigned to
    // leaf.  This differs slightly across the exponential family
    // because different ways to do data augmentation.
    virtual double draw_mean(Bart::TreeNode *leaf) = 0;

    // Returns the log density of the set of Y's described by suf,
    // conditional on sigma, but integrating mu out over the prior.
    // That is,
    //
    // log  \int p(Y | \mu, \sigma) p(\mu | \sigma) d \mu
    //
    // Derived classes may omit normalizing constants that will cancel
    // in MH acceptance ratios, such as factors of 2*pi or constant
    // variance terms.
    virtual double log_integrated_likelihood(
        const Bart::SufficientStatisticsBase &suf) const = 0;

    // Returns the log integrated likelihood for the subtree
    // consisting of *node and its descendants.  This is the sum of
    // the log integrated likelihoods for all the leaves under *node.
    double subtree_log_integrated_likelihood(Bart::TreeNode *node) const;

    // Returns the log likelihood associated with the given set of
    // complete data sufficient statistics.  ***NOTE*** the outupt of
    // this function will be compared to the output of
    // log_integrated_likelihood for some MH moves.  If a derived
    // class's implementation of log_integrated_likelihood omits
    // constant that canel in MH ratios, those same constants must be
    // omitted here.
    virtual double complete_data_log_likelihood(
        const Bart::SufficientStatisticsBase &suf) const = 0;

    // Clear the vector of residuals (make it empty).
    virtual void clear_residuals() = 0;

    // Returns the number of observations stored in the residual vector.
    virtual int residual_size() const = 0;

    // Creates and stores the residual observation corresponding to
    // observation i in the model.  The model needs to have data
    // assigned before this function can be called.
    virtual Bart::ResidualRegressionData *create_and_store_residual(int i) = 0;

    // Returns a pointer to the concrete instance of
    // ResidualRegressionData associated with data point i.
    virtual Bart::ResidualRegressionData *residual(int i) = 0;

    // Create the type of sufficient statistics that go along with the
    // type of your data.
    virtual Bart::SufficientStatisticsBase *create_suf() const = 0;

    //----------------------------------------------------------------------
    // Verify that the vector of residuals has been allocated and
    // computed, and that the trees owned by model_ are populated.  If
    // the trees don't have the right data, clear them and put insert
    // the residual data.  It should only be necessary to call this
    // function once (it is designed to be a no-op most of the time).
    void check_residuals();

    // To be called with a new tree.
    void fill_tree_with_residual_data(Bart::Tree *tree);

    //--------------------------------------------------------------
    // Moves used to implement draw.

    // A MH move that tries to modify the tree structure, and to
    // sample the terminal means.
    void modify_tree(Bart::Tree *tree);

    // Does one MH step on the structure of 'tree', conditional on
    // sigma, but integrating over the mean parameters.
    void modify_tree_structure(Bart::Tree *tree);

    // Attempt a birth move on the specified tree using Metropolis Hastings.
    void split_move(Bart::Tree *tree);

    // Attempt a death move on the specified tree using Metropolis Hastings.
    void prune_split_move(Bart::Tree *tree);

    // Choose a random interior node and attempt to swap its decision
    // rule with its parent.
    void swap_move(Bart::Tree *tree);

    void grow_branch_move(Bart::Tree *tree);
    void prune_branch_move(Bart::Tree *tree);

    // Choose a random interior node and attempt to change its
    // splitting rule.
    void change_cutpoint_move(Bart::Tree *tree);

    // Modifies the cutpoint for a randomly chosen interior node using
    // slice sampling.
    void slice_sample_cutpoint(Bart::Tree *tree);
    void slice_sample_continuous_cutpoint(Bart::TreeNode *node);
    void slice_sample_discrete_cutpoint(Bart::TreeNode *node);

    // Conditional on the tree structure and sigma, sample the mean
    // parameters at the leaves.
    void draw_terminal_means_and_adjust_residuals(Bart::Tree *tree);

    //----------------------------------------------------------------------
    // Birth and death at the tree level

    // Propose a tree with a single split in a MH move.
    void tree_birth_move();
    double tree_birth_log_acceptance_probability(Bart::Tree *proposal);

    // Select a tree with at most one split.  Propose removing it in a
    // Metropolis Hastings move.
    void tree_death_move();

    // Compute the log of the prior probability of splitting (or not
    // splitting) at the given depth.  The root is depth zero.  Its
    // children are depth 1, etc.  This is the a/(1 + d)^b probability
    // mentioned in the preamble comments.
    double log_probability_of_split(int depth) const;
    double probability_of_split(int depth) const;
    double log_probability_of_no_split(int depth) const;
    double probability_of_no_split(int depth) const;

    // The mean prior variance is the variance of the prior
    // distribution for the mean parameter at an individual leaf.  Its
    // value is is total_prediction_variance / number_of_trees.
    double mean_prior_variance() const;

    const MoveAccounting &move_accounting() const { return MH_accounting_; }

    // Grow a new branch on tree, starting from 'leaf', which must be
    // a leaf node owned by 'tree'.  An exception will be thrown if
    // 'leaf' is not really a leaf.  If the tree is fully saturated,
    // so that no further splits from leaf are possible, the function
    // returns false.  Otherwise it returns true.
    //
    // The nodes in the new branch (including 'leaf') will have their
    // splitting rules simulated from the prior (uniform over
    // available splits).  Leaves will not have sensible mean
    // parameters assigned.
    //
    //
    bool grow_branch_from_prior(Bart::Tree *tree, Bart::TreeNode *leaf);

    bool assign_random_split_rule(Bart::TreeNode *leaf);
    bool assign_random_split_rule_from_subset(Bart::TreeNode *leaf,
                                              Selector &included_variables);

   protected:
    // Removes all pointers to residuals_ from the trees owned by
    // model_.
    void clear_data_from_trees();

    //----------------------------------------------------------------------
    // Compute the log of the Metropolis-Hastings ratio for the split
    // move.  The log ratio for the prune_split move is -1 times this
    // number.
    double split_move_log_metropolis_ratio(Bart::Tree *tree,
                                           Bart::TreeNode *leaf);

    // Computes the log of the Metropolis-Hastings ratio for the
    // grow_branch move.  The log ratio for the prune_branch_move is
    // -1 times this number.  If the tree branch is a single split
    // then this function forwards to split_move_log_metropolis_ratio.
    double grow_branch_log_metropolis_ratio(Bart::Tree *tree,
                                            Bart::TreeNode *branch_root);

   private:
    BartModelBase *model_;
    // Alpha is kept on both the log scale and the raw scale.
    double log_prior_tree_depth_alpha_;
    double prior_tree_depth_alpha_;
    double prior_tree_depth_beta_;

    // The total prediction variance is the variance of the prediction
    // across all the trees.  It defines the prior for the mean
    // parameter at an individual leaf, which is
    // N(0, total_prediction_variance / number_of_trees)
    double total_prediction_variance_;

    // Functor returning log P(model_->number_of_trees).
    std::function<double(int)> log_prior_number_of_trees_;

    // The types of moves (for manipulating a single tree) considered
    // by the Metropolis-Hastings algorithm.
    enum TreeStructureMoveType {
      SPLIT = 0,
      PRUNE_SPLIT = 1,
      GROW_BRANCH = 2,
      PRUNE_BRANCH = 3,
      SWAP = 4,
      CHANGE_CUTPOINT = 5
    };

    MoveAccounting MH_accounting_;

    // The vector of move_probabilities_ must be the same length as
    // the number of elements in the MoveType enum.
    Vector move_probabilities_;
  };

}  // namespace BOOM
#endif  // BART_POSTERIOR_SAMPLER_BASE_HPP_
