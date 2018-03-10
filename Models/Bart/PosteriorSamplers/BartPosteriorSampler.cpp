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

#include "Models/Bart/PosteriorSamplers/BartPosteriorSampler.hpp"
#include "LinAlg/Selector.hpp"
#include "Models/Bart/ResidualRegressionData.hpp"
#include "Samplers/ScalarSliceSampler.hpp"
#include "cpputil/math_utils.hpp"
#include "distributions.hpp"

namespace {
  // Returns the log of the integer d.  This is a compiler
  // optimization to pre-compute the logs of the first 100 integers.
  inline double log_integer(const int d) {
    switch (d) {
      case 1:
        return log(1.0);
      case 2:
        return log(2.0);
      case 3:
        return log(3.0);
      case 4:
        return log(4.0);
      case 5:
        return log(5.0);
      case 6:
        return log(6.0);
      case 7:
        return log(7.0);
      case 8:
        return log(8.0);
      case 9:
        return log(9.0);
      case 10:
        return log(10.0);
      case 11:
        return log(11.0);
      case 12:
        return log(12.0);
      case 13:
        return log(13.0);
      case 14:
        return log(14.0);
      case 15:
        return log(15.0);
      case 16:
        return log(16.0);
      case 17:
        return log(17.0);
      case 18:
        return log(18.0);
      case 19:
        return log(19.0);
      case 20:
        return log(20.0);
      case 21:
        return log(21.0);
      case 22:
        return log(22.0);
      case 23:
        return log(23.0);
      case 24:
        return log(24.0);
      case 25:
        return log(25.0);
      case 26:
        return log(26.0);
      case 27:
        return log(27.0);
      case 28:
        return log(28.0);
      case 29:
        return log(29.0);
      case 30:
        return log(30.0);
      case 31:
        return log(31.0);
      case 32:
        return log(32.0);
      case 33:
        return log(33.0);
      case 34:
        return log(34.0);
      case 35:
        return log(35.0);
      case 36:
        return log(36.0);
      case 37:
        return log(37.0);
      case 38:
        return log(38.0);
      case 39:
        return log(39.0);
      case 40:
        return log(40.0);
      case 41:
        return log(41.0);
      case 42:
        return log(42.0);
      case 43:
        return log(43.0);
      case 44:
        return log(44.0);
      case 45:
        return log(45.0);
      case 46:
        return log(46.0);
      case 47:
        return log(47.0);
      case 48:
        return log(48.0);
      case 49:
        return log(49.0);
      case 50:
        return log(50.0);
      case 51:
        return log(51.0);
      case 52:
        return log(52.0);
      case 53:
        return log(53.0);
      case 54:
        return log(54.0);
      case 55:
        return log(55.0);
      case 56:
        return log(56.0);
      case 57:
        return log(57.0);
      case 58:
        return log(58.0);
      case 59:
        return log(59.0);
      case 60:
        return log(60.0);
      case 61:
        return log(61.0);
      case 62:
        return log(62.0);
      case 63:
        return log(63.0);
      case 64:
        return log(64.0);
      case 65:
        return log(65.0);
      case 66:
        return log(66.0);
      case 67:
        return log(67.0);
      case 68:
        return log(68.0);
      case 69:
        return log(69.0);
      case 70:
        return log(70.0);
      case 71:
        return log(71.0);
      case 72:
        return log(72.0);
      case 73:
        return log(73.0);
      case 74:
        return log(74.0);
      case 75:
        return log(75.0);
      case 76:
        return log(76.0);
      case 77:
        return log(77.0);
      case 78:
        return log(78.0);
      case 79:
        return log(79.0);
      case 80:
        return log(80.0);
      case 81:
        return log(81.0);
      case 82:
        return log(82.0);
      case 83:
        return log(83.0);
      case 84:
        return log(84.0);
      case 85:
        return log(85.0);
      case 86:
        return log(86.0);
      case 87:
        return log(87.0);
      case 88:
        return log(88.0);
      case 89:
        return log(89.0);
      case 90:
        return log(90.0);
      case 91:
        return log(91.0);
      case 92:
        return log(92.0);
      case 93:
        return log(93.0);
      case 94:
        return log(94.0);
      case 95:
        return log(95.0);
      case 96:
        return log(96.0);
      case 97:
        return log(97.0);
      case 98:
        return log(98.0);
      case 99:
        return log(99.0);
      case 100:
        return log(100.0);
      default:
        return log(static_cast<double>(d));
    }
  }
}  // namespace

namespace BOOM {
  using Bart::ResidualRegressionData;
  using Bart::Tree;
  using Bart::TreeNode;
  using Bart::VariableSummary;

  //----------------------------------------------------------------------
  BartPosteriorSamplerBase::BartPosteriorSamplerBase(
      BartModelBase *model, double total_prediction_sd,
      double prior_tree_depth_alpha, double prior_tree_depth_beta,
      const std::function<double(int)> &log_prior_number_of_trees,
      RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        log_prior_tree_depth_alpha_(log(prior_tree_depth_alpha)),
        prior_tree_depth_alpha_(prior_tree_depth_alpha),
        prior_tree_depth_beta_(prior_tree_depth_beta),
        total_prediction_variance_(square(total_prediction_sd)),
        log_prior_number_of_trees_(log_prior_number_of_trees) {
    if (prior_tree_depth_alpha <= 0 || prior_tree_depth_alpha >= 1) {
      report_error(
          "The prior_tree_depth_alpha parameter "
          "must be strictly between 0 and 1.");
    }
    if (prior_tree_depth_beta < 0) {
      report_error(
          "The prior_tree_depth_beta parameter "
          " must be non-negative");
    }
    if (total_prediction_sd <= 0) {
      report_error("total_prediction_sd must be positive");
    }
    set_default_move_probabilities();
  }

  //----------------------------------------------------------------------
  BartPosteriorSamplerBase::~BartPosteriorSamplerBase() {
    //    clear_data_from_trees();

    // I would like to call clear_data_from_trees, so that when the
    // posterior sampler gets destroyed, another one can be used in
    // its place.  However, if this gets destroyed because the Ptr
    // that owns it is killed by the model's destructor, then the
    // trees that own the data are in an undetermined state (at least
    // undetermined by me).  In that case calling
    // clear_data_from_trees() can lead to a crash.
    //
    // It seems to be a better idea to leave the data in the trees
    // (which will point to deallocated memory when *this is
    // destoyed).
  }

  //----------------------------------------------------------------------
  void BartPosteriorSamplerBase::set_default_move_probabilities() {
    move_probabilities_.resize(6);
    double probability = 1.0 / move_probabilities_.size();
    move_probabilities_[SPLIT] = probability;
    move_probabilities_[PRUNE_SPLIT] = probability;
    move_probabilities_[GROW_BRANCH] = probability;
    move_probabilities_[PRUNE_BRANCH] = probability;
    move_probabilities_[SWAP] = probability;
    move_probabilities_[CHANGE_CUTPOINT] = probability;
  }

  //----------------------------------------------------------------------
  double BartPosteriorSamplerBase::logpri() const {
    // Implementing logpri would involve a sum over trees of
    // P(split | parents) * p(mean).  Not hard, really?????
    report_error(
        "logpri() is not yet implemented for "
        "BartPosteriorSamplerBase, and it probably won't "
        "be any time soon.");
    return -1;
  }

  //----------------------------------------------------------------------
  void BartPosteriorSamplerBase::draw() {
    check_residuals();
    for (int i = 0; i < model_->number_of_trees(); ++i) {
      modify_tree(model_->tree(i));
    }
    tree_death_move();
    tree_birth_move();
  }

  //----------------------------------------------------------------------
  double BartPosteriorSamplerBase::subtree_log_integrated_likelihood(
      Bart::TreeNode *node) const {
    if (node->is_leaf()) {
      return log_integrated_likelihood(node->compute_suf());
    } else {
      return subtree_log_integrated_likelihood(node->left_child()) +
             subtree_log_integrated_likelihood(node->right_child());
    }
  }

  //----------------------------------------------------------------------
  // It should only be necessary to call check_residuals once.
  void BartPosteriorSamplerBase::check_residuals() {
    if (residual_size() != model_->sample_size()) {
      clear_residuals();
      clear_data_from_trees();
      for (int i = 0; i < model_->sample_size(); ++i) {
        Bart::ResidualRegressionData *data = create_and_store_residual(i);
        for (int j = 0; j < model_->number_of_trees(); ++j) {
          model_->tree(j)->populate_data(data);
        }
      }
      for (int i = 0; i < model_->number_of_trees(); ++i) {
        model_->tree(i)->populate_sufficient_statistics(create_suf());
      }
    }
  }

  //----------------------------------------------------------------------
  void BartPosteriorSamplerBase::fill_tree_with_residual_data(Tree *tree) {
    for (int i = 0; i < residual_size(); ++i) {
      tree->populate_data(residual(i));
    }
  }

  //----------------------------------------------------------------------
  void BartPosteriorSamplerBase::modify_tree(Tree *tree) {
    tree->remove_mean_effect();
    modify_tree_structure(tree);
    draw_terminal_means_and_adjust_residuals(tree);
  }

  //----------------------------------------------------------------------
  void BartPosteriorSamplerBase::modify_tree_structure(Tree *tree) {
    TreeStructureMoveType move =
        TreeStructureMoveType(rmulti_mt(rng(), move_probabilities_));
    switch (move) {
      case SPLIT:
        split_move(tree);
        break;
      case PRUNE_SPLIT:
        prune_split_move(tree);
        break;
      case GROW_BRANCH:
        grow_branch_move(tree);
        break;
      case PRUNE_BRANCH:
        prune_branch_move(tree);
        break;
      case SWAP:
        swap_move(tree);
        break;
      case CHANGE_CUTPOINT:
        slice_sample_cutpoint(tree);
        break;
      default:
        report_error(
            "An impossible move type was attempted in "
            "BartPosteriorSamplerBase::modify_tree_structure");
    }
  }

  //----------------------------------------------------------------------
  // Returns false if no split is possible.  Returns true otherwise.
  bool BartPosteriorSamplerBase::grow_branch_from_prior(Tree *tree,
                                                        TreeNode *leaf) {
    if (runif_mt(rng()) < probability_of_no_split(leaf->depth())) {
      // Bail if the leaf does not want to split.
      return true;
    }

    // Examples where can_split could be false include very deep trees
    // with small numbers of predictors that are all binary features.
    // If an ancestor splits on a binary feature, none of its
    // descendants can split on the same feature.  Thus it is possible
    // (though unlikely) to run out of features to split on.
    bool can_split = assign_random_split_rule(leaf);
    if (!can_split) {
      return false;
    }
    tree->grow(leaf);
    return grow_branch_from_prior(tree, leaf->left_child()) &&
           grow_branch_from_prior(tree, leaf->right_child());
  }

  //----------------------------------------------------------------------
  void BartPosteriorSamplerBase::grow_branch_move(Tree *tree) {
    MoveTimer timer = MH_accounting_.start_time("grow_branch");

    // Choose a leaf uniformly at random and simulate a full branch
    // from the prior.
    TreeNode *leaf = tree->random_leaf(rng());
    bool okay = grow_branch_from_prior(tree, leaf);
    if (!okay) {
      // No further splits on leaf are possible.
      return;
    }

    if (leaf->is_leaf()) {
      // No splits occurred in the proposed new branch.
      return;
    }

    double log_alpha = grow_branch_log_metropolis_ratio(tree, leaf);

    if (log(runif_mt(rng())) < log_alpha) {
      // accept the draw by doing nothing
      MH_accounting_.record_acceptance("grow_branch");
    } else {
      // reject the draw by reverting back to the way things were.
      tree->prune_descendants(leaf);
      MH_accounting_.record_rejection("grow_branch");
    }
  }

  //----------------------------------------------------------------------
  // Compute the log of the Metropolis-Hastings ratio for the tree
  // containing a branch grown from branch root.
  //
  // Args:
  //   tree: The tree containing branch_root.  It is assumed that the
  //     branch growing from branch_root was simulated from the prior
  //     distribution.
  //   branch_root: A node in tree that was a leaf before the
  //     proposal.  Now the root of a branch.
  double BartPosteriorSamplerBase::grow_branch_log_metropolis_ratio(
      Tree *tree, TreeNode *branch_root) {
    // Handle special cases first.
    if (branch_root->is_leaf()) {
      // If the branch root is a leaf then the proposal distribution
      // didn't generate any splits, so we reject the proposal.
      return negative_infinity();
    } else if (branch_root->left_child()->is_leaf() &&
               branch_root->right_child()->is_leaf()) {
      // If the branch contains a single split, then we need to handle
      // the possibility that the same tree could have been generated
      // by the split move.
      return split_move_log_metropolis_ratio(tree, branch_root);
    }

    double log_likelihood_ratio =
        subtree_log_integrated_likelihood(branch_root) -
        log_integrated_likelihood(branch_root->compute_suf());

    int depth = branch_root->depth();
    double log_prior_ratio = -log_probability_of_no_split(depth);

    double log_transition_density_numerator = log(
        move_probabilities_[PRUNE_BRANCH] / tree->number_of_interior_nodes());

    double log_transition_density_denominator =
        log(move_probabilities_[GROW_BRANCH] /
            tree->number_of_leaves_after_pruning(branch_root));

    double log_transition_density_ratio =
        log_transition_density_numerator - log_transition_density_denominator;

    return log_likelihood_ratio + log_prior_ratio +
           log_transition_density_ratio;
  }

  //----------------------------------------------------------------------
  void BartPosteriorSamplerBase::prune_branch_move(Tree *tree) {
    MoveTimer timer = MH_accounting_.start_time("prune_branch");
    // Choose a random interior node
    TreeNode *node = tree->random_interior_node(rng());
    if (!node) {
      MH_accounting_.record_special("prune_branch", "no interior node");
      return;
    }
    double log_alpha = -1 * grow_branch_log_metropolis_ratio(tree, node);
    if (log(runif_mt(rng())) < log_alpha) {
      tree->prune_descendants(node);
      MH_accounting_.record_acceptance("prune_branch");
    } else {
      MH_accounting_.record_rejection("prune_branch");
      // reject by doing nothing.
    }
  }

  //----------------------------------------------------------------------
  bool BartPosteriorSamplerBase::assign_random_split_rule(TreeNode *leaf) {
    // The variable_index is the proposal for which variable should be
    // used in the new splitting rule.
    int variable_index =
        random_int_mt(rng(), 0, model_->number_of_variables() - 1);
    leaf->set_variable(variable_index);
    bool success =
        model_->variable_summary(variable_index).random_cutpoint(rng(), leaf);
    if (success) {
      return true;
    } else {
      Selector potential_variables(model_->number_of_variables());
      potential_variables.drop(variable_index);
      return assign_random_split_rule_from_subset(leaf, potential_variables);
    }
  }

  //----------------------------------------------------------------------
  bool BartPosteriorSamplerBase::assign_random_split_rule_from_subset(
      TreeNode *leaf, Selector &potential_variables) {
    if (potential_variables.nvars() == 0) {
      return false;
    }
    int variable_index = potential_variables.random_included_position(rng());
    if (variable_index < 0) {
      report_error(
          "Something went wrong in "
          "'assign_random_split_rule_from_subset'");
    }
    leaf->set_variable(variable_index);
    bool success =
        model_->variable_summary(variable_index).random_cutpoint(rng(), leaf);

    if (success) {
      return true;
    } else {
      potential_variables.drop(variable_index);
      return assign_random_split_rule_from_subset(leaf, potential_variables);
    }
  }

  //----------------------------------------------------------------------
  void BartPosteriorSamplerBase::split_move(Tree *tree) {
    // Choose a leaf uniformly at random, from among the leaves with
    // at least 5 observations.
    //
    // Select a variable at random from the set of available
    // variables.  Select a cutpoint at random from the set of
    // available cutpoints.
    //
    // Note that as you go deeper in the tree, some variables can
    // become impossible to split on because one side of the split
    // would necessarily be empty.  For example, you would not split
    // on a dummy variable if an ancestor split on the same variable.
    // Likewise, if an ancestor split on variable 1 at cutpoint 3.2,
    // and you're on the right hand path from that ancestor, you can't
    // split on a value of variable 1 less than 3.2.
    //
    // Step 1: Sample from the proposal distribution by randomly
    // selecting a leaf from the set of available leaves, and a
    // splitting rule from the set of rules available for that leaf.
    MoveTimer timer = MH_accounting_.start_time("split");
    TreeNode *leaf = NULL;
    bool node_can_split = false;
    while (!node_can_split) {
      // Select a node and a variable uniformly at random.  Select a
      // cutpoint uniformly at random from the set of cutpoints
      // available to that node for that variable.
      leaf = tree->random_leaf(rng());

      node_can_split = assign_random_split_rule(leaf);
    }

    // Step 2: Compute the MH ratio (on the log scale).
    tree->grow(leaf);
    double log_alpha = split_move_log_metropolis_ratio(tree, leaf);

    double logu = log(runif_mt(rng()));
    if (logu < log_alpha) {
      // Accept the proposal by doing nothing.
      MH_accounting_.record_acceptance("split");
    } else {
      // Reject the proposal.
      tree->prune_descendants(leaf);
      MH_accounting_.record_rejection("split");
    }
  }

  //----------------------------------------------------------------------
  //  The prune_split_move selects a random parent of leaves and
  //  proposes removing the leaves (making the parent a leaf).  The MH
  //  acceptance ratio is the inverse of the MH ratio for the
  //  split_move.
  void BartPosteriorSamplerBase::prune_split_move(Tree *tree) {
    MoveTimer timer = MH_accounting_.start_time("prune_split");
    TreeNode *candidate_leaf = tree->random_parent_of_leaves(rng());
    if (!candidate_leaf) {
      // No such node is available.  Reject this move.
      MH_accounting_.record_rejection("prune_split");
      return;
    }
    double log_alpha =
        -1 * split_move_log_metropolis_ratio(tree, candidate_leaf);
    if (log(runif_mt(rng())) < log_alpha) {
      tree->prune_descendants(candidate_leaf);
      MH_accounting_.record_acceptance("prune_split");
    } else {
      // Reject the move by doing nothing.
      MH_accounting_.record_rejection("prune_split");
    }
  }

  //----------------------------------------------------------------------
  // Compute the log of the Metropolis-Hastings ratio for the split
  // move.  The log ratio for the prune_split move is -1 times this
  // ratio.
  // Args:
  //   tree:  The tree that has been split at leaf.
  //   leaf:  The leaf where the split has taken place.
  double BartPosteriorSamplerBase::split_move_log_metropolis_ratio(
      Tree *tree, TreeNode *leaf) {
    // If you prune a parent of leaves, you always reduce the number
    // of leaves by 1.
    int original_number_of_leaves = tree->number_of_leaves() - 1;
    int depth = leaf->depth();

    double log_likelihood_ratio =
        log_integrated_likelihood(leaf->left_child()->compute_suf()) +
        log_integrated_likelihood(leaf->right_child()->compute_suf()) -
        log_integrated_likelihood(leaf->compute_suf());

    // The prior_ratio omits a factor of p(variable, cutpoint) that
    // cancels with the transition distribution.
    double log_prior_ratio = log_probability_of_split(depth) +
                             2 * log_probability_of_no_split(depth + 1) -
                             log_probability_of_no_split(depth);

    double log_transition_density_numerator = log(
        move_probabilities_[PRUNE_SPLIT] / tree->number_of_parents_of_leaves() +
        move_probabilities_[PRUNE_BRANCH] / tree->number_of_interior_nodes());
    double log_transition_density_denominator =
        log(move_probabilities_[SPLIT] / original_number_of_leaves +
            (move_probabilities_[GROW_BRANCH] / original_number_of_leaves *
             probability_of_split(depth) *
             square(probability_of_no_split(depth + 1))));
    double log_transition_density_ratio =
        log_transition_density_numerator - log_transition_density_denominator;

    double log_alpha =
        log_likelihood_ratio + log_prior_ratio + log_transition_density_ratio;
    return log_alpha;
  }

  //----------------------------------------------------------------------
  // Choose a non-root interior node, and swap decision rules with its parent.
  void BartPosteriorSamplerBase::swap_move(Tree *tree) {
    MoveTimer timer = MH_accounting_.start_time("swap");

    if (tree->number_of_nodes() == 1) {
      MH_accounting_.record_special("swap", "single_node");
      return;
    }
    TreeNode *node = tree->random_interior_node(rng());
    if (node == tree->root()) {
      MH_accounting_.record_special("swap", "selected_root");
      return;
    }

    double original_log_integrated_likelihood =
        subtree_log_integrated_likelihood(node->parent());

    node->swap_splitting_rule(node->parent());
    node->parent()->refresh_subtree_data();
    const VariableSummary &parent_variable_summary(
        model_->variable_summary(node->parent()->variable_index()));
    const VariableSummary &child_variable_summary(
        model_->variable_summary(node->variable_index()));
    if (!parent_variable_summary.is_legal_configuration(node->parent()) ||
        !child_variable_summary.is_legal_configuration(node)) {
      node->swap_splitting_rule(node->parent());
      node->parent()->refresh_subtree_data();
      MH_accounting_.record_special("swap", "cant_split");
      return;
    }

    double proposal_log_integrated_likelihood =
        subtree_log_integrated_likelihood(node->parent());
    double log_alpha =
        proposal_log_integrated_likelihood - original_log_integrated_likelihood;
    if (log(runif_mt(rng())) < log_alpha) {
      // Do nothing.  Accept the tree.
      MH_accounting_.record_acceptance("swap");
    } else {
      // Reject the proposal.  Switch back to the original tree.
      node->swap_splitting_rule(node->parent());
      node->parent()->refresh_subtree_data();
      MH_accounting_.record_rejection("swap");
    }
  }

  //----------------------------------------------------------------------
  void BartPosteriorSamplerBase::change_cutpoint_move(Tree *tree) {
    MoveTimer timer = MH_accounting_.start_time("change_cutpoint");
    TreeNode *node = tree->random_interior_node(rng());
    if (!node) {
      MH_accounting_.record_special("change_cutpoint", "no interior node");
      return;
    }

    double original_log_likelihood = subtree_log_integrated_likelihood(node);
    double original_cutpoint = node->cutpoint();
    int variable = node->variable_index();
    const VariableSummary &variable_summary(model_->variable_summary(variable));
    double candidate_cutpoint = variable_summary.random_cutpoint(rng(), node);
    node->set_variable_and_cutpoint(variable, candidate_cutpoint);
    node->refresh_subtree_data();
    double candidate_log_likelihood = subtree_log_integrated_likelihood(node);

    double log_alpha = candidate_log_likelihood - original_log_likelihood;
    double logu = log(runif_mt(rng()));
    if (logu < log_alpha) {
      MH_accounting_.record_acceptance("change_cutpoint");
    } else {
      MH_accounting_.record_rejection("change_cutpoint");
      node->set_variable_and_cutpoint(variable, original_cutpoint);
      node->refresh_subtree_data();
    }
  }

  //----------------------------------------------------------------------
  void BartPosteriorSamplerBase::slice_sample_cutpoint(Tree *tree) {
    MoveTimer timer = MH_accounting_.start_time("slice_cutpoint");
    TreeNode *node = tree->random_interior_node(rng());
    if (!node) {
      return;
    }
    int variable = node->variable_index();
    const VariableSummary &variable_summary(model_->variable_summary(variable));
    if (variable_summary.is_continuous()) {
      slice_sample_continuous_cutpoint(node);
    } else {
      slice_sample_discrete_cutpoint(node);
    }
    MH_accounting_.record_acceptance("slice_cutpoint");
  }

  //----------------------------------------------------------------------
  class ContinuousCutpointLogLikelihood {
   public:
    ContinuousCutpointLogLikelihood(BartPosteriorSamplerBase *sampler,
                                    TreeNode *node, double lower_cutpoint_bound,
                                    double upper_cutpoint_bound)
        : sampler_(sampler),
          node_(node),
          lower_cutpoint_bound_(lower_cutpoint_bound),
          upper_cutpoint_bound_(upper_cutpoint_bound) {}

    double operator()(double cutpoint) {
      if (cutpoint < lower_cutpoint_bound_) {
        return negative_infinity();
      } else if (cutpoint > upper_cutpoint_bound_) {
        return negative_infinity();
      }
      node_->set_variable_and_cutpoint(node_->variable_index(), cutpoint);
      node_->refresh_subtree_data();
      return sampler_->subtree_log_integrated_likelihood(node_);
    }

   private:
    BartPosteriorSamplerBase *sampler_;
    TreeNode *node_;
    double lower_cutpoint_bound_;
    double upper_cutpoint_bound_;
  };

  void BartPosteriorSamplerBase::slice_sample_continuous_cutpoint(
      TreeNode *node) {
    int variable = node->variable_index();
    const VariableSummary &variable_summary(model_->variable_summary(variable));
    Vector range = variable_summary.get_cutpoint_range(node);
    ContinuousCutpointLogLikelihood logf(this, node, range[0], range[1]);
    ScalarSliceSampler slice(logf);
    slice.set_limits(range[0], range[1]);
    double cutpoint = slice.draw(node->cutpoint());
    node->set_variable_and_cutpoint(variable, cutpoint);
    node->refresh_subtree_data();
  }

  void BartPosteriorSamplerBase::slice_sample_discrete_cutpoint(
      TreeNode *node) {
    int variable = node->variable_index();
    const VariableSummary &variable_summary(model_->variable_summary(variable));
    Vector potential_cutpoint_values =
        variable_summary.get_cutpoint_range(node);
    if (potential_cutpoint_values.empty()) {
      report_error(
          "Started with an illegal configuration in "
          "slice_sample_discrete_cutpoint");
    } else if (potential_cutpoint_values.size() == 1) {
      // There is only one choice.  We need to stay where we are.
      return;
    }

    double logf_slice =
        subtree_log_integrated_likelihood(node) - rexp_mt(rng(), 1.0);
    Selector possible_cutpoint_positions(potential_cutpoint_values.size(),
                                         true);
    double logp = logf_slice - 1;
    while (logp < logf_slice && possible_cutpoint_positions.nvars() > 0) {
      int pos = possible_cutpoint_positions.random_included_position(rng());
      if (pos < 0) {
        report_error(
            "Something went wrong when sampling cutpoints in "
            "'slice_sample_discrete_cutpoint'");
      }
      double cutpoint = potential_cutpoint_values[pos];
      node->set_variable_and_cutpoint(variable, cutpoint);
      node->refresh_subtree_data();
      logp = subtree_log_integrated_likelihood(node);
      possible_cutpoint_positions.drop(pos);
    }
    if (logp < logf_slice && possible_cutpoint_positions.nvars() == 0) {
      report_error(
          "Ran out of choices for cutpoints when slice sampling "
          "a discrete variable.");
    }
  }

  //----------------------------------------------------------------------
  void BartPosteriorSamplerBase::draw_terminal_means_and_adjust_residuals(
      Bart::Tree *tree) {
    for (Tree::NodeSetIterator it = tree->leaf_begin(); it != tree->leaf_end();
         ++it) {
      Bart::TreeNode *leaf = *it;
      double mean = draw_mean(leaf);
      leaf->set_mean(mean);
      leaf->replace_mean_effect();
    }
  }

  //----------------------------------------------------------------------
  // A Metropolis Hastings move attempting to add a new stump.
  void BartPosteriorSamplerBase::tree_birth_move() {
    // Bail if the birth move is certain to fail.  It would be nice to
    // avoid the extra evaluation of log_prior_number_of_trees(), but
    // this is probably cheaper than the work leading up to the MH
    // accept probability calculation.
    if (log_prior_number_of_trees_(model_->number_of_trees() + 1) ==
        negative_infinity()) {
      return;
    }

    std::shared_ptr<Tree> proposal(new Tree(0.0));
    proposal->populate_sufficient_statistics(create_suf());
    fill_tree_with_residual_data(proposal.get());

    TreeNode *root = proposal->root();
    bool node_can_split = false;
    // Make sure that the variable chosen as the root is splittable.
    // For example, if was the constant term in a design matrix then
    // it would not be splittable.
    while (!node_can_split) {
      int variable_index =
          random_int_mt(rng(), 0, model_->number_of_variables() - 1);
      const VariableSummary &variable_summary(
          model_->variable_summary(variable_index));
      node_can_split = variable_summary.random_cutpoint(rng(), root);
    }

    // Split the root with probability determined by the prior, but
    // don't allow further splits.  Keep all mean parameters at 0 for
    // the moment so the split does not affect the residuals from the
    // current model.
    if (log(runif_mt(rng())) < log_probability_of_split(0)) {
      proposal->grow(root);
    }

    double logu = log(runif_mt(rng()));
    if (logu < tree_birth_log_acceptance_probability(proposal.get())) {
      model_->add_tree(proposal);
      draw_terminal_means_and_adjust_residuals(proposal.get());
    }
  }

  //----------------------------------------------------------------------
  // Computes the log of the MH acceptance ratio for moving from the
  // current model to the current model plus *proposal.  The MH ratio
  // is not truncated above by 1, so the test for acceptance is
  // log(runif()) < tree_birth_log_acceptance_probability().
  // Args:
  //   proposal: The proposed tree to add to the current ensemble.
  //     The proposal is assumed to have at most one split, otherwise
  //     this function returns negative_infinity.  The mean parameters
  //     at the leaves of proposal have not been set.
  double BartPosteriorSamplerBase::tree_birth_log_acceptance_probability(
      Bart::Tree *proposal) {
    // The prior omits factors that cancel with current_log_prior
    // (e.g. the priors for the trees they both share), as well as
    // factors that cancel with the transition probability (e.g. the
    // prior for the variable and cutpoint associated with a split, if
    // there is one).
    //
    // Formally, let theta[F+1] denote the proposed new tree
    // (integrating over mean parameters at the leaves), let theta[j]
    // denote a tree in the current model, let Theta denote the full
    // current model, and let Theta* = Union(current model, proposal).
    // The factors of the prior are
    //
    // p(Theta*) = p(F+1) * p(theta[1] | F+1) * ...
    //                    * p(theta[F] | F + 1)
    //                    * p(theta[F+1] | F+1)
    // p(Theta)  =   p(F) * p(theta[1] | F) * ...
    //                    * p(theta[F] | F)
    //
    // Because the prior distribution on mean parameters depends on
    // the number of trees, the p(theta[j]) factors don't cancel,
    // because the number of trees changes.  However, p(theta[j] | F)
    // / p(theta[j] | G) can be evaluated in terms of sufficient
    // statistics.
    //
    // We evaulate p(F) and p(F+1) using the prior on the number of
    // trees.  p(theta[F+1]) contains a prior over the tree topology
    // as well as a prior on the sets of variables and cutpoints.  The
    // prior over the variables and cutpoints will cancel with the
    // transition proposal p(Theta -> Theta*), because the proposal
    // samples variables and cutpoints from the prior.  The prior on
    // tree topology does not cancel, because the proposal is limited
    // to at most one split.
    double proposal_log_prior =
        log_prior_number_of_trees_(model_->number_of_trees() + 1);
    switch (proposal->number_of_leaves()) {
      case 1:
        proposal_log_prior += log_probability_of_no_split(0);
        break;
      case 2:
        proposal_log_prior +=
            log_probability_of_split(0) + 2 * log_probability_of_no_split(1);
        break;
      default:
        report_error(
            "tree_birth_log_acceptance_probability called with a "
            "proposal containing more than one split.");
    }

    double current_log_prior =
        log_prior_number_of_trees_(model_->number_of_trees());

    // This part computes the ratio
    //                      p(M[j] | F + 1)
    // \prod_{j = 1} ^ F -------------------------
    //                      p(M[j] | F)
    //
    // where M[j] is all the mean parameters in tree j.  This ratio
    // can be written
    //
    //   [(F + 1) / F]^(n/2) \exp(-0.5 * sum_{i = 1}^n (mu[i]^2)/ tau^2)
    //
    // where F is the number of trees in the current model, n is the
    // total number of leaves in the current model, and tau^2 is the
    // total_prediction_variance.
    double current_number_of_trees = model_->number_of_trees();
    double proposed_number_of_trees = current_number_of_trees + 1;
    GaussianSuf suf = model_->mean_effect_sufstats();
    int number_of_leaves = suf.n();
    double log_prior_mean_ratio =
        .5 * number_of_leaves *
            log(proposed_number_of_trees / current_number_of_trees) -
        .5 * suf.sumsq() / total_prediction_variance_;

    // As mentioned above, the only component of the prior we consider
    // here is tree topology, because the other elements of the
    // proposal probability (variable and cutpoint) cancel with the
    // prior.
    double log_proposal_transition_probability =
        proposal->number_of_leaves() == 1 ? log_probability_of_no_split(0)
                                          : log_probability_of_split(0);

    // The reverse transition probability is the probability of
    // proposing the current configuration given current + proposal.
    // The reverse proposal is generated by choosing a random stump
    // from the set of available stumps.  The starting point for the
    // reverse transition always includes at least one stump, because
    // proposal is a stump.
    double log_reverse_transition_probability =
        log(1.0 / (1 + model_->number_of_stumps()));

    double proposal_loglike = 0;
    for (Bart::Tree::ConstNodeSetIterator it = proposal->leaf_begin();
         it != proposal->leaf_end(); ++it) {
      proposal_loglike += log_integrated_likelihood((*it)->compute_suf());
    }

    // Any root will do here, since they all start with the same set
    // of data.
    double current_loglike =
        complete_data_log_likelihood(proposal->root()->compute_suf());

    double log_numerator = proposal_loglike + proposal_log_prior -
                           log_proposal_transition_probability;
    double log_denominator = current_loglike + current_log_prior -
                             log_reverse_transition_probability;

    return log_numerator + log_prior_mean_ratio - log_denominator;
  }

  //----------------------------------------------------------------------
  // The tree death move operates by choosing a random stump as a
  // proposal, and killing the proposed stump with a probability
  // determined by the MH algorithm.  A stump is a tree with at most
  // one split.
  void BartPosteriorSamplerBase::tree_death_move() {
    Bart::Tree *stump = model_->random_stump(rng());
    if (!stump) {
      // Return if no stumps found.
      return;
    }
    // When computing the six components of the MH ratio ([likelihood,
    // prior, proposal density] X [candidate, proposal] ), we ignore
    // factors that cancel in the MH probability.  The omitted factors
    // in the log likelihood calculations are documented in
    // complete_data_log_likelihood for each of the derived classes.
    //
    // Notation: Let F be the number of trees (size of the 'F'orest).
    // Let theta[F] denote the stump we're proposing to delete.  Let
    // Theta = theta[1], ..., theta[F] denote the ensemble of trees in
    // the current model, including all mean parameters.  Let Theta* =
    // theta[1], ..., theta[F-1] denote the ensemble in the model that
    // we would get if theta[F] were removed.
    //
    // The priors are:
    // p(Theta*) = p(F-1) * p(theta[1] | F - 1) * ...
    //                    * p(theta[F-1] | F - 1)
    // p(Theta)  =   p(F) * p(theta[1] | F) * ...
    //                    * p(theta[F-1] | F)
    //                    * p(theta[F] | F)
    //
    // The components of p(theta[F]) include tree topology (splits),
    // variables, cutpoints, and mean parameters.  The variables and
    // cutpoints will cancel with the reverse proposal density,
    // because the reverse proposal simulates them from the prior.
    // The topology will partly (but not completely) cancel with the
    // reverse proposal.  The topology, variables, and cutpoints prior
    // components will cancel for theta[1]...theta[F-1] but the mean
    // parameters will not, because eliminating a tree creates a
    // looser prior on the remaining mean parameters.  The mean
    // parameters will partially cancel, though, so we compute their
    // ratio instead of computing them separately.  The mean
    // parameters for theta[F] must be considered as well.

    int current_number_of_trees = model_->number_of_trees();
    int proposed_number_of_trees = current_number_of_trees - 1;
    if (proposed_number_of_trees == 0) {
      return;
    }

    // Other than mean effects and tree size, everything else cancels
    // out of proposal log prior.
    double proposal_log_prior =
        log_prior_number_of_trees_(proposed_number_of_trees);

    // The proposed transition is killing a random stump.  We check
    // for the no-stump case above, so no additional error checking
    // here.
    double log_proposal_transition_probability =
        -log(model_->number_of_stumps());

    // This section computes the contribution of the ratio of mean
    // effect distributions to the log MH probability.
    GaussianSuf current_mean_suf = model_->mean_effect_sufstats();
    GaussianSuf proposal_mean_suf = current_mean_suf;
    GaussianSuf stump_mean_suf;
    for (Tree::ConstNodeSetIterator it = stump->leaf_begin();
         it != stump->leaf_end(); ++it) {
      // This loop is over at most 2 elements.
      double mu = (*it)->mean();
      proposal_mean_suf.remove(mu);
      stump_mean_suf.update_raw(mu);
    }

    double log_prior_mean_ratio =
        -0.5 * proposal_mean_suf.n() *
            log(current_number_of_trees / proposed_number_of_trees) +
        0.5 * proposal_mean_suf.sumsq() / total_prediction_variance_;

    // Now compute the current prior and reverse transition
    // probability.
    double current_log_prior =
        log_prior_number_of_trees_(current_number_of_trees);

    double log_reverse_transition_probability = 0;
    double current_mean_sd =
        sqrt(total_prediction_variance_ / current_number_of_trees);
    switch (stump->number_of_leaves()) {
      case 1: {
        double logp_nosplit0 = log_probability_of_no_split(0);
        log_reverse_transition_probability += logp_nosplit0;
        current_log_prior += logp_nosplit0;
        current_log_prior +=
            dnorm(stump->root()->mean(), 0, current_mean_sd, true);
        break;
      }
      case 2: {
        double logp_split0 = log_probability_of_split(0);
        double logp_nosplit1 = log_probability_of_no_split(1);
        log_reverse_transition_probability += logp_split0;
        current_log_prior += logp_split0 + 2 * logp_nosplit1;
        current_log_prior += dnorm(stump->root()->left_child()->mean(), 0,
                                   current_mean_sd, true);
        current_log_prior += dnorm(stump->root()->right_child()->mean(), 0,
                                   current_mean_sd, true);
        break;
      }
      default:
        report_error(
            "The 'stump' proposed in the tree_death move isn't "
            "really a stump");
    }

    // Compute the likelihood contributions for the proposal and the
    // current model.

    double current_loglike =
        complete_data_log_likelihood(stump->root()->compute_suf());

    stump->remove_mean_effect();
    double proposal_loglike =
        complete_data_log_likelihood(stump->root()->compute_suf());

    double log_numerator = proposal_loglike + proposal_log_prior -
                           log_proposal_transition_probability;
    double log_denominator = current_loglike + current_log_prior -
                             log_reverse_transition_probability;

    double log_acceptance_probability =
        log_numerator + log_prior_mean_ratio - log_denominator;
    if (log(runif_mt(rng())) < log_acceptance_probability) {
      model_->remove_tree(stump);
    } else {
      stump->replace_mean_effect();
    }
  }

  //----------------------------------------------------------------------
  // p(split) = alpha / (1 + d)^beta
  double BartPosteriorSamplerBase::log_probability_of_split(int depth) const {
    return log_prior_tree_depth_alpha_ -
           prior_tree_depth_beta_ * log_integer(1 + depth);
  }

  double BartPosteriorSamplerBase::probability_of_split(int depth) const {
    return prior_tree_depth_alpha_ / pow(1 + depth, prior_tree_depth_beta_);
  }

  //----------------------------------------------------------------------
  // log( 1 - p(split) )
  double BartPosteriorSamplerBase::log_probability_of_no_split(
      int depth) const {
    return log(probability_of_no_split(depth));
  }

  double BartPosteriorSamplerBase::probability_of_no_split(int depth) const {
    return 1 - probability_of_split(depth);
  }

  double BartPosteriorSamplerBase::mean_prior_variance() const {
    double ntrees = model_->number_of_trees();
    if (ntrees < 1) ntrees = 1;
    return total_prediction_variance_ / ntrees;
  }

  //----------------------------------------------------------------------
  void BartPosteriorSamplerBase::clear_data_from_trees() {
    for (int i = 0; i < model_->number_of_trees(); ++i) {
      model_->tree(i)->clear_data_and_delete_suf();
    }
  }

}  // namespace BOOM
