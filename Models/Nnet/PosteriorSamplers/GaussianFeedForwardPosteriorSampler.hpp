#ifndef BOOM_GAUSSIAN_FEEDFORWARD_POSTERIOR_SAMPLER_HPP_
#define BOOM_GAUSSIAN_FEEDFORWARD_POSTERIOR_SAMPLER_HPP_
/*
  Copyright (C) 2005-2018 Steven L. Scott

  This library is free software; you can redistribute it and/or modify it under
  the terms of the GNU Lesser General Public License as published by the Free
  Software Foundation; either version 2.1 of the License, or (at your option)
  any later version.

  This library is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
  details.

  You should have received a copy of the GNU Lesser General Public License along
  with this library; if not, write to the Free Software Foundation, Inc., 51
  Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
*/

#include "Models/Nnet/GaussianFeedForwardNeuralNetwork.hpp"
#include "Models/Nnet/PosteriorSamplers/HiddenLayerImputer.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"

namespace BOOM {

  class GaussianFeedForwardPosteriorSampler
      : public PosteriorSampler {
   public:
    explicit GaussianFeedForwardPosteriorSampler(
        GaussianFeedForwardNeuralNetwork *model,
        RNG &seeding_rng = GlobalRng::rng);

    double logpri() const override;
    void draw() override;

   private:
    //---------------------------------------------------------------------------
    // This section contains implementation for the 'draw' method.
    void impute_hidden_layer_outputs(RNG &rng);

    // Simulate from the posterior distribution of model parameters, conditional
    // on the imputed {0, 1} values at the hidden nodes.
    void draw_parameters_given_hidden_nodes();

    // Remove imputed data from the models used to implement the hidden and
    // terminal layers.
    void clear_latent_data();

    // The un-normalized conditional distribution for the inputs to the
    // termainal layer (i.e. the outputs from the final hidden layer).  This
    // distribution conditions on the model parameters and observed data, and
    // integrates over the latent data from previous hidden layers.
    //
    // Args:
    //   response: The observed value for this observation.
    //   binary_inputs: The inputs to the terminal layer (outputs from the final
    //     hidden layer).
    //   logprob: The log of the probability that each node in the final hidden
    //     layer is 'on', conditional on the predictors for the observation.
    //   logprob_complement: The log of the probability that each node in the
    //     final hidden layer is 'off', conditional on the predictors for the
    //     observation.
    double terminal_inputs_log_full_conditional(
        double response,
        const Vector &binary_inputs,
        const Vector &logprob,
        const Vector &logprob_complement) const;

    // Ensure that the proper data structures have been built for storing latent
    // data.
    void ensure_space_for_latent_data();

    // Ensure that each hidden layer in the model has a HiddenLayerImputer
    // allocated to manage it.
    void ensure_imputers();

    // Implementation for impute_hidden_layer_outputs.  Don't call these from
    // elsewhere.
    void impute_terminal_layer_inputs(RNG &rng,
                                      double response,
                                      std::vector<bool> &inputs,
                                      Vector &wsp1, Vector &wsp2);

    //----------------------------------------------------------------------
    // Data section.
    GaussianFeedForwardNeuralNetwork *model_;

    // Each imputer is responsible for one hidden layer.
    std::vector<HiddenLayerImputer> imputers_;

    // imputed_hidden_layer_outputs_[i][layer][node] indicates whether the
    // specified node in the specified hidden layer is 'on' for observation i.
    std::vector<Nnet::HiddenNodeValues> imputed_hidden_layer_outputs_;
  };

}  // namespace BOOM

#endif  //  BOOM_GAUSSIAN_FEEDFORWARD_POSTERIOR_SAMPLER_HPP_
