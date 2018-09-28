#ifndef BOOM_NNET_GAUSSIAN_FEEDFORWARD_NEURAL_NETWORK_HPP_
#define BOOM_NNET_GAUSSIAN_FEEDFORWARD_NEURAL_NETWORK_HPP_
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

#include "Models/Nnet/Nnet.hpp"
#include "Models/Glm/RegressionModel.hpp"

namespace BOOM {

  class GaussianFeedForwardNeuralNetwork
      : public FeedForwardNeuralNetwork,
        public IID_DataPolicy<RegressionData>
  {
   public:
    GaussianFeedForwardNeuralNetwork();

    GaussianFeedForwardNeuralNetwork(const GaussianFeedForwardNeuralNetwork &rhs);
    GaussianFeedForwardNeuralNetwork(GaussianFeedForwardNeuralNetwork &&rhs) = default;
    GaussianFeedForwardNeuralNetwork &operator=(
        const GaussianFeedForwardNeuralNetwork &rhs);
    GaussianFeedForwardNeuralNetwork &operator=(
        GaussianFeedForwardNeuralNetwork &&rhs) = default;
    GaussianFeedForwardNeuralNetwork *clone() const override {
      return new GaussianFeedForwardNeuralNetwork(*this);
    }

    int number_of_observations() const override { return dat().size(); }

    double predict(const ConstVectorView &predictors) const {
      FeedForwardNeuralNetwork::fill_prediction_workspace(predictors);
      return terminal_layer_->predict(terminal_layer_inputs());
    }
    double predict(const VectorView &predictors) const {
      return predict(ConstVectorView(predictors));
    }
    double predict(const Vector &predictors) const {
      return predict(ConstVectorView(predictors));
    }

    Ptr<RegressionModel> terminal_layer() {return terminal_layer_;}

    double residual_sd() const {return terminal_layer_->sigma();}
    
    // Args:
    //   dim: The number of outputs in the last hidden layer (the layer farthest
    //     from the input data).
    void restructure_terminal_layer(int dim) override;
    
   private:
    Ptr<RegressionModel> terminal_layer_;
  };
  
}  // namespace BOOM 

#endif  //  BOOM_NNET_GAUSSIAN_FEEDFORWARD_NEURAL_NETWORK_HPP_
