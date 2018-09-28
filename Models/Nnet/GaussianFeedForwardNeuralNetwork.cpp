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

namespace BOOM {

  namespace {
    using GFFNN = GaussianFeedForwardNeuralNetwork;
  }

  GFFNN::GaussianFeedForwardNeuralNetwork()
      : terminal_layer_(new RegressionModel(1.0))
  {
    ParamPolicy::add_model(terminal_layer_);
  }

  GFFNN::GaussianFeedForwardNeuralNetwork(const GFFNN &rhs)
      : FeedForwardNeuralNetwork(rhs),
        DataPolicy(rhs),
        terminal_layer_(rhs.terminal_layer_->clone())
  {
    ParamPolicy::add_model(terminal_layer_);
  }

  GFFNN & GFFNN::operator=(const GFFNN &rhs) {
    if (&rhs != this) {
      ParamPolicy::clear();
      FeedForwardNeuralNetwork::operator=(rhs);
      terminal_layer_.reset(rhs.terminal_layer_->clone());
      ParamPolicy::add_model(terminal_layer_);
    }
    return *this;
  }
  
  void GFFNN::restructure_terminal_layer(int dim) {
    if (dim != terminal_layer_->xdim()) {
      ParamPolicy::drop_model(terminal_layer_);
      double sigsq = terminal_layer_->sigsq();
      terminal_layer_.reset(new RegressionModel(dim));
      terminal_layer_->set_sigsq(sigsq);
      ParamPolicy::add_model(terminal_layer_);
    }
  }
  
}  // namespace BOOM
