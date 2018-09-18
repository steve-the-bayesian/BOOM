# Copyright 2018 Steven L. Scott. All Rights Reserved.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA

library(BoomSpikeSlab)
# if (require(testthat) && require(mlbench)) {
require(testthat)
require(mlbench)
  seed <- 8675309
  set.seed(seed)

  data(BostonHousing)

  hidden.layers <- list(
    HiddenLayer(5, MvnPrior(rep(0, 13), diag(rep(1, 13)))),
    HiddenLayer(5, MvnPrior(rep(0, 5), diag(rep(1, 5)))),
    HiddenLayer(5, MvnPrior(rep(0, 5), diag(rep(1, 5)))))

  model <- BayesNnet(medv ~ ., layers = hidden.layers, niter = 100, data = BostonHousing)
}


