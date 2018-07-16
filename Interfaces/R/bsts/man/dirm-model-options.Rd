% Copyright 2018 Steven L. Scott. All Rights Reserved.
%
% This library is free software; you can redistribute it and/or
% modify it under the terms of the GNU Lesser General Public
% License as published by the Free Software Foundation; either
% version 2.1 of the License, or (at your option) any later version.
%
% This library is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
% Lesser General Public License for more details.
%
% You should have received a copy of the GNU Lesser General Public
% License along with this library; if not, write to the Free Software
% Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA

\alias{DirmModelOptions}
\name{dirm-model-optoins}
\title{Specify Options for a Dynamic Intercept Regression Model}
\Rdversion{1.1}
\description{
  Specify modeling options for a dynamic intercept regression model.
}

\usage{
DirmModelOptions(timeout.seconds = Inf,
                 high.dimensional.threshold.factor = 1.0)
}

\arguments{

  \item{timeout.seconds}{The overall time budget for model fitting.  If
    the MCMC algorithm takes longer than this number, the current
    iteration will complete, and then the fitting algorithm will return
    with however many MCMC iterations were managed during the allotted
    time. }

  \item{high.dimensional.threshold.factor}{ When doing Kalman filter
    updates for the model, Sherman-Morrisson-Woodbury style updates are
    applied for high dimensional data, while direct linear algebra is
    used for low dimensional data.  The definition of "high dimensional"
    is relative to the dimension of the state.  An observation is
    considered high dimensional if its dimension exceeds the state
    dimension times this factor.
  }

}

\value{
  An object of class \code{DirmModelOptions}, which is simply a list
  containing values of the function arguments.

  The value of using this function instead of making a list "by hand" is
  that argument types are properly checked, and list names are sure to
  be correct.
}

\keyword{models}
\keyword{regression}
