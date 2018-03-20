% Copyright 2018 Google LLC. All Rights Reserved.
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

\name{replist}

\alias{RepList}

\Rdversion{1.0}

\title{
  Repeated Lists of Objects
}

\description{
  Produces repeated copies of an object.
}

\usage{
RepList(object, times)
}

\arguments{
  \item{object}{ The object to be replicated. }
  \item{times}{The desired number of replications.}
}

\value{
  Returns a list containing \code{times} copies of \code{object}.
}

\examples{
  alist <- list(x = "foo", y = 12, z = c(1:3))
  three.copies <- RepList(alist, 3)
}

\author{
  Steven L. Scott
}
