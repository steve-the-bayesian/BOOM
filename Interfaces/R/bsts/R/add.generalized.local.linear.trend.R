# Copyright 2011 Google Inc. All Rights Reserved.
# Author: stevescott@google.com (Steve Scott)

AddGeneralizedLocalLinearTrend <- function (state.specification = list(),
                                            y = NULL,
                                            level.sigma.prior = NULL,
                                            slope.mean.prior = NULL,
                                            slope.ar1.prior = NULL,
                                            slope.sigma.prior = NULL,
                                            initial.level.prior = NULL,
                                            initial.slope.prior = NULL,
                                            sdy = NULL,
                                            initial.y = NULL) {
  warning("AddGeneralizedLocalLinearTrend is deprecated (because it was a ",
          "terrible name!). Please use AddSemilocalLinearTrend instead.")
  return(AddSemilocalLinearTrend(
      state.specification,
      y,
      level.sigma.prior,
      slope.mean.prior,
      slope.ar1.prior,
      slope.sigma.prior,
      initial.level.prior,
      initial.slope.prior,
      sdy,
      initial.y))
}
