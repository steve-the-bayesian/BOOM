update.bsts <- function(model, newdata, trials.or.exposure=1, na.action=na.exclude, seed = NULL) {
  ## Update the latent state draws for a bsts model in response to new data.
  ## This is essentially "rolling" the state forward.  Parameter draws are not
  ## updated, but future calls to "predict" jump off from the update state
  ## rather than the state saved at train time.

  stopifnot(inherits(model, "bsts"))
  if (model$has.regression) {
    stopifnot(inherits(newdata, "dataframe"))
  } else {
    stopifnot(is.numeric(newdata))
  }

  update.data <- .FormatBstsUpdateData(model, newdata, trials.or.exposure=trials.or.exposure, na.action=na.action)

  updates <- .Call("analysis_common_r_update_bsts_final_state_",
                   model,
                   update.data,
                   seed = seed)

  stopifnot(is.matrix(updates))
  return(updates)
}

###======================================================================

.FormatBstsUpdateData <- function(object, newdata, trials.or.exposure, na.action) {
  formatted.data <- .FormatBstsPredictionData(object, newdata, trials.or.exposure, na.action)
  ## TODO(steve):  This will work for the POC, but lots of edge cases to handle here.
  formatted.data$response <- newdata
  formatted.data$response.is.observed <- !is.na(newdata)
  return(formatted.data)
}
