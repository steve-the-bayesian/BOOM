.ValidateHolidayList <- function(holiday.list) {
  ## Args:
  ##   holiday.list:  a list of objects of class Holiday.
  ## Returns:
  ##   If holiday.list is a list of objects of class Holiday then it is returned
  ##   unchanged.  Otherwise an error is raised.
  stopifnot(is.list(holiday.list),
            all(sapply(holiday.list, inherits, "Holiday")))
}

.DefaultRegressionHolidayModelCoefficientPrior <- function(sdy) {
  ## Args:
  ##   sdy: The standard deviation of the target series.
  ## Returns:
  ##   A default prior of class NormalPrior, describing variation (across days
  ##   and across holidays) among daily holiday effects.
  return(NormalPrior(0, sdy))
}

AddRegressionHoliday <- function(state.specification = NULL,
                                 y,
                                 holiday.list,
                                 time0 = NULL,
                                 coefficient.prior = NULL,
                                 sdy = sd(as.numeric(y), na.rm = TRUE)) {
  ## Add a regression-based state model describing the effects of one or more
  ## holidays.  Each day of each holiday exerts a constant effect, specific to
  ## that day, on the observed time series.
  ##
  ## Args:
  ##   state.specification: A list of state components.  If omitted, an empty
  ##    list is assumed.
  ##   y:  A numeric vector.  The time series being modeled.  See 'sdy' below.
  ##   holiday.list: A list of objects of type 'Holiday'.  See ?Holiday.  The
  ##     width of the influence window should be the same number of days for all
  ##     the holidays in this list.  See below if this case does not apply.
  ##   time0: Either NULL, or an object convertible to class Date giving the
  ##     date of the first day in the training data y.  If NULL and y is of type
  ##     'zoo' then an attempt will be made to infer time0 from the index of y.
  ##   coefficient.prior: An object of type NormalPrior describing the a priori
  ##     expected variation among holiday effects.
  ##   sdy: The standard deviation of the 'y' argument, used to speicfy default
  ##     priors in case coefficient.prior is missing.
  ##
  ## Returns:
  ##   A list containing the information needed to specify this state model to
  ##   the underlying C++ code, in the expected format.
  .ValidateHolidayList(holiday.list)
  if (missing(state.specification)) {
    state.specification <- list()
  }

  if (is.null(coefficient.prior)) {
    coefficient.prior <- .DefaultRegressionHolidayModelCoefficientPrior(sdy)
  }
  stopifnot(inherits(coefficient.prior, "NormalPrior"))
  
  spec <- list(name = "RegressionHolidays",
               holidays = holiday.list,
               time0 = as.Date(.SetTimeZero(time0, y)),
               coefficient.prior = coefficient.prior)
  class(spec) <- c("RegressionHolidayStateModel.", "StateModel")
  state.specification[[length(state.specification) + 1]] <- spec
  return(state.specification)
}

AddHierarchicalRegressionHoliday <- function(
    state.specification = NULL,
    y,
    holiday.list,
    coefficient.mean.prior = NULL,
    coefficient.variance.prior = NULL,
    time0 = NULL,
    sdy = sd(as.numeric(y), na.rm = TRUE)) {
  ##
  ##
  ## Args:
  ##   state.specification: A list of state components.  If omitted, an empty
  ##    list is assumed.
  ##   y:  A numeric vector.  The time series being modeled.  See 'sdy' below.
  ##   holiday.list: A list of objects of type 'Holiday'.  See ?Holiday.  The
  ##     width of the influence window should be the same number of days for all
  ##     the holidays in this list.  See below if this case does not apply.
  ##   coefficient.mean.prior: An object of type MvnPrior giving the hyperprior
  ##     for the average effect of a holiday in each day of the influence window.
  ##   coefficient.variance.prior: An object of type InverseWishartPrior
  ##     describing the prior belief about the variation in holiday effects from
  ##     one holiday to the next.
  ##   time0: Either NULL, or an object convertible to class Date giving the
  ##     date of the first day in the training data y.  If NULL and y is of type
  ##     'zoo' then an attempt will be made to infer time0 from the index of y.
  ##   sdy: The standard deviation of the 'y' argument, used to speicfy default
  ##     priors in case either coefficient.mean.prior or
  ##     coefficient.variance.prior are missing.  If both coefficient.mean.prior
  ##     and coefficient.variance.prior are specified then neither 'y' nor 'sdy'
  ##     is needed.  If 'sdy' is specified then 'y' is not needed.
  ##
  ## Returns:
  ##   A list containing the information needed to specify this state model to
  ##   the C++ code, in the expected format.

  ## Make sure the holiday list is a list of Holiday objects, and that it
  ## contains enough holidays to be useful.
  .ValidateHolidayList(holiday.list)
  if (length(holiday.list) < 3) {
    stop("You need 3 or more holidays to fit the hierarchical model in ",
         "AddHierarchicalRegressionHolidayModel.")
  } 
  if (missing(state.specification)) {
    state.specification <- list()
  }

  if (is.null(coefficient.mean.prior) || is.null(coefficient.variance.prior)) {
    if (missing(sdy)) {
      stopifnot(is.numeric(y))
      sdy <- sd(as.numeric(y), na.rm = TRUE)
    }
    max.window.width <- unique(sapply(holiday.list, MaxWindowWidth))
    if (length(max.window.width) != 1) {
      stop("All holidays must have the same window width.")
    }
  }

  if (is.null(coefficient.mean.prior)) {
    coefficient.mean.prior <- MvnPrior(
      mean = rep(0, max.window.width),
      variance = diag(rep(sdy, max.window.width))
    )
  }
  stopifnot(inherits(coefficient.mean.prior, "MvnPrior"))

  if (is.null(coefficient.variance.prior)) {
    coefficient.variance.prior <- InverseWishartPrior(
      variance.guess = diag(rep(sdy / 10, max.window.width)),
      variance.guess.weight = max.window.width + 1)
  }
  stopifnot(inherits(coefficient.variance.prior,
    "InverseWishartPrior"))
  
  spec <- list(name = "HierarchicalRegressionHolidays",
               holidays = holiday.list,
               time0 = as.Date(.SetTimeZero(time0, y)),
               coefficient.mean.prior = coefficient.mean.prior,
               coefficient.variance.prior = coefficient.variance.prior)
  class(spec) <- c("HierarchicalRegressionHolidayStateModel.", "StateModel")
  state.specification[[length(state.specification) + 1]] <- spec
  return(state.specification)
}
                                               
