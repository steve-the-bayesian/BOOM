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
                                 prior = NULL,
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
  ##   prior: An object of type NormalPrior describing the a priori
  ##     expected variation among holiday effects.
  ##   sdy: The standard deviation of the 'y' argument, used to speicfy default
  ##     priors in case prior is missing.
  ##
  ## Returns:
  ##   A list containing the information needed to specify this state model to
  ##   the underlying C++ code, in the expected format.
  .ValidateHolidayList(holiday.list)
  if (missing(state.specification)) {
    state.specification <- list()
  }
  if (is.null(prior)) {
    prior <- .DefaultRegressionHolidayModelCoefficientPrior(sdy)
  }
  stopifnot(inherits(prior, "NormalPrior"))
  spec <- list(name = "RegressionHolidays",
               holidays = holiday.list,
               time0 = as.Date(.SetTimeZero(time0, y)),
               size = 1,
               prior = prior)
  class(spec) <- c("RegressionHolidayStateModel", "HolidayStateModel", "StateModel")
  state.specification[[length(state.specification) + 1]] <- spec
  return(state.specification)
}

.PlotHolidayRegressionCoefficients <- function(coefficients, ylim, ...) {
  ## A driver function used to implement common plotting needs for
  ## plot.RegressionHolidayStateModel and
  ## plot.HierarchicalRegressionHolidayStateModel.
  ##
  ## Args:
  ##   coefficients: A matrix of holiday coefficients. Each row is a Monte Carlo
  ##     draw, and each column is a specific holiday coefficient.  The column
  ##     names for the matrix are expected to give the name of the holiday
  ##     corresponding to each coefficient.  All coefficients for a given
  ##     holiday window are expected to be grouped together in ascending order.
  ##   ylim:  Limits for the vertical axis.
  ##   ...: Extra arguments passed to boxplot.
  ##
  ## Side Effects:
  ##   A plot is added to the current graphics device.  Side-by-side boxplots
  ##   show the posterior distribution of the effect of each holiday.
  ##
  ## Returns:
  ##   invisible(NULL)
  if(is.null(ylim)) {
    ylim = range(coefficients)
  }
  variable.names <- colnames(coefficients)
  old.mai <- par("mai")
  old.mai[1] <- max(strheight(variable.names, units = "inches"))
  old.mai[2] <- max(strwidth(variable.names, units = "inches")) +
    strwidth("        -- ", units = "inches")
  oldpar <- par(mai = old.mai)
  on.exit(par(oldpar))

  ## Reversing the order of the coefficients means the plot can be read from top
  ## to bottom, which is the natural order.
  new.holiday.index <- (1:ncol(coefficients))[colnames(coefficients) != ""][-1]
  coefficients <- coefficients[, ncol(coefficients):1]
  new.holiday.index <- 1 + ncol(coefficients) - new.holiday.index

  boxplot(coefficients, horizontal = TRUE, ylim = ylim, las = 1, cex = .6, ...)
  abline(v = 0, lty = 3)
  abline(h = new.holiday.index + .5, lty = 3)
  return(invisible(NULL))
}

plot.RegressionHolidayStateModel <- function(x,
                                             bsts.object,
                                             burn = NULL,
                                             time = NULL,
                                             style = NULL,
                                             ylim = NULL,
                                             ...) {
  ## S3 method for plotting a RegressionHolidayStateModel
  ##
  ## Args:
  ##   x: An object inheriting from RegressionHolidayStateModel.
  ##   bsts.object: A bsts model that includes state.specification in its state
  ##     specification.
  ##   burn:  The number of MCMC iterations to discard as burn-in.
  ##   time: Not used.  Here to match the signature of plot.StateModel.
  ##   style: Not used.  Here to match the signature of plot.StateModel.
  ##   ylim:  Limits on the vertical axis.
  ##   ...: Extra arguments passed boxplot
  ##
  ## Side Effects:
  ##   A plot is added to the current graphics device.  Side-by-side boxplots
  ##   show the posterior distribution of the effect of each holiday.
  ##
  ## Returns:
  ##   invisible(NULL)
  state.specification <- x
  stopifnot(inherits(state.specification, "RegressionHolidayStateModel"))
  stopifnot(inherits(bsts.object, "bsts"))
  if (is.null(.FindStateSpecification(state.specification, bsts.object))) {
    stop("The state specification is not part of the bsts object.")
  }

  holidays <- state.specification$holidays
  coefficients <- NULL
  for (i in 1:length(holidays)) {
    new.coefficients <- bsts.object[[holidays[[i]]$name]]
    colnames(new.coefficients) <-
      c(holidays[[i]]$name, rep("", ncol(new.coefficients) - 1))
    coefficients <- cbind(coefficients, new.coefficients)
  }
  if (is.null(burn)) {
    burn <- 0
  }
  stopifnot(is.numeric(burn))
  if (burn > 0) {
    coefficients <- coefficients[-(1:burn), , drop = FALSE]
  }
  .PlotHolidayRegressionCoefficients(coefficients, ylim, ...)
  title("Regression Holiday Effects")
  return(invisible(NULL))
}

plot.HierarchicalRegressionHolidayStateModel <- function(x,
                                                         bsts.object,
                                                         burn = NULL,
                                                         time = NULL,
                                                         style = NULL,
                                                         ylim = NULL,
                                                         ...) {
  ## S3 method for plotting a HierarchicalRegressionHolidayStateModel.
  ##
  ## Args:
  ##   x: An object inheriting from HierarchicalRegressionHolidayStateModel.
  ##   bsts.object: A bsts model that includes state.specification in its state
  ##     specification.
  ##   burn:  The number of MCMC iterations to discard as burn-in.
  ##   time: Not used.  Here to match the signature of plot.StateModel.
  ##   style: Not used.  Here to match the signature of plot.StateModel.
  ##   ylim:  Limits on the vertical axis.
  ##   ...: Extra arguments passed boxplot
  ##
  ## Side Effects:
  ##   A plot is added to the current graphics device.  Side-by-side boxplots
  ##   show the posterior distribution of the effect of each holiday.
  ##
  ## Returns:
  ##   invisible(NULL)
  state.specification <- x
  stopifnot(inherits(
    state.specification, "HierarchicalRegressionHolidayStateModel"))
  stopifnot(inherits(bsts.object, "bsts"))
  if (is.null(.FindStateSpecification(state.specification, bsts.object))) {
    stop("The state specification is not part of the bsts object.")
  }

  coefficient.array <- bsts.object$holiday.coefficients
  if (is.null(burn)) {
    burn <- 0
  }
  if (burn > 0) {
    coefficient.array <- coefficient.array[-(1:burn), , , drop = FALSE]
  }
  holiday.window.size <- dim(coefficient.array)[3]
  number.of.holidays <- dim(coefficient.array)[2]
  coefficients <- matrix(nrow = dim(coefficient.array)[1],
    ncol = holiday.window.size * number.of.holidays)
  colnames(coefficients) <- rep("", ncol(coefficients))
  colnames(coefficients)[seq(from = 1, by = holiday.window.size,
    len = number.of.holidays)] <- dimnames(coefficient.array)[[2]]

  start <- 0
  for (i in 1:number.of.holidays) {
    coefficients[, start + (1:holiday.window.size)] <- coefficient.array[, i, ]
    start <- start + holiday.window.size
  }

  .PlotHolidayRegressionCoefficients(coefficients, ylim, ...)
  title("Hierarchical Regression\nHoliday Effects")
  return(invisible(NULL))
}

AddHierarchicalRegressionHoliday <- function(
    state.specification = NULL,
    y,
    holiday.list,
    coefficient.mean.prior = NULL,
    coefficient.variance.prior = NULL,
    time0 = NULL,
    sdy = sd(as.numeric(y), na.rm = TRUE)) {
  ## Args:
  ##   state.specification: A list of state components.  If omitted, an empty
  ##    list is assumed.
  ##   y:  A numeric vector.  The time series being modeled.  See 'sdy' below.
  ##   holiday.list: A list of objects of type 'Holiday'.  See ?Holiday.  The
  ##     width of the influence window should be the same number of days for all
  ##     the holidays in this list.  See below if this case does not apply.
  ##   coefficient.mean.prior: An object of type MvnPrior giving the hyperprior
  ##     for the average effect of a holiday in each day of the influence
  ##     window.
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
               size = 1,
               coefficient.mean.prior = coefficient.mean.prior,
               coefficient.variance.prior = coefficient.variance.prior)
  class(spec) <- c("HierarchicalRegressionHolidayStateModel",
    "HolidayStateModel", "StateModel")
  state.specification[[length(state.specification) + 1]] <- spec
  return(state.specification)
}
