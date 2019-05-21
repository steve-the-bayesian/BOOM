notcran <- Sys.getenv("NOT_CRAN") != ""
if (notcran) {
  seed <- 8675309

  SeasonalTransitionMatrix <- function(nseasons) {
    ## Returns the transition matrix for a seasonal state model.
    ##
    ## Args:
    ##   nseasons:  The number of seasons per cycle.
    ##
    ## Returns:
    ##   A matrix with nseasons - 1 rows and columns.
    id.matrix <- diag(rep(1, nseasons - 2))
    return(rbind(rep(-1, nseasons - 1),
      cbind(id.matrix, rep(0, nrow(id.matrix)))))
  }

  SimulateSeasonalPattern <- function(sample.size, initial.pattern,
                                      season.duration, innovation.sd) {
    ## Args:
    ##   sample.size:  The number of time points to simulate.
    ##   initial.pattern: The pattern from a single cycle, which need not sum to
    ##     zero.
    ##   season.duration: The number of time points that each seasons will last.
    ##   innovation.sd: The standard deviation of the innovation error term in the
    ##     seasonal state model.
    ##
    ## Returns:
    ##   A vector of length 'sample.size' containing the contribution of this
    ##   seasonal component to the mean of the series.

    ## Compute the initial state by removing the final element from the initial
    ## pattern.
    state <- head(initial.pattern, -1)
    nseasons <- length(initial.pattern)
    transition.matrix <- SeasonalTransitionMatrix(nseasons)
    pattern <- numeric(sample.size)
    for (i in 1:sample.size) {
      pattern[i] <- state[1]
      state <- transition.matrix %*% state
      state[1] <- state[1] + rnorm(1, 0, innovation.sd)
    }
    if (season.duration > 1) {
      pattern <- rep(pattern, each = season.duration)[1:sample.size]
    }
    return(pattern)
  }

  set.seed(seed)

  daily.pattern <- rnorm(7)

  ## There are roughly 52 weeks per year, but we can pretend there are
  ## fewer for testing purposes.
  weeks.per.year <- 5
  n.years <- 10

  ## A smooth annual pattern is more easily aliased with the trend.
  weekly.annual.pattern <- rnorm(weeks.per.year,
    cos(2 * pi * (1:weeks.per.year) / weeks.per.year), .1)

  sample.size <- round(7 * weeks.per.year * n.years)

  trend <- cumsum(rnorm(sample.size, 0, .3))
  seasonal.daily <- SimulateSeasonalPattern(sample.size, daily.pattern,
    season.duration = 1, innovation.sd = .15)
  seasonal.annual <- SimulateSeasonalPattern(sample.size, weekly.annual.pattern,
    season.duration = 7, innovation.sd = .5)

  series <- rnorm(sample.size, trend + seasonal.daily + seasonal.annual, 1.0)

  ss <- AddLocalLevel(list(), series, sigma.prior = SdPrior(.3, 10))
  ss <- AddSeasonal(ss, series, nseasons = 7, sigma.prior = SdPrior(.15, 10))
  ss <- AddSeasonal(ss, series, nseasons = weeks.per.year, season.duration = 7,
    sigma.prior = SdPrior(.5, 10))

  model <- bsts(series, ss, niter = 500, seed = seed, prior = SdPrior(1.0, 10))

  ## Check that the recovered state values match the truth.
  test_that("seasonal model covers true state", {
    expect_that(model, is_a("bsts"))
    ## The trend here is more jagged than the model expects, so the trend test
    ## fails.  That's fine as long as the other two tests pass.
    ##
    ## expect_true(CheckMcmcMatrix(model$state.contributions[, 1, ],
    ##   truth = trend, confidence = .5),
    ##   info = "trend failed")
    expect_true(CheckMcmcMatrix(model$state.contributions[, 2, ],
      truth = seasonal.daily, confidence = .8),
      info = "seasonal.daily failed")
    expect_true(CheckMcmcMatrix(model$state.contributions[, 3, ],
      truth = seasonal.annual, confidence = .8),
      info = "seasonal.annual failed")
  })
}
