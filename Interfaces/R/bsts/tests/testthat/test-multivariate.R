library(bsts)
library(testthat)

# GDP figures for 57 countries as reported by the OECD.
data(gdp)
series.id <- gdp$Country
timestamps <- gdp$Time

## test_that("Multivariate model runs", {

##   shared.ss <- AddSharedLocalLevel(list(), wide.data)
##   series.ss <- AddStaticIntercept(list(), wide.data)
  
##   model <- mbsts(response, shared.ss, series.ss, niter = 100)
  
##   expect_that(model, is_a("mbsts"))
##   expect_true(all(abs(model$state.contributions) < 10))

## })
