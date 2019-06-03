set.seed(8675309)
library(bsts)
data(goog)

## This works
## ss0 <- AddSemilocalLinearTrend(list(), as.numeric(goog))
## model0 <- bsts(as.numeric(goog), ss0, niter = 50)


# This example has lots of missing data but no timestamps
## pattern <- c(T, T, T, T, T, F, F)
## pattern <- rep(pattern, len = (7/5) * length(goog))
## if (sum(pattern) > length(goog)) {
##   pattern <- head(pattern, -(sum(pattern) - length(goog)))
## }
## extended.goog <- rep(NA, length(pattern))
## extended.goog[pattern] <- goog
## ss1 <- AddSemilocalLinearTrend(list(), extended.goog)
## model1 <- bsts(extended.goog, ss1, niter = 50)


## This does not.
goog <- zoo(as.numeric(goog), index(goog))
ss <- AddSemilocalLinearTrend(list(), goog)
model <- bsts(goog, ss, niter = 50)


