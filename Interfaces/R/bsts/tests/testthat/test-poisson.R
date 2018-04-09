
library(bsts)
data(shark)

test_that("Poisson bsts runs without crashing", {
ss <- AddLocalLevel(list(), y = log(1 + shark$Attacks))
model <- bsts(shark$Attacks, ss, niter = 1000, family = "poisson")
expect_that(model, is_a("bsts"))
})
