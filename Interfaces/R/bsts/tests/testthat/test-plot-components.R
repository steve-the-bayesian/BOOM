data(iclaims)
ss <- AddLocalLinearTrend(list(), initial.claims$iclaimsNSA)
ss <- AddSeasonal(ss, initial.claims$iclaimsNSA, nseasons = 52)
model <- bsts(iclaimsNSA ~ ., state.specification = ss,
  data = initial.claims, niter = 100)

test_that("PlotBstsComponents handles errors correctly", {
  expect_error(plot(model, "comp", burn = 10, components = 99), ".* is not TRUE")
  expect_error(plot(model, "comp", burn = 10, components = 1:2), NA)
  expect_error(plot(model, "comp", burn = 10, components = 2:1), NA)
  expect_error(plot(model, "comp", burn = 10, components = c(-1, 2)), ".* is not TRUE")
  expect_error(plot(model, "comp", burn = 10, components = numeric(0)), ".* is not TRUE")
  })

