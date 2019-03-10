library(bsts)
set.seed(12345)

trend <- cumsum(rnorm(1095, 0, .1))
dates <- seq.Date(from = as.Date("2014-01-01"), length = length(trend), by = "day")
y <- zoo(trend + rnorm(length(trend), 0, .2), dates)

AddHolidayEffect <- function(y, dates, effect) {
  ## Adds a holiday effect to simulated data.
  ## Args:
  ##   y: A zoo time series, with Dates for indices.
  ##   dates: The dates of the holidays.
  ##   effect: A vector of holiday effects of odd length.  The central effect is
  ##     the main holiday, with a symmetric influence window on either side.
  ## Returns:
  ##   y, with the holiday effects added.
  time <- dates - (length(effect) - 1) / 2
  for (i in 1:length(effect)) {
    y[time] <- y[time] + effect[i]
    time <- time + 1
  }
  return(y)
}

## Define some holidays.
memorial.day <- NamedHoliday("MemorialDay")
memorial.day.effect <- c(-.75, -2, -2)
memorial.day.dates <- as.Date(c("2014-05-26", "2015-05-25", "2016-05-25"))
y <- AddHolidayEffect(y, memorial.day.dates, memorial.day.effect)

## The holidays can be in any order.
holiday.list <- list(memorial.day)

## Let's train the model to just before MemorialDay
cut.date = as.Date("2016-05-20")
train.data <- y[time(y) < cut.date]
test.data <- y[time(y) >= cut.date]
ss <- AddLocalLevel(list(), train.data)
ss <- AddRegressionHoliday(ss, train.data, holiday.list = holiday.list)
model <- bsts(train.data, state.specification = ss, niter = 500, ping = 0)
## Now let's make a prediction covering MemorialDay
my.horizon = 15
## Note adding the time stamps here doesn't help either
pred <- predict(object = model, horizon = my.horizon)
## Make a data frame for plotting
plot(pred)

## plot.info <- data.frame(Date = time(y), 
##                         value = y, 
##                         predict.mean = NA,
##                         predict.upper = NA,
##                         predict.lower = NA
##                        )
## plot.info[plot.info$Date %in% time(test.data)[1:my.horizon],]$predict.mean = pred$mean
## plot.info[plot.info$Date %in% time(test.data)[1:my.horizon],]$predict.lower = pred$interval[1,]
## plot.info[plot.info$Date %in% time(test.data)[1:my.horizon],]$predict.upper = pred$interval[2,]

## ## Let's make a pretty plot to demonstrate the problem
## filter(plot.info, Date > time(test.data)[1] - 25 & Date < time(test.data)[my.horizon] + 10)  %>% 
##     ggplot(aes(x = Date, y = value)) +
##     geom_line() +
##     geom_line(aes(y = predict.mean), col = "Forest Green") + # The prediction
##     geom_line(aes(y = predict.lower), col = "Forest Green", lty = 2) + # lower bound
##     geom_line(aes(y = predict.upper), col = "Forest Green", lty = 2)  # upper bound
