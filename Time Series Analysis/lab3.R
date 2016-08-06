
google.data <- read.csv('/Users/bshur/School/Time Series Analysis/lab3/google_correlate_flight.csv')
google.data$DateFmt <- as.Date(as.character(google.data$Date), format="%m/%d/%y", tz='UTC')
fp <- ts(google.data[,c('flight.prices')], 
                          start=c(2004,1), 
                          end=c(2016,1),
                          frequency=52)

# plot time series 
par(mfrow=c(2,1))
plot.ts(fp, main='Flight Price Searches, Time Series')
plot.ts(filter(fp, rep(1,4)/4, sides=2), main='Flight Price Searches, 4-week Moving Average')
dev.off()

# experiment with differential
par(mfrow=c(2,1))
fp.diff <- diff(fp, lag=1)
plot.ts(fp.diff, main='First Order Differential') 
plot.ts(filter(fp.diff, rep(1,4)/4, sides=2), main='First Order Differential, 4-week Moving Average')
dev.off()

# calculate means
mean(fp.diff)
sd(fp.diff)

# plot ACF and PCF
par(mfrow=c(2,1))
plot.ts(acf(fp.diff, lag.max=120), main='Flight Price Searches, ACF')
plot.ts(pacf(fp.diff, lag.max=120), main='Flight Price Searches, PACF')
dev.off()

# arima model 
get.best.arima <- function(x.ts, maxord = c(1,1,1,1,1,1))
{
  best.aic <- 1e8
  n <- length(x.ts)
  for (p in 0:maxord[1]) for(d in 0:maxord[2]) for(q in 0:maxord[3])
  {
    for (P in 0:maxord[4]) for(D in 0:maxord[5]) for(Q in 0:maxord[6])
    {
      fit <- arima(x.ts, order = c(p,d,q),
                   seas = list(order = c(P,D,Q),
                               frequency(x.ts)), method = "CSS")
      fit.aic <- -2 * fit$loglik + (log(n) + 1) * length(fit$coef)
      if (fit.aic < best.aic)
      {
        best.aic <- fit.aic
        best.fit <- fit
        best.model <- c(p,d,q,P,D,Q)
      }
    }
  }
  list(best.aic, best.fit, best.model)
}
N <- length(fp)
test <- (N - 52):(N)
train <- 1:(N - 53)
get.best.arima(fp[train])
fp.best_arima <- arima(x = fp[train], order=c(0,0,1), seasonal=list(order=c(1,0,0)))
