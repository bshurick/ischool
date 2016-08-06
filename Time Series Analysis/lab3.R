#####################################################################
# Directory         : /Users/bshur/School/Time Series Analysis
# Program Name      : lab3.R
# Original Developer: Brandon Shurick
# Last Updated by   : Brandon Shurick
# Last Updated      : 8/5/2015
# -------------------------------------------------------------------
#####################################################################


#####################################################################
## Setup

# load libraries
require(forecast)

# load dataset
google.data <- read.csv('/Users/bshur/School/Time Series Analysis/lab3/google_correlate_flight.csv')
fp <- ts(google.data[,c('flight.prices')], 
                          start=c(2004,1), 
                          end=c(2016,1),
                          frequency=52)

#####################################################################


#####################################################################
## Data Exploration

# plot time series 
dev.off()
par(mfrow=c(2,2))
plot.ts(fp, main='Flight Price Searches', lty=2, col='navy')
lines(filter(fp, rep(1,12)/12, sides=2), 
      main='Flight Price Searches, 12-week Moving Average',
      ylab='Flight Searches',
      xlab='Week',
      col="blue")

# add legend
leg.txt <- c("Original Series", "Moving Average")
legend("topleft", legend=leg.txt, lty=c(2,1), col=c("navy","blue"),
       bty='n', cex=1)

# experiment with differential
fp.diff <- diff(fp, lag=1)
plot.ts(fp.diff, main='First Difference of Flight Price Searches', lty=2) 
lines(filter(fp.diff, rep(1,12)/12, sides=2), 
     main='First Order Differential, 12-week Moving Average',
     ylab='Flight Searches',
     xlab='Week',
     col='blue')

# add legend
leg.txt <- c("Original Series", "Moving Average")
legend("topleft", legend=leg.txt, lty=c(2,1), col=c("navy","blue"),
       bty='n', cex=1)

# plot ACF and PCF
acf(fp.diff, lag.max=120, 
        main='ACF of First Difference')
pacf(fp.diff, lag.max=120, 
        main='PACF of First Difference')

#####################################################################


#####################################################################
## Fit Arima Model

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

#####################################################################


#####################################################################
## Evaluate ARIMA Model

# plot model in-sample residuals
dev.off()
par(mfrow=c(2,2))
plot(fp.best_arima$residuals, main='ARIMA (0,0,1)(1,0,0) In-sample Residuals')
hist(fp.best_arima$residuals, main='ARIMA (0,0,1)(1,0,0) In-sample Residuals')
acf(fp.best_arima$residuals, main='ACF: ARIMA (0,0,1)(1,0,0) In-sample Residuals')
pacf(fp.best_arima$residuals, main='PACF: ARIMA (0,0,1)(1,0,0) In-sample Residuals')

# summary 
summary(fp.best_arima$residuals)

# make forecast 
fp.best_arima.fcast <- forecast.Arima(fp.best_arima, h=52)

# RMSE 
rmse <- sqrt(mean(fp.best_arima.fcast$residuals**2))
print(paste0('RMSE: ',round(rmse,4)))

# Plot forecast vs original
dev.off()
par(mfrow=c(1,1))
xlimits <- c(0, 628)
ylimits <- c(-3, 6)
plot(fp.best_arima.fcast, lty=2, xlim=xlimits,ylim=ylimits,
     main="Out-of-Sample Forecast",
     ylab="Original, Estimated, and Forecast Values")
par(new=T)
plot.ts(fitted(fp.best_arima.fcast), 
        col="blue",lty=1,axes=F, xlim=xlimits,ylim=ylimits,ylab='')
par(new=T)
plot.ts(fp[train], col="navy",axes=F,xlim=xlimits,ylim=ylimits,ylab="", lty=2)

# add legend
leg.txt <- c("Original Series", "Fitted series", "Forecast")
legend("topleft", legend=leg.txt, lty=c(2,1,1),
       col=c("navy","blue","blue"), lwd=c(1,1,2),
       bty='n', cex=1)

#####################################################################

