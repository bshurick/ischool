#####################################################################
# Directory         : /Users/bshur/School/Time Series Analysis
# Program Name      : lab3.R
# Original Developer: Brandon Shurick
# Last Updated by   : Brandon Shurick
# Last Updated      : 8/5/2015
# -------------------------------------------------------------------
#####################################################################


#####################################################################
#########                      PART 1                       #########
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
acf(fp, lag.max=120, 
        main='ACF')
pacf(fp, lag.max=120, 
        main='PACF')

#####################################################################


#####################################################################
## Fit Arima Model

# arima model 
get.best.arima <- function(x.ts, test.ts=NA, method='AIC', maxord = c(1,1,1,1,1,1))
{
  best.aic <- 1e8
  best.rmse <- 1e8
  n <- length(x.ts)
  H <- length(test.ts)
  for (p in 0:maxord[1]) for(d in 0:maxord[2]) for(q in 0:maxord[3])
  {
    for (P in 0:maxord[4]) for(D in 0:maxord[5]) for(Q in 0:maxord[6])
    {
      fit <- arima(x.ts, order = c(p,d,q),
                   seas = list(order = c(P,D,Q),
                               frequency(x.ts)), method = "CSS")
      fit.aic <- -2 * fit$loglik + (log(n) + 1) * length(fit$coef)
      if (method=='RMSE')
      {
        fit.fcast <- forecast.Arima(fit, h=H)
        fit.rmse <- sqrt(mean((fit.fcast$mean-test.ts)**2))
        if (fit.rmse < best.rmse)
        {
          best.rmse <- fit.rmse
          best.aic <- fit.aic
          best.fit <- fit
          best.model <- c(p,d,q,P,D,Q)
        }
      }
      else 
      {
        if (fit.aic < best.aic)
        {
          best.rmse <- NA
          best.aic <- fit.aic
          best.fit <- fit
          best.model <- c(p,d,q,P,D,Q)
        }
      }
    }
  }
  list(best.rmse, best.aic, best.fit, best.model)
}
N <- length(fp)
test <- (N - 51):(N)
train <- 1:(N - 52)
get.best.arima(fp[train], fp[test], method='RMSE')

# fit best model
fp.best_arima <- arima(x = fp, 
                       order=c(0,0,1), 
                       seasonal=list(order=c(0,0,1),  
                                     frequency(fp)),
                       method = "CSS")

#####################################################################


#####################################################################
## Evaluate ARIMA Model

# plot model in-sample residuals
dev.off()
par(mfrow=c(2,2))
plot(fp.best_arima$residuals, main='ARIMA (0,0,1)(0,0,1) In-sample Residuals')
hist(fp.best_arima$residuals, main='ARIMA (0,0,1)(0,0,1) In-sample Residuals')
acf(fp.best_arima$residuals, main='ACF: ARIMA (0,0,1)(0,0,1) In-sample Residuals')
pacf(fp.best_arima$residuals, main='PACF: ARIMA (0,0,1)(0,0,1) In-sample Residuals')

# summary 
summary(fp.best_arima$residuals)

# make forecast 
fp.best_arima.fcast <- forecast.Arima(fp.best_arima, h=52)

# Plot forecast vs original
dev.off()
par(mfrow=c(1,1))
xlimits <- c(2004, 2017)
ylimits <- c(-3, 6)
plot(fp.best_arima.fcast, lty=2, xlim=xlimits,ylim=ylimits,
     main="Out-of-Sample Forecast",
     ylab="Original, Estimated, and Forecast Values")
par(new=T)
plot.ts(fitted(fp.best_arima.fcast), 
        col="blue",lty=1,axes=F, xlim=xlimits,ylim=ylimits,ylab='')
par(new=T)
plot.ts(fp, col="navy",axes=F,xlim=xlimits,ylim=ylimits,ylab="", lty=2)

# add legend
leg.txt <- c("Original Series", "Fitted series", "Forecast")
legend("topleft", legend=leg.txt, lty=c(2,1,1),
       col=c("navy","blue","blue"), lwd=c(1,1,2),
       bty='n', cex=1)

#####################################################################


#####################################################################
#########                      PART 2                       #########
#####################################################################


#####################################################################
## Setup

# Load libraries
require(tseries)

# Read dataset 
load('/Users/bshur/School/Time Series Analysis/lab3/gasOil.Rdata')
str(gasOil)

#####################################################################


####################################################################
# Evaluate Dataset

# Take summary statistics
summary(gasOil)

# Create time-series objects
gasOil.Production.ts <- ts(gasOil$Production, start=c(1978, 1), frequency=12)
gasOil.Price.ts <- ts(gasOil$Price, start=c(1978, 1), frequency=12)

####################################################################


####################################################################
# Reproduce Correlation Results

# Model Price~Production
plot(gasOil[, c('Production','Price')])
m <- lm(Price~Production, data=gasOil)
summary(m)

####################################################################


####################################################################
# Evaluate Dataset

# Plot Time Series
dev.off()
par(mfrow=c(2,1))
plot.ts(gasOil.Price.ts, main='')
plot.ts(gasOil.Production.ts)

# Evaluate Correlograms
dev.off()
par(mfrow=c(2,1))
acf(gasOil.Price.ts)
acf(gasOil.Production.ts)

# Test for unit roots
adf.test(gasOil.Price.ts)
adf.test(gasOil.Production.ts)
pp.test(gasOil.Price.ts)
pp.test(gasOil.Production.ts)

# Test for co-integration
po.test(cbind(gasOil.Price.ts, gasOil.Production.ts))

# Fails to reject null, go with univariate model
dev.off()
par(mfrow=c(2,2))
plot.ts(gasOil.Price.ts, main='Gas Prices', lty=2, col='navy', ylab='Gas Prices')
lines(filter(gasOil.Price.ts, rep(1,12)/12, sides=2), 
      main='Gas Prices, 12-month Moving Average',
      ylab='Gas Prices',
      xlab='Month',
      col="blue")

# add legend
leg.txt <- c("Original Series", "Moving Average")
legend("topleft", legend=leg.txt, lty=c(2,1), col=c("navy","blue"),
       bty='n', cex=1)

# experiment with differential
gasOil.Price.ts.diff <- diff(gasOil.Price.ts, lag=1)
plot.ts(gasOil.Price.ts.diff, 
        main='First Difference of Gas Prices', 
        lty=2,
        ylab='Gas Price Differential') 
lines(filter(gasOil.Price.ts.diff, rep(1,12)/12, sides=2), 
      main='Gas Prices, 12-month Moving Average',
      ylab='Gas Prices',
      xlab='Month',
      col='blue')

# add legend
leg.txt <- c("Original Series", "Moving Average")
legend("topleft", legend=leg.txt, lty=c(2,1), col=c("navy","blue"),
       bty='n', cex=1)

# plot ACF and PCF
acf(gasOil.Price.ts, lag.max=120, 
    main='ACF')
pacf(gasOil.Price.ts, lag.max=120, 
     main='PACF')

#####################################################################


#####################################################################
## Fit Arima Model

# Fit model using AIC 
get.best.arima(gasOil.Price.ts)

# fit best model
gasOil.Price.best_arima <- arima(x = gasOil.Price.ts, 
                                 order=c(0,1,1), 
                                 seasonal=list(order=c(1,0,1),  
                                               frequency(gasOil.Price.ts)),
                                 method = "CSS")

#####################################################################


#####################################################################
## Evaluate ARIMA Model

# plot model in-sample residuals
dev.off()
par(mfrow=c(2,2))
plot(gasOil.Price.best_arima$residuals, main='ARIMA (0,1,1)(1,0,1) In-sample Residuals')
hist(gasOil.Price.best_arima$residuals, main='ARIMA (0,1,1)(1,0,1) In-sample Residuals')
acf(gasOil.Price.best_arima$residuals, main='ACF: ARIMA (0,1,1)(1,0,1) In-sample Residuals')
pacf(gasOil.Price.best_arima$residuals, main='PACF: ARIMA (0,1,1)(1,0,1) In-sample Residuals')

# summary 
summary(gasOil.Price.best_arima$residuals)

# make forecast 
gasOil.Price.best_arima.fcast <- forecast.Arima(gasOil.Price.best_arima, h=12*4-2)

# Plot forecast vs original
dev.off()
par(mfrow=c(1,1))
xlimits <- c(1978, 2016)
ylimits <- c(0, 7.5)
plot(gasOil.Price.best_arima.fcast, lty=2, 
     xlim=xlimits, ylim=ylimits,
     main="Out-of-Sample Forecast",
     ylab="Original, Estimated, and Forecast Values")
par(new=T)
plot.ts(fitted(gasOil.Price.best_arima.fcast), 
        col="blue",lty=1,axes=F, xlim=xlimits,ylim=ylimits,ylab='')
par(new=T)
plot.ts(gasOil.Price.ts,col="navy",axes=F,xlim=xlimits,ylim=ylimits,ylab="", lty=2)

# add legend
leg.txt <- c("Original Series", "Fitted series", "Forecast")
legend("topleft", legend=leg.txt, lty=c(2,1,1),
       col=c("navy","blue","blue"), lwd=c(1,1,2),
       bty='n', cex=1)

#####################################################################

