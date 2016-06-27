#####################################################################
# Directory         : C:/Users/K/z_Teach/MIDS_AdvStat/pgms
# Program Name      : tsa5.R
# Original Developer: Jeffrey Yau
# Last Updated by   : Jeffrey Yau
# Last Updated      : 5/5/2015
# -------------------------------------------------------------------
# Main Topics Covered:
#
# 1. Random walks
# 2. ARIMA models in-depth Discussion
#    a. Simulation of a series of ARIMA model (manually)
#    b. Examine the empirical patterns of these models
#    c. Examine the ACF and PACF of AR-generated realizations
#    d. Model estimation
#    e. Model diagnostics using estimated residuals 
#    f. Forecasting
#    g. Forecast Evaluation
#####################################################################

#####################################################################
# Setup

# Set working directory
setwd("C:/Users/K/z_Teach/MIDS_AdvStat/pgms")
getwd()

# Set Numeric Value Display
# See reference from https://stat.ethz.ch/R-manual/R-devel/library/base/html/options.html
options(digits=2) # Set the printed number of digits to 2. Note: It is a suggestion only. Default is 7. 
#options("scipen" = 10)

# Set memory limit
memory.limit(50000000)

# Remove all the existing variables in the working directory
rm(list=ls())
ls()

# Type "Ctrl+L" to clear the console 

################################################################
# Load Libraries

require(forecast)

################################################################

################################################################
# Recap:
# Remember that classical regression is insufficient for
# explaining all of the interesting dynamics of a time
# series, meaning that there could be additional 
# structure of the data that is not captured.
#
# Introducing correlation through lagged linear relations with
# a series' past values and white noises leads to the ARMA models
# introduced in the last 2 lectures, starting with AR and MA models
# separately. Recall that to achieve the principle of parsimony, 
# however, a lower order ARMA model is better than AR and MA models.
# 
# It is important to keep in mind that for practical purposes,
# this class of models applies only to stationary processes, leading
# to the class of Autoregressive Integrated Moving Average (ARIMA)
# models, popularized by Box and Jenkins (1970). This class of 
# models, when combining with techniques to account for trends 
# and cyclicality, can accomodate a wide range of non-stationary
# time series. In fact, we will discuss how to use the Box-Jenkins
# approachto build ARIMA models.
#
# We will continue to use the arima() function to model 
# ARIMA models, including those with seasonal components
# (i.e. Seasonal ARIMA models)
# 
# Remember that non-stationarity often comes in several forms:
# trend, seasonality, and time-varying variance. ARIMA models
# can be used to "account for" trend and seasonality. However,
# it still assumes a constant (unconditional and conditional) variance.
# To model conditional heteroskedasticity, we will use the 
# Generalized Autoregressive Conditional Heteroskedasticity (GARCH)
# models, which we will study in the next lecture.
################################################################


################################################################
## ARIMA Models: Quantitative Analysis

# ---------------
#  1. Simulation
# ---------------
set.seed(898) # Set the simulation to use the same random number sequence
# for all of the simluation to ensure that difference
# in results is not due to difference in the random
# number sequences generated

# To simulate an ARMA model, we will use the function arima.sim
#
# arima.sim(model, n, rand.gen = rnorm, innov = rand.gen(n, ...),
#          n.start = NA, start.innov = rand.gen(n.start, ...),
#          ...)

# For our purpose, we will specify 
#    (1) the length of the simulated series (n) 
#    (2) the orders of autoregressive and moving average parameters (model)
#
# In simulation, it is very important to keep in mind
# the distribution from which the random numbers are drawn. ALWAYS CHECK
# WHAT THE DEFAULT OPTION PROVIDES, AND DO NOT ALWAYS
# JUST ACCEPT THE DEFAULT OPTION AVAILABLE. The default random
# number generator in the arima.sim() function draws from a normal(0,1)
# distribution. Because ARMA models use indepedent Gaussian white noise,
# we will use the default random number generation option. We will
# also need to specify 

# Remember that when using arima.sim(), the underlying ARIMA model
# be modeled uses the arima() function. The default option for an ARMA model
# (i.e. arima(p,d=0,q)), assumes the series is demeaned
# (i.e. include.mean is true). So, the formula applies to x-m rather than x.
#
# For ARIMA models (i.e. arima(p,d<>0,q)), the differenced series follows a
# zero-mean ARMA model.

##################################################################
# Simulation of Random Walk

# Random Walk without Drift
set.seed(898)
x1 <- w <- rnorm(100)
for (i in 2:100) x1[i] <- x1[i-1] + w[i]
  str(x1)
  str(w)
  head(cbind(x1,w),20)
  summary(x1)
plot.ts(x1, main="Random Walk without Drift (100 Simulations)", 
        col="blue", xlab="Simulated Time Period",
        ylab="Values of x1")

# Random Walk with Drift
x2 <- w
  head(cbind(x2,w),20)
drift <- 0.5
for (i in 2:100) x2[i] <- drift + x2[i-1] + w[i]
  head(cbind(x2,w), 20)
  summary(x2)
plot.ts(x2, main="Random Walk with Drift=0.5",
        col="navy", lty=2, 
        xlab="Simulated Time Period",
        ylab="Values of x2")


# Overlaid one graph on the other
plot.ts(x2, main="Random Walk Processes with and without Drift",
        col="navy", lty=2, ylim=c(-10,60),
        xlab="Simulated Time Period",
        ylab="Values of x1 and x2")
lines(x1, col="blue")
leg.txt <- c("Random Walk with Drift=0.5", "Random Walk without Drift")
legend("topleft", legend=leg.txt, lty=c(2,1), col=c("navy","blue"),
       bty='n', cex=1)


##################################################################
# Part 2: ARIMA models
# ----------------------------------------------------------------

rm(list=ls())
ls()

# (example from the book, pp. 131):
set.seed(898)
x1 <- w <- rnorm(100)
for (i in 3:100) x1[i] <- 0.5*x1[i-1] + x1[i-1] - 0.5*x1[i-2] + w[i] + 0.3*w[i-1]

  str(x1)
  summary(x1)
  length(x1)

png("C:/Users/K/z_Teach/MIDS_AdvStat/notes/sim_series1.jpg", width=400, height=350)
par(mfrow=c(3,2))
  plot.ts(x1); title("Fig 1: Simulated Series")
  plot.ts(diff(x1)); title("Fig 2: First Difference of the Simulated Series")
  acf(x1, main=""); title("Fig 3: ACF of the Simulated Series")
  acf(diff(x1), main=""); title("Fig 4: ACF of the Differenced Simulated Series")
  pacf(x1, main=""); title("Fig 5: PACF of the Simulated Series")
  pacf(diff(x1), main=""); title("Fig 6: PACF of the Differenced Simulated Series")
dev.off()

# Estimation: Estimate the simulated series
    fit1 <- arima(x1, order=c(1,1,1)) #ARIMA(p=1, d=1, q=1)
    summary(fit1)

    #fit1b <- arima(diff(x1), order=c(1,0,1))
    #summary(fit1b)
    #
    #x1c <- x1
    #for (i in 3:1000) x1c[i] <- 0.5*x1c[i-1] + w[i] + 0.3*w[i-1]
    #fit1c <- arima(x1c, order=c(1,1,1))
    #summary(fit1c)

# Model Diagnostics: Residuals
  summary(fit1$resid)
  Box.test(fit1$resid, type="Ljung-Box")

  par(mfrow=c(2,2))
  plot(fit1$resid, col="blue", main="Residual Series")
  hist(fit1$resid, col="gray", main="Residuals")
  acf(fit1$resid , ylim=c(-0.4,1), main="ACF: Residual Series")
  pacf(fit1$resid, ylim=c(-0.4,1), main="PACF: Residual Series")

# Model Performance Evaluation: In-Sample Fit

  df <- data.frame(cbind(x1,fitted(fit1),fit1$resid ))
  class(df)  
  stargazer(df, type="text", title="Descriptive Stat", digits=1)

  summary(x1)
  summary(fit1$resid)
  par(mfrow=c(1,1))
  plot.ts(x1, col="navy", 
        main="ARMA Simulated vs a ARIMA Estimated Series with Resdiauls",
        ylab="Original and Estimated Values",
        ylim=c(-5,30), pch=1, lty=2)
  par(new=T)
  plot.ts(fitted(fit1),col="blue",axes=T,xlab="",ylab="",
          ylim=c(-5,30), lty=1) 
  leg.txt <- c("Original Series", "Estimated Series", "Residuals")
  legend("topleft", legend=leg.txt, lty=c(1,1,2), 
         col=c("navy","blue","green"), bty='n', cex=1)
  par(new=T)
  plot.ts(fit1$resid,axes=F,xlab="",ylab="",col="green",
          ylim=c(-5,30), lty=2, pch=1, col.axis="green")
  axis(side=4, col="green")
  mtext("Residuals", side=4, line=2,col="green")

# Forecasting

fit1.fcast <- forecast.Arima(fit1, h=12)

  str(fit1.fcast)
  length(fit1.fcast$mean)

  ts(rbind(melt(x1),melt(fit1.fcast$mean)))
  fit1.fcast

summary(fit1.fcast$mean)

#----------------------------------------------------------------
jpeg("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec11_TSA5/images/sim_ARIMA_fcst.jpg")
plot(fit1.fcast,
     main="12-Step Ahead Forecast and Original & Estimated Series",
     xlab="Simulated Time Period", 
     ylab="Original, Estimated, and Forecasted Values",
     xlim=c(0,112),ylim=c(0,25.0), lty=2,lwd=1.5)
par(new=T)
plot.ts(fitted(fit1),col="blue", 
        lty=2, lwd=2, xlab="",ylab="",xlim=c(0,112),ylim=c(0,25.0))
leg.txt <- c("Original Series", "Estimated Series", "Forecast")
legend("topleft", legend=leg.txt, lty=c(2,2,1), lwd=c(1,2,2),
       col=c("black","blue","blue"), bty='n', cex=1)
dev.off()
#----------------------------------------------------------------

# 7. Backtesting and Out-of-Sample Forecasting

# Re-estimate the model holding out 10 observations
# Based on teh ACF and PACF, we will use an ARIMA(1,1,0) model instead
  str(x1)
  fit <- Arima(x1[1:(length(x1)-10)], order=c(1,1,0))
  summary(fit)

  #--------------------------------------
  # Plot the original and estimate series 
  jpeg("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec11_TSA5/images/sim_ARIMA_bktst_fit.jpg")
  par(mfrow=c(1,1))
  plot.ts(x1[1:(length(x1)-10)], col="navy", 
          main="Original vs an ARIMA(1,1,0) Estimated Series with Resdiauls",
          ylab="Original and Estimated Values",
          ylim=c(-3,25), pch=1)
  par(new=T)
  plot.ts(fitted(fit),col="blue",axes=T,xlab="",ylab="",
          ylim=c(-3,25))

  leg.txt <- c("Original Series", "Estimated Series", "Residuals")
  legend("topleft", legend=leg.txt, lty=1, col=c("navy","blue","green"),
         bty='n', cex=1)

  par(new=T)
  plot.ts(fit$resid,axes=F,xlab="",ylab="",col="green",
          ylim=c(-3,25), pch=1)
          axis(side=4, col="green")
  mtext("Residuals", side=4, line=2,col="green")
  dev.off()
  #--------------------------------------

  fit.fcast <- forecast.Arima(fit, h=20)
  str(fit.fcast)
  length(fit.fcast$mean)
  fit.fcast

  par(mfrow=c(1,1))
  jpeg("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec11_TSA5/images/sim_ARIMA110_bktst_fit2.jpg")
  plot(fit.fcast,lty=2,
       main="Out-of-Sample Forecast",
       xlim=c(0,110), ylim=c(0,40),
       ylab="Original, Estimated, and Forecast Values")
  par(new=T)
  plot.ts(fitted(fit), col="blue",axes=F,xlim=c(0,110),ylim=c(0,40),
          ylab="", lty=1)
  par(new=T)
  plot.ts(x1, col="navy",axes=F,xlim=c(0,110),ylim=c(0,40),
          ylab="", lty=2)
  leg.txt <- c("Original Series", "Fitted series", "Forecast")
  legend("topleft", legend=leg.txt, lty=c(2,1,1),
         col=c("navy","blue","blue"), lwd=c(1,1,2),
         bty='n', cex=1)
  dev.off()

