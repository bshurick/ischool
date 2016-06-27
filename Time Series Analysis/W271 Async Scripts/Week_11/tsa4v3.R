###################################################################### Directory         : C:/Users/K/pgms/_advstat
# Directory         : C:/Users/K/z_Teach/MIDS_AdvStat/pgms
# Program Name      : tsa4v3.R
# Original Developer: Jeffrey Yau
# Last Updated by   : Jeffrey Yau
# Last Updated      : 5/7/2015
# -------------------------------------------------------------------
# Main Topics Covered:
#
# MA models and ARMA in-depth Discussion
# 1. Simulation of a series of AR(1) and AR(2) models
# 2. Examine the empirical patterns of these models
# 3. Examine the ACF and PACF of AR-generated realizations
# 4. Model estimation
# 5. Model diagnostics using estimated residuals 
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

# Type "Ctrl+L" to clear the console (if you want)

################################################################
# Load Libraries

library(astsa)    # Time series package by Shummway and Stoffer
library(zoo)      # time series package
library(forecast)
#require(tseries)
library(quantmod) # Financial time series package
#require(MASS)
#require(graphics)
#require(corrgram)

################################################################

################################################################
# Recap:
# Remember that classical regression is insufficient for
# explaining all of the interesting dynamics of a time
# series, meaning that there could be additional 
# structure of the data that is not captured.
#
# Introducing correlation through lagged linear relations
# leads to the ARMA models introduced in the last 3
# lectures, starting with AR and MA models separately.
# To achieve parsimony, however, a lower order ARMA model
# is better than AR and MA models.
# 
# It is important to keep in mind that for practical purposes,
# this class of models applies only to stationary processes.
#
# In this lecture, we will finish off studying stationary models,
# focusing on MA and ARMA models. As in the discussions
# of AR models in the last lecture, we will use the followin
# packages. The URL is provided here, but each function can also
# be searched for within R using the help function, as long as 
# the package is already installed.
#
# Package Used : stats
#           URL: https://stat.ethz.ch/R-manual/R-patched/library/stats/html/00Index.html
# Major Functions Used: ariam.sim
#                       arima
#                       acf
#                       pacf
#
# Package Used : 
################################################################

################################################################
# Part 1: MA(q) Model
# Unlike AR(p) models, the moving average process is stationary
# for any values of the parameters of the model! 
#
# Let's simulate different MA(q) models using different number
# of parameters and values of the parameters.
#
################################################################

#---------------------------------------------------------------
#Theoretical ACF of MA(1):
theta = seq(-2,2,0.01)
rho = theta/(1+theta^2)
plot(theta, rho, xlab=expression(~~~theta),
     ylab=expression(~~~theta/(1+theta^2)),
     main = "Theoretical ACF of MA(1)")
# ==> Show that abs(rho) <=1/2 for all values of theta.

#---------------------------------------------------------------
# -----------
# Exercise 1: 
# -----------
#  Plot the theoretical ACF of MA(2) models

#---------------------------------------------------------------

#---------------------------------------------------------------
# Simulations
# and Using the Simulated Data to illustrate
#  Estimation
#  Model Diagnosis and Assumption Testing
#  Model Performance Evaluation
#  Forecasting / Statistical Inference
#---------------------------------------------------------------

# ----------------
# MA(1) - 4 models
# ----------------
set.seed(898)
ma1a <- arima.sim(list(order=c(0,0,1), ma=.9), n=100)
set.seed(898)
ma1b <- arima.sim(list(order=c(0,0,1), ma=-.9), n=100)
set.seed(898)
ma1c <- arima.sim(list(order=c(0,0,1), ma=.5), n=100)
set.seed(898)
ma1d <- arima.sim(list(order=c(0,0,1), ma=-.5), n=100)
  str(ma1a)
  
par(mfrow = c(2,2))

# Time-series plots
hist(ma1a, ylab="value", main=(expression(MA(1)~~~theta==+.9)))
hist(ma1b, ylab="x", main=(expression(MA(1)~~~theta==-.9)))
hist(ma1c, ylab="x", main=(expression(MA(1)~~~theta==+.5)))
hist(ma1d, ylab="x", main=(expression(MA(1)~~~theta==-.5)))


# Time-series plots
plot(ma1a, ylab="value", main=(expression(MA(1)~~~theta==+.9)))
plot(ma1b, ylab="x", main=(expression(MA(1)~~~theta==-.9)))
plot(ma1c, ylab="x", main=(expression(MA(1)~~~theta==+.5)))
plot(ma1d, ylab="x", main=(expression(MA(1)~~~theta==-.5)))

# ACF Plots
acf(ma1a,main="Fig1: ACF of MA1(ma=0.9)")
acf(ma1b,main="Fig2: ACF of MA1(ma=-0.9)")
acf(ma1c,main="Fig3: ACF of MA1(ma=0.5)")
acf(ma1d,main="Fig4: ACF of MA1(ma=-0.5)")

#PACF Plots
pacf(ma1a,main="Fig1: PACF of MA1(ma=0.9)")
pacf(ma1b,main="Fig2: PACF of MA1(ma=-0.9)")
pacf(ma1c,main="Fig3: PACF of MA1(ma=0.5)")
pacf(ma1d,main="Fig4: PACF of MA1(ma=-0.5)")



# ----------------
# MA(2) - 8 models
# ----------------
set.seed(898)
ma2a1 <- arima.sim(list(order=c(0,0,2), ma=c(0.9,  0.4)), n=100)
set.seed(898)
ma2a2 <- arima.sim(list(order=c(0,0,2), ma=c(0.9, -0.4)), n=100)

set.seed(898)
ma2b1 <- arima.sim(list(order=c(0,0,2), ma=c(-.9, 0.4)), n=100)
set.seed(898)
ma2b2 <- arima.sim(list(order=c(0,0,2), ma=c(-.9, -0.4)), n=100)

set.seed(898)
ma2c1 <- arima.sim(list(order=c(0,0,2), ma=c(0.5, 0.4)), n=100)
set.seed(898)
ma2c2 <- arima.sim(list(order=c(0,0,2), ma=c(0.5, -0.4)), n=100)

set.seed(898)
ma2d1 <- arima.sim(list(order=c(0,0,2), ma=c(-0.5, 0.4)), n=100)
set.seed(898)
ma2d2 <- arima.sim(list(order=c(0,0,2), ma=c(-0.5, -0.4)), n=100)

par(mfrow = c(2,2))
# Time-series plots
plot(ma2a1, ylab="simulated value", main="MA2(0.9, 0,4)")
plot(ma2a2, ylab="simulated value", main="MA2(0.9,-0.4)")
plot(ma2b1, ylab="simulated value", main="MA2(-0.9,0.4)")
plot(ma2b2, ylab="simulated value", main="MA2(-0.9,-0.4)")

plot(ma2c1, ylab="simulated value", main="MA2(0.5, 0.4)")
plot(ma2c2, ylab="simulated value", main="MA2(0.5,-0.4)")
plot(ma2d1, ylab="simulated value", main="MA2(0.5, 0.4)")
plot(ma2d2, ylab="simulated value", main="MA2(0.5,-0.4)")

par(mfrow=c(3,2))
plot(ma1a,  ylab="simulated value", main=(expression(MA(1)~~~theta==+.9)))
 acf(ma1a,  ylab="Autocorrelation", main="ACF of MA1(0.9)")

plot(ma2a1, ylab="simulated value", main="MA2(0.9, 0,4)")
 acf(ma2a1, ylab="Autocorrelation", main="ACF of MA2(0.9, 0,4)")

plot(ma2a2, ylab="simulated value", main="MA2(0.9,-0.4)")
 acf(ma2a2, ylab="Autocorrelation value", main="ACF MA2(0.9,-0.4)")

# Observations:
# 1. It is not easy to distinguish among MA(1) and MA(2) models
#    based only on the time-series plots if their first MA parameter 
#    have the same value, although the MA(2) model with a negative 2nd
#    parameter value appear to be a lot more volatile.
# 2. The ACFs show the difference between MA(1) and MA(2) models; 
#    as seen in the discussion of the mathematical properties of MA models
#    their ACFs drop off at the time lag correponding to the model.
# 3. The ACF sharply drops off after two lags for the MA2 models
# 4. Realizations from MA models generally do not show
#    any apparent trends

par(mfrow=c(2,2))
plot(ma2a1, ylab="simulated value", main="MA2(0.9,0.4)")
acf(ma2a1, ylab="Autocorrelation value", main="ACF MA2(0.9,0.4)")

plot(ma2a2, ylab="simulated value", main="MA2(0.9,-0.4)")
 acf(ma2a2, ylab="Autocorrelation", main="ACF MA2(0.9,-0.4)")

par(mfrow=c(2,2))
plot(ma2b1, ylab="simulated value", main="MA2(-0.9,0.4)")
acf(ma2b1, ylab="Autocorrelation value", main="ACF MA2(-0.9,0.4)")

plot(ma2b2, ylab="simulated value", main="MA2(-0.9,-0.4)")
acf(ma2b2, ylab="Autocorrelation", main="ACF MA2(-0.9,-0.4)")

# ACF Plots
acf(ma2a1,main="Fig1: ACF of MA2(0.9,0.4)")
acf(ma2a2,main="Fig2: ACF of MA2(0.9,-0.4)")
acf(ma2b1,main="Fig3: ACF of MA2(-0.9,0.4)")
acf(ma2b2,main="Fig4: ACF of MA2(-0.9,-0.4)")

#PACF Plots
pacf(ma2a1,main="Fig1: PACF of MA2(0.9,0.4)")
pacf(ma2a2,main="Fig2: PACF of MA2(0.9,-0.4)")
pacf(ma2b1,main="Fig3: PACF of MA2(-0.9,0.4)")
pacf(ma2b2,main="Fig4: PACF of MA2(-0.9,-0.4)")


# ----------------------------------------------
# Estimation 
# ----------------------------------------------

par(mfrow=c(2,2))
plot(ma2c2, ylab="simulated value", main="MA2(0.5,-0.4)")
hist(ma2c2, col="grey", ylab="simulated value", main="MA2(0.5,-0.4)")
acf(ma2c2, ylab="simulated value", main="MA2(0.5,-0.4)")
pacf(ma2c2, ylab="simulated value", main="MA2(0.5,-0.4)")

# Estimation using simulated data
# Let's consider estimating a model using the simluated data
# (ma2c2), which come from the model MA2(0.5,-0.4)

ma2.fit <- arima(ma2c2, order=c(0,0,2))
  ma2.fit
  summary(ma2.fit)
  # Observations:
  # Both of the MA parameters are not statistically different from the 
  # true theoretical values (0.5, -0.4)

 # auto.arima???
 #fitted.Arima

# Diagnosis using residuals
  # Visualization
  head(cbind(ma2c2, fitted(ma2.fit), ma2.fit$resid),10)

  df<-data.frame(cbind(ma2c2, fitted(ma2.fit), ma2.fit$resid))
  library(stargazer)
  stargazer(df, type="text")
  summary(ma2.fit$resid)

  par(mfrow=c(2,2))
    plot.ts(ma2.fit$resid, main="Residual Series",
            ylab="residuals", col="navy")
    hist(ma2.fit$resid, col="gray", main="Residuals")
    acf(ma2.fit$resid, main="ACF of Residuals")
    pacf(ma2.fit$resid, main="PACF of Residuals")

    #Observations: All of the evidence point to the residuals mimicing white noise

  # Ljung-Box test of residual dynamics (or lack thereof)
  # Reference: https://stat.ethz.ch/R-manual/R-patched/library/stats/html/box.test.html
    # These tests are sometimes applied to the residuals from an ARMA(p, q) fit,
    # in which case the references suggest a better approximation to the null-hypothesis 
    # distribution is obtained by setting fitdf = p+q, provided of course that lag > fitdf.
  Box.test(ma2.fit$resid, type="Ljung-Box") # Box-Pierce test
    # Observations: the test cannot rejects the null hypothesis of independence 
    #               of the residual series.


# Model Performance Evaluation Using In-Sample Fit
par(mfrow=c(1,1))
  plot(ma2c2, col="navy", 
       main="Original vs Estimated Series (MA2(0.5,-0.4))",
       ylab="Simulated and Estimated Values", lty=2)
  lines(fitted(ma2.fit),col="blue")
  leg.txt <- c("Original Series", "Estimated Series")
  legend("topright", legend=leg.txt, lty=c(2,1), 
         col=c("navy","blue"), bty='n', cex=1)

  #lines(ma2.fit$resid, col="green")
  #lines(ma2.fit$fcast)

  df <- cbind(ma2c2, fitted(ma2.fit), ma2.fit$resid)
    #Observations: 

# Forecast
  # Given the underlying assumptions of the model is correct, we can proceed to 
  # to do forecast
  ma2.fit.fcast <- forecast.Arima(ma2.fit, 10)
  summary(ma2.fit.fcast)

  plot(ma2.fit.fcast, main="10-Step Ahead Forecast and Original & Estimated Series",
      xlab="Simulated Time Period", ylab="Original, Estimated, and Forecasted Values",
      xlim=c(), lty=2, col="navy")
  lines(fitted(ma2.fit),col="blue")  


################################################################
# An Example from a real world series:
################################################################
# --------------------------
# Example 1:
# US-NZ Dollar Exchange Rate
# --------------------------

# Instead of using the data provide by the textbook, I downloaded
# the data from the Federal Reserve website:
# http://research.stlouisfed.org/fred2/series/EXUSNZ/downloaddata
# 
# Alternatively, one could use quantmod to stream the data 
# directly, which I will do in the next lecture.

exusnz <- read.csv("C:/Users/K/z_Teach/MIDS_AdvStat/data/EXUSNZ.csv", 
                   header=TRUE, stringsAsFactors=FALSE)

# 1. Examien the data
  str(exusnz)
  cbind(head(exusnz), tail(exusnz))
    # Observation: the last observation is missing.
  exusnz2 <- exusnz[-532,] # omit the last observation because it is missing
  str(exusnz2)
  cbind(head(exusnz2), tail(exusnz2))
  #exusnz2<-as.numeric(exusnz2[,2]) # convert to numeric if it is not already
  str(exusnz2) # check to make sure it is convert to the numeric format needed
  # Convert the data into a time series object
  nz <- ts(exusnz2[,2], start=c(1971,1), end=c(2015,4), freq=12)
    str(nz)  
    head(nz, 10)

# 2. Data Visualization
# Descriptive statistics

  length(nz)
  summary(as.numeric(nz))
  v <- cbind(length(nz), mean(as.numeric(nz)), sd((as.numeric(nz))), IQR(as.numeric(nz)))
  str(v)
  names(v)=c("N","Mean","SD","IQR")
  v
  quantile(as.numeric(nz), c(.01,.05,.1,.25,.5,.75,.9,.95,.99))


  rbind(melt(rep(1,25)),melt(rep(2,25)),melt(rep(3,25)),melt(rep(4,25)))
  chunk<-split(nz, 4)
  str(chunk)
      boxplot(nz~chunk)
  
par(mfrow=c(2,2))
plot.ts(nz, main="US/NZ Exchange Rate",
        ylab="NZD per 1 USD",
        col="blue")
hist(nz, col="gray")
acf(nz, main="ACF of US/NZ Exchange Rate")
pacf(nz, main="PACF of US/NZ Exchange Rate")
  #Observations:
  # 1. The USD/NZD exchange rate series does not appear to the realization of a MA models 
  #    at all. It is very persistent - when it trended down, it trended down for a 
  #    long period of time, and when it trended up, it trended up for a long period of time.
  #    It does not appear to "revert" or "converge" to any constant level.
  # 2. The ACF looks very much like that of a random walk with drift. Even after 2 years
  #    (remember, this is a monthly series, so each of the observation represents 
  #    a monthly exchange rate), the correlation still remains about 0.8.
  # 3. The PACF drops off very sharply even at the first lag.
  # 4. The distribution of the series is skewed to the right, but as I mentioned multipled
  #    times already, a density-type plot masks the time component of the series. In fact,
  #    the high value all happened at the begining of the series (i.e. early 70s) and the two

# 3. Estimation
  # Based on the visual displays, we know that 

  # We will "try" an MA(4) model as an "approximation" to the series
  nz<-as.numeric(nz)
  ma4.nzfit <- arima(nz, order=c(0,0,4))
  ma4.nzfit
  summary(ma4.nzfit)

  # Observations:
  # All of the MA parameters are hightly statistically significant, which is not 
  # surprising, given the fact that the series is so persistent.
  # However, I strongly doubt that the model provide a "good fit" to the data.
  # Let's check out the residuals and the forecast
  
# 4. Diagnostics using residuals
# Visualization
  head(ma4.nzfit$resid, 10)
  summary(ma4.nzfit$resid)
  par(mfrow=c(2,2))
    plot(ma4.nzfit$resid, fitted(ma4.nzfit), 
         main="Residuals vs Fitted Series", 
         ylab="Residuals", xlab="Fitted Values")  
    plot.ts(ma4.nzfit$resid, main="Residual Series", ylab="Residuals")
    acf(ma4.nzfit$resid , main="ACF of the Residual Series")
    pacf(ma4.nzfit$resid, main="PACF of the Residual Series")

    # Check for conditional heteroscedasticity
    acf(ma4.nzfit$resid^2 , main="ACF of the Residual Series")
    pacf(ma4.nzfit$resid^2, main="PACF of the Residual Series")

    # Observations: 
    # 1. The residuals clearly show a relationship with the fitted series,
    #    indicating that there are aspects of the series that are left unexplained
    # 2. The residual series' t-plot does not look like a realization of a white noise
    # 3. Both ACF and PACF show strong autocorrelations and partial autocorrelations

  # Ljung-Box test of residual dynamics (or lack thereof)
  # Reference: https://stat.ethz.ch/R-manual/R-patched/library/stats/html/box.test.html
  # These tests are sometimes applied to the residuals from an ARMA(p, q) fit,
  # in which case the references suggest a better approximation to the null-hypothesis 
  # distribution is obtained by setting fitdf = p+q, provided of course that lag > fitdf.
  Box.test(ma4.nzfit$resid, type="Ljung-Box") # Box-Pierce test
  # Observations: the null hypothesis of independence of the residual series is strongly
  #               rejected.

  # Testing the correlation of the original series
  lag1_nz = lag(nz,-1)
    str(nz)
    head(cbind(nz,lag1_nz))
  cor.test(nz[1:531],lag1_nz[2:length(nz)])
  # Observation: 

# inverse of the roots of the 

# 5. Model Performance Evaluation
  summary(ma4.nzfit$resid)
  par(mfrow=c(1,1))
  plot(nz, col="navy", 
       main="Original vs a MA4 Estimated Series with Resdiauls",
       ylab="Simulated and Estimated Values",
       ylim=c(-0.2,1.4)
  )
  leg.txt <- c("Original Series", "Estimated Series", "Residuals")
  legend("topright", legend=leg.txt, lty=1, col=c("black","blue","green"),
         bty='n', cex=1)
  lines(fitted(ma4.nzfit),col="blue")
  lines(ma4.nzfit$resid, col="green")
  # Observation: Surprisingly, the in-sample fit looks reasonable

df <- cbind(nz, fitted(ma4.nzfit), ma4.nzfit$resid)
#Observations: 

# 6. Forecast / Statistical Inference
# As we noted above, the model's underlying assumption is incorrect.
# However, let's continue with the exercise anyway.

  rm(ma4.nzfit.fcast)
  ma4.nzfit.fcast <- forecast.Arima(ma4.nzfit, h=24)
  #ma4.nzfit.fcast <- predict(ma4.nzfit, n.ahead=24)
  summary(ma4.nzfit.fcast$pred)
  plot(ma4.nzfit.fcast, main="24-Step Ahead Forecast and Original & Estimated Series",
     xlab="Simulated Time Period", ylab="Original, Estimated, and Forecasted Values",
     xlim=c())
  lines(fitted(ma4.nzfit),col="blue")  
  tail(nz,24)

# 7. Model Evaluation - An Altervative Method called Backtesting

  # Step 1: Re-estimate the model leaving out the last 10% of the observations.
  #         For this series, I leave out 48 observations, which is 4 years worth
  #         of data
  fit2 <- Arima(nz[1:(length(nz)-48)], order=c(0,0,4))
    summary(fit2)
    fitted(fit2)  
    cbind(nz[1:(length(nz)-48)], fitted(fit2), fit2$resid)
    plot(fit2$resid)

  # Step 2: Forecast
    fit2.fcast <- forecast.Arima(fit2, h=60)
    plot(fit2.fcast)
    lines(nz[484:])
    cbind(nz[485:length(nz)], fit2.fcast$mean[1:48])
    z1<-ts(nz[485:length(nz)])
    z2<-ts(fit2.fcast$mean[1:48])
    ts.plot(z1,z2, gpars = list(col = c("black", "blue")))

    library(reshape2)
    z2b<- ts(rbind(melt(nz[1:484]),melt(fit2.fcast$mean[1:48])))
    ts.plot(nz,z2b, gpars = list(col = c("black", "blue")))
   ts.plot(z2b)
cbind(nz,z2b)

length(nz)
length(z2b)

# ----------------------------------------------------------
# Example 2:
# BP-NZ Dollar Exchange Rate Series provided by the Textbook
# ----------------------------------------------------------

bpnz<-read.table("C:/Users/K/z_Teach/MIDS_AdvStat/data/pounds_nz.txt", header=T)

# 1. Examining the Data

  str(bpnz)
  #bpnz<-as.numeric(bpnz)
  summary(bpnz)
  head(bpnz, 10)

# 2. Data Visualization
  par(mfrow=c(2,2))
  plot.ts(bpnz, main="NZ/BP Exchange Rate",
          ylab="NZD per 1 British Pounds",
          col="blue")
  hist(bpnz$xrate, col="gray", main="NZ/BP Exchange Rate")
  acf(bpnz, main="ACF of NZ/BP Exchange Rate")
  pacf(bpnz, main="PACF of NZ/BP Exchange Rate")

# 3. Estimation
  # Based on the graphs, it does not appear that MA models 
  # could capture the dynamics of the data series
  # However, I would estimate a MA(5) model

  ma5.bpnzfit <- arima(bpnz, order=c(0,0,5))
    ma5.bpnzfit
    summary(ma5.bpnzfit)

  # Observations:
  # The first four MA parameters are hightly statistically significant, which is not 
  # surprising, given the fact that the series is so persistent.
  # However, let's see if the model provide a "good fit" to the data.
  # Let's check out the residuals and the forecast

# 4. Diagnostics using residuals
# Visualization
  ma5.bpnzfit$resid
  fitvalue <- fitted(ma5.bpnzfit)
  class(fitvalue)
  class(ma5.bpnzfit$resid)
  summary(ma5.bpnzfit$resid)
  head(cbind(ma5.bpnzfit$resid, fitted(ma5.bpnzfit), bpnz$xrate))
  jpeg("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec11_TSA4/images/bpnz_resid.jpg")
  par(mfrow=c(2,2))
    plot.ts(ma5.bpnzfit$resid, main="Residual Series", 
            ylab="Residuals", pch=19)
    hist(bpnz$xrate, col="gray", main="Residual Series")
    acf(ma5.bpnzfit$resid , main="ACF of the Residual Series")
    pacf(ma5.bpnzfit$resid, main="PACF of the Residual Series")
  dev.off()


  summary(ma5.bpnzfit$resid)
  hist(ma5.bpnzfit$resid)
  Box.test(ma5.bpnzfit$resid, type="Ljung-Box")

# 5. Model Performance Evaluation: 
  # Comparing Fitted and Original Series
  
  
  # Create a dafaframe for producing descriptive stats
  df <- data.frame(cbind(bpnz$xrate,fitted(ma5.bpnzfit),ma5.bpnzfit$resid ))
  class(df)  
  stargazer(df, type="text", title="Descriptive Stat", digits=1)

  #----------------------------------------------------------------
  par(mfrow=c(1,1))
  plot.ts(bpnz$xrate, col="navy", 
          main="Original vs a MA5 Estimated Series with Resdiauls",
          ylab="Original and Estimated Values",
          ylim=c(2.0,4.0), pch=1)
  par(new=T)
  plot.ts(fitted(ma5.bpnzfit),col="blue",axes=T,xlab="",ylab="",
          ylim=c(2.0,4.0)) 
  leg.txt <- c("Original Series", "Estimated Series", "Residuals")
  legend("bottomright", legend=leg.txt, lty=c(1,1,2), 
         col=c("navy","blue","green"), bty='n', cex=1)
  par(new=T)
  plot.ts(ma5.bpnzfit$resid,axes=F,xlab="",ylab="",col="green",
          ylim=c(-0.5,0.5), lty=2, pch=1, col.axis="green")
  axis(side=4, col="green")
  mtext("Residuals", side=4, line=2,col="green")
  #----------------------------------------------------------------

  # Another look at the graph (by adjusting the axes)
  par(mfrow=c(1,1))
  plot.ts(bpnz$xrate, col="navy", 
       main="Original vs a MA5 Estimated Series with Resdiauls",
       ylab="Original and Estimated Values",
       ylim=c(-1.0,4.0), pch=1)
  par(new=T)
  plot.ts(fitted(ma5.bpnzfit),col="blue",axes=T,xlab="",ylab="",
          ylim=c(-1.0,4.0)) 
  leg.txt <- c("Original Series", "Estimated Series", "Residuals")
  legend("top", legend=leg.txt, lty=1, col=c("navy","blue","green"),
         bty='n', cex=1)
  par(new=T)
  plot.ts(ma5.bpnzfit$resid,axes=F,xlab="",ylab="",col="green",
        ylim=c(-1.0,4.0), pch=1)
  #axis(side=4, col="green")
  mtext("Residuals", side=4, line=2,col="green")
  #Observation: Surprisingly, the in-sample fit looks reasonable
  #----------------------------------------------------------------

  # 6. Forecast / Statistical Inference

  # 6-Step ahead Forecast
  ma5.bpnzfit.fcast <- forecast.Arima(ma5.bpnzfit, h=6)

  str(ma5.bpnzfit.fcast)
  length(ma5.bpnzfit.fcast$mean)

  # 
  library(reshape2)
  ts(rbind(melt(bpnz$xrate),melt(ma5.bpnzfit.fcast$mean)))
  
  #----------------------------------------------------------------
  jpeg("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec11_TSA4/images/bpnz_fcst.jpg")
  plot(ma5.bpnzfit.fcast, 
       main="6-Step Ahead Forecast and Original & Estimated Series",
       xlab="Simulated Time Period", 
       ylab="Original, Estimated, and Forecasted Values",
       xlim=c(0,46),ylim=c(2.0,4.0), lty=2,lwd=1.5)
  par(new=T)
  plot.ts(fitted(ma5.bpnzfit),col="blue", 
          lty=2, lwd=2, xlab="",ylab="",xlim=c(0,46),ylim=c(2.0,4.0))
  leg.txt <- c("Original Series", "Estimated Series", "Forecast")
  legend("topleft", legend=leg.txt, lty=c(2,2,1), lwd=c(1,2,2),
         col=c("black","blue","blue"), bty='n', cex=1)
  dev.off()
  #----------------------------------------------------------------

# 7. Model Evaluation - An Altervative Method called Backtesting

  # Step 1: Re-estimate the model leaving out the last 10% of the observations.
  #         For this series, I leave out 6 observations, 
  #         which is 6 months worth of data
  fit <- Arima(bpnz$xrate[1:(length(bpnz$xrate)-6)], order=c(0,0,5))
  summary(fit)
  length(fitted(fit))
  length(fit$resid)
  cbind(bpnz$xrate[1:(length(bpnz$xrate)-6)], fitted(fit), fit$resid)

  # Plot the original and estimate series 
  jpeg("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec11_TSA4/images/bpnz_bktst_fit.jpg")
  par(mfrow=c(1,1))
  plot.ts(bpnz$xrate[1:(length(bpnz$xrate)-6)], col="navy", 
           main="Original vs a MA5 Estimated Series with Resdiauls",
           ylab="Original and Estimated Values",
           ylim=c(2.0,4.0), pch=1)
  par(new=T)
  plot.ts(fitted(fit),col="blue",axes=T,xlab="",ylab="",
        ylim=c(2.0,4.0)) 
  leg.txt <- c("Original Series", "Estimated Series", "Residuals")
  legend("top", legend=leg.txt, lty=1, col=c("navy","blue","green"),
       bty='n', cex=1)
  par(new=T)
  plot.ts(fit$resid,axes=F,xlab="",ylab="",col="green",
        ylim=c(-0.5,0.5), pch=1)
  axis(side=4, col="green")
  mtext("Residuals", side=4, line=2,col="green")
  dev.off()

  # Step 2: Out-of-Sample Forecast
  fit.fcast <- forecast.Arima(fit, h=12)
    str(fit.fcast)
    length(fit.fcast$mean)

  par(mfrow=c(1,1))
  jpeg("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec11_TSA4/images/bpnz_bktst_fit2.jpg")
  plot(fit.fcast,lty=2,
       main="Out-of-Sample Forecast",
       ylab="Original, Estimated, and Forecast Values")
  par(new=T)
  plot.ts(bpnz$xrate, col="navy",axes=F,xlim=c(1,45),ylab="", lty=1)
  leg.txt <- c("Original Series", "Forecast series")
  legend("top", legend=leg.txt, lty=1, col=c("black","blue"),
         bty='n', cex=1)
  dev.off()

  melt(fit.fcast)


################################################################
# Part 2: ARMA Models: Quantitative Analysis
################################################################

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
# In simulation, it is always important to keep in mind
# from what distribution the random numbers are drawn. ALWAYS CHECK
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

# We will simluate a series of models and then plot the first 100 simulated
# obserations

# Simulating different AR and ARMA models

# AR(1): (x_t - m) =  0.6(x_{t-1} - m) + w_t
set.seed(898)
x1 <- arima.sim(n = 100, list(ar=0.6))
# AR(1): (x_t - m) = -0.6(x_{t-1} - m) + w_t
set.seed(898)
x2 <- arima.sim(n = 100, list(ar=-0.6))
# AR(1): (x_t - m) = 0.8(x_{t-1} - m) + w_t
set.seed(898)
x3 <- arima.sim(n = 100, list(ar=0.8))
# AR(1): (x_t - m) = -0.8(x_{t-1} - m) + w_t
set.seed(898)
x4 <- arima.sim(n = 100, list(ar=-0.8))

# ARMA(1,1) (x_t - m) = 0.6(x_{t-1} - m) + w_t + 0.5w_{t-1}
set.seed(898)
x5 <- arima.sim(n = 100, list(ar=0.6, ma=0.5))
# ARMA(1,1) (x_t - m) = -0.6(x_{t-1} - m) + w_t + 0.5w_{t-1}
set.seed(898)
x6 <- arima.sim(n = 100, list(ar=-0.6, ma=0.5))
# ARMA(1,1) (x_t - m) =  0.8(x_{t-1} - m) + w_t + 0.5w_{t-1}
set.seed(898)
x7 <- arima.sim(n = 100, list(ar=0.8, ma=0.5))
# ARMA(1,1) (x_t - m) = -0.8(x_{t-1} - m) + w_t + 0.5w_{t-1}
set.seed(898)
x8 <- arima.sim(n = 100, list(ar=-0.8, ma=0.5))

# ARMA(2,1) (x_t - m) = 0.6(x_{t-1} - m) - 0.5(x_{t-2} - m) + w_t + 0.5w_{t-1}
set.seed(898)
x9 <- arima.sim(n = 100, list(ar = c(0.6, -0.5), ma = c(0.5)))
# ARMA(2,2) (x_t - m) = -0.6(x_{t-1} - m)  - 0.5(x_{t-2} - m) + w_t + 0.5w_{t-1} - 0.2w_{t-2}
set.seed(898)
x10 <- arima.sim(n = 100, list(ar = c(-0.6, -0.5), ma = c(0.5,-0.2)))

set.seed(898) # note that the second AR parameter cannot be (much) bigger
x11 <- arima.sim(n = 100, list(ar = c(0.6, 0.3), ma = c(0.5)))

set.seed(898) # note that the second AR parameter cannot be (much) bigger
x12 <- arima.sim(n = 100, list(ar = c(0.8, 0.1), ma = c(0.5)))


#-----------------------------------------------------------------
## Non-stationary models!
# ARMA(2,1) (x_t - m) = 0.8(x_{t-1} - m) + 0.2(x_{t-2} - m) + w_t + 0.5w_{t-1}
set.seed(898)
arima.sim(n = 100, list(ar = c(0.8, 0.2), ma = c(0.5)))
  #NOTE: This model is not stationary!

# ARMA(2,2) (x_t - m) = -0.8(x_{t-1} - m) + 0.2(x_{t-2} - m) + w_t + 0.5w_{t-1} - 0.2w_{t-2}
set.seed(898)
arima.sim(n = 100, list(ar = c(-0.8, 0.2), ma = c(0.5,-0.2)))
  #NOTE: This model is not stationary!

# ARMA(2,1) (x_t - m) = 0.6(x_{t-1} - m) + 0.5(x_{t-2} - m) + w_t + 0.5w_{t-1}
set.seed(898)
arima.sim(n = 100, list(ar = c(0.6, 0.5), ma = c(0.5)))
  #==> In fact, this model is not stationary either
#-----------------------------------------------------------------


# Let's visualize the simulated series
png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec11_TSA4/images/sim_AR_plot2.jpg", width=400, height=350)
par(mfrow=c(2,2))
  ts.plot(x1, main="AR(1): ar=0.6")
  ts.plot(x2, main="AR(1): ar=-0.6")
  ts.plot(x3, main="AR(1): ar=0.8")
  ts.plot(x4, main="AR(1): ar=-0.8")
dev.off()

png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec11_TSA4/images/sim_ARMA_plot1.jpg", width=400, height=350)
par(mfrow=c(2,2))
  ts.plot(x5, main="ARMA(ar=0.6, ma=0.5)")
  ts.plot(x6, main="ARMA(ar=0.6, ma=-0.5)")
  ts.plot(x7, main="ARMA(ar=0.8, ma=0.5)")
  ts.plot(x8, main="ARMA(ar=-0.8, ma=0.5)")
dev.off()

png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec11_TSA4/images/sim_ARMA_plot2.jpg", width=400, height=350)
par(mfrow=c(2,2))
  ts.plot(x9,  main="ARMA(ar=(0.6,-0.5), ma=0.5)")
  ts.plot(x10, main="ARMA(ar=(-0.6,-0.5), ma=(0.5,-0.2)")
  ts.plot(x11, main="ARMA(ar=(0.6, 0.3), ma=(0.5)")
  ts.plot(x12, main="ARMA(ar=(0.8,0.1), ma=(0.5)")
dev.off()

## Examine the ACF and PACF of the 10 simulated series 
#ACF of 4 AR(1) Models
png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec11_TSA4/images/sim_AR_ACF_plot1.jpg", width=400, height=350)
par(mfrow=c(2,2))
  acf(x1,20, main="ACF: AR(ar=0.6)")
  acf(x2,20, main="ACF: AR(ar=-0.6)")
  acf(x3,20, main="ACF: AR(ar=0.8)")
  acf(x4,20, main="ACF: AR(ar=-0.8)")
dev.off()

#ACF of 4 ARMA(1,1) Models
png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec11_TSA4/images/sim_ARMA_ACF_plot1.jpg", width=400, height=350)
par(mfrow=c(2,2))
  acf(x5, main="ACF: ARMA(ar= 0.6, ma=0.5)")
  acf(x6, main="ACF: ARMA(ar=-0.6, ma=0.5)")
  acf(x7, main="ACF: ARMA(ar= 0.8, ma=0.5)")
  acf(x8, main="ACF: ARMA(ar=-0.8, ma=0.5)")
dev.off()

#ACF of 4 other ARMA Models
png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec11_TSA4/images/sim_ARMA_ACF_plot2.jpg", width=400, height=350)
par(mfrow=c(2,2))
  acf(x9 , main="ARMA(ar=(0.6 , -0.5), ma=0.5)")
  acf(x10, main="ARMA(ar=(-0.6, -0.5), ma=(0.5,-0.2)")
  acf(x11, main="ARMA(ar=(0.6 , 0.3), ma=(0.5)")
  acf(x12, main="ARMA(ar=(0.8, 0.1), ma=(0.5)")
dev.off()

#PACF of 4 AR(1) Models
png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec11_TSA4/images/sim_AR_PACF_plot1.jpg", width=400, height=350)
par(mfrow=c(2,2))
  pacf(x1,20, main="PACF: AR(ar=0.6)")
  pacf(x2,20, main="PACF: AR(ar=-0.6)")
  pacf(x3,20, main="PACF: AR(ar=0.8)")
  pacf(x4,20, main="PACF: AR(ar=-0.8)")
dev.off()

#PACF of 4 ARMA(1,1) Models
png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec11_TSA4/images/sim_ARMA_PACF_plot1.jpg", width=400, height=350)
par(mfrow=c(2,2))
  pacf(x5, main="PACF: ARMA(ar= 0.6, ma=0.5)")
  pacf(x6, main="PACF: ARMA(ar=-0.6, ma=0.5)")
  pacf(x7, main="PACF: ARMA(ar= 0.8, ma=0.5)")
  pacf(x8, main="PACF: ARMA(ar=-0.8, ma=0.5)")
dev.off()

#PACF of 4 other ARMA Models
png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec11_TSA4/images/sim_ARMA_PACF_plot2.jpg", width=400, height=350)
par(mfrow=c(2,2))
  pacf(x9 , main="PACF: ARMA(ar=( 0.6 , -0.5), ma=0.5)")
  pacf(x10, main="PACF: ARMA(ar=(-0.6 , -0.5), ma=(0.5,-0.2)")
  pacf(x11, main="PACF: ARMA(ar=( 0.6 ,  0.3), ma=(0.5)")
  pacf(x12, main="PACF: ARMA(ar=( 0.8 ,  0.1), ma=(0.5)")
dev.off()


# One last example ...

set.seed(898)
ar1.1 <- arima.sim(n = 100, list(ar = c(0.4)))
set.seed(898)
ar1.2 <- arima.sim(n = 100, list(ar = c(0.8)))
set.seed(898)
ar1.3 <- arima.sim(n = 100, list(ar = c(0.9)))
set.seed(898)
arma21 <- arima.sim(n = 100, list(ar = c(0.6, 0.3), ma = c(0.5)))
set.seed(898)
arma22 <- arima.sim(n = 100, list(ar = c(0.6, 0.3), ma = c(0.5,0.5)))

par(mfrow=c(2,2))
plot.ts(ar1.1, ylim=c(-4,4))
plot.ts(ar1.2, ylim=c(-4,4))
plot.ts(ar1.3, ylim=c(-4,4))
plot.ts(arma21, ylim=c(-4,4))

acf(ar1.1, ylim=c(-0.2,1.0))
acf(ar1.2, ylim=c(-0.2,1.0))
acf(ar1.3, ylim=c(-0.2,1.0))
acf(arma21, ylim=c(-0.2,1.0))

pacf(ar1.1, ylim=c(-0.2,1.0))
pacf(ar1.2, ylim=c(-0.2,1.0))
pacf(ar1.3, ylim=c(-0.2,1.0))
pacf(arma21, ylim=c(-0.2,1.0))

###################################################################
# ----------------------------------------------------
# Example: British Pound - New Zealand Dollar Exchange
#          Rate Re-visit Using ARMA Models
# ----------------------------------------------------

fit$aic #ma5
fit2 <- Arima(bpnz$xrate[1:(length(bpnz$xrate)-6)], order=c(1,0,1))
fit2$aic

fit3 <- Arima(bpnz$xrate[1:(length(bpnz$xrate)-6)], order=c(2,0,1))
fit3$aic

fit4 <- Arima(bpnz$xrate[1:(length(bpnz$xrate)-6)], order=c(2,0,2))
fit4$aic

fit5 <- Arima(bpnz$xrate[1:(length(bpnz$xrate)-6)], order=c(2,0,3))
fit5$aic

fit6 <- Arima(bpnz$xrate[1:(length(bpnz$xrate)-6)], order=c(3,0,3))
fit6$aic

fit7 <- Arima(bpnz$xrate[1:(length(bpnz$xrate)-6)], order=c(1,0,1))
fit7
fit

fit8a <- Arima(bpnz$xrate, order=c(0,0,5))
fit8a

# 3. Estimation - Using a ARMA model
fit8 <- Arima(bpnz$xrate, order=c(1,0,1))
  summary(fit8)

# 4. Model Diagnostic
par(mfrow=c(2,2))
  plot(fit8$resid, main="Residual Series", col="navy")
  hist(fit8$resid, col="gray")
  acf(fit8$resid,main="ACF: Residaul Series")
  pacf(fit8$resid,main="PACF: Residaul Series")

  summary(fit8$resid)
  Box.test(fit8$resid, type="Ljung-Box")

# 5. Model Performance Evaluation: In-Sample Fit
par(mfrow=c(1,1))
  plot.ts(bpnz$xrate,col="navy",lty=2,
          main="Original vs a ARMA(1,1) Estimated Series with Resdiauls",
          ylim=c(2.0,4.0),xlim=c(0,40),ylab="Original and Estimate Values")
  par(new=T)
  plot(fitted(fit8),col="blue",axes=F,ylab="",
       ylim=c(2.0,4.0),xlim=c(0,40))
  leg.txt <- c("Original Series", "Estimated Series", "Residuals")
  legend("topleft", legend=leg.txt, lty=c(2,1,2), 
         col=c("navy","blue","green"),
         bty='n', cex=1)
  par(new=T)
  plot.ts(fit8$resid,axes=F,xlab="",ylab="",col="green",
          ylim=c(-0.5,0.5), xlim=c(0,40), pch=1, lty=2)
  axis(side=4, col="green")
  mtext("Residuals", side=4, line=2,col="green")

# 6. Forecast / Statistical Inference

  # 6-Step ahead Forecast
  fit8.fcast <- forecast.Arima(fit8, h=6)

  str(fit8.fcast)
  length(fit8.fcast$mean)
  #ts(rbind(melt(bpnz$xrate),melt(fit8.fcast$mean)))

  #----------------------------------------------------------------
  jpeg("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec11_TSA4/images/bpnz_ARMA_fcast.jpg")
  plot(fit8.fcast, 
       main="6-Step Ahead Forecast and Original & Estimated Series",
       xlab="Simulated Time Period", 
       ylab="Original, Estimated, and Forecasted Values",
       xlim=c(0,46),ylim=c(2.0,4.0), lty=2,lwd=1.5)
  par(new=T)
  plot.ts(fitted(fit8),col="blue", 
          lty=2, lwd=2, xlab="",ylab="",xlim=c(0,46),ylim=c(2.0,4.0))
  leg.txt <- c("Original Series", "Estimated Series", "Forecast")
  legend("topleft", legend=leg.txt, lty=c(2,2,1), lwd=c(1,2,2),
         col=c("black","blue","blue"), bty='n', cex=1)
  dev.off()
  #----------------------------------------------------------------

# 7. Model Evaluation - An Altervative Method called Backtesting

# Step 1: Re-estimate the model leaving out the last 10% of the observations.
#         For this series, I leave out 6 observations, 
#         which is 6 months worth of data

fit8b <- Arima(bpnz$xrate[1:(length(bpnz$xrate)-6)], order=c(1,0,1))
  summary(fit8b)
  length(fitted(fit8b))
  length(fit$resid)
  cbind(bpnz$xrate[1:(length(bpnz$xrate)-6)], fitted(fit8b), fit8b$resid)

# Plot the original and estimate series 
  jpeg("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec11_TSA4/images/bpnz_bktst_fit.jpg")
  par(mfrow=c(1,1))
  plot.ts(bpnz$xrate[1:(length(bpnz$xrate)-6)], col="navy", 
          main="Original vs a ARMA(1,1) Estimated Series with Resdiauls",
          ylab="Original and Estimated Values",
          ylim=c(2.0,4.0), xlim=c(0,45), pch=1)
  par(new=T)
  plot.ts(fitted(fit8b),col="blue",axes=T,xlab="",ylab="",
          ylim=c(2.0,4.0), xlim=c(0,45)) 


# Step 2: Out-of-Sample Forecast
fit8b.fcast <- forecast.Arima(fit8b, h=12)
  str(fit8b.fcast)
  length(fit8b.fcast$mean)

  par(mfrow=c(1,1))
  jpeg("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec11_TSA4/images/bpnz_ARMA_bktst.jpg")
  plot(fit8b.fcast,lty=2,
       main="Out-of-Sample Forecast",
       ylab="Original, Estimated, and Forecast Values")

  par(new=T)
  plot.ts(bpnz$xrate, col="navy",axes=F,xlim=c(1,45),ylab="", lty=1)

  leg.txt <- c("Original Series", "Forecast series")
  legend("top", legend=leg.txt, lty=1, col=c("black","blue"),
         bty='n', cex=1)

  dev.off()

melt(fit.fcast)
