#####################################################################
# Directory         : C:/Users/K/pgms/_advstat
# Program Name      : tsa3.R
# Original Developer: Jeffrey Yau
# Last Updated by   : Jeffrey Yau
# Last Updated      : 5/5/2015
# -------------------------------------------------------------------
# Main Topics Covered:
#
# AR models in-depth Discussion
# 1. Simulation of a series of AR(1) and AR(2) models
# 2. Examine the empirical patterns of these models
# 3. Examine the ACF and PACF of AR-generated realizations
# 4. Model estimation
# 5. Model diagnostics using estimated residuals 
#####################################################################

#####################################################################
# Recap:
# Remember that classical regression is insufficient for
# explaining all of the interesting dynamics of a time
# series, meaning that there could be additional 
# structure of the data that is not captured.
#
# In the last two lectures, we studied how to use various graphical
# techniques to examine and identify the key patterns of time series data.
# We also learned about how to measure dependency
# structures using autocorrelation functions (i.e. correlogram),
# the notion of stationarity, and how to spot check them through
# graphs. In addition, we discussed how to simulate time series
# using the most fundamental time series models -
# linear and other deterministic trends, white noise, moving
# averges, autoregressive models, and random walk (with and without drift).
#
# In this lecture, we will study in-depth oautoregressive models.
# We will learn about identification of the order of dependency in
# AR and MA models using ACF and PACF, lag-scatterplot matrix,
# estimation, diagonsis of residuals (after the model is estimated),
# model assumption testing, model performance evaluation, and
# forecasting.
# 
# We will also start to learn about and use extensively the concepts 
# of parsimonious in building time series models and will continue 
# to develop this important principle in the next several lectures.
# 
# It is very important to keep in mind that for practical purposes,
# this class of models applies only to stationary processes. 
# Therefore, always check for stationarity before applying
# AR(p) models to the data.
#
# Package Used : stats
#           URL: https://stat.ethz.ch/R-manual/R-patched/library/stats/html/00Index.html
# Major Functions Used: ariam.sim
#                       arima
#                       acf
#                       pacf
#
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

#####################################################################
# Load Libraries

library(astsa)    # Time series package by Shummway and Stoffer
library(zoo)      # time series package
library(forecast)
#require(tseries)
#library(quantmod) # Financial time series package
#require(MASS)
#require(graphics)
#require(corrgram)

################################################################

################################################################
# Part 1: Simulate AR(p) Models for various order of dependency
#         and study their dependency properties using ACF and PACF
# --------------------------------------------------------------

set.seed(898) # Set the simulation to use the same random number sequence
# for all of the simluation to ensure that difference
# in results is not due to difference in the random
# number sequences generated

# To simulate an AR, MA, or other ARIMA-type models, we can use the function
# arima.sim() function:
#
# arima.sim(model, n, rand.gen = rnorm, innov = rand.gen(n, ...),
#          n.start = NA, start.innov = rand.gen(n.start, ...),
#          ...)
# Make sure you read the documentation of this function
# URL: http://www.inside-r.org/r-doc/stats/arima.sim
# Also read the documentation associated with the arima() function
# URL: http://www.inside-r.org/r-doc/stats/arima


# (i) Simulate 1000 data points for each of the models
# AR(1): (x_t - m) = 0.9(x_{t-1} - m) + w_t
  x1a <- arima.sim(n = 1000, list(ar=0.9))
    str(x1a)
    summary(x1a)
    par(mfrow=c(2,2))
    hist(x1a)
    plot(x1a, type="l", 
         main="1000 Simulated AR(ar=0.9) Realizations",
         ylab="Simulated Values", xlab="Simulated Time Period")
    acf(x1a, main="1000 Simulated AR(ar=0.9) Realizations")
    pacf(x1a, main="1000 Simulated AR(ar=0.9) Realizations")


# As we did in the last lecture, we can always perform the simulation manually
# In general, I strongly recommend that you know the underlying statistical
# model from which you simulate. For learning a new model, it is always
# a good idea to write codes to construct the model from scratch.

 set.seed(898)
    x <- w <- rnorm(1000)
    for (t in 2:1000) x[t] <- 0.9*x[t-1] + w[t] # a zero-mean AR(1) process
    summary(x)
    plot(x, type="l")
    acf(x)

# AR(1): (x_t - m) = 0.4(x_{t-1} - m) + w_t
  x2a <- arima.sim(n = 1000, list(ar=0.4))
# AR(1): (x_t - m) = -0.9(x_{t-1} - m) + w_t
  x1b <- arima.sim(n = 1000, list(ar=-0.9))
# AR(1): (x_t - m) = -0.4(x_{t-1} - m) + w_t 
  x2b <- arima.sim(n = 1000, list(ar=-0.4))

# (ii) Visualize the simulated data points
# Patterns to observe
# ==> The high persistence of the AR1(ar=0.9) series
#     and the much less presistence of the AR1(ar=0.4) series,
#     as expected, because of its coefficient being 0.4 instead
#     of 0.9
# ==> The jagged, earthquake-movement-like series generated
#     by the two AR(1) models with negative coefficient

# Time Plots
png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec10_TSA3/images/sim_AR1_tplot1a.jpg")
par(mfrow=c(2,2))
ts.plot(x1a, main="Simulated Series of AR1(ar=0.9)",
        ylab="Simulated Values",
        xlab="Simulated Time Period")
dev.off()

png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec10_TSA3/images/sim_AR1_ACF1a.jpeg",
    width=500, height=300)
acf(x1a, main="ACF of the Simulated Series of AR1(ar=0.9)")
dev.off()

png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec10_TSA3/images/sim_AR1_PACF1a.jpeg",
    width=500, height=300)
pacf(x1a, main="PACF of the Simulated Series of AR1(ar=0.9)")
dev.off()


png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec10_TSA3/images/sim_AR1_tplot2a.jpeg",
ts.plot(x2a, main="Simulated Series of AR1(ar=0.4)")
dev.off()

png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec10_TSA3/images/sim_AR1_ACF2a.jpeg",
acf(x2a, main="ACF of Simulated Series of AR1(ar=0.4)")
dev.off()

png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec10_TSA3/images/sim_AR1_PACF2a.jpeg",
    width=500, height=300)
pacf(x2a, main="PACF of Simulated Series of AR1(ar=0.4)")
dev.off()
acf(x1b, main="ACF of the Simulated Series of AR1(ar=-0.9)")
pacf(x1b, main="ACF of the Simulated Series of AR1(ar=-0.9)")

ts.plot(x2b, main="Simulated Series of AR1(ar=-0.4)")
acf(x2a, main="ACF of Simulated Series of AR1(ar=-0.4)")
pacf(x2a, main="PACF of Simulated Series of AR1(ar=-0.4)")


par(mfrow=c(2,2))
ts.plot(x1a, main="Simulated Series of AR1(ar=0.9)")
ts.plot(x2a, main="Simulated Series of AR1(ar=0.4)")
ts.plot(x1b, main="Simulated Series of AR1(ar=-0.9)")
ts.plot(x2b, main="Simulated Series of AR1(ar=-0.4)")

# ACF
png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec10_TSA3/images/sim_AR1_ACF_plot1,jpeg", 
    width=400, height=350)
par(mfrow=c(2,2))
  acf(x1a,20, main="ACF: AR1(ar=0.9)")
  acf(x2a,20, main="ACF: AR1(ar=0.4)")
  acf(x1b,20, main="ACF: AR1(ar=-0.9)")
  acf(x2b,20, main="ACF: AR1(ar=-0.4)")
dev.off()
# Patterns to observe
# 1. The ACF monotonically decay, slower for the more persistent series
#    and faster for the less persistent.
# 2. They do not suddenly cut off at zero 
# 3. The ACF of the AR(1) models with positive coefficients
#    show only positive autocorrelations, but the models
#    with negative coefficients show alternate positive and
#    negative correlations
# 4. The blue dotted lines represent the 95% confidence interval
#    of each of the correlation, as we covered in the lecture

# PACF
png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec10_TSA3/images/sim_AR1_PACF_plot1,jpg", 
    width=400, height=350)
par(mfrow=c(2,2))
  pacf(x1a,20, main="PACF: AR1(ar=0.9)")
  pacf(x2a,20, main="PACF: AR1(ar=0.4)")
  pacf(x1b,20, main="PACF: AR1(ar=-0.9)")
  pacf(x2b,20, main="PACF: AR1(ar=-0.4)")
dev.off()
# Patterns to observe
# 1. The PACF of teh AR(1) process cuts of abruptly

# ---------------------------------------
# Part 2:  ACF and PACF of an AR(2) Model
# ---------------------------------------

# 1. Simulate a model
set.seed(898)
x3 <- arima.sim(n = 1000, list(ar = c(1.5, -.9), ma=0))
  str(x3)
  summary(x3)
  head(x3, 10)

  # Visualization
  par(mfrow=c(2,2))
  plot.ts(x3, main="Realizations from Simulated AR(ar1=1.5,ar2=-0.9) Model")
  hist(x3, breaks="FD", col="blue",main="Simulated Series of AR(ar1=1,5, ar2=-0.9)")
  acf(x3, main="ACF of x3")
  pacf(x3, main="PACF of x3")  

  # Estimate the model just simulated
  test.fit<-arima(x3, order=c(2,0,0)) #one way to estimate the model is to use arima()
                                      #which we will use extensively in the next few lectures
  summary(test.fit)
  test.fit
  aic = -2*test.fit$loglik + 2*4           # Manaually calculate AIC
  aic

  test.fit2<-arima(x3, order=c(5,0,0))
  test.fit2
  test.fit2$coef

# Compute the roots of the characteristic polynomials
polyroot(c(1, -1.5,0.9))

par(mfrow=c(2,2))
hist(x3, breaks="FD", col="blue",main="Simulated Series of AR(ar1=1,5, ar2=-0.9)")
ts.plot(x3, main="Simulated Series of AR(ar1=1,5, ar2=-0.9)")
acf(x3, 30, main="ACF : AR(ar=1.5, ar2=-0.9)")
pacf(x3,30, main="PACF: AR(ar=1.5, ar2=-0.9)")


#acf  = ARMAacf(ar=c(1.5,-.75),ma=0,24)[-1]
#pacf = ARMAacf(ar=c(1.5,-.75),ma=0,24, pacf=TRUE)

#png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec10_TSA3/images/AR2_ACF_PACF.jpg", width=400, height=350)
#par(mfrow=c(2,2))
#plot(acf, type="h", xlab="lag", ylim=c(-.8,1)); abline(h=0)
#plot(pacf, type="h", xlab="lag", ylim=c(-.8,1)); abline(h=0)
#plot.ts(x[1:100], main="First 100 Point of ARMA(ar=(1.5,-0.75)")

#dev.off()

# -------------------------------
# 2. Estimate AR models
# -------------------------------
x3.arfit <- ar(x3, method = "mle") #AR estimates a series of AR models up to order 12 (by default)
  #summary(x3.arfit)
str(x3.arfit)
x3.arfit$order # order of the AR model with lowest AIC
x3.arfit$ar    # parameter estimate
x3.arfit$aic   # AICs of the fit models
sqrt(x3.arfit$asy.var) # asy. standard error

x3.arfit$mean
x3.arfit$var.pred

  #--------------------------
  ar(x3, method="mle", order.max=2) #an example of setting the order of the model

  x3.arfit<-auto.arima(x3) # Yet another way to use "auto" model selection based on some
                         # in-sample fit 
  summary(x3.arfit)
  str(x3.arfit)
  #--------------------------

  # Confidence Interval of the AR parameters:
x3.arfit$ar + c(-2,2)*sqrt(x3.arfit$asy.var)

# manually calculate log(AIC):
# -2*log-likelihood + 2*k (k = number of parameters estimated)
# For AR(1) model in R, it estimates the AR coefficient, the mean,
# and the variance of the series.
-2*(-1446.94) + 2*(3)

# Examine the residuals:
head(x3.arfit$resid, 15)
head(x3.arfit$resid[-c(1:4)], 15)
par(mfrow=c(2,2))
plot(rnorm(1000), type="l", main="Gaussian White Noise")
plot(x3.arfit$resid[-c(1:4)], type="l", main="Residuals: t-plot")
acf(x3.arfit$resid[-c(1:4)], main="ACF of the Residual Series")
pacf(x3.arfit$resid[-c(1:4)], main="ACF of the Residual Series")

polyroot(c(1, -1.4787, 0.9085, 0.0659, -0.1367, 0.01021 ))


par(mfrow=c(1,1))
hist(x3.arfit$resid[-c(1:5)], breaks="FD", col="blue",
     main="Residual Series", ylim=c(0,100))
qqnorm(x3.arfit$resid[-c(1:5)], main="Normal Q-Q Plot of the Residuals",
       type="p");
qqline(x3.arfit$resid[-c(1:5)], col="blue")


  #----------------------------------------------------------
  # Exercise 1
  #----------------------------------------------------------

  # Repeat all the steps using the simulated data series x
  # a: Estimate the model using the function ar()
  # b: Examine the estimation results (like the way I did in the example above)
  # c: Examine the estimated residuals (like the way I did in the example above)
  # Note: All the graphs must be well-titled and well-labeled.

  #----------------------------------------------------------

x <- w <- rnorm(1000)
x <- arima.sim(n = 1000, list(ar=c(0.7), ma=0))
  str(x)
  summary(x)
  par(mfrow=c(2,2))
  hist(x, breaks="FD", col="blue", main="Simulated AR(1) Series")
  plot(x, type="l", main="Simulated AR(1) Series")
  acf(x, main="Simulated AR(1) Series")
  pacf(x, main="Simulated AR(1) Series")


x.ar <- ar(x, method = "mle")
summary(x.ar)
x.ar$order # order of the AR model with lowest AIC
x.ar$ar # parameter estimate
sqrt(x.ar$asy.var) # asy. standard error
x.ar$aic
x.ar$mean

# Confidence Interval of the AR parameters:
x.ar$ar + c(-2,2)*sqrt(x.ar$asy.var)

# manually calculate log(AIC):
# -2*log-likelihood + 2*k (k = number of parameters estimated)
# For AR(1) model in R, it estimates the AR coefficient, the mean,
# and the variance of the series.
-2*(-1446.94) + 2*(3)

x.ar$mean
x.ar$n.used
plot(x)

# Examine the residuals:
  head(x.ar$resid)
  par(mfrow=c(2,2))
  plot(x, main="Simulated AR1 Model t-plot")
  plot(x.ar$resid[-1], type="l", main="Residuals t-plot")
  acf(x.ar$resid[-1], main="ACF of the Residual Series")
  pacf(x.ar$resid[-1], main="ACF of the Residual Series")

  par(mfrow=c(1,1))
  hist(x.ar$resid[-1], breaks="FD", col="blue",
       main="Residual Series", ylim=c(0,100))
  qqnorm(x.ar$resid[-1], main="Normal Q-Q Plot of the Residuals",
         type="p");
  qqline(x.ar$resid[-1], col="blue")


#y <- rt(200, df = 5)
#qqnorm(y); qqline(y, col = 2)
#qqplot(y, rt(300, df = 5))


head(x.ar$order.max)
plot(x.ar$aic[-1], main="AICs of a Series of AR(p) Models",
     xlab="Order of a Particular AR Model",
     ylab="Relative AIC")

# 95% CI, which concludes that the null hypothesis that
# parameter estimate is equal to 0.7 cannot be rejeccted
x.ar$ar + c(-2,2)*sqrt(x.ar$asy.var)

# Try another fitting method to see if we get the same answer
arima(x, c(1,0,0))




#----------------------------------------------------------------------------
# Exercise 2
#----------------------------------------------------------------------------

# Repeat all the steps using the NZ-US Exchange rate series in us_xrates.txt
# a: Estimate the model using the function ar()
# b: Examine the estimation results (like the way I did in the example above)
# c: Examine the estimated residuals (like the way I did in the example above)
# Note: All the graphs must be well-titled and well-labeled.

#----------------------------------------------------------------------------


#----------------------------------------------------------------------------
# Exercise 3
#----------------------------------------------------------------------------

# Repeat all the steps using the XXX data series in  XXX.txt (TBD later)
# a: Estimate the model using the function ar()
# b: Examine the estimation results (like the way I did in the example above)
# c: Examine the estimated residuals (like the way I did in the example above)
# Note: All the graphs must be well-titled and well-labeled.

#----------------------------------------------------------------------------





