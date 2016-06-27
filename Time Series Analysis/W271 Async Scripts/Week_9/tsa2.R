#####################################################################
# Directory         : C:/Users/K/pgms/_advstat
# Program Name      : tsa2.R
# Original Developer: Jeffrey Yau
# Last Updated by   : Jeffrey Yau
# Last Updated      : 5/5/2015
# -------------------------------------------------------------------
# Main Topics Covered:
#
# 1. Autocovariance, Autocorrelation, and Correlogram
# 2. Smoothing Techniques
#      - Symmetric Moving Average Smoothing
#      - Regression Smoothing
#      - Kernel Smoothing
#      - Smoothing Splines
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

require(astsa)    # Time series package by Shummway and Stoffer

#####################################################################


#####################################################################
# Part 1. Autocovariance, Autocorrelation, and Correlogram
#####################################################################

#-----------------------
# Example: Wave Height
#-----------------------

# Import data
wave<-read.table("C:/Users/K/z_Teach/MIDS_AdvStat/data/wavetank.txt", 
                 header = TRUE)
str(wave)
dim(wave)
head(wave)
summary(wave)

# Let's examine the series for the existance of trend and 
# seasonal pattern
par(mfrow=c(1,1))
plot(ts(wave), main="Wave Height Series Over 39.7 Seconds")
plot(ts(wave[1:60,]), main="Wave Tank Series  Over 6 Seconds")

# The autocorrelations of x are strored in teh vector acf(x)$acf
# with lag k autocorrelation located in acf(x)$acf[k+1]
par(mfrow=c(1,1))

# ACF and AVCF up to 26 lags:
acf(wave)$acf
acf(wave, type=c("covariance"))$acf

# Scatterplot Matrix of Wave Height and its Own Lags
lag.plot(wave, lags=9, layout=c(3,3), 
         diag=TRUE, disg.col="red",
         main="Autocorrelation between Wave Height and its Own Lags")

#Correlogram
acf(wave)$acf

#####################################################################
# Examine the mean, variance, autocovariance, 
# and autocorrelation of the foundational time series models

#---------------------------------------------------------
# 1. White Noise
#---------------------------------------------------------
# Simulation
set.seed(898)
sigma_w = 1
w <- rnorm(500,0,sigma_w)

par(mfrow=c(2,2))
plot(w, type="l")
title("Simulated White Noise Series", "500 Simulations")

acf(w, main="")
title("Correlogram of the Simulated White Noise Series")

#---------------------------------------------------------
# 2. A Stochastic Model with a Deterministic Linear Trend
#---------------------------------------------------------
# Simulation
set.seed(898)
sigma_w = 10
beta0 = 1
beta1 = 0.5
t = seq(1,500)
w <- rnorm(500,0,sigma_w)
x2 <- beta0 + beta1*t + w
cbind(t, x2, w)
summary(x2)
mean(x2)
sd(x2)

png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec09_TSA2/images/sim_stochastic_lin_trend.jpg")
par(mfrow=c(1,1))
plot(x2, type="l")
title("Simulated Stochatic Model with a Linear Trend", "500 Simulations")
dev.off()


acf(x2, main="")
title("Correlogram of the Simulated Stochatic Model with a Linear TrendSeries")

#---------------------------------------------------------
# 3. AR(1) Model
#---------------------------------------------------------

  #------------------------------------------------------------
  # Examine the exponential decay behavior of AR(1) correlogram
  # Theoretical ACF of an AR(1) Model
    rho <- function(k, alpha) alpha^k
    par(mfrow=c(1,1))

    png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec09_TSA2/images/ar1_theoretical_acf1.jpg")
    plot(0:10, rho(0:10,  0.7), type="b", main="rho=0.7")
    dev.off()

    png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec09_TSA2/images/ar1_theoretical_acf2.jpg")
    plot(0:10, rho(0:10, -0.7), type="b", main="rho=-0.7")
    dev.off()
  #------------------------------------------------------------

# Simulation
set.seed(898)
sigma_w = 1
alpha0 = 0
alpha1 = 0.8
x3 <- w <- rnorm(500,0,sigma_w)

for (t in 2:500) x3[t] <- alpha0 + alpha1*x3[t-1] + w[t]

mean(x3)
sd(x3)

par(mfrow=c(2,2))
plot(x3, type="b", xlab="Time Index")
title("Simulated AR(1) Model", "500 Simulations")


acf(x3, main="")
title("Correlogram of aSimulated AR(1) Model Series")


alpha1b = -0.8
x3b <- w <- rnorm(500,0,sigma_w)

for (t in 2:500) x3b[t] <- alpha0 + alpha1b*x3b[t-1] + w[t]

mean(x3b)
sd(x3b)

par(mfrow=c(2,2))
plot(x3b, type="l")
title("Simulated AR(1) Model", "500 Simulations")

acf(x3b, main="")
title("Correlogram of aSimulated AR(1) Model Series")

#---------------------------------------------------------
# 4. MA(1) Model
#---------------------------------------------------------

# Simulation
set.seed(898)
beta1 = 0.8
x4 <- w <- rnorm(500,0,sigma_w)

for (t in 2:500) x4[t] <- w[t] + beta1*w[t-1]
summary(x4)
mean(x4)
sd(x4)

par(mfrow=c(2,2))
plot(x4, type="b", xlab="Time Index")
title("Simulated MA(1) Model", "500 Simulations")

acf(x4, main="")
title("Correlogram of aSimulated MA(1) Model Series")


beta1 = -0.8
x4b <- w <- rnorm(500,0,sigma_w)

for (t in 2:500) x4b[t] <- w[t] + beta1*w[t-1]

mean(x4b)
sd(x4b)

par(mfrow=c(2,2))
plot(x4b, type="b", xlab="Time Index")
title("Simulated MA(1) Model", "500 Simulations")

acf(x4b, main="")
title("Correlogram of aSimulated MA(1) Model Series")

#---------------------------------------------------------
# 5. A Zero-Drift Random Walk Model
#---------------------------------------------------------

# -----------
# Exercise 1:
# -----------
# a) Generate a zero-drift random walk model using 500 simulation
# b) Provide the descriptive statistics of the simulated realizations.
#    The descriptive statistics should include the mean, standard
#    deviation, 25th, 50th, and 75th quantiles, minimum, and maximum
# c) Plot the time-series plot of the simulated realizations
#    Note: Make sure that the graph is well-labeled and all the 
#          axes are well-defined. A poorly-labeled graph
#          will not receive any credits
# d) Plot the autocorrelation graph
# e) Plot the partial autocorrelation graph

#---------------------------------------------------------
# 6. A Random Walk with Drift Model
#---------------------------------------------------------

# -----------
# Exercise 2:
# -----------
# a) Generate arandom walk with drift model using 500 simulation,
#    with the drift = 0.5
# b) Provide the descriptive statistics of the simulated realizations.
#    The descriptive statistics should include the mean, standard
#    deviation, 25th, 50th, and 75th quantiles, minimum, and maximum
# c) Plot the time-series plot of the simulated realizations
#    Note: Make sure that the graph is well-labeled and all the 
#          axes are well-defined. A poorly-labeled graph
#          will not receive any credits
# d) Plot the autocorrelation graph
# e) Plot the partial autocorrelation graph

#---------------------------------------------------------


#####################################################################
# Part 2: Smoothing Techniques
#####################################################################

#---------------------------------------------------------
# Example: Cardiovascular Mortality
#---------------------------------------------------------
library(astsa)

# This data set is included in the library astsa

par(mfrow=c(1,1))
plot(cmort, main="Cardiovascular Mortality (Weekly Series)", 
     pch=4, lty=5, lwd=1, xlab="Year", 
     ylab="Number of deaths per week")


# 1. Moving Average Smoothing:
ma5 = filter(cmort, sides=2, rep(1,5)/5)
ma53 = filter(cmort, sides=2, rep(1,53)/53)
plot(cmort, main="Cardiovascular Mortality (Weekly Series) and Moving Average Smoothing", 
     pch=4, lty=5, lwd=1, xlab="Year", 
     ylab="Number of deaths per week")
lines(ma5, lty=1, lwd=1.5, col="green")
lines(ma53, lty=1, lwd=1.5, col="blue")
# Add Legend
leg.txt <- c("Original Series", "5-Point Symmetric Moving Average", "53-Point Symmetric Moving Average")
legend("topright", legend=leg.txt, lty=c(1,1,1), col=c("black","green","blue"),
       bty='n', cex=1, merge = TRUE, bg=336)


# 2. Regression Smoothing
  # wk is essentially t/52 centered at zero  
head(cbind(time(cmort),mean(time(cmort))),10)

wk = time(cmort) - mean(time(cmort))  
wk2 = wk^2 
wk3 = wk^3
cs = cos(2*pi*wk)  
sn = sin(2*pi*wk)
reg1 = lm(cmort~wk + wk2 + wk3, na.action=NULL)
reg2 = lm(cmort~wk + wk2 + wk3 + cs + sn, na.action=NULL)
plot(cmort, main="Cardiovascular Mortality (Weekly Series) and Regression Smoothing", 
     pch=4, lty=5, lwd=1, xlab="Year", 
     ylab="Number of deaths per week")
lines(fitted(reg1), lty=1, lwd=1.5, col="green")
lines(fitted(reg2), lty=1, lwd=1.5, col="blue")
# Add Legend
leg.txt <- c("Original Series", "Cubic Trend Regression Smoothing", "Periodic Regression Smoothing")
legend("topright", legend=leg.txt, lty=c(1,1,1), col=c("black","green","blue"),
       bty='n', cex=1, merge = TRUE, bg=336)


# 3. Kernel Smoothing
plot(cmort, main="Cardiovascular Mortality (Weekly Series) and Kernel Smoothing", 
     pch=4, lty=5, lwd=1, xlab="Year", 
     ylab="Number of deaths per week")
lines(ksmooth(time(cmort), cmort, "normal", bandwidth=5/52),lty=1, lwd=1.5, col="green")
lines(ksmooth(time(cmort), cmort, "normal", bandwidth=2),lty=1, lwd=1.5, col="blue")
# Add Legend
leg.txt <- c("Original Series", "Kernel Smoothing: bandwidth=5/52", "Kernel Smoothing: bandwidth=2")
legend("topright", legend=leg.txt, lty=c(1,1,1), col=c("black","green","blue"),
       bty='n', cex=1, merge = TRUE, bg=336)


# 4a. Nearest Neighborhood and Lowess Smoothing
plot(cmort, main="Cardiovascular Mortality (Weekly Series) and Nearest Neighborhood Smoothing", 
     pch=4, lty=5, lwd=1, xlab="Year", 
     ylab="Number of deaths per week")
lines(supsmu(time(cmort), cmort, span=.01),lty=1, lwd=1.5, col="green")
lines(supsmu(time(cmort), cmort, span=.5),lty=1, lwd=1.5, col="blue")
# Add Legend
leg.txt <- c("Original Series", "NN Smoothing: bandwidth=.01", "NN Smoothing: bandwidth=.5")
legend("topright", legend=leg.txt, lty=c(1,1,1), col=c("black","green","blue"),
       bty='n', cex=1, merge = TRUE, bg=336)


# 4b. Lowess Smoothing
plot(cmort, main="Cardiovascular Mortality (Weekly Series) and LOWESS Smoothing", 
     pch=4, lty=5, lwd=1, xlab="Year", 
     ylab="Number of deaths per week")
lines(lowess(cmort, f=.02),lty=1, lwd=1.5, col="green")
lines(lowess(cmort, f=2/3),lty=1, lwd=1.5, col="blue")
# Add Legend
leg.txt <- c("Original Series", "LOWESS Smoothing: bandwidth=.02", "LOWESS Smoothing: bandwidth=2/3")
legend("topright", legend=leg.txt, lty=c(1,1,1), col=c("black","green","blue"),
       bty='n', cex=1, merge = TRUE, bg=336)


# 5. Smoothing Splines
plot(cmort, main="Cardiovascular Mortality (Weekly Series) and Smoothing Splines", 
     pch=4, lty=5, lwd=1, xlab="Year", 
     ylab="Number of deaths per week")
lines(smooth.spline(time(cmort), cmort, spar=0.05),lty=1, lwd=1.5, col="green")          
lines(smooth.spline(time(cmort), cmort, spar=0.9),lty=1, lwd=1.5, col="blue")  
# Add Legend
leg.txt <- c("Original Series", "Spline: Smoothing Parameter=.05", "Spline: Smoothing Parameter=0.8")
legend("topright", legend=leg.txt, lty=c(1,1,1), col=c("black","green","blue"),
       bty='n', cex=1, merge = TRUE, bg=336)

# -----------
# Exercise 3:
# -----------

# Repeat the above analysis using the initial jobless claims data
# The data is saved in INJCJC.csv file

# a) Load the data
# b) Examine the basic structure of the data
#    using str(), dim(), head(), and tail() functions
# c) Convert the variables INJCJC into a time series object
#    frequency=52, start=c(1990,1,1), end=c(2014,11,28)
# d) Examine the converted data series
#    Define a variable using the command INJCJC.time<-time(INJCJC)
#    Using the following command to examine the first 10 rows of the 
#    data. Change the parameter to examine different number of rows
#    of data
#    head(cbind(INJCJC.time, INJCJC),10)
# e1) Plot the time series plot of INJCJC. Remember that the graph
#    must be well labelled.
# e2) Plot the histogram of INJCJC. What is shown and not shown
#    in a histogram? How do you decide the number of bins used?
# e3) Plot the autocorrelation graph of INJCJC series
# e4) Plot the partial autocorrelation graph of INJCJC series
# e5) Plot a 3x3 Scatterplot Matrix of correlation against lag values
# f1) Generate two symmetric Moving Average Smoothers. Choose 
#     the number of moving average terms such that one of the smoothers
#     is very smoother and the other one can trace through the
#     dynamics of the series.
#     Plot the smoothers and the original series in one graph.
# f2) Generate two regression smoothers, one being a cubic trend
#     regression and the other being a periodic regression.
#     Plot the smoothers and the original series in one graph.
# f3) Generate kernel smoothers. Choose the smoothing parametrs 
#     such that one of the smoothers is very smoother and the
#     the other one can trace through the dynamics of the series
#     Plot the smoothers and the original series in one graph.
# f4) Generate two nearest neighborhood smoothers. Choose the smoothing parametrs 
#     such that one of the smoothers is very smoother and the
#     the other one can trace through the dynamics of the series
#     Plot the smoothers and the original series in one graph.
# f5) Generate two LOWESS smoothers. Choose the smoothing parametrs 
#     such that one of the smoothers is very smoother and the
#     the other one can trace through the dynamics of the series
#     Plot the smoothers and the original series in one graph.
# f5) Generate two spline smoothers. Choose the smoothing parametrs 
#     such that one of the smoothers is very smoother and the
#     the other one can trace through the dynamics of the series
#     Plot the smoothers and the original series in one graph.

