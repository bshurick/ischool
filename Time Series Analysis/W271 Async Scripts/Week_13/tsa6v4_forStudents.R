################################################################
# Directory         : C:/Users/K/z_Teach/MIDS_AdvStat/pgms
# Program Name      : tsa6v4.R
# Original Developer: Jeffrey Yau
# Last Updated by   : Jeffrey Yau
# Last Updated      : 5/7/2015
# -------------------------------------------------------------------
# Main Topics Covered:
#
# Variance Models:
#    ARCH and GARCH Models
#    a. Characteristics of ARCH and GARCH process
#    b. Build a ARMA/GARCH models
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

library(fGarch)
library(quantmod)

################################################################

################################################################
# Recap:
# All the models introduced so far assume a constant variance dynamics;
# in fact, both the unconditional and conditional variance is the 
# same. This is not an appropriate assumption of series with 
# time-varying variance (or so-called "volatility" in the field of
# finance).
#
# In this lecture, we will study GARCH models, which is the most 
# popular model to capture conditional variance dynamics
# 
################################################################

################################################################
# Part 1: Introdution
# --------------------------------------------------------------

#Make a new environment to store data
econData <- new.env()

#Specify what date to get the prices from
startDate = as.Date("2010-01-01") 

# U.S. / British Pound Exchange Rate
# One British Pound to U.S. Dollar, Daily Series, Not Seasonally Adjusted
getSymbols("DEXUSUK", src="FRED", env=econData, from=startDate)

# S&P 500 Daily Index
getSymbols("^GSPC", src="yahoo", env=econData, from=startDate)

# Examine US/Euro Exchange Rate
head(econData$DEXUSUK,10)
tail(econData$DEXUSUK,10)
str(econData$DEXUSUK)
dim(econData$DEXUSUK)
#plot.ts(econData$DEXUSUK)


# Examine SP500 Index
head(econData$GSPC)
str(econData$GSPC)
dim(econData$GSPC)
head(econData$GSPC[,6])
head(econData$GSPC[,1])

# Fix the window of the data (S&P 500 Index)
SP500 <- window(econData$GSPC[,6],start=startDate) 
xUSUK <- window(econData$DEXUSUK,start=startDate) 

str(SP500)
head(SP500)
tail(SP500)

head(cbind(SP500, diff(SP500)),20)
head(cbind(SP500, diff(SP500), 
           diff(SP500)[2:length(SP500)]/SP500[2:length(SP500)]), 20)

SPret = diff(SP500)[2:length(SP500)]/SP500[2:length(SP500)]

str(xUSUK)
head(xUSUK)
tail(xUSUK)
xUSUK = xUSUK[2:length(xUSUK)]

head(cbind(xUSUK, diff(xUSUK)),20)
xUSUK_diff = diff(xUSUK)
xUSUK_diff = na.omit(xUSUK_diff)
head(xUSUK_diff, 20)

n_SP  = length(SPret)
year_SP = 2010 + (1:n_SP)* (2015-2010)/n_SP
plot(year_SP,abs(SPret),main="S&P 500 Daily Absolute Return",
     xlab="year",type="l",
     cex.axis=1.5,cex.lab=1.5,cex.main=1.5,ylab="|log return|")
mod = loess( abs(SPret)~year_SP,span=.25)
lines(year_SP,predict(mod), col=gray(.7) )

n_xUSUK=length(xUSUK_diff)
year_xUSUK = 2010 + (1:n_xUSUK)* (2015-2010)/n_xUSUK
plot(year_xUSUK,abs(xUSUK_diff),main="US/BP Daily Absolute 
     Exchange Rate Difference",
     xlab="year",type="l",
     cex.axis=1.5,cex.lab=1.5,cex.main=1.5,ylab="|log return|")
mod = loess( abs(xUSUK_diff)~year_xUSUK,span=.25)
lines(year_xUSUK,predict(mod), col=gray(.7) )

################################################################
# Part 2.1: Simulation of an ARCH(1) Process
# --------------------------------------------------------------

set.seed("8988")

# parameters
n = 110
e = rnorm(n)
a = e
u = e
x = e

sig2= e^2
omega = 1
alpha = .95
phi = .8
mu = .1

for (i in 2:n)
{
  sig2[i+1] = omega + alpha * a[i]^2
  a[i] = sqrt(sig2[i])*e[i]
  u[i] = mu + phi*(u[i-1]-mu) + a[i] #AR(1)/ARCH(1)
  x[i] = mu + phi*(x[i-1]-mu) + e[i] #AR(1)/White Noise 
}

summary(u[11:n])
summary(x[11:n])

par(mfrow=c(2,2))
plot(e[11:n],type="l",xlab="t",ylab=expression(epsilon),main="(a) white noise")
plot(sqrt(sig2[11:n]),type="l",xlab="t",ylab=expression(sigma[t]),
     main="(b) Conditional S.D.")
plot(a[11:n],type="l",xlab="t",ylab="a",main="(c) ARCH")
plot(u[11:n],type="l",xlab="t",ylab="u",main="(d) AR/ARCH")

par(mfrow=c(2,2))
plot(u[11:n],type="l",xlab="t",ylab="u",main="(d) AR/ARCH")
plot(x[11:n],type="l",xlab="t",ylab="u",main="(e) AR/White Noise")
acf(u[11:n]  , main="ACF of AR(1)/ARCH(1)")
acf(u[11:n]^2, main="ACF of AR(1)/ARCH(1) Squared")

par(mfrow=c(1,1))
plot(u[11:n],type="l",xlab="Simulated Time Period",
     ylab=expression(u[t]), 
     ylim=c(-7,10),
     main="AR/ARCH vs AR/White Noise",
     col="blue",lty=2)
par(new=T)
plot(x[11:n],type="l",xlab="",ylab="", ylim=c(-7,10),
     main="", axes=F,
     col="green")
leg.txt <- c("AR/ARCH", "AR/White Noise")
legend("topleft", legend=leg.txt, lty=c(2,1), col=c("blue","green"),
       bty='n', cex=1)


################################################################
# Part 2.2: Simulation of a GARCH Process
# --------------------------------------------------------------

n = 110

set.seed("898")
e = rnorm(n)
a =e
u = e
sig2= e^2

omega = 1
alpha = 0.95
beta  = .9
phi = .8
mu = .1

for (i in 2:n)
{
  #sig2[i+1] = omega + alpha * a[i]^2
  sig2[i+1] = omega + alpha * a[i]^2 + beta*sig2[i]
  a[i] = sqrt(sig2[i])*e[i]
  u[i] = mu + phi*(u[i-1]-mu) + a[i]
}

par(mfrow=c(2,2))
plot(e[11:n],type="l",xlab="t",
     ylab=expression(epsilon),
     main="(a) White Noise")
plot(sqrt(sig2[11:n]),type="l",
     xlab="t",ylab=expression(sigma[t]),
     main="(b) Conditional S.D.")
plot(a[11:n],type="l",xlab="t",ylab="a",main="(c) GARCH(1,1)")
par(mfrow=c(1,1))
plot(u[11:n],type="l",xlab="t",ylab="u",main="(d) AR(1)/GARCH(1,1)")

par(mfrow=c(2,2))
acf(a[11:n],main="(e) GARCH(1,1)")
acf(a[11:n]^2,main="(f) GARCH(1,1) squared")
acf(u[11:n],main="(g) AR(1)/GARCH(1,1)")
acf(u[11:n]^2,main="(h) AR(1)/GARCH(1,1) squared")

summary(u)

################################################################
# Part 3: Buiding ARMA(pA,qA)/GARCH(pG,qG) Models
# --------------------------------------------------------------

# ----------------------------------------
# Example: Volatility in Climate Series
# ----------------------------------------
# Data: The Southern Hemisphere Temperature Series
#
# This examples studies the volatility in climate series (e.g. Romilly 2005)

stemp = scan(file="C:/Users/K/z_Teach/MIDS_AdvStat/data/southern_temp.txt")
  str(stemp)
  head(stemp)
stemp.ts = ts(stemp, start=1850, freq=12)
  plot(stemp.ts)

# Cool code for choosing the best ARIMA(p,d,q) model using AIC
# Use this if you think you need a first or second order difference
# Use if by typing get.best.arima(yourts, maxord=c(n,n,n)) where
# you specify the name of your time series object and the maximum order
# of p, d and q that you would like to test.
# Reference: http://www.colorado.edu/geography/class_homepages/geog_4023_s11/TS_Code.R
#
# Alternatively, one may use auto.arima{forecast}
# URL: http://www.inside-r.org/packages/cran/forecast/docs/auto.arima

get.best.arima <- function(x.ts, maxord = c(1,1,1))  # don't change any of this code
{
  best.aic <- 1e8
  n <- length(x.ts)
  for (p in 0:maxord[1]) for(d in 0:maxord[2]) for(q in 0:maxord[3]) 
  {
    fit <- arima(x.ts, order = c(p,d,q))
    fit.aic <- -2 * fit$loglik + (log(n) + 1) * length(fit$coef)
    if (fit.aic < best.aic) 
    {
      best.aic <- fit.aic
      best.fit <- fit
      best.model <- c(p,d,q) 
    }
  }
  list(best.aic, best.fit, best.model)
}

stemp.best = get.best.arima(stemp.ts, maxord = rep(2,6))
stemp.best[[3]]# ARIMA(p=1,d=1,q=2)

# Estimate a Seasonal ARIMA (SARIMA) Model
stemp.arima = arima(stemp.ts, order=c(1,1,2),
                    seas = list(order = c(2,0,1), 12))
summary(stemp.arima)
t( confint(stemp.arima) )

  # Observations: the 2nd seasonal AR component is not significantly different
  #               from zero

# Re-estimate the model leaving out the seasonal component:
# Be very careful when doing this:

stemp.arima = arima(stemp.ts, order=c(1,1,2),
                    seas = list(order = c(1,0,1),12))
summary(stemp.arima)
t( confint(stemp.arima) )

acf(stemp.arima$resid, main="")
  title("ACF of the Residuals of Estimated SARIMA(1,1,2)|seas=(1,0,1) Model")
acf((stemp.arima$resid)^2, main="")
  title("ACF of the Squared Residuals of Estimated SARIMA(1,1,2)|seas=(1,0,1) Model")

# Observation: The squared estimated residuals' ACF shows evidence 
# of time-varying volatility

# Estimate a GARCH model to the residual series

stemp.garch <- garch(stemp.res, trace=F)
t(confint(stemp.garch))
stemp.garch.res <- resid(stemp.garch)[-1]

acf(stemp.garch.res)
acf(stemp.garch.res^2)

# Observation: the correlogram of the residuals shows no obvious patterns
# or significant values. A satisfactory fit has been obtained.


################################################################






