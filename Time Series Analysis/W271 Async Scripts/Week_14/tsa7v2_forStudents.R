################################################################
# Directory   : DIRECTORY TO BE HERE
# Program Name: tsa7.R
# Analyst     : Jeffrey Yau
# Last Updated: 4/2/2015 
################################################################
# -------------------------------------------------------------------
# Main Topics Covered:
#
# 1. Spurious Correlation
# 2. Testing for Unit Roots
# 3. Cointegration
# 4. Vector Autoregression Models
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

################################################################
# Load Libraries

require(forecast)
require(tseries)
require(fGarch)   # Garch package
require(quantmod) # Financial time series package


# fBasics, fGarch, quantmod, 
# fUtilities, fUnitRoots, MTS, net, evir, urca)

################################################################

################################################################
# 1. Spurious Regressions

# --------------------------------------------------------
# Example 1: Simulated Series
# --------------------------------------------------------
set.seed(14) # this seed is chosen (by trial and error) to produce
             # a high dependence between the two spurious regression
  #Create two independent white noise series
  x <- y <- rnorm(1000)
    cor(x,y)
    head(cbind(x,y))
    mean(x-y)
    sd(x-y)
  y <- rnorm(1000)
  cor(x,y)

  # Create two independent random walks
  for (i in 2:1000) {
    x[i] <- x[i-1] + rnorm(1)
    y[i] <- y[i-1] + rnorm(1) 
  }
  cor(x,y)
    # Observation: Interestingly, even though these are two independent random walks,
    #              they have a very high negative correlation!

  head(cbind(x,y), 10)

png("C:/Users/K/z_Teach/MIDS_AdvStat/notes/sim_ind_rw1.jpg", width=400, height=350)
par(mfrow=c(2,2))
  plot.ts(x); title("Time-Series Plot of X")
  plot.ts(y); title("Time-Series Plot of Y")
  ts.plot(ts(x),ts(y)); title("Time-Series Plot of X and Y")
  plot(x,y); title("X vs Y")
dev.off()

png("C:/Users/K/z_Teach/MIDS_AdvStat/notes/sim_ind_rw2.jpg", width=400, height=350)
par(mfrow=c(2,2))
  acf(x, main=""); title("ACF of X")  
  acf(y, main=""); title("ACF of Y")
  pacf(x, main=""); title("PACF of X")
  pacf(y, main=""); title("PACF of Y")
dev.off()

# --------------------------------------------------------
  library(tseries)
  adf.test(x)
  str(x)
  summary(x)

  par(mfrow=c(2,2))
  plot.ts(x)
  hist(x)
  acf(x)
  pacf(x)
# --------------------------------------------------------


# --------------------------------------------------------
# Example 2: (Chocolate and Electricity Production series)
# --------------------------------------------------------
cbe <- read.table("C:/Users/K/z_Teach/MIDS_AdvStat/data/cbe.txt", header = TRUE)

  str(cbe) # Look at the structure of the data
           # ==> 3 variables and 396 observations
           # Unfortunately, the calendar time is not included
  # Extract the electricity production series
  elec.ts <- ts(cbe[,3], start=1958, freq=12) 
  # Extract the chocolate production series
  choc.ts <- ts(cbe[,1], start=1958, freq=12)
  length(elec.ts); length(choc.ts)
  t=c(1:length(elec.ts))


# Correlation between the Electricity and Chocolate Production Series
cor(elec.ts,choc.ts)
# Observation: The correlation between two apparent independent series is 0.81.

# Plot the Series
layout(1,1)
png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec14_TSA7/images/elec_choc_tplot1.jpg", width=400, height=350)
  ts.plot(choc.ts, elec.ts, gpars=list(xlab="year", 
                   ylab="Production Series", 
                   lty=c(2:3), pch=c(1,4),
                   col=c("blue","black")))
  title("Electricity and Chocolate Production Series")
  leg.txt <- c('choc', 'elec')
  legend('topleft', leg.txt, lty=1, col=c('blue', 'black'), bty='n', cex=.75)
dev.off()

# Aggregate the month series into an annual series
# To learn about the what aggregate() does, read the documetnation
#
## S3 method for class 'ts':
#  aggregate((x, nfrequency = 1, FUN = sum, ndeltat = 1,
#             ts.eps = getOption("ts.eps"), ...))
# Reference: http://www.inside-r.org/r-doc/stats/aggregate

# To get a feel of what it does, simply list the two series
# and their aggregate version

  cbind(choc.ts[1:12],elec.ts[1:12])
  cbind(aggregate(choc.ts), aggregate(elec.ts))
  cbind(sum(choc.ts[1:12]),sum(elec.ts[1:12]))
  tempx<-aggregate(choc.ts); tempy<-aggregate(elec.ts)
  cbind(tempx[1],tempy[1]) # which is the same as sum(choc.ts[1:12]) and sum(elec.ts[1:12])

# Plot the annual production series
png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec14_TSA7/images/elec_choc_tplot2.jpg", width=400, height=350)
  plot(as.vector(tempx), as.vector(tempy), 
       xlim=c(0,100000), ylim=c(0,160000),
       xlab="Chocolate Production", ylab="Electricity Production",
       col="blue")
  title("Annual Electricity Production vs. Annual Chocolate Production")
dev.off()
rm(tempx, tempy)
cor(tempx,tempy)

png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec14_TSA7/images/elec_choc_acf.jpg", width=400, height=350)
  par(mfrow=c(2,2))
  acf(choc.ts, 36, main=""); title("ACF of Chocolate Production Series")  
  acf(elec.ts, 36, main=""); title("ACF of Electricity Production Series")
  pacf(choc.ts, 36, main=""); title("PACF of Chocolate Production Series")
  pacf(elec.ts, 36, main=""); title("PACF of Electricity Production Series")
dev.off()

# Let's compute the correlation between the 2 series
cat("Corr(Chocolate , Electricity): ", cor(choc.ts,elec.ts))
  #==> As shown in the graphs above, the two series both have
  #    a strong seasonal component and are trending.
  #    As such, strong correlation between the two series
  #    does not imply that they are casually related. In general,
  #    correlation does not imply causation, but in the case of
  #    trending time series, one has to pay special attention
  #    to the 

# --------------------------------------------------------
# Example 3: UK pounds, Euro, and New Zealand Dollars
# --------------------------------------------------------

# First, we use the cleaned data set provided in CM2009
us_xrates <- read.table("C:/Users/K/z_Teach/MIDS_AdvStat/data/us_xrates.txt", header = TRUE)
  str(us_xrates) # check the structure of the data
                 # 1003 observations and 3 varibles
  us_xrates[1:5,]

# Extract the currency series
UK.ts <- ts(us_xrates[,1], start=2004, freq=252) 
NZ.ts <- ts(us_xrates[,2], start=2004, freq=252)
EU.ts <- ts(us_xrates[,3], start=2004, freq=252) 

length(UK.ts); length(NZ.ts); length(EU.ts)
t=c(1:length(UK.ts))

png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec14_TSA7/images/xrates_tplot1.jpg", width=400, height=350)
layout(1:1)
ts.plot(UK.ts, NZ.ts, EU.ts, 
        gpars=list(xlab="year", 
                   #axes=FALSE,
        ylab="Exchange Rate Series", 
        lty=c(1:3), pch=c(1,4),
        col=c("blue","black","Navy")))
#axis(2, ylim=c(0,1), col="blue", las=1)
#axis(4, ylim=c(1,2), col="black",las=1)
#axis(1, pretty(range(time),10))
title("British Pound, New Zealand Dollar, and Euro Currency Series")
leg.txt <- c('UK', 'NZ', 'EU')
legend('topright', leg.txt, lty=1, col=c('blue', 'black', "Navy"), bty='n', cex=.75)
dev.off()

png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec14_TSA7/images/xrates_acf.jpg", width=400, height=350)
par(mfrow=c(3,2))
  acf(UK.ts, 50, main=""); title("ACF of British Pound")  
  pacf(UK.ts, 50, main=""); title("PACF of British Pound")
  acf(NZ.ts, 50, main=""); title("ACF of New Zealand Dollar")
  pacf(NZ.ts, 50, main=""); title("PACF of New Zealand Dollar")
  acf(EU.ts, 50, main=""); title("ACF of Euro")
  pacf(EU.ts, 50, main=""); title("PACF of Euro")
dev.off()
  # The ACF and PACF suggested that all three series appear to be 
  # random walks.

# Scatterplot Matrix
png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec14_TSA7/images/xrates_splotmtx.jpg", width=400, height=350)
splom(us_xrates, 
      main = "Pairwise Scatterplot of British Pounds, NZ Dollar, and Euro")
dev.off()

library(GGally)
png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec14_TSA7/images/xrates_splotmtx2.jpg", width=400, height=350)
ggpairs(us_xrates, main="NZ")
title("Pairwise Scatterplot of British Pounds, NZ Dollar, and Euro")
dev.off()

# Correlation Matrix
cor(us_xrates)
corrgram(us_xrates, order=NULL,
         lower.panel=panel.shade,
         upper.panel=panel.pie,
         text.panel=panel.txt)
  #==> the pounds and Euro are highly correlated!

layout(1,1)
plot(density(UK.ts))
plot(density(NZ.ts))
plot(density(EU.ts))
densityplot(~UK.ts + NZ.ts + EU.ts)


# --------------------------------------------------------
# Testing for Unit Roots for the Currency Series
# --------------------------------------------------------
# Augmented Dickey-Fuller Test
adf.test(UK.ts)
adf.test(NZ.ts)
adf.test(EU.ts)
#==> All these tests fail to reject the null hypothesis that
#    each of these series is unit root.

# Phillips-Perron Unit Root Test
pp.test(UK.ts)
pp.test(NZ.ts)
pp.test(EU.ts)
#==> Likewise, none of these tests fail to reject the null hypothesis
#    that each of these series is unit root.

################################################################
# 2. Cointegration
# Test for cointegration of two series using Phillips-Ouliaris
# test

#-------------------------------------------------
# Example (using simulation) from the text pp. 217
#-------------------------------------------------

x <- y <- mu <- rep(0, 1000) # generate 3 vectors (of length 1000) of 0's
# Construct the common underlying stochastic trends which is a
# random walk
  str(x); str(y); str(mu);
  summary(x); summary(y); summary(mu)

for (i in 2:1000) mu[i] <- mu[i-1] + rnorm(1)
  str(mu); summary(mu)

# Generate the two series sharing this stochastic trend
w_x <- rnorm(1000)  # note that this and the next white noise
                    # series are not the same. Run the command 
                    # below twice and you will see the difference
w_y <- rnorm(1000)
  dump <- rnorm(100); summary(dump); rm(dump)
  dump <- rnorm(100); summary(dump); rm(dump)

x <- mu + w_x 
y <- mu + w_y
  
# Always visualize the data before running any statistical test
summary(x); summary(y); summary(mu)
head(cbind(x,y,mu,w_x, w_y), 10)

png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec14_TSA7/images/sim_coin_tplot1.jpg", width=400, height=350)
layout(1:1)
ts.plot(ts(x), ts(y), ts(mu), 
        gpars=list(xlab="t", 
                   #axes=FALSE,
                   ylab="Series Values", 
                   lty=c(1:3), pch=c(1,4),
                   col=c("blue","black","Navy")))
#axis(2, ylim=c(0,1), col="blue", las=1)
#axis(4, ylim=c(1,2), col="black",las=1)
#axis(1, pretty(range(time),10))
title("x, y, and mu")
leg.txt <- c('x', 'y', 'mu')
legend('topright', leg.txt, lty=1, col=c('blue', 'black', "Navy"), bty='n', cex=.75)
dev.off()

png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec14_TSA7/images/sim_coin_splotmtx.jpg", width=400, height=350)
ggpairs(cbind(x,y, mu))
dev.off()

# Run the Augmented Dickey-Fuller Test

adf.test(x)
adf.test(y)
adf.test(mu)

  #==> adf tests on these series cannot reject the null hypothesis
  #    that they are not stationary

po.test(cbind(x,y))

#-------------------------------------------------------------
# Example using the exchange rate series from the text pp. 218
#-------------------------------------------------------------

po.test(cbind(UK.ts, EU.ts))

#Let's investigate the cointegrated model:

# 1. Fit a linear regression model
ukeu.lm <- lm(UK.ts ~ EU.ts)
  summary(ukeu.lm)
  # Not surprisingly, the slop coefficient estimate is highly 
  # significant and the adjusted R-squared is very high.

# 2. Obtain the residuals
png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec14_TSA7/images/ukeu_resid.jpg")
ukeu.res <- resid(ukeu.lm)
  summary(ukeu.res)  
  plot(ukeu.res, xlab="t", ylab="Residuals", 
                 lty=1, pch=1, col="blue")
  title("Residuals from the Linear Regression of British Pounds on Euro ")
dev.off()

  # the residuals clearly display a presistence


png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec14_TSA7/images/ukeu_resid2..jpg")
par(mfrow=c(2,2))
  plot(ukeu.res, xlab="t", ylab="Residuals", 
       lty=1, pch=1, col="blue"); title("Residuals of Pounds on Euro")
  plot(density(ukeu.res), main="Kernel Density of Residuals")
  acf(ukeu.res, main="ACF of Residuals")
  pacf(ukeu.res, main="PACF of Residuals")
dev.off()

# 3. Build a series of AR models

ukeu.res.ar <- ar(ukeu.res)
  summary(ukeu.res.ar)
  ukeu.res.ar$order
  resid<-na.omit(ukeu.res.ar$resid)  
    summary(resid)
    length(resid)

png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec14_TSA7/images/resid_ar3.jpg")
par(mfrow=c(2,2))
  plot(resid, xlab="t", ylab="Residuals", 
       lty=1, pch=1, col="blue"); title("Residuals of AR(3) model")
  plot(density(resid), main="Kernel Density of Residuals")
  acf(resid, main="ACF of Residuals")
  pacf(resid, main="PACF of Residuals")
dev.off()
#==> These plots show that the residuals of the AR(3) model appears
#    to be white noise.



################################################################
# 3. Multivariate Time Series


# To learn more about the library, use the help() function
help(package=astsa)

# This data is included in the astra library:
# The data
# gtemp: Global (annual) mean land-ocean temperature deviations (from 1951-1980 average),
#        measured in degrees centigrade, for the years 1880-2009.
# Original Source: http://data.giss.nasa.gov/gistemp/graphs/
# Related data: Global Annual Mean Surface Air Temperature Change
# Tabulated data is available in http://data.giss.nasa.gov/gistemp/graphs_v3/Fig.A2.txt


data(gtemp)
  str(gtemp)
  head(gtemp)

# Fitting a linear trend to the Global Annual Mean Land-Ocean Temperature Deviation
# Regress gtemp on time
summary(fit_lm <- lm(gtemp ~ time(gtemp))) 
  fit_lm$coeff
  fit_lm$qr
  fit_lm$fitted.values


png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec14_TSA7/images/gtemp_fit.jpg",)
     width=400, height=350)
plot(gtemp, type="o", col="blue",
     main="",
     xlab="Year",
     ylab="Global Temperature Deviation")
  title("Global Annual Average Land-Ocean Temperature Deviation")
abline(fit_lm) # add the fitted regression line to the plot
dev.off()

################################################################
# 4. Vector Autoregressive Models 

# Findind the absolute value of the roots of a polynomial:
# 1 - 0.5x -0.02x^2
Mod(polyroot(c(1, -0.5, -0.02)))

#--------------------------------------------
# Example 4.1 Simulation of an VAR(1) Process
#--------------------------------------------

# Simulate a bivariate white noise:
library(mvtnorm)
cov.mat <- matrix(c(1, 0.8, 0.8, 1), nr=2) # Define the covariance matrix
# Simulate 1000 observations from a bivariate normal distribution
w <- rmvnorm(1000, sigma = cov.mat) 
dim(w)

png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec14_TSA7/images/binormal.jpg",)
width=400, height=350)
plot(w, col="navy",
     main="Simulated Observations from a Bivariate Normal Distribution")
dev.off()

cov(w) # Notice that the variance-covariance matrix of the bivariate
       # normal distribution is not the same theoretial var-cov
       # matrix defined above, although it is very close.

wx <- w[,1] # Create the 1st component of the bivariate white noise
wy <- w[,2] # Create the 2nd component of the bivariate white noise
# Note that these two white noise process is contemporaneously correlated
# with each other but they do not form any lead-lag relationship,
# as shown in the cross-correlationship.
ccf(wx,wy,main="Cross-Correlation Function of Bivariate White Noise") 

# Simulate a VAR(1) process
x <- y <- rep(0,1000) # create a vector of 1000 and store it with zeros
x[1] <- wx[1] # Define the initial value to be used in the recursive
              #  formular below
y[1] <- wy[1] # Define the initial value
for (i in 2:1000) {
  x[i] = 0.4*x[i-1] + 0.3*y[i-1] + wx[i]
  y[i] = 0.2*x[i-1] + 0.1*y[i-1] + wx[i]
}
plot(x)

# Examine the bivariate time series:

png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec14_TSA7/images/sim_bivar_series.jpg", 
    width=400, height=350)
  ggpairs(cbind(x,y))
dev.off()

png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec14_TSA7/images/sim_bivar_series_tplot.jpg", 
    width=400, height=350)
layout(1:1)
ts.plot(ts(x), ts(y), 
        gpars=list(xlab="Simulated Time Period",
                   ylab="Series Values",
                   #axes=FALSE,
                   lty=c(1:3), pch=c(1,4),
                   col=c("blue","black","Navy")))
#axis(2, ylim=c(0,1), col="blue", las=1)
#axis(4, ylim=c(1,2), col="black",las=1)
#axis(1, pretty(range(time),10))
title("Simulated Bivariate Time Series")
leg.txt <- c('x', 'y')
legend('topright', leg.txt, lty=1, col=c('blue', 'black'), bty='n', cex=.75)
dev.off()

png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec14_TSA7/images/sim_bivar_series_acf.jpg", 
    width=400, height=350)
  par(mfrow=c(2,2))
  acf(x); acf(y)
  pacf(x); pacf(y)
dev.off()

png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec14_TSA7/images/sim_bivar_series_ccf.jpg", 
    width=400, height=350)
#par(mfrow=c(1,1))
  ccf(x,y, main="Cross-correlation of X and Y")
dev.off()

# Estimate a VAR(1) model
xy.ar <- ar(cbind(x,y))
# Display the coefficient estimates
# Note that is is quite close to our theoretial model which we use
# to simulate the bivariate time series
xy.ar$ar[, , ]

#-------------------------------------------------------------------
# Example 4.2 Cardioviscular Mortality, Temperature, and Particulate
#-------------------------------------------------------------------
library(vars)
x = cbind(cmort, tempr, part) # This data comes with the library astsa
str(x)
time(cmort)
summary(VAR(x, p=1, type="both"))  # "both" fits constant + trend

# continued from 5.10
VARselect(x, lag.max=10, type="both")
summary(fit <- VAR(x, p=2, type="both"))

acf(resid(fit), 52)
serial.test(fit, lags.pt=12, type="PT.adjusted")

(fit.pr = predict(fit, n.ahead = 24, ci = 0.95))  # 4 weeks ahead
fanchart(fit.pr)  # plot prediction + error

(fit.pr = predict(fit, n.ahead = 48, ci = 0.95))  # 8 weeks ahead
fanchart(fit.pr)  # plot prediction + error


#-----------------------------------------------
# Example 4.3 Quarterly US Economic Time Series
#-----------------------------------------------
# The example is obtain from the textbook section 11.6.1
# on page 222.
# Use the economic series that comes with library tseries
# The dataset is called USeconomic and contains both GNP
# and M1 money supply from 1954 to 1987

data(USeconomic)
  str(USeconomic)
  head(USeconomic)
  head(cbind(GNP, M1), 10)

# Convert them to time series
GNP.ts <- ts(GNP, start=1954, freq=4) 
M1.ts  <- ts(M1, start=1954, freq=4)
head(cbind(GNP.ts, M1.ts))

# Examine the data using various graphical and tabular techniques 

summary(cbind(GNP,M1)) #Some of these statistics are meaningless
# I still look at them because I need to get the minimum and
# maximum values to define the axis tickers later


# This shows that histogram and correlation may not be effective
# tools for time series data
png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec14_TSA7/images/USecon_series_splotmtx.jpg", 
    width=400, height=350)
ggpairs(cbind(GNP.ts,M1.ts))
dev.off()

par(mfrow=c(2,2))
  ts.plot(GNP.ts, ylim=c(1000,4000), col="blue", las=1)
  title("GNP")
  ts.plot(M1.ts, ylim=c(400,600), col="navy", las=1)
  title("M1")

png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec14_TSA7/images/USecon_series_acf.jpg", 
    width=400, height=350)
par(mfrow=c(2,2))
acf(GNP.ts); acf(M1.ts)
pacf(GNP.ts); pacf(M1.ts)
dev.off()

png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec14_TSA7/images/USecon_series_ccf.jpg", 
    width=400, height=350)
par(mfrow=c(1,1))
ccf(GNP.ts, M1.ts, main="Cross-correlation between GNP and M1")
dev.off()

# Estimate the model
US.ar <- ar(cbind(GNP, M1), method="ols", dmean=T, intercept=F)
summary(US.ar) # Objects of the estimation results
US.ar$ar
  #==> it suggests that the best fitting VAR model is of order 3

# Diagnosis using the estimated residuals
dim(US.ar$res)
summary(US.ar$res)
US.ar$res #list the residual

png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec14_TSA7/images/USecon_res_splotmtx.jpg")
ggpairs(US.ar$res) # the residuals do look "fairly" normal and
                   # not correlated with each other
dev.off()

png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec14_TSA7/images/USecon_res_tplot.jpg")
ts.plot(US.ar$res[,1], US.ar$res[,2],
        gpars=list(xlab="Simulated Time Period",
                   ylab="Series Values",
                   #axes=FALSE,
                   lty=c(1:2), pch=c(1,4),
                   col=c("blue","black")))
#axis(2, ylim=c(-100,100), col="blue", las=1)
#axis(4, ylim=c(-25,25), col="black",las=1)
#axis(1, pretty(range(time),10))
title("Estimated Resduals from the VAR(3) Model")
leg.txt <- c('GNP', 'M1')
legend('topleft', leg.txt, lty=1, col=c('blue', 'black'), bty='n', cex=.75)
dev.off()

png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec14_TSA7/images/USecon_res_acf.jpg")
par(mfrow=c(2,2))
  ts.plot(US.ar$res[-c(1:3),1], col="blue", 
          main="GNP Residuals from a VAR(3) Model")
  ts.plot(US.ar$res[-c(1:3),2], col="black",
          main="M1 Residuals from a VAR(3) Model")
  acf(US.ar$res[-c(1:3),1], col="blue", main="ACF of GNP Residuals")
  acf(US.ar$res[-c(1:3),2], main="ACF of M1 Residuals")
dev.off()

# Since ar() function does not provide standard erros of the VAR 
# parameters from the ar objects, we will have to use another 
# package called vars

library(vars)
US.var <- VAR(cbind(GNP,M1), p=3, type="trend")
summary(US.var)
coef(US.var)

# 4-Step ahead forecast or 1-year forecast
US.pred <- predict(US.var, n.ahead=4)
US.pred

GNP.pred <- ts(US.pred$fcst$GNP[,1], st=1988, fr=4)
M1.pred  <- ts(US.pred$fcst$M1[,1], st=1988, fr=4)

png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec14_TSA7/images/USecon_pred.jpg")
  ts.plot(cbind(window(GNP, start=1981), GNP.pred), lty=1:2)
  ts.plot(cbind(window(M1, start=1981), M1.pred), lty=1:2)
dev.off










