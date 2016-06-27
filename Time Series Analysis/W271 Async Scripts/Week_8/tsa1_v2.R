#####################################################################
# Directory         : C:/Users/K/pgms/_advstat
# Program Name      : tsa1.R
# Original Developer: Jeffrey Yau
# Last Updated by   : Jeffrey Yau
# Last Updated      : 5/5/2015
# -------------------------------------------------------------------
# Main Topics Covered:
#
# 1. Introduction to Exploratory Time Series Data Analysis
# 2. Simulation of Some Basic Time Series Models
# 3. Time Series Decomposition
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

#install.packages("astsa") # install the library if the library is not already installed
library(astsa) # Applied Statistical Time Series Analysis package by Shumway and Stoffer
               # We will use a few datasets contained in this library
               # but we will not use the functions in this library

               

#####################################################################
# Topic 1: Exploratory Time Series Data Analysis (ETSDA)
#
# Load the data
#load("C:/Users/K/z_Teach/MIDS_AdvStat/data/tsa3.rda")

# Note that the data is already embedded in the astsa package
# Example 1: Johnson & Johnson Earnings
plot(jj,type="o", main="J&J Earnings",
     ylab="Quarterly Earnings per Share",
     xlab="Year", col="blue")
# Observe: 1. gradually trend up
#          2. regular variation around the trend
#          3. variation increase as the serie trends up


# Example 2: Global Temperature Trend and Variation
plot(gtemp,type="o", main="Annual Average Global Temperature Change",
     ylab="Temp Deviations", xlab="Year", col="blue")
# 1. Upward trend since 1970
# 2. Leveling off at around 1935 then sharp upward trend at about 1970
# Here, the question is more on 'trend'

# Example 3: NYSE Stock Returns
plot(nyse,type="o", main="NYSE Daily Returns",
     ylab="NYSE Returns")
# 1. The mean of the series appears to be stable around 0
# 2. The volatility changes over time
# ==> ARCH/GARCH model and stochastic volatility models

# Example 4: El Nino and Fish Population
par(mfrow=c(1,1))
plot(soi, ylab="", xlab="", main="Southern Oscillation Index")
plot(rec, ylab="", xlab="", main="Recruitment")
# 1. Repetitive behavior / regularly repeating cycles
# 2.
#==> Transfer function model

# Example 5: fMRI Imaging
par(mfrow=c(2,1), mar=c(3,2,1,0)+.5, mgp=c(1.6,.6,0))
ts.plot(fmri1[,2:5], lty=c(1,2,4,5),ylab="BOLD", xlab="", main="Cortex")
ts.plot(fmri1[,6:9], lty=c(1,2,4,5),ylab="BOLD", xlab="", main="Thalamus & Cerebellum")
mtext("Time (1 pt = 2sec)", side=1, line=2)


#####################################################################
# Topic 2: Simulation of Basic Time Series Models

# -------------------------------------------------------------------
# 1. Simulate a White Noise Series and a MoVing Average Series

w=rnorm(500,0,1) # Make 500 independent random draw from a standard normal distribution
png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec08_TSA1/images/white_noise_sim.jpg", width=400, height=350)
  plot.ts(w, main="Simulated White Noise", col="navy",
          ylab="Simulated values", xlab="Simulated Time Period")
dev.off()

hist(w,main="Simulated White Noise", col="blue",
     xlab="Simulated Values")

# -------------------------------------------------------------------
# 2. Simulate a MoVing Average Series
v=filter(w, sides=2, rep(1/3,3))
png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec08_TSA1/images/mv_sim.jpg", width=400, height=350)
  plot.ts(v, main="Simulated Moving Average", col="navy",
          ylab="Simulated values", xlab="Simulated Time Period")
dev.off()

hist(w,main="Simulated Moving Average", col="blue",
     xlab="Simulated Values")

# Putting the two simulated series together
par(mfrow=c(2,2))
  plot.ts(w, main="Simulated White Noise", col="navy",
          ylab="Simulated values", xlab="Simulated Time Period")
  hist(w,main="Simulated White Noise", col="blue",
       xlab="Simulated Values")
  plot.ts(v, main="Simulated Moving Average", col="navy",
          ylab="Simulated values", xlab="Simulated Time Period")
  hist(w,main="Simulated Moving Average", col="blue",
       xlab="Simulated Values")

# -------------------------------------------------------------------
# 3. Simulate a zero-mean AR(1) Series

length(w)
z <- w
# We are going to construct a simulated AR(1) model manually instead of
# using built-in function from R
for (t in 2:length(w)){
  z[t] <- 0.7*z[t-1] + w[t] # use the same random normal sequence generated above
}

png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec08_TSA1/images/AR_hist.jpg", width=400, height=350)
par(mfrow=c(1,1))
hist(z, breaks="FD",
     main="AR(ar=c(0.7))",
     xlab="Values of a Simluated Zero-Mean AR(1) Series",
     col="blue", labels=TRUE)
dev.off()

png("C:/Users/K/z_Teach/MIDS_AdvStat/_Lec08_TSA1/images/AR_sim.jpg", width=400, height=350)
plot.ts(z, main="Simulated AR(ar=c(0.7)) Series", col="navy",
        ylab="Values of the Simluated Series",
        xlab="Simulated Time Period")
dev.off()         

# -------------------------------------------------------------------
# 4. Random walk with and without drift

# Random walk with zero drift
x=cumsum(w) 

# Random walk with drift = 0.2
wd = 0.2 + w; 
xd = cumsum(wd) 

# Check out the numbers to see if they make sense
head(cbind(w,x,wd,xd),20)

# Set graphic parameters
#GRAPH_BLUE<-rgb(43/255, 71/255,153/255)
#par(bg="grey95") # Background color for the plot

par(mfrow=c(1,1))
plot.ts(xd, main="Random Walk with Drift, Random Walk without Drift, Deterministic Trend",
        col="blue", ylab="Values", xlab="Simulated Time Period", bg=38)
lines(0.2*(1:length(xd)), lty="dashed", col="navy")
lines(x, col="purple")
  # Add vertical lines
  abline(v=c(100,200,300,400),col=3,lty=3)
  # Add Legend
  leg.txt <- c("RW with Drift", "Deterministic Linear Trend", "RW without Drift")
  legend("topleft", legend=leg.txt, lty=c(1,2,1), col=c("blue","navy","purple"),
         bty='n', cex=1, merge = TRUE, bg=336)

par(mfrow=c(2,2))
plot.ts(xd, main="Random Walk with Drift, Random Walk without Drift, Deterministic Trend",
        col="blue", ylab="Values", xlab="Simulated Time Period", bg=38)
lines(0.2*(1:length(xd)), lty="dashed", col="navy")
lines(x, col="purple")
leg.txt <- c("RW with Drift", "Deterministic Linear Trend", "RW without Drift")
legend("topleft", legend=leg.txt, lty=c(1,2,1), col=c("blue","navy","purple"),
       bty='n', cex=1, merge = TRUE, bg=336)

hist(xd, main="RW with Drift", col="blue")
hist(0.2*(1:length(xd)), main="Deterministic Linear Trend", col="navy")
hist(x, main="RW without Drift", col="purple")

# -------------------------------------------------------------------


#####################################################################
# Topic 3: Time Series Decomposition

# Import data
CBE<-read.table("C:/Users/K/z_Teach/MIDS_AdvStat/data/cbe.txt", header = TRUE)
  edit(CBE)

# Check the data
str(CBE)
dim(CBE)
head(CBE, 10)
summary(CBE)

# Create time series object for the three variables in the data set
# Convert numeric vectors into R time series objects
Elec.ts <- ts(CBE[,3], start=1958, freq=12)
Beer.ts <- ts(CBE[,2], start=1958, freq=12)
Choc.ts <- ts(CBE[,1], start=1958, freq=12)

# Examine the Series Before Performing Any Other Operations
par(mfrow=c(1,1))
plot(cbind(Elec.ts, Beer.ts, Choc.ts), col="navy",
     main="Australian Chocolate, Beer, and Electricity Production
     (Jan 1958 - Dec 1990)", xlab="Year")

# Decomposition of the Electricity Production Series
plot(decompose(Elec.ts, type= "additive", filter=NULL))
plot(decompose(Elec.ts, type= "multiplicative", filter=NULL))

# Decomposition of the Beer Production Series
plot(decompose(Beer.ts, type= "additive", filter=NULL))
plot(decompose(Beer.ts, type= "multiplicative", filter=NULL))

# Decomposition of the Chocolate Production Series
plot(decompose(Choc.ts, type= "additive", filter=NULL))
plot(decompose(Choc.ts, type= "multiplicative", filter=NULL))





