#####################################################################
# Directory   : 
# Program Name: Instrumental_Variables.R
# Analyst     : Paul Laskowski
# Last Updated: 4/3/2015
#
# Purpose: Instrumental Variables Estimation
#####################################################################

#####################################################################
# Setup

setwd("~/Desktop/data_w251")
getwd()

#####################################################################
# Load Libraries

library(car)
library(AER)
library(sandwich)
library(ivpack) # use ivpack to get robust standard errors for IV regression
library(stargazer) # use stargazer for professional regression tables



#####################################################################
# Analysis of wage data

# In this section, we examine the dataset used by Angrist and Krueger in
# their famous paper that uses quarter of birth as an instrument to
# measure returns to education.

#Load the data and examine it
library(foreign)
Q_full = read.dta("QOB_full.dta")

# Observe that the variables have uninformative names
names(Q_full)

# Pull out key variables and rename according to instructions at
# http://economics.mit.edu/faculty/angrist/data1/data/angkru1991
Q = Q_full[, c(1,2,4,5,6,9,10,11,12,13,16,18,19,20,21,24,25,27)]
names(Q) = c("age", "ageq", "educ", "enocent", "esocent", "lwklywge", 
             "married", "midatl", "mt", "neweng", "census", "qob", 
             "race", "smsa", "soatl", "wnocent", "wsocent", "yob")

summary(Q)


# Examine year of birth variable
summary(Q$yob)
hist(Q$yob)
# Notice that some entries include century and some don't.
# we remove the century for all entries.

Q$yob2 = Q$yob %% 100
hist(Q$yob2)
# Now we have a nice smooth distribution

# Examine ageq variable - age measured at quarterly level.
# to match Angrist and Krueger, we measure ageq from 1900
# this makes the coefficients of ageq and ageq^2 different,
# thought the fitted quadratic is the same.
summary(Q$ageq)
hist(Q$ageq)
Q$ageq2= Q$ageq %% 100
summary(Q$ageq2)
hist(Q$ageq2)
# Notice that the ageq2 variable only spans two decades,
# while year of birth spans 3.
# To investigate this further, let's plot one variable against
# the other.
# If the plot command takes too long, sample a few rows of
# data as in the following:
plot(Q[sample(nrow(Q), 1000),c("yob2","ageq2")])

# To understand what's happening, I read the description of the dataset
# at http://economics.mit.edu/faculty/angrist/data1/data/angkru1991
# Data for men born in the 20's are taken from the 1970 census,
# Data for other men are taken from the 1980 census, so the men born in 
# the 20's have their age recorded 10 years earlier.


# Examine log weekly wage
summary(Q$lwklywge)
# To get a sense of scale, take the exponent of this variable
summary(exp(Q$lwklywge))
# Wages range from 10 cents to 75 thousand dollars.
hist(Q$lwklywge, breaks = 100)

# There are a few outliers on both sides of the distribution
# We can look at a few of the smallest values to get a sense of
# whether they're meaningful
head(sort(exp(Q$lwklywge)), 100)
# The lowest wage is about 10 cents an hour. We might worry if this
# is an error, but notice that the values increase fairly smoothly
# above this point.




# Let's visualize the relationship between yob and log weekly wage
by(Q$lwklywge, Q$yob2, mean)
plot( c(1920:1949), by(Q$lwklywge, Q$yob2, mean), xlab = "Year of Birth", 
      ylab = "Log Wage")
# Notive the discontinuity at 1930.  Once again, this is because data for
# men born in the 20's comes from the 1970 census, while data for other men
# comes from the 1980 census, so these two groups are not comparable.


# In accordance with Angrist and Krueger (1991), we look at one decade
# of individuals at a time.  As the authors explain, the subset of men born
# in the 1930's is especially fruitful for study, because the average wage
# for men is roughly flat through this decade, then declines for men born
# in the 40's.
Q30 = Q[Q$yob2 > 29 & Q$yob2 < 40,]
summary(Q30)
nrow(Q30)

# Lets create a graph of log wages by quarter of birth, to visually
# assess whether there's a relationship.
# To do this, we examine the quarter of birth variable, to see how
# it relates to age in quarters.
summary(Q30$qob)
head(Q30[,c("ageq2","qob")], 10)
# Starting from the lowest value of ageq2, which is 40.25, quarter of
# birth is 4, 3, 2, 1, 4 and so on.

# We can now create the graph
plot( seq(40.25,50, by = .25), by(Q30$lwklywge, Q30$ageq2, mean), xlab = "Age in Quarters", 
      ylab = "Log Wage", col = c("red", "green","blue", "black"),
      pch = c(15, 16, 17, 18))
lines( seq(40.25,50, by = .25), by(Q30$lwklywge, Q30$ageq2, mean) , lty=2, col="red")
legend("bottomright", c("4th qtr.", "3rd qtr.", "2nd qtr.", "1st qtr."), cex=1.0, bty="n",
       col=c("red", "green","blue", "black"),pch = c(15, 16, 17, 18))

# Notice the consistent pattern, with the black diamonds representing
# first quarter births consistently lower than the others.


##### Instrumental Variable Estimation
# Let's begin with the simple OLS regression with no covariates
# Since ability and other variables are omitted, we expect OLS
# to provide a biased estimate of the return to education.
ols_model0 = lm(lwklywge ~ educ, data = Q30)
plot(ols_model0)
coeftest(ols_model0, vcov = vcovHC)
# In the naive regression, each year of education is associated
# with about a 7% increase in wage.

# We want to use quarter of birth as an instrument.
# Let's confirm whether it's related to education.
# We can write down the first stage regression.
first_stage = lm(educ ~ factor(qob), data = Q30)
coeftest(first_stage, vcov = vcovHC)
# There's a significant relationship: men born in the
# 4th quarter get an average of 0.15 more years of education
# than men born in the 1st quarter.

# We could do the second stage regression directly, by putting
# the fitted values from the first stage directly into our
# linear model.
# WARNING: The standard errors produced by this regression
# are invalid.
second_stage = lm(Q30$lwklywge ~ first_stage$fitted)
summary(second_stage)
# Our estimate of returns to education has slightly increased,
# to around 10% per year.

# More commonly, we will have R do both stages for us, using
# the ivreg command.
tsls_model0 = ivreg(lwklywge ~ educ | factor(qob), data = Q30)

# To test coefficients and get robust standard errors, we use
# the robust.se command.
se_tsls0 = robust.se(tsls_model0)
se_tsls0

# Notice that we can pull our the second column to get just the numeric
# vector of standard errors.  This will be useful to pass into stargazer
# for output.
se_tsls0[,2]


##### Alternate Model Specifications
# In this section, we recreate table V from Angrist and Krueger (1991)
# We include several OLS and several TSLS specifications, including
# different sets of covariates in each one.

# OLS model with year-of-birth dummies
ols_model1 = lm(lwklywge ~ educ + factor(yob2), data = Q30)
se_ols1 = robust.se(ols_model1)[,2]

# OLS with age covariates
ols_model2 = lm(lwklywge ~ educ + factor(yob2) + ageq2 + I(ageq2**2), data = Q30)
se_ols2 = robust.se(ols_model2)[,2]

# OLS with race, city size, married, and region dummies
ols_model3 = lm(lwklywge ~ educ + factor(yob2) + race + married + smsa + neweng 
                + midatl + enocent + wnocent + soatl +esocent + wsocent
                + mt, data = Q30)
se_ols3 = robust.se(ols_model3)[,2]

# OLS with all covariates
ols_model4 = lm(lwklywge ~ educ + factor(yob2) + race + married + smsa + neweng 
                + midatl + enocent + wnocent + soatl +esocent + wsocent
                + mt + ageq2 + I(ageq2**2), data = Q30)
se_ols4 = robust.se(ols_model4)[,2]

# TSLS regression
tsls_model1 = ivreg(lwklywge ~ educ + factor(yob2) | factor(yob2) * factor(qob), data = Q30)
se_tsls1 = robust.se(tsls_model1)[,2]

# TSLS with age covariates
tsls_model2 = ivreg(lwklywge ~ educ + factor(yob2) + ageq2 + I(ageq2**2) | factor(yob2) * factor(qob) + ageq2 + I(ageq2**2), data = Q30)
se_tsls2 = robust.se(tsls_model2)[,2]

# TSLS with race, city size, married, and region dummies
tsls_model3 = ivreg(lwklywge ~ educ + factor(yob2) +  race + married + smsa + neweng 
                    + midatl + enocent + wnocent + soatl +esocent + wsocent
                    + mt| factor(yob2) * factor(qob) + race + married + smsa + neweng 
                    + midatl + enocent + wnocent + soatl +esocent + wsocent
                    + mt, data = Q30)
se_tsls3 = robust.se(tsls_model3)[,2]

# TSLS with all covariates
tsls_model4 = ivreg(lwklywge ~ educ + factor(yob2) +  race + married + smsa + neweng 
                    + midatl + enocent + wnocent + soatl +esocent + wsocent
                    + mt + ageq2 + I(ageq2**2) | factor(yob2) * factor(qob) 
                    + race + married + smsa + neweng + midatl + enocent 
                    + wnocent + soatl +esocent + wsocent + mt + ageq2 
                    + I(ageq2**2), data = Q30)
se_tsls4 = robust.se(tsls_model4)[,2]



stargazer(ols_model1, tsls_model1, ols_model2, tsls_model2, 
          ols_model3, tsls_model3, ols_model4, tsls_model4,
          se = list(se_ols1, se_tsls1, se_ols2, se_tsls2, 
                    se_ols3, se_tsls3, se_ols4, se_tsls4),
          covariate.labels=c("education", "age", "age squared", "race (1 = black)", 
                             "married", "smsa (1 = city center)"),
          dep.var.labels = "Log Weekly Wage",
          omit = c("factor(yob2)*","neweng|midatl|enocent|wnocent|soatl|esocent|wsocent|mt"), 
          out = "QOB_table.htm", df= F,
          omit.labels = c("9 year-of-birth dummies", "8 region-of-residence dummies"))


##### The Wald Estimator
# Here's how we would use a Wald estimator in this setting.
# The Wald estimator requires a binary treatment variable.
# We'll use a dummy variable for being born in the first quarter.

Q30$q1 = Q30$qob == 1

# We can specifically examine the difference in wage between these groups
wage_change = by(Q30$lwklywge, Q30$q1, mean)[2] - by(Q30$lwklywge, Q30$q1, mean)[1]
wage_change

# We can also look at the difference in education levels between the groups.
educ_change = by(Q30$educ, Q30$q1, mean)[2] - by(Q30$educ, Q30$q1, mean)[1]
educ_change

# The ratio is the Wald estimator
wage_change/educ_change

# We can get the same thing by running the instrumental variable regression
# in the usual way.
# This also makes computing standard errors much easier.
wald_model = ivreg(lwklywge ~ educ | q1, data = Q30)
robust.se(wald_model)
