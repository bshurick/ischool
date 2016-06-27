#####################################################################
# Directory   : 
# Program Name: OLS_estimation.R
# Analyst     : Paul Laskowski
# Last Updated: 4/3/2015
#
# Purpose:
# OLS Estimation
#####################################################################

#####################################################################
# Setup

setwd("~/Desktop/data_w251")
getwd()

#####################################################################
# Part 1: Simulation

# Simulating data is a valuable technique that has many uses in
# statistics.  Here, we will simulate a data set, then perform a linear
# regression as a way to test our methods and highlight the difference
# between parameters and estimators.

# The key feature of a simulation is that we know exactly what
# the population is, since we create it ourselves.

# The true population model will be y = 1 + 0.5 * x + u,
# where we assume u has a normal distribution with mean 0
# and standard deviation 1.
# we'll also assume the marginal distribution of x is normal
# with mean 10 and standard deviation 5.

# setting the seed will ensure different users generate the
# same random numbers 
set.seed(898)

# generate x values
x = rnorm(100, 10, 5)
x
# generate y values
y = 1 + 0.5 * x + rnorm(100, 0, 1)

# Let's visualize the data we generated
plot(x,y)

# Next, we will switch roles and try to estimate the paramters
# of the model.
# fit a linear model
simmodel1 = lm(y ~ x)
summary(simmodel1)

# We can superimpose our fitted model over the data
abline(simmodel1)

# Our estimator for the slope coefficient is 0.519 (the true value is 0.5)
# Our estimator for the intercet is 0.775 (the true value is 1.0)

# Observe that our estimators don't equal the true parameters exactly.
# Next week, we will learn to estimate how much the estimators may vary
# from the true values.


#####################################################################
# Part 2: Analysis of GPA data

#Load the data and examine it
load("GPA1.rdata")

# see what the variables are
ls()
# alternately call:  objects()

# read the description of variables
desc

# examine a few rows of data
head(data)

#####################################################################
# Fit a univariate linear model
# An interesting question is what effect ACT score has on college GPA
# note that this variable is highly endogenous, students that score
# higher on the ACT are certainly different than students that score
# lower.  In particular, we expect that they have higher ability
# Here, we fit a linear model with just the ACT variable

# examine the ACT variable
summary(data$ACT)
hist(data$ACT, breaks = 20)
# notice that there are no missing values, the histogram shows no
# unusual spikes, and all values are within the expected range of
# 1-36

# examine the colGPA variable
summary(data$colGPA)
hist(data$colGPA, breaks = 20)
# Notice again that we have no missing values, and no unusual spikes
# in the histogram, and all values are within the expected range
# of 0-4

# let's visualize the relationship
plot(data$ACT, data$colGPA, xlab = "ACT score", ylab = "College GPA", main = "College GPA versus ACT score")

# fit the linear model
model1 = lm(colGPA ~ ACT, data = data)
summary(model1)
abline(model1)

# notice that the coefficient of ACT is .027.
# Interpretation: each additional point on the ACT is associated with
# .027 more GPA points.
# A student that scores 4 points higher would be expected to have a GPA
# just over 0.1 points higher.

# Note that we have a large sample, so the most important assumption for 
# estimation is exogeneity.  As long as we don't care about causality, assuming
# exogeneity just means we're looking for the least squares fit line, which is
# fine for our purposes here.

# when we move on to inference, we'll have to check more assumptions,
# and we'll have to look at our regression diagnostic plots,
# but for now we know that our estimates are consistent.

#####################################################################
# Fit a bivariate linear model
# Here, we recreate the regression from the lecture and from
# Woodridge chapter 3.
# we predict colGPA from both ACT and high school GPA (hsGPA)


# fit the linear model
model2 = lm(colGPA ~ ACT + hsGPA, data = data)
summary(model2)

# Notice that the coefficent for ACT has fallen by about a factor of 3,
# to .0094.  The coefficient for hsGPA is .453 -almost 1/2.  This means
# that a student with a letter grade higher average in high school but 
# the same ACT score is expected to have a half-letter grade higher 
# average in college.

# what's happening is that ACT score in the simple regression was probably
# picking up some of the effect of high school GPA, since students
# with a higher ACT score likely have a higher GPA.  This regression
# suggests that high school GPA is likely a better predictor of 
# college GPA than ACT score is.

