#####################################################################
# Directory   : 
# Program Name: OLS_inference.R
# Analyst     : Paul Laskowski
# Last Updated: 4/3/2015
#
# Purpose:
# OLS Inference
#####################################################################

#####################################################################
# Setup

setwd("~/Desktop/data_w251")
getwd()


#####################################################################
# Load Libraries

library(car)
library(lmtest)
library(sandwich)

#####################################################################
# Part 1: Simulation

# We will use today's simulation to demonstrate OLS sampling
# distributions and their properties for large samples.

# As before, let the true population model be y = 1 + 0.5 * x + u,
# To make asymptotic properties clearer, we assume u has a very
# non-normal distribution.  Suppose u is a random variable that takes
# on the value -1 with probabilty 1/2 and +1 with probability 1/2.
# We will also use the variable n to represent the sample size
n = 100

# As we did last week, assume the marginal distribution of x is normal
# with mean 10 and standard deviation 5.

set.seed(898)


# generate x values
x = rnorm(n, 10, 5)

# generate errors
u = sample(c(-1,1), n, replace = T)
u
# generate y values
y = 1 + 0.5 * x + u

# Let's visualize the data we generated
plot(x,y)
# notice the distinctive effect of the discrete error distribution

# Next, we will switch roles and try to estimate the paramters
# of the model.

# fit a linear model
simmodel1 = lm(y ~ x)
summary(simmodel1)
abline(simmodel1)

# We can also get the slope coefficient this way
simmodel1$coefficients[2]

# Our estimator for the slope coefficient is 0.4947 (the true value is 0.5)
# Our estimator for the intercept is 1.2737 (the true value is 1.0)

# Our work up to this point represents the entirety of a real-world
# study.  To visualize a sampling distribution, we have to perform
# a thought experiment and imagine performing our study a large number
# of times.  This is a great use of simulation.

# write a function to perform the procedure we followed above
# and return the slope coefficent for our estimated model.
# we'll pass in our vector of x values as a parameter
coefsim = function(x) {
  n = length(x)
  
  # generate errors
  u = sample(c(-1,1), n, replace = T)
  
  # generate y values
  y = 1 + 0.5 * x + u
  
  #fit a model and return the first coefficient.
  simmodel1 = lm(y ~ x)
  return(simmodel1$coefficients[2])
}

# Let's begin with a rather small sample size to make asymptotic properties
# clearer.  we'll make a vector of 5 x-values
n = 5
x = rnorm(n, 10, 5)
draws = replicate(1000, coefsim(x))
draws
hist(draws, breaks = 100)

sd(draws)

# the histogram represents (an approximation of) the sampling
# distribution of our statistic.  Notice 
# 1. it is centered around the true value of 0.5.  We've met
#    the first 4 Gauss Markov assumptions, so our coefficient
#    must be unbiased.
# 2. The sampling distribution is not normal.
# 3. it has a standard deviation of 0.195

# Let's repeat the process for a large sample.
# the central limit theorem tells us that the sampling distribution
# should become normal.

n = 100
x = rnorm(n, 10, 5)
draws = replicate(1000, coefsim(x))
hist(draws, breaks = 100)
sd(draws)

# Observe 
# 1. the distribution is much more normal
# 2. the standard deviation has fallen to 0.0199

#####################################################################
# Part 2: Analysis of wage data.
# We look at the wage dataset to demonstrate heteroskedasticity

load("Wage1.rdata")
ls()

# desc includes descriptions of each variable
desc

# data includes the actual observations
summary(data)
str(data)
nrow(data)

# Examine the wage variable
summary(data$wage)
hist(data$wage, breaks = 50)
# There are no missing values, and no suspicious features in the histogram

# Similarly, examine the educ and exper variables
summary(data$educ)
hist(data$educ, breaks = 50)
summary(data$exper)
hist(data$exper, breaks = 50)
# Again, we find no missing values, and no suspicious observations

# fit the linear model
model1 = lm(wage ~ educ + exper, data = data)

# we could use the summary command, but the errors are not
# robust to heteroskedasticity.
summary(model1)

# get the residual vs. fitted value and scale-location plot
plot(model1)

# Look at the residuals directly
hist(model1$residuals, breaks = 50)
# Note the positive skew

# Notice that we seem to have a violation of zero-conditional mean,
# homoskedasticity, and normality of errors.

# Because we have a large sample, we can rely on OLS asymptotics. 
# If we set aside causality and just look for the best fit line,
# exogeneity tells us that our estimates are consistent.
# We also get normality of our sampling distributions from the 
# large sample size.

# To address heteroskedasticity, we use robust standard errors.
coeftest(model1, vcov = vcovHC)

# Each year of education is associated with $0.64 more hourly
# earnings

# To test overall model significance, we use the wald test,
# which generalizes the usual F-test of overall significance,
# but allows for a heteroskedasticity-robust covariance matrix
waldtest(model1, vcov = vcovHC)

# An alternative is to use the linear Hypothesis function in the
# car package.
linearHypothesis(model1, c("educ", "exper"), vcov = vcovHC)

# Even though it's not necessary given the large sample size, 
# researchers usually enter wage into linear models in logarithmic form.
# Here, we examine this alternate specification:

model2 = lm(log(wage) ~ educ + exper, data = data)

plot(model2)
# Note how the residuals are dramatically more normal
# Also notice how the residuals vs fitted values
# and scale-location plots show much less heteroskedasticity.

# Look at the residuals directly
hist(model2$residuals, breaks = 50)
# Notice the reduced skew.

# We continue using robust standard errors, because it's good practice
coeftest(model2, vcov = vcovHC)
waldtest(model1, vcov = vcovHC)
# Each year of education is associated with about 9.7% higher
# wages
