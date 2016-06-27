#####################################################################
# Directory   : 
# Program Name: OLS_causality.R
# Analyst     : Paul Laskowski
# Last Updated: 4/3/2015
#
# Purpose:
# OLS Causality
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
# Analysis of unemployment data
# We examine the dataset used by David Card and Alan Krueger in
# their seminal paper, "Minimum Wages and Employment: A Case Study
# of the Fast-Food Industry in New Jersey and Pennsylvania."

# In April 1992, New Jersey raised its minimum wage from $4.25 to
# $5.05, while neighboring Pennsylvania kepts its minimum fixed.
# The authors argue that Pennsylvania forms a natural control 
# against which to compare the trend in New Jersey's employment at
# the time of the increase.  They survey fast-food restaurants in
# both states and record the number of workers (full-time equivalents)
# in each one, both before and after the increase.

# This is a difference-in-difference design, in which we are interested
# not in the change in employment in New Jersey, but in how that change
# compares to the analogous change in Pennsylvania.

# Read in the data.
MW = read.table("minwage.dat")

# We name our variables according to the codebook available at
# http://davidcard.berkeley.edu/data_sets.html
names(MW) = c("sheet", "chain", "co_owned", "state", "southj", 
              "centralj", "northj", "pa1", "pa2", "shore", "ncalls", 
              "empft", "emppt", "nmgrs", "wage_st", "inctime", 
              "firstinc", "bonus", "pctaff", "meal", "open", 
              "hrsopen", "psoda", "pfry", "pentree", "nregs", 
              "nregs11", "type2", "status2", "date2", "ncalls2", 
              "empft2", "emppt2", "nmgrs2", "wage_st2", "inctime2", 
              "firstin2", "special2", "meals2", "open2r", "hrsopen2", 
              "psoda2", "pfry2", "pentree2", "nregs2", "nregs112")
summary(MW)

# Examine numbers of employees
summary(MW$empft)
# Note that these have been read in as a factor. Also note the presence of
# "." values.  Take a closer look at these.
MW[MW$empft ==  ".",]
# Notice that these restaurants all had over 20 employees in the second
# time period, thus . is unlikely to mean zero, and we interpret these
# as missing values.

# Next notice that as.numeric does not correctly translate this variable
# to numbers
summary(as.numeric(MW$empft))
# There are no missing values, and we can see that the maximum is only
# 49 while there are restaurants with 50 and 60 full-time employees.
# R is extracting the factor levels instead of converting to numbers
# notice that all "."s are assigned to 1
as.numeric(MW$empft)[MW$empft=="."]
# as.numeric(as.character)) does the right thing.
summary(as.numeric(as.character(MW$empft)))

# We scan through other key variables to see if there are other suspicious
# values.
summary(MW$emppt)
summary(MW$nmmgrs)
summary(MW$empft2)
summary(MW$emppt2)
summary(MW$nmmgrs2)

# Compute total employment as half of part-time plus full-time plus managers
MW$emptot = as.numeric(as.character(MW$empft)) + 
  0.5 * as.numeric(as.character(MW$emppt)) + 
  as.numeric(as.character(MW$nmgrs))
summary(MW$emptot)
hist(MW$emptot, breaks = 50)

MW$emptot2 = as.numeric(as.character(MW$empft2)) + 
  0.5 * as.numeric(as.character(MW$emppt2)) + 
  as.numeric(as.character(MW$nmgrs2))
summary(MW$emptot2)
hist(MW$emptot2, breaks = 50)

# Compute the change in employment - the first difference.
MW$chgemp = MW$emptot2 - MW$emptot

# Investigate status2
summary(MW$status2)
summary(factor(MW$status2))
# According to the codebook, 3 represents "closed permanently"
# For this exercise, we want to treat closed
# permanently as have zero employees, but closed temporarily
# and the other categories aside from "answered second interview"
# as missing data.  The authors follow this strategy, but also
# conduct their calculations under alternate assumptions for
# comparison and compute similar estimates.

# Observe that the locations closed permanently are already coded
# as having zero employees.
MW[MW$status2 == 3,]

# And the other categories have numbers of employees listed as "."
# which we correctly interpreted as missing.
MW[MW$status2 %in% c(0,2,4,5),]

# Examine state variable, according to the codebook, 1 is New Jersey,
# 2 is Pennsylvania.  We create a dummy named newjersey for clarity
summary(MW$state)
MW$newjersey = MW$state == 1
summary(MW$newjersey)

# Let's look at the differences directly
# we could find the difference in difference directly
# using commands like
meanchg = by(MW$chgemp, MW$newjersey, mean, na.rm=T)
meanchg
means = c(meanchg, meanchg[2] - meanchg[1])
names(means) = c("Penn", "NJ", "NJ - Penn")
means

# We find that the diff-in-diff is 2.75.  Relative
# to the change in Pennsylvania, the change in employment
# in New Jersey was 2.75 greater.

# We could also go straight to a regression framework.
# An advantage is that we get standard errors without any
# extra work.
model0 = lm(chgemp ~ newjersey, data = MW)
plot(model0)
coeftest(model0, vcov = vcovHC)

# Note that the parameters correspond to those we found manually, above.
# We find that the diff-in-diff is statistically significant.

# We are missing the standard error for the mean employment in NJ,
# but we could get it by running another regression, removing the
# intercept:
model1 = lm(chgemp ~ newjersey - 1, data = MW)
coeftest(model1, vcov = vcovHC)

# We can pull a vector of standard errors out of the previous outputs
errors = c(coeftest(model1, vcov = vcovHC)[,2], coeftest(model0, vcov = vcovHC)[2,2])
errors

# place results in a matrix
results = cbind(means, errors)
results
colnames(results) = c("change in employment", "std. error")

# The final summary table
results

# An advantage of the regression framework is that we can improve our
# estimates and check robustness by including covariates.
# Here is another specification provided by Card and Krueger
model2 = lm(chgemp ~ newjersey + factor(chain) + co_owned, data = MW)
coeftest(model2, vcov = vcovHC)

# The diff-in-diff estimate, 2.78 is quite similar to before.
# Several more specifications can be compared in the original paper.

# At times, our data will appear in long format,
# This means that each resaurant will appear twice, once
# for each time period.  Here, we see what a long-format
# dataset would look like, and how to estimate a diff-in-diff
# in this setting.

# We first transform our dataset to long format using the
# reshape command.
# To simplify our output, we also pull out the just the columns we
# actually want to work with
?reshape
MW2 = reshape(MW[,c("sheet", "chain", "newjersey", "emptot", "emptot2", 
                    "empft", "empft2", "emppt", "emppt2", "nmgrs", "nmgrs2")],
              varying = list(c("emptot", "emptot2"), c("empft", "empft2"),  
                                c("emppt", "emppt2"), c("nmgrs", "nmgrs2") ),
            timevar = "time", times= c(0,1), 
            v.names = c("emptot","empft", "emppt", "nmgrs"), 
            direction = "long")
summary(MW2)

# To see what happened, we can sort by sheet to see how each restaurant
# now has two records.  Here, the sheet variable is the restaurant id.
head(MW2[order(MW2$sheet),])

# To find our diff-in-diff estimate, we include both time and newjersey
# interacted with time in our model.  Notice that the change in employment
# in Pennsylvania according to the model is the coefficient for time, while
# the change in New Jersey is the coefficient for time plus the coefficient
# for the interaction term. The interaction term coefficient is therefore 
# our estimate of the difference-in-difference.
model3 = lm(emptot ~ time + newjersey + newjersey * time, data= MW2)
coeftest(model3, vcov = vcovHC)
# Again, we find an estimate of 2.75.
