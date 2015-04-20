### A demonstration of multivariate regression in R

## 1. Preparation

# use car for scatterplots
library(car)

# Load the video data
load("Videos_clean.Rdata")
summary(Videos)


# If we were exploring, rather than testing a specific
# hypothesis, we could begin by checking the correlation
# matrix
# However, do NOT do this and pick out promising variables,
# then test their significance in regression.  This creates
# inflated error rates, breaking down the machinery
# of hypothesis testing.
cor(Videos[,c(3,5:9)], use = "pairwise.complete.obs")


# Let's investigate the determinants of something any
# video creator might care about - the number of views
hist(Videos$views)

# First, we probably want to transform views to make
# it more normal.
hist(log10(Videos$views))

# Suppose our theory tells us that the rate
# (representing video quality) and the category
# are important determinants of the number of views
# We also believe that it makes sense to control for 
# age, since older videos obviously have more time to 
# accumulate views.

# Our plan is therefore to run a hierarchical regression,
# beginning with age, and then adding category and rate.
# Since we are planning to compare our models against
# each other, we should first pull out the rows that have
# no missing values across all of these variables.  Otherwise,
# we'd have to delete cases as we add each new varible, and
# we won't be able to compare models with ANOVA.
# We can get the rows that have values for all of these
# variables with:
lim_rows = complete.cases(Videos$views, Videos$age, Videos$category, Videos$rate)
lim_rows

# Pull those rows out to create a subset of the dataset
Videos_lim = Videos[lim_rows,]


# 2. Regression Analysis

# The scatterplot shows a strong relationship
scatterplot(Videos_lim$age, log10(Videos_lim$views))

# Run a simple regression
model1 = lm(log10(views) ~ age, data = Videos_lim)
summary(model1)

# use the plot command for common diagnostics
plot(model1)

# Let's add in the category variable.
# This is automatically dummy-coded
# for us.  Can see dummy variables with
contrasts(Videos$category)


model2 = lm(log10(views) ~ age + category, data = Videos_lim)
summary(model2)
plot(model2)

# it seems like the category matters for the number of views,
# especially for Entertainment, Film and Animation, Music,
# Gaming, and Sports

# check the improvement from model 1 to model 2 is statistically
# significant
anova(model1, model2)

# we could also compare the R squares for the models, or
# compare them with Akaike's information criterion
AIC(model1)
AIC(model2)
# We want a smaller AIC - that indicates a better 
# fitting or more parsimonious model

# Let's see what happens when we add the rate variable
model3 = lm(log10(views) ~  age + category + rate, data = Videos_lim)
summary(model3)
plot(model3)

# Notice that rate has a highly significant effect
# also, look at what happened to the effects of the
# categories!

# compare the model improvement with anova
anova(model2, model3)

# and check the AIC
AIC(model2)
AIC(model3)

