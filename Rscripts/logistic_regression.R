### A demonstration of logistic regression in R

# In this analysis, we'll examine factors that predict
# whether a country has a female head of state.

## 1. Preparation

# First load the Countries data
load("Countries3.Rdata")
summary(Countries)


# We want to add a variable denoting whether the country
# has a female head of state.
# The following list is taken from Wikipedia
f_led= c("Germany", "Liberia", "Argentina", "Bangladesh", "Lithuania", "Costa Rica", "Trinidad and Tobago", "St Maarten", "Bermuda", "Brazil", "Kosovo", "Thailand", "Denmark", "Jamaica", "Malawi", "South Korea", "San Marino", "Norway")


# notice that the match function finds the index of items in a vector
match(c("Germany"), Countries$Country)

# Look to see which countries are in our dataset
match(f_led, Countries$Country)

# get a vector of just the missing countries
f_led[is.na(match(f_led, Countries$Country))]

# Let's scan the dataframe to see if the missing
# countries are there, but simply misspelled
View(Countries)
# It seems these countries are just missing, so we'll
# ignore them for this analysis.

# Add a female_led variable to the dataset
Countries$female_led = Countries$Country %in% f_led

summary(Countries$female_led)


# Suppose we have a theory that the prevalence of 
# contraception predicts whether a country has a
# female head of state.  We also believe that gdp
# and infant mortality could have significant effects
# and we may want to control for these.
# Our plan is to run a hierarchical regression,
# beginning with just the contraception variable,
# then adding in our control variables.

# before we begin, we manually remove all the rows
# that have missing values for any of our variables.
# First get the rows that have complete values
lim_rows = complete.cases(Countries$contraception, Countries$gdp, Countries$infant.mortality)
lim_rows

# Pull out just those rows and resave as Countries_lim.
Countries_lim = Countries[lim_rows,]


## 2. Regression Analysis

# We begin with a bivariate logistic regression
model1 = glm(female_led ~ contraception, data=Countries_lim, family=binomial())
summary(model1)


# The coefficient for contraception is significant,
# but hard to interpret.  It can help to look at the
# expodent of the coefficient, which we can get as
# follows:
exp(model1$coef)
# This is the proportional increase in odds (about 3.6%)
# that results from a one-unit increase in the 
# contraception rate.

# We use analysis of deviance to see if the model
# overall is significant (better than just the mean)
anova(model1, test="Chisq")
# Looks like the improvement over the mean is significant.

# We then add controls for gdp and infant mortality.
# The idea here is to see how robust our result for
# contraception is.
model2 = glm(female_led ~ contraception + log10(gdp) + infant.mortality, data=Countries_lim, family=binomial())
summary(model2)

# The coefficient for contraception is considerably
# larger now, and highly significant.  Some of the effect
# of contraception was being suppressed by the other
# variables.

# check the model improvement with analysis of deviance
anova(model1, model2, test="Chisq")

