# Techniques for testing basic assumptions in R:
# normality and homogeneity of variance

library(ggplot2)
library(car)
library(psych)

# load the countries dataset, including 
# corruption and internet growth variables
load("Countries2.Rdata")
summary(Countries)

# use a histogram to see if the distribution of gdp looks normal
graph1 = ggplot(Countries, aes(gdp))
graph1 + geom_histogram()

describe(Countries$gdp)

# The less flexible, but quick way to make a histogram
hist(Countries$gdp)

# check normality using a qqplot
qqplot = qplot(sample = Countries$gdp, stat="qq")
qqplot

# Finally, use a Shapiro-Wilk test to see if normality is a plausible hypothesis
shapiro.test(Countries$gdp)



# Next, let's do the same thing with the log of gdp
# This is a very common transformation in econometrics
Countries$loggdp = log10(Countries$gdp)

# Begin with the Shapiro-Wilk test
shapiro.test(Countries$loggdp)

# But look at the shape of the qqplot
qqplot = qplot(sample = Countries$loggdp, stat="qq")
qqplot

# use a histogram to see if the distribution of loggdp looks normal
graph1 = ggplot(Countries, aes(loggdp))
graph1 + geom_histogram()



# Let's use our high_cpi variable to compare loggdp between 
# two groups of countries: the more corrupt and more trustworthy

# First, check the means
by(Countries$loggdp, Countries$high_cpi, mean, na.rm = TRUE)

# check if the variances are the same for both groups
by(Countries$loggdp, Countries$high_cpi, var, na.rm = TRUE)

# use a Levene test to see if equal variances is a plausible hypothesis
leveneTest(Countries$loggdp, Countries$high_cpi)