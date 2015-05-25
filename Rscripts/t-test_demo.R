### T-tests to compare two means in R

# In this script, we'll practice using t-tests to
# see if two means are significantly different from
# each other


# Use the Countries dataset, including takedown variables
load("Countries3.Rdata")
summary(Countries)


## 1. Let's take another look at log gdp between the corrupt and
# trustworthy Country groups
Countries$loggdp = log10(Countries$gdp)

# The means look different between groups
by(Countries$loggdp, Countries$high_cpi, mean, na.rm = TRUE)

# But is this statistically significant?

# From the qqplot, it's not clear if loggdp is normally distributed
qqnorm(Countries$loggdp)

# The Shapiro test suggests that it's not
shapiro.test(Countries$loggdp)

# But we have a large sample size, so we can rely on 
# the central limit theorem and use a regular t.test
t.test(Countries$loggdp ~ Countries$high_cpi, Countries)


## Computing effect sizes
# We can manually compute Cohen's d, a common measure of effect
# size for the difference between two means.
# Quite simply, Cohen's d is the difference between the means
# divided by their pooled standard error.
# We'll place our code in a function so we can use it again later
cohens_d <- function(x, y) {
	# this function takes two vectors as inputs, and compares
	# their means
	
	# first, compute the pooled standard error
  lx = length(subset(x,!is.na(x)))
  ly = length(subset(y,!is.na(y)))
	# numerator of the pooled variance:
	num = (lx-1)*var(x, na.rm=T) + (ly-1)*var(y, na.rm=T)
	pooled_var = num / (lx + ly - 2) # variance
	pooled_sd = sqrt(pooled_var)
	
	# finally, compute cohen's d
	cd = abs(mean(x, na.rm=T) - mean(y, na.rm=T)) / pooled_sd
	return(cd)
}

# get the vectors of loggdp for each of our two groups
loggdp_c = Countries$loggdp[Countries$high_cpi=="Corrupt"]
loggdp_t = Countries$loggdp[Countries$high_cpi=="Trustworthy"]

# plug them into our cohen's d function
cohens_d(loggdp_c, loggdp_t)

# We could also compute the effect size correlation
# this is, quite simply, the correlation between the our metric
# variable and our grouping variable (suitably dummy-coded)
cor.test(Countries$loggdp, as.numeric(factor(Countries$high_cpi)))

## 2. Suppose we were just looking at countries in the Americas
Americas = Countries[Countries$region == "Americas",]
summary(Americas)

# We may ask whether the more corrupt countries in this 
# group issue more or less takedown requests than the 
# more trustworthy ones
by(Americas$total.takedowns, Americas$high_cpi, mean, na.rm = TRUE)

# Notice that total takedowns is not at all normal.
qqnorm(Americas$total.takedowns)

# Use the Wilcoxon rank-sum test to compare means
wilcox.test(Americas$total.takedowns ~ Americas$high_cpi)

# we can compute cohen's d using the function we wrote earlier
takedowns_c = Americas$total.takedowns[Americas$high_cpi == "Corrupt"]
takedowns_t = Americas$total.takedowns[Americas$high_cpi == "Trustworthy"]
cohens_d(takedowns_c, takedowns_t)

## 3. Let's finally compare the number of takedown requests
# issued by courts, with those issued by executives / police
mean(Countries$Court.Orders, na.rm = T)
mean(Countries$Executive, na.rm = T)

# Because there is just one group of countries, with two
# variables per country, we need a paired-samples test 
# (paired = TRUE)
#
# In general, we need a paired-sample t-test whenever
# we can pair each observation in one sample with an
# observation in the other sample, and when we expect
# the observations in each pair to vary together to
# some extent.
#
# The pairing could be formed in several ways:
#
# 1. We have two variables for each unit of analysis
# The classic example here is giving a test twice to
# the same group of individuals (pretest-posttest).
# But we could also take two different measurements at
# the same time - such as court ordered takedowns and
# executive-ordered takedowns in our example.
#
# 2. We have a natural pairing between units of analysis
# This could be the case for measurements on twins, or
# spouses.
#
# 3. We create a matched sample by pairing units of
# analysis with similar characteristics


# Because of the large sample size, we can use the parametric
# t-test
t.test(Countries$Court.Orders, Countries$Executive, paired = T)

# effect size
cohens_d(Countries$Court.Orders, Countries$Executive)
