#####################################################################
# Directory         : /Users/bshur/School/Time Series Analysis/
# Program Name      : lab2.R
# Original Developer: Brandon Shurick
# Last Updated by   : Brandon Shurick
# Last Updated      : 7/12/2016
# -------------------------------------------------------------------
##################################################################### 
library(lmtest)
library(car)
library(sandwich)
library(corrplot)

#####################################################################
# 
# Load data and set args
# 
#####################################################################
reload_data <- function() {
  lab.folder <<- '/Users/bshur/School/Time Series Analysis/lab2 for students/'
  lab.file.q1 <<- 'saratoga.rdata'
  load(paste0(lab.folder,lab.file.q1), .GlobalEnv)
  lab.file.q2 <<- 'public_opinon_us_primary.csv'
  lab.data.q2 <<- read.csv(paste0(lab.folder,lab.file.q2))
  currentyear <<- as.numeric(format(Sys.Date(),"%Y"))
}


#####################################################################
# 
# Question #1
# 
#####################################################################

# Part 1 --
# Begin with a thorough exploratory data analysis. 
# For each item presented, provide a
# discussion of any observations and insights you find.
# ------
reload_data()
str(saratoga)
summary(saratoga) # reveals one NA value for Acra
saratoga <- na.omit(saratoga)
N <- nrow(saratoga)

scatterplotMatrix(~Price + ., data=saratoga)

saratoga <- within(saratoga, {
  has_fireplace <- Fireplace=='Yes'
})
cor(saratoga[,colnames(saratoga)!='Fireplace'], method='pearson')
# - One NA value, omitted
# - Living area seems to be highly correlated with baths and bedrooms,
# which both may be proxies for living area
# - Price and living area seem to have a non-linear relationship
# - Age looks to have a decreasing impact on price


# Part 2 --
# Fit a model that uses size to predict price, 
# denote this as model #1.
# ------
model1.ff <- Price ~ Living.Area
model1.lm <- lm(model1.ff, data=saratoga)
# plot(model1.lm)
c <- summary(model1.lm)


# Part 2a. --
# Is there evidence the line does not pass through the origin? 
# Answer this question using a confidence interval.
# ------
CI <- c[1,1] +             # Intercept
      c[1,2]*c(-1.96,1.96) # Confidence Interval
print(paste('Intercept confidence interval is between'
            ,round(CI[1],2),'and',round(CI[2],2)))


# Part 2b. --
# If the line passes through the origin, 
# then the slope is a proxy for the price per square foot. 
# Is there evidence the price per square foot is less than 
# $100 per square foot? 
# Answer this question using a hypothesis test.
# ------
# H_0: B_1 - 100 = 0
# H_1: B_1 - 100 < 0
tval <- (c$coefficients[2,1]-100)/c$coefficients[2,2]
pval <- pt(tval, N-1-1)
print(paste('P-value is',round(pval,6)))


# Part 2c. --
# Is there evidence the residuals do not have a Normal distribution?
# Answer this question with the appropriate 
# visualization and hypothesis test.
# ------
plot(model1.lm)
shapiro.test(model1.lm$residuals)
# null hypothesis = normally distributed
# we reject the null hypothesis (show QQ plot)


# Part 2d. --
# Is there evidence the fireplace variable is needed in the model? 
# Answer this question with the appropriate visualization 
# and numerical statistics. 
# If you find that the fireplace variable is needed in the model, 
# what condition is violated for model #1?
# ------
model1.lm.fireplace <- lm(Price ~ Living.Area + has_fireplace, data=saratoga)
ssr_ur <- sum(model1.lm.fireplace$residuals**2)
ssr_r <- sum(model1.lm$residuals**2)
q <- 1
df_ur <- N-3
fval <- ((ssr_r-ssr_ur)/q) / (ssr_ur/df_ur)
pval <- 1 - pf(fval, 1, N-3)
anova(model1.lm, model1.lm.fireplace)
print(paste('F-value of',round(fval,2),
            'is significant at p value of',round(pval,4)))
# F-Test is significant, meaning the new model has significantly
# higher explained variance


# Part 3 --
# Fit a model that uses the fireplace 
# variable to predict price, denote this as model #2.
# ------
lmodel2.lm <- lm(Price ~ has_fireplace, data=saratoga)
summary(lmodel2.lm)


# Part 3a --
# What is the baseline or reference group?
# ------
# The baseline is no fireplace and zero living area


# Part 3b --
# Is there evidence the change in the average price 
# is not zero dollars when changing
# from homes without a fireplace to homes with a 
# fireplace? Answer this question
# using a hypothesis test.
# ------


# Part 3c --
# Refer to the previous part. 
# What statistical procedure is the hypothesis test
# equivalent to? 
# Specify the corresponding competing hypotheses.
# ------
# This is equivalent to a t-test of the coefficient
# Beta_fireplace, where H_0: B_f = 0 and H_1: B_f != 0


# Part 4 --
# Fit a model that uses all of the numeric variables 
# to predict the price, denote this as model #3.
# ------
lmodel3.lm <- lm(Price ~ Living.Area + Baths + Bedrooms + Acres + Age, data=saratoga)


# Part 4a --
# Is there evidence of collinear predictors? 
# Answer this question with the appropriate 
# visualization and numerical statistics.
# ------
vif(lmodel3.lm)
# VIF is moderately high for Bedrooms, Baths, and Living Area
# None are above 5, which is generally considered to
# be highly collinear
corrplot(cor(saratoga[,colnames(saratoga)!='Fireplace']), method='ellipse')

# Part 4b --
# Is there evidence at least one of the acreage or 
# age variables are needed in the model? 
# Answer this question using a hypothesis test.
# ------
lmodel3.restricted <- lm(Price ~ Living.Area + Baths + Bedrooms, data=saratoga)
lmodel3.restricted1 <- lm(Price ~ Living.Area + Baths + Bedrooms + Age, data=saratoga)
lmodel3.restricted2 <- lm(Price ~ Living.Area + Baths + Bedrooms + Acres, data=saratoga)
anova(lmodel3.restricted, lmodel3.lm)
anova(lmodel3.restricted1, lmodel3.lm)
anova(lmodel3.restricted2, lmodel3.lm)



# Part 4c --
# Is there evidence the variation of the residuals is heteroskedastic? 
# Answer this question with the appropriate visualization and hypothesis test.
# ------
plot(lmodel3.lm)
bptest(lmodel3.lm)
# Yes, the Residuals vs Fitted plot is curved, 
# and the Breusch-Pagan test is significant with
# p-value < 2.2e-16


# Part 5 --
# Fit a model that uses the size, number of baths, 
# number of bedrooms and the fireplace
# variable to predict the price, denote this as model #4.
# ------
lmodel4.lm <- lm(Price ~ Living.Area + Baths + Bedrooms + has_fireplace, saratoga)


# Part 5a --
# Is there evidence the change in the average price is not zero dollars 
# when changing from homes without a fireplace to homes with a 
# fireplace? Answer this question using a hypothesis test.
# ------
summary(lmodel4.lm, vcov=vcovHC)
# There is not evidence that fireplace changes 
# the price, as the coefficient estimate for fireplace
# is now 5143, and the standard error is 3756, which 
# means that the 95% confidence interval contains zero.
# The t-value of this estimate is 1.369, which is a 
# p-value of 0.1712, so there is not enough evidence
# to reject the null hypothesis that the coefficent is zero.


# Part 5b --
# Refer to model #2 and part 3 (b). 
# Explain why the results are different using model #4.
# ------
coeftest(lmodel2.lm)
# Compared to model 2, the coefficient for fireplace is 
# higher in model 4, which indicates that there was 
# omitted variable bias in model 2. Adding the new
# variables to the model removes some of the positive
# bias, as those variables are positively correlated
# with fireplace and the coefficient was positive.


# Part 5c --
# From model #4, identify any outliers. 
# Explain what it means for an observation 
# to be an outlier in this context.
# ------
plot(lmodel4.lm)
outliers <- c('724','726','422','409','423')
# The residuals indicate that the current model cannot
# properly explain these outlier points.


# Part 6 --
# Fit a model that uses the size, number of baths, 
# number of bedrooms, fireplace variable,
# and an interaction between the size and 
# fireplace variable to predict the price, denote
# this as model #5.
# ------
lmodel5.lm <- lm(Price ~ Living.Area + Baths + Bedrooms + has_fireplace + has_fireplace*Living.Area, data=saratoga)


# Part 6a --
# For homes with a fireplace, what is the slope 
# between size and price.
# ------
summary(lmodel5.lm)


# Part 6b --
# Is there evidence the interaction term is needed in the model? 
# Answer this question using a hypothesis test.
# ------
lmodel5.restricted <- lm(lm(Price ~ Living.Area + Baths + Bedrooms + has_fireplace, data=saratoga))
anova(lmodel5.restricted, lmodel5.lm)
# Yes, the F-statistic is 43 with p-value=7.68e-11


# Part 6c --
# Explain what an interaction between the size and fireplace 
# variables means in the context of the problem.
# ------
# It means that the effect of fireplace on price is dependent
# on the value of living area. As living area increases, 
# the effect of fireplace increases by 40.73.



#####################################################################
# 
# Question #2
# 
#####################################################################


# Part 1 -- 
# Does Hillary Clinton rate relatively higher compared to 
# Bernie Sanders among individuals who have a higher feeling 
# thermometer rating for minority groups? 
# ------
reload_data()

# Data cleaning & manipulation
lab.data.q2 <- within(lab.data.q2, {
  female        <- (gender == 2)*1           # create female dummy variable
  presjob_lh    <- 8-presjob                 # flip rating from low->high
  econnow_lh    <- 6-econnow                 # flip rating from low->high
  age           <- currentyear - birthyr     # add age variable
  ftobama       <- ifelse(ftobama<=100,      # set NAs
                          ftobama, 
                          NA)
  ftsanders     <- ifelse(ftsanders<=100,    # set NAs
                          ftsanders, 
                          NA)  
  fthrc         <- ifelse(fthrc<=100,        # set NAs
                          fthrc, 
                          NA)  
  ftwhite       <- ifelse(ftwhite<=100,      # set NAs
                          ftwhite, 
                          NA)    
  fthisp        <- ifelse(fthisp<=100,       # set NAs
                          fthisp, 
                          NA)   
  ftblack       <- ifelse(ftblack<=100,      # set NAs
                          ftblack, 
                          NA)    
  hc_over_bs    <- fthrc - ftsanders         # calculate diff of HC->BS
  bs_over_hc    <- ftsanders - fthrc         # calculate diff of BS->HC
  ftminority    <- (ftblack+fthisp)/2        # create mean value of minorities
  minority_ntil <- quantile(ftminority       # create quantile with minority rating
                   , seq(0,1,.25),na.rm=T)
})
summary(lab.data.q2)

# model building
ff <- hc_over_bs ~ ftminority
lmodel <- lm(ff, data=lab.data.q2)
c <- coeftest(lmodel, vcov=vcovHC); c
w <- waldtest(lmodel, vcov = vcovHC); w
c[2,1]+c[2,2]*c(-1.96,1.96)

# parameter ftminority is not a significant
# predictor of rating Hillary over Sanders


# Part 2 -- 
# How does the inclusion of respondents’ perception of 
# President Obama and the economy (well known predictors 
# of presidential elections) impact your answer 
# to the first question? 
# ------
ff <- hc_over_bs ~ ftminority + ftobama + econnow_lh 
lmodel <- lm(ff, data=lab.data.q2)
coeftest(lmodel, vcov=vcovHC)
waldtest(lmodel, vcov = vcovHC)
# adding in parameters for rating of Obama 
# and the economy (to a much lesser extent)
# reduces noise in the coefficient for minority
# and now signifies that -- holding approval 
# for Obama and rating of economy constant --
# that a higher rating for minorities decreases
# the likelihood of rating Hillary over Sanders

