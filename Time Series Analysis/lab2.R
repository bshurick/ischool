#####################################################################
# Directory         : /Users/bshur/School/Time Series Analysis/
# Program Name      : lab2.R
# Original Developer: Brandon Shurick
# Last Updated by   : Brandon Shurick
# Last Updated      : 7/8/2016
# -------------------------------------------------------------------
##################################################################### 
library(lmtest)
library(car)
library(sandwich)


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
reload_data()
summary(saratoga)
saratoga <- within(saratoga, {
  has_fireplace <- Fireplace=='Yes'
  Acres <- NULL
  Fireplace <- NULL
})
N <- nrow(saratoga)
scatterplotMatrix(~Price + ., data=saratoga)
cor(saratoga)
# - Living area seems to be highly correlated with baths and bedrooms,
# which both may be proxies for living area
# - Price and living area seem to have a non-linear relationship
# - Age looks to have a decreasing impact on price

# Part 2 --
# Fit a model that uses size to predict price, 
# denote this as model #1.
model1.ff <- Price ~ Living.Area
model1.lm <- lm(model1.ff, data=saratoga)
# plot(model1.lm)
summary(model1.lm)
c <- coeftest(model1.lm, vcov=vcovHC); c
w <- waldtest(model1.lm, vcov=vcovHC); w

# Part 2a. --
# Is there evidence the line does not pass through the origin? 
# Answer this question using a confidence interval.
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
# H_0: B_1 - 100 = 0
# H_1: B_1 - 100 < 0
tval <- (c[2,1]-100)/c[2,2]
pval <- pt(tval, N-2)
print(paste('P-value is',round(pval,6)))


#####################################################################
# 
# Question #2
# 
#####################################################################


# Part 1 -- 
# Does Hillary Clinton rate relatively higher compared to 
# Bernie Sanders among individuals who have a higher feeling 
# thermometer rating for minority groups? 
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
})
summary(lab.data.q2)

# model building
ff <- hc_over_bs ~ ftminority
lmodel <- lm(ff, data=lab.data.q2)
coeftest(lmodel, vcov=vcovHC)
waldtest(lmodel, vcov = vcovHC)
# parameter ftminority is not a significant
# predictor of rating Hillary over Sanders


# Part 2 -- 
# How does the inclusion of respondents’ perception of 
# President Obama and the economy (well known predictors 
# of presidential elections) impact your answer 
# to the first question? 
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
# the likelyhood of rating Hillary over Sanders

