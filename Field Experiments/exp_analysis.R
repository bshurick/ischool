
require(plyr)
comb <- read.csv('~/Downloads/CombinatoricsPilot.csv',header=T)
Renamecols <- function(df) {
  colnames(df)<- c(
    "ResponseID",
    "ResponseSet",
    "Name",
    "ExternalDataReference",
    "EmailAddress",
    "IPAddress",
    "Status",
    "StartDate",
    "EndDate",
    "Finished",
    "random",
    "leveledfclick",
    "leveledlclick",
    "leveledpagesubmit",
    "leveledclickcount",
    "leveled",
    "skillmathfclick",
    "skillmathlclick",
    "skillmathpagesubmit",
    "skillmathclickcount",
    "skillmath",
    "englishreadingfclick",
    "englishreadinglclick",
    "englishreadingpagesubmit",
    "englishreadingclickcount",
    "englishreading",
    "englishlisteningfclick",
    "englishlisteninglclick",
    "englishlisteningpagesubmit",
    "englishlisteningclickcount",
    "englishlistening",
    "ageyearsfclick",
    "ageyearslclick",
    "ageyearspagesubmit",
    "ageyearsclickcount",
    "ageyears",
    "gender1mfclick",
    "gender1mlclick",
    "gender1mpagesubmit",
    "gender1mclickcount",
    "gender1m",
    "countrybirthfclick",
    "countrybirthlclick",
    "countrybirthpagesubmit",
    "countrybirthclickcount",
    "countrybirth",
    "countryresidencefclick",
    "countryresidencelclick",
    "countryresidencepagesubmit",
    "countryresidenceclickcount",
    "countryresidence",
    "SurveyBegin",
    "pre1_10fclick",
    "pre1_10lclick",
    "pre1_10pagesubmit",
    "pre1_10clickcount",
    "pre1_10",
    "pre2_10fclick",
    "pre2_10lclick",
    "pre2_10pagesubmit",
    "pre2_10clickcount",
    "pre2_10",
    "pre3_6fclick",
    "pre3_6lclick",
    "pre3_6pagesubmit",
    "pre3_6clickcount",
    "pre3_6",
    "pre4_10fclick",
    "pre4_10lclick",
    "pre4_10pagesubmit",
    "pre4_10clickcount",
    "pre4_10",
    "pre5_cfclick",
    "pre5_clclick",
    "pre5_cpagesubmit",
    "pre5_cclickcount",
    "pre5_c",
    "pre6_4fclick",
    "pre6_4lclick",
    "pre6_4pagesubmit",
    "pre6_4clickcount",
    "pre6_4",
    "textmathfclick",
    "textmathlclick",
    "textmathpagesubmit",
    "textmathclickcount",
    "textmath",
    "videomathfclick",
    "videomathlclick",
    "videomathpagesubmit",
    "videomathclickcount",
    "videomath",
    "placebofclick",
    "placebolclick",
    "placebopagesubmit",
    "placeboclickcount",
    "placebo",
    "post1_15fclick",
    "post1_15lclick",
    "post1_15pagesubmit",
    "post1_15clickcount",
    "post1_15",
    "post2_3fclick",
    "post2_3lclick",
    "post2_3pagesubmit",
    "post2_3clickcount",
    "post2_3",
    "post3_6fclick",
    "post3_6lclick",
    "post3_6pagesubmit",
    "post3_6clickcount",
    "post3_6",
    "post4_35fclick",
    "post4_35lclick",
    "post4_35pagesubmit",
    "post4_35clickcount",
    "post4_35",
    "post5_cfclick",
    "post5_clclick",
    "post5_cpagesubmit",
    "post5_cclickcount",
    "post5_c",
    "post6_4fclick",
    "post6_4lclick",
    "post6_4pagesubmit",
    "post6_4clickcount",
    "post6_4",
    "confirmationcode",
    "LocationLatitude",
    "LocationLongitude",
    "LocationAccuracy"
  )
  return(df)
}

comb <- Renamecols(comb)[, c("ResponseID",
                   "ResponseSet",
                   "Name",
                   "ExternalDataReference",
                   "EmailAddress",
                   "IPAddress",
                   "Status",
                   "StartDate",
                   "EndDate",
                   "Finished",
                   "random",
                   "leveled",
                   "skillmath",
                   "englishreading",
                   "englishlistening",
                   "ageyears",
                   "gender1m",
                   "countrybirth",
                   "countryresidence",
                   "pre1_10",
                   "pre2_10",
                   "pre3_6",
                   "pre4_10",
                   "pre5_c",
                   "pre6_4",
                   "textmath",
                   "videomath",
                   "placebo",
                   "post1_15",
                   "post2_3",
                   "post3_6",
                   "post4_35",
                   "post5_c",
                   "post6_4",
                   "confirmationcode",
                   "LocationLatitude",
                   "LocationLongitude",
                   "LocationAccuracy")]

comb[is.na(comb$textmath),'textmath'] <- 0
comb[is.na(comb$videomath),'videomath'] <- 0
comb[is.na(comb$placebo),'placebo'] <- 0

comb <- within(comb,{
  pre1 <- (pre1_10==10)*1
  pre2 <- (pre2_10==10)*1
  pre3 <- (pre3_6==6)*1
  pre4 <- (pre4_10==10)*1
  pre5 <- (tolower(pre5_c)=='c')*1
  pre6 <- (pre6_4==4)*1
  post1 <- (post1_15==15)*1
  post2 <- (post2_3==3)*1
  post3 <- (post3_6==6)*1
  post4 <- (post4_35==35)*1
  post5 <- (tolower(post5_c)=='c')*1
  post6 <- (post6_4==4)*1
  prescore <- pre1+pre2+pre3+pre4+pre5+pre6
  postscore <- post1+post2+post3+post4+post5+post6
  d_in_d <- postscore-prescore
  testtime <- as.POSIXlt(EndDate)-as.POSIXlt(StartDate)
})

with(comb,mean(d_in_d[textmath==1],na.rm=T)-mean(d_in_d[placebo==1],na.rm=T))
with(comb,mean(d_in_d[videomath==1],na.rm=T)-mean(d_in_d[placebo==1],na.rm=T))
with(comb,mean(d_in_d[videomath==1],na.rm=T)-mean(d_in_d[textmath==1],na.rm=T))

# > with(comb,mean(d_in_d[textmath==1],na.rm=T)-mean(d_in_d[placebo==1],na.rm=T))
# [1] -0.4662338
# > with(comb,mean(d_in_d[videomath==1],na.rm=T)-mean(d_in_d[placebo==1],na.rm=T))
# [1] -0.1238095
# > with(comb,mean(d_in_d[videomath==1],na.rm=T)-mean(d_in_d[textmath==1],na.rm=T))
# [1] 0.3424242

ddply(comb,.(textmath,videomath,placebo),summarize,
      avg_testtime = mean(testtime,na.rm=T))

# textmath videomath placebo   avg_testtime
# 1        0         0       1  9.855833 mins
# 2        0         1       0 17.050980 mins
# 3        1         0       0 17.172840 mins

summary(with(comb,lm(d_in_d~textmath+videomath+placebo+skillmath)))

# Call:
#   lm(formula = d_in_d ~ textmath + videomath + placebo + skillmath)
# 
# Residuals:
#   Min      1Q  Median      3Q     Max 
# -4.8242 -0.9782  0.0732  0.9231  4.0732 
# 
# Coefficients: (1 not defined because of singularities)
# Estimate Std. Error t value Pr(>|t|)
# (Intercept)  -0.4166     0.4386  -0.950    0.345
# textmath     -0.5488     0.4294  -1.278    0.205
# videomath    -0.2488     0.3996  -0.623    0.535
# placebo           NA         NA      NA       NA
# skillmath     0.1974     0.1459   1.353    0.180
# 
# Residual standard error: 1.562 on 83 degrees of freedom
# (14 observations deleted due to missingness)
# Multiple R-squared:  0.03554,  Adjusted R-squared:  0.0006794 
# F-statistic: 1.019 on 3 and 83 DF,  p-value: 0.3883

ddply(comb,.(textmath,videomath,placebo),summarize,
      avg_skillmath = mean(skillmath,na.rm=T))

# textmath videomath placebo avg_skillmath
# 1        0         0       1      2.475000
# 2        0         1       0      3.117647
# 3        1         0       0      2.814815

