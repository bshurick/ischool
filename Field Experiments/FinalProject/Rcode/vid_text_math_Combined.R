setwd("C:/Data/Github/DS241_VideoTextMath/")
source("Rcode/vidmathDatamapper.R")


library(gplots)
library(car)
library(ggplot2)
library(plotrix)
library(stargazer)
library(memisc) 

# Import data into Dataframes and create combined dataframe
  comb <- read.csv("Data/CombinatoricsPilot.csv", header=TRUE, strip.white=TRUE)
  comb = comb[-1,]
  combNoplacebo <- read.csv("Data/CombinatoricsPilot_08122015.csv", header=TRUE)
  combNoplacebo = combNoplacebo[-1,]
  dfcombined <- rbind(GetAnalysisSubset1(Renamecols(comb)), GetAnalysisSubset1(RenamecolsNoplacebo(combNoplacebo)))


# Simple Analysis of the combined dataframe
  colnames(dfcombined)
  head(dfcombined)
  nrow(dfcombined)
  nrow(dfcombined[dfcombined$textmath==1,])
  nrow(dfcombined[dfcombined$videomath==1,])
  nrow(dfcombined[dfcombined$placebo==1,])
  
  ATE_text = mean(dfcombined$d_in_d[dfcombined$textmath=="1"])
  ATE_text
  SE_text = std.error(dfcombined$d_in_d[dfcombined$textmath=="1"])
  SE_text
  ATE_video = mean(dfcombined$d_in_d[dfcombined$videomath=="1"])
  ATE_video
  SE_video = std.error(dfcombined$d_in_d[dfcombined$videomath=="1"])
  SE_video
  ATE_placebo = mean(dfcombined$d_in_d[dfcombined$placebo=="1"])
  ATE_placebo
  SE_placebo = std.error(dfcombined$d_in_d[dfcombined$placebo=="1"])
  SE_placebo
  
  ATE = ATE_video - ATE_text
  ATE

#Regression

  treatmentVideo <- dfcombined$d_in_d[dfcombined$videomath == 1]
  treatmentText <- dfcombined$d_in_d[dfcombined$textmath == 1]
  control <- dfcombined$d_in_d[dfcombined$placebo == 1]
  var.test(treatmentVideo,treatmentText)
  var.test(treatmentVideo,control)
  var.test(treatmentText,control)

  # Anova 
  fit = lm(d_in_d ~ factor(treatmentgroup),data=dfcombined)
  summary(fit)
  plot(fit)
  #stargazer(fit,title="Regression Results", align=TRUE)
  #mtable(fit)
  # Normality tests
  shapiro.test(control)
  shapiro.test(treatmentText)
  shapiro.test(treatmentVideo)  
  kruskal.test(d_in_d ~ factor(treatmentgroup), data = dfcombined) 
  t.test(control,treatmentVideo)
  treamentvt <- c(treatmentVideo,treatmentText)
  t.test(control,treamentvt)
  #Homogeneity of Variance test
  leveneTest(d_in_d ~ factor(treatmentgroup), data=dfcombined)    
  anova(fit)
  plotmeans(dfcombined$d_in_d~dfcombined$treatmentgroup, digits=2, ccol="red", xlab="Treatment Group", ylab="DID test score", mean.labels=T, main="Plot of testscore means by treatment group")
  #boxplot(dfcombined$d_in_d ~ dfcombined$treatmentgroup, main="test scores by treatment group (mean is black dot)", xlab="treatment group", ylab="d_in_d", col=rainbow(7),yaxt="n" , ylim = c(-1,1) )
  #axis(side=2,at=seq(-1.0,1.0,by=0.5))
    min.mean.sd.max <- function(x) {
      r <- c(min(x), mean(x) - sd(x), mean(x), mean(x) + sd(x), max(x))
      names(r) <- c("ymin", "lower", "middle", "upper", "ymax")
      r
    }
  p1 <- ggplot(aes(y = d_in_d, x = factor(treatmentgroup)), data = dfcombined)
  p1 <- p1 + stat_summary(fun.data = min.mean.sd.max, geom = "boxplot") + geom_jitter(position=position_jitter(width=.2), size=3) + ggtitle("Boxplot Treatment groups, 95%CI, test scores") + xlab("Treatment Group") + ylab("d_in_d Test Scores")
  plot(p1)
  
  # Clean up covariates for HTE Check
  dfcombined$countrybirth[dfcombined$countrybirth=="UAS"] = "USA"
  dfcombined$countrybirth[dfcombined$countrybirth=="US"] = "USA"
  dfcombined$countrybirth[dfcombined$countrybirth=="USA"] = "USA"
  dfcombined$countrybirth[dfcombined$countrybirth=="USA "] = "USA"
  dfcombined$countrybirth[dfcombined$countrybirth=="us"] = "USA"
  dfcombined$countrybirth[dfcombined$countrybirth=="usa"] = "USA"
  dfcombined$countrybirth[dfcombined$countrybirth=="united states"] = "USA"
  dfcombined$countrybirth[dfcombined$countrybirth=="United States"] = "USA"
  dfcombined$countrybirth[dfcombined$countrybirth=="United states"] = "USA"
  dfcombined$countrybirth[dfcombined$countrybirth=="The United States of America"] = "USA"
  dfcombined$countrybirth[dfcombined$countrybirth=="United States of America"] = "USA"
  dfcombined$countrybirth[dfcombined$countrybirth=="india"] = "India"
  dfcombined$countryresidence[dfcombined$countryresidence=="UAS"] = "USA"
  dfcombined$countryresidence[dfcombined$countryresidence=="US"] = "USA"
  dfcombined$countryresidence[dfcombined$countryresidence=="USA"] = "USA"
  dfcombined$countryresidence[dfcombined$countryresidence=="USA "] = "USA"
  dfcombined$countryresidence[dfcombined$countryresidence=="us"] = "USA"
  dfcombined$countryresidence[dfcombined$countryresidence=="usa"] = "USA"
  dfcombined$countryresidence[dfcombined$countryresidence=="united states"] = "USA"
  dfcombined$countryresidence[dfcombined$countryresidence=="United States"] = "USA"
  dfcombined$countryresidence[dfcombined$countryresidence=="United states"] = "USA"
  dfcombined$countryresidence[dfcombined$countryresidence=="The United States of America"] = "USA"
  dfcombined$countryresidence[dfcombined$countryresidence=="United States of America"] = "USA"
  dfcombined$countryresidence[dfcombined$countryresidence=="india"] = "India"
  dfcombined$agedecade=as.integer(as.integer(as.character(dfcombined$ageyears))/10)*10
  
  # HTE Regression (fishing trip!)
   
  dfcombined$high_ed <- as.numeric(dfcombined$leveled) > mean(as.numeric(dfcombined$leveled))
  dfcombined$high_math <- as.numeric(dfcombined$skillmath) > mean(as.numeric(dfcombined$skillmath))
  dfcombined$high_reading <- as.numeric(dfcombined$englishreading) > mean(as.numeric(dfcombined$englishreading))
  dfcombined$high_listening <- as.numeric(dfcombined$englishlistening) > mean(as.numeric(dfcombined$englishlistening))
  dfcombined$high_age <- as.numeric(dfcombined$ageyears) > mean(as.numeric(dfcombined$ageyears))
  
  ### TEST HTE FOR HIGH ED
  hte_high_ed_vid <-  lm(d_in_d ~ videomath + videomath * high_ed ,data=dfcombined)
  summary(hte_high_ed_vid)
  
  hte_high_ed_text <-  lm(d_in_d ~ textmath + textmath * high_ed ,data=dfcombined)
  summary(hte_high_ed_text)
  
  ### TEST HTE FOR HIGH SKILLMATH
  hte_high_math_vid <-  lm(d_in_d ~ videomath + videomath * high_math ,data=dfcombined)
  summary(hte_high_math_vid)
  
  hte_high_math_text <-  lm(d_in_d ~ textmath + textmath * high_math ,data=dfcombined)
  summary(hte_high_math_text)
  
  ### TEST HTE FOR HIGH ENGLISH READING
  hte_high_read_vid <-  lm(d_in_d ~ videomath + videomath * high_reading ,data=dfcombined)
  summary(hte_high_read_vid)
  
  hte_high_read_text <-  lm(d_in_d ~ textmath + textmath * high_reading ,data=dfcombined)
  summary(hte_high_read_text)
  
  ### TEST HTE FOR HIGH ENGLISH LISTENING
  hte_high_listen_vid <-  lm(d_in_d ~ videomath + videomath * high_listening ,data=dfcombined)
  summary(hte_high_listen_vid)
  
  hte_high_listen_text <-  lm(d_in_d ~ textmath + textmath * high_listening ,data=dfcombined)
  summary(hte_high_listen_text)
  
  ### TEST HTE FOR HIGH ENGLISH LISTENING
  hte_high_age_vid <-  lm(d_in_d ~ videomath + videomath * high_age ,data=dfcombined)
  summary(hte_high_age_vid)
  
  hte_high_age_text <-  lm(d_in_d ~ textmath + textmath * high_age ,data=dfcombined)
  summary(hte_high_age_text)
  
  ### SATURATED HTE MODELS
  
  hte_textmath = lm(d_in_d ~ textmath + textmath * factor(leveled) + textmath * factor(skillmath) + textmath * factor(englishreading) + textmath * factor(englishlistening) + textmath * factor(agedecade) + textmath * factor(gender1m) + textmath * factor(countrybirth) + textmath * factor(countryresidence),data=dfcombined)
  summary(hte_textmath)
  
  hte_videomath = lm(d_in_d ~ videomath + videomath * factor(leveled) + videomath * factor(skillmath) + videomath * factor(englishreading) + videomath * factor(englishlistening) + videomath * factor(agedecade) + videomath * factor(gender1m) + videomath * factor(countrybirth) + videomath * factor(countryresidence),data=dfcombined)
  summary(hte_videomath)
  
  hte_placebo = lm(d_in_d ~ placebo + placebo * factor(leveled) + placebo * factor(skillmath) + placebo * factor(englishreading) + placebo * factor(englishlistening) + placebo * factor(agedecade) + placebo * factor(gender1m) + placebo * factor(countrybirth) + placebo * factor(countryresidence),data=dfcombined)
  summary(hte_placebo)

  dcherrypick<-dfcombined[dfcombined$d_in_d > 2.5,]
  #dcherrypick
  #nrow(dcherrypick[dcherrypick$treatmentgroup == "Videomath",])
  #nrow(dcherrypick)
  nrow(dcherrypick[dcherrypick$treatmentgroup == "Videomath",])/nrow(dcherrypick)
  fitcherry = lm(d_in_d ~ factor(treatmentgroup),data=dcherrypick)
  summary(fitcherry) 
  

  
