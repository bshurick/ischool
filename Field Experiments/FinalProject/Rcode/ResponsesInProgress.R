setwd("C:/Data/Github/DS241_VideoTextMath/")
source("RCode/vidmathDatamapper.R")

# Import data into Dataframes
inprogress <- read.csv("Data/CombinatoricsPilot-Responses in Progress.csv", header=TRUE)
inprogress = inprogress[-1,]

# Analysis for Responses in Progress
  # Map Column names and get subset needed for analysis.
  colnames(inprogress)
  dfMappedinprogress <- RenamecolsNoplacebo(inprogress)
  colnames(dfMappedinprogress)
  dfMappedinprogress <- GetAnalysisSubset1(dfMappedinprogress)
  colnames(dfMappedinprogress)
  head(dfMappedinprogress)
  nrow(dfMappedinprogress)
  summary(factor(dfMappedinprogress$ResponseSet))
  summary(factor(dfMappedinprogress$textmath))
  summary(factor(dfMappedinprogress$videomath))
  summary(factor(dfMappedinprogress$placebo))

  # Ross you can add your code from here.


