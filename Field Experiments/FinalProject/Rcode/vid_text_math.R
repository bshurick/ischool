setwd("C:/Data/Github/DS241_VideoTextMath/")
source("RCode/vidmathDatamapper.R")

# Import data into Dataframes
comb <- read.csv("Data/CombinatoricsPilot.csv", header=TRUE)
comb = comb[-1,]
combNoplacebo <- read.csv("Data/CombinatoricsPilot_08122015.csv", header=TRUE)
combNoplacebo = combNoplacebo[-1,]
# Analysis for first survey
  # Map Column names and get subset needed for analysis.
  colnames(comb)
  dfMapped <- Renamecols(comb)
  colnames(dfMapped)
  dfsubset <- GetAnalysisSubset1(dfMapped)
  colnames(dfsubset)
  head(dfsubset)
  nrow(dfsubset)
  
  ATE_text = mean(dfsubset$postscore[dfsubset$textmath=="1"] - dfsubset$prescore[dfsubset$textmath=="1"])
  ATE_text
  ATE_video = mean(dfsubset$postscore[dfsubset$videomath=="1"] - dfsubset$prescore[dfsubset$videomath=="1"])
  ATE_video
  ATE_placebo = mean(dfsubset$postscore[dfsubset$placebo=="1"] - dfsubset$prescore[dfsubset$placebo=="1"])
  ATE_placebo
  

  #summary(factor(dfsubset$ResponseSet))

  nrow(dfsubset[dfsubset$textmath==1,])
  nrow(dfsubset[dfsubset$videomath==1,])
  nrow(dfsubset[dfsubset$placebo==1,])

# Analysis for second (Noplacebo) survey
  # Map Column names and get subset needed for analysis.
  colnames(combNoplacebo)
  head(combNoplacebo)
  dfMappedNoplacebo <- RenamecolsNoplacebo(combNoplacebo)
  colnames(dfMappedNoplacebo)
  dfsubsetNoplacebo <- GetAnalysisSubset1(dfMappedNoplacebo)
  colnames(dfsubsetNoplacebo)
  head(dfsubsetNoplacebo)
  nrow(dfsubsetNoplacebo)
  
  ATE_text = mean(dfsubsetNoplacebo$postscore[dfsubsetNoplacebo$textmath=="1"] - dfsubsetNoplacebo$prescore[dfsubsetNoplacebo$textmath=="1"])
  ATE_text
  ATE_video = mean(dfsubsetNoplacebo$postscore[dfsubsetNoplacebo$videomath=="1"] - dfsubsetNoplacebo$prescore[dfsubsetNoplacebo$videomath=="1"])
  ATE_video
  
  #summary(factor(dfsubsetNoplacebo$ResponseSet))
  nrow(dfsubsetNoplacebo[dfsubsetNoplacebo$textmath==1,])
  nrow(dfsubsetNoplacebo[dfsubsetNoplacebo$videomath==1,])
  nrow(dfsubsetNoplacebo[dfsubsetNoplacebo$placebo==1,])



