#  
#  Image Processing 
#  Kaggle https://www.kaggle.com/c/facial-keypoints-detection/details/getting-started-with-r
# 

# Load libraries
install.packages('doMC')
require('doMC')

# Register multicore backend
registerDoMC()

# Set Args
data.dir <- '/Users/bshur/School/Extras/Data/'
test.file <- 'imageTest.csv'
train.file <- 'imageTraining.csv'

# Read training & test data
d.train <- read.csv(paste0(data.dir,train.file),stringsAsFactors=F)
d.test <- read.csv(paste0(data.dir,test.file),stringsAsFactors=F)

# Prepare image data
im.train <- d.train$Image
im.test <- d.test$Image
d.train$Image <- NULL
d.test$Image <- NULL

# Vectorize raw image data with parallel processing
im.train <- foreach(im = im.train, .combine=rbind) %dopar% {
  as.integer(unlist(strsplit(im, " ")))
}
im.test <- foreach(im = im.test, .combine=rbind) %dopar% {
  as.integer(unlist(strsplit(im, " ")))
}

# Save data 
save(d.train, im.train, d.test, im.test, file=paste0(data.dir,'data.Rd'))
