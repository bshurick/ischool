# Verify the Randomization of Qualtrics with RI
vector <-c()
groupcount <- data.frame( "maxcountineachgroup" = integer(), "pgreaterthanmaxcount" = integer(), stringsAsFactors=FALSE)
for (maxcount in 30:45 ) {
  count = 0 
  testsize = 10000
  for (i in 1:testsize ) {
    s <- sample(c(1,2,3),size = 100, replace =TRUE)
    n1 = length(s[s==1]) 
    n2 = length(s[s==2])
    n3 = length(s[s==3])
    nlist <- c(n1,n2,n3)
    max(nlist)
    if(max(nlist) > maxcount) {
      count = count +1
    }
  }
  count
  groupcount[nrow(groupcount) + 1, ] <- c( maxcount, (count/testsize)*100)
  #vector <- c(vector, cbind((maxcount,(count/testsize)*100)))
}

plot(groupcount$maxcountineachgroup,groupcount$pgreaterthanmaxcount)