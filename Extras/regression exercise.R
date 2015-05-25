library(MASS)
N = 1000
#creating independent variables
### first, the covariance matrix for our multi-variate normal 
### this determines the correlation of our independent variables.
sig  = matrix(c(2,.5,.25,.5,1,0,.25,0,1) , nrow=3) 
### now the variables:
M = mvrnorm(n = N, mu = rep(1,3), Sigma = sig )

y.cont = 1 + 2* M[,1] - 5 * M[,2]  + M[,3] + rnorm(N)
y.bin  = as.numeric ( y.cont > 0 ) 


############## PART 2

### RSS returns total sum of squares of residuals
RSS <- function(beta,y,x) {
  y.hat = x %*% beta
  sum((y-y.hat)^2)
}


### find b that minimizes RSS
regress <- function(y,x) {
  if (!is.matrix(x))  x=matrix(x, nrow=length(y) )  
  k=dim(x)[2]
  beta0 = rep(0,k)
  best = optim(par = beta0 , fn = RSS, y=y, x=x)
  ### we should CHECK optim's convergence (but I'm not doing it)
  return(best$par)
}

### test the function 
### include the intercept in your independent variables
X = cbind (1, M )
y = y.cont 
regress(y,X)
lm(y~0+X)
lm(y~M)

####### PART 3
LL<- function(b,y,x){
  xb = x %*% b
  sum( ifelse(y==0,
         -xb - log( 1+exp(-xb)  ), ## in case of zero
         0   - log( 1+exp(-xb)  ) ## in case of one
         ) )
  }
  
### find b that maximizes LL
logit <- function(y,x) {
  if (!is.matrix(x))  x=matrix(x, nrow=length(y) )  
  k=dim(x)[2]
  beta0 = rep(1,k)
  best = optim(par = beta0 , fn = LL, control = list(fnscale=-1), y=y, x=x)
  if (best$convergence == 0) 
    return(best$par)
  else return(NA) 
}

X = cbind (1, M )
logit(y.bin,X)
glm(y.bin~0+X,family = binomial)



############ PART ONE 
reg = function(y,x)
{
  n = length (y)
  x = matrix(x, nrow=n)
  if ( dim(x)[2] == 1) { 
    b = sum(y*x)/sum(x^2) 
    e = y - x*b
  }
  else {
    r0 = reg(y,x[,-1])
    r1 = reg(x[,1],x[,-1])
    r2 = reg(r0$e,r1$e)
    r3 = reg( y - x[,1] * r2$b, x[,-1] )
    b =  rbind(r2$b,r3$b)
    e = y - x %*% b
  }
  return(list(b=b,e=e))
}



##### TESTING
reg(y,X) $b
coef( lm(y~0+X) )


