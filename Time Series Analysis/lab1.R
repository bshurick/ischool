
# Question 4
x <- 1-pbinom(0, 12, 0.2)
print(paste0('Probability of 1+ juror: ',round(x*100,2),'%'))

x <- 1-pbinom(1, 12, 0.2)
print(paste0('Probability of 2+ jurors: ',round(x*100,2),'%'))


# Question 8
players <- c('Mark Price','Trent Tucker','Dale Ellis','Craig Hodges'
            ,'Danny Ainge','Byron Scott','Reggie Miller','Larry Bird'
            ,'Jon Sundvold','Brian Taylor')
fga <- c(429,833,1149,1016,1051,676,416,1206,440,417)
fgm <- c(188,345,472,396,406,260,159,455,166,157)
theta <- fgm/fga
Y <- mean(theta)
MP <- theta[1]
n <- sum(fga)
seY <- sqrt(Y*(1-Y)/n)
Z <- (Y-MP)/seY
p_value <- pnorm(Z)
print(paste0('SD of theta: ',round(sd(theta),6)))
print(paste0('SD of FGA: ',round(sd(fga),2)))
print(paste0('P-value: ',round(p_value,16)))


# Question 9 
yes_no <- c(rep(1,130),rep(0,70))
print(paste0('SD: ',round(sd(yes_no),6)))
z <- (0.575-0.65)/(0.4781665/sqrt(200))
p_value = pnorm(z)
print(paste0('P-value: ',round(p_value,16)))


# Question 10
z <- (0.394-0.4)/(sqrt(0.394*(1-0.394)/419))
p_value = pnorm(z)
