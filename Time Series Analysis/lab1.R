
# Question 4
x <- 1-pbinom(0, 12, 0.2)
print(paste0('Probability of 1+ juror: ',round(x*100,2),'%'))

x <- 1-pbinom(1, 12, 0.2)
print(paste0('Probability of 2+ jurors: ',round(x*100,2),'%'))


