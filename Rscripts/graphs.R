library(ggplot2)
load("Countries.Rdata")
summary(Countries)

# demonstration of histogram with gdp
graph1 = ggplot(Countries, aes(gdp))
graph1 + geom_histogram()

graph1 + geom_histogram() + scale_x_log10()

graph1 + geom_histogram(binwidth=.2) + scale_x_log10()

# investigation of corruption and growth of internet users

Corruption = read.csv("2012_Corruption_Index.csv", header=TRUE)
summary(Corruption)
Countries = merge(Countries, Corruption, by="Country", all=TRUE)

Countries$internet_growth = (Countries$internet_users_2011 - Countries$internet_users_2010) / Countries$internet_users_2010

graph2 = ggplot(Countries, aes(internet_growth))
graph2 + geom_histogram()

graph3 = ggplot(Countries, aes(cpi, internet_growth))
graph3 + geom_point(shape=4)

graph3 + geom_point() + scale_y_log10()

# recode corruption as a binary variable

Countries$high_cpi = Countries$cpi > mean(Countries$cpi, na.rm = TRUE)

Countries$high_cpi = ifelse(Countries$cpi > mean(Countries$cpi, na.rm = TRUE), "Trustworthy", "Corrupt")
Countries$high_cpi

graph4 = ggplot(Countries[! is.na(Countries$high_cpi),], aes( high_cpi, internet_growth))
graph4 + geom_boxplot()

graph4 + stat_summary(fun.y = mean, geom = "bar", fill="White", colour = "Black") + stat_summary(fun.data = mean_cl_normal, geom = "errorbar", width = 0.2)

# invesigate the effect of high and low gdp

#code a new high_gdp variable
Countries$high_gdp = ifelse(Countries$gdp > median(Countries$gdp, na.rm = TRUE), "High", "Low")
Countries$high_gdp = factor( Countries$high_gdp)

Countries_lim = Countries[!(is.na(Countries$high_gdp) | is.na(Countries$high_cpi)), ]

#create bar chat of internet growth against both high corruption and high gdp
graph5 = ggplot(Countries_lim, aes( high_gdp, internet_growth, fill=high_cpi))

graph5 + stat_summary(fun.y = mean, geom="bar", colour="Black", position = "dodge") + stat_summary(fun.data = mean_cl_normal, geom="errorbar", width=0.2 , position = position_dodge(width= 0.9))

# create scatter of internet by internet users and high cpi

graph6 = ggplot(Countries_lim, aes( internet_users_2010, internet_growth, colour=high_cpi))
graph6 + geom_point() + geom_smooth()
