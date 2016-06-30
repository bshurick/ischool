
# read and parse data
sd <- read.csv('~/School/Extras/Kaggle/Speeddating/Data/Speed\ Dating\ Data.csv')
sd <- within(sd, {
  male <- gender
  gender <- NULL
  age_diff <- age - age_o
  goout_frequently <- go_out==1 | go_out==2 | go_out==3
  date_frequently <- date==1 | date==2 | date==3
  has_met <- met==1
  income <- as.numeric(income)
  likes_clubbing <- clubbing>median(clubbing, na.rm=T)
  likes_gaming <- gaming>median(gaming, na.rm=T)
  likes_exercising <- exercise>median(exercise, na.rm=T)
})
ff <- match ~ int_corr +
              date_frequently +
              goout_frequently +
              age +
              expnum +
              age*expnum +
              int_corr*expnum +
              likes_gaming +
              likes_clubbing +
              likes_exercising +
              has_met +
              imprace 
m <- glm(ff, data=sd, family=binomial(logit))
summary(m)

exp(m$coefficients)-1

