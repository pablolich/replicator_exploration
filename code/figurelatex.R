library(tidyverse)

#data your dataframe before writing
data<-data.frame(Ratio=c(5,6,3,3,4,4),
                 Number=c(65,74,43,34,23,12))
a<-rep(NA,dim(data)[2])
data[2,] = a
datana = rbind(data, a)

write.table(data, 
            file = "data_na.dat", 
            row.names = FALSE,
            na=" ",
            col.names=FALSE,
            sep=" ")


#load results

solution_lat = read.table("../solution_latent.csv")
solution_disc = read.table("../solution_disc.csv")
tpoints = read.table("../timepoints.csv")
evalpoints = read.table("../evalpoints.csv")

solution_lat["t"] = tpoints

latentTex = solution_lat %>% 
  pivot_longer(!c(t)) %>% 
  group_by(t) %>% 
  mutate(evalpoints = evalpoints$V1) %>%
  arrange(t) %>% 
  group_by(evalpoints) %>% 
  group_modify(~ add_row(.x,.before=0)) %>% 
  mutate(evalpoints = if_else(is.na(t), NA, evalpoints)) %>% 
  select(c(t,evalpoints, value))

write.table(latentTex[2:nrow(latentTex),], 
            file = "sol_lat.dat", 
            row.names = FALSE,
            na=" ",
            col.names=FALSE,
            sep=" ")

#initialize matrix
latexplot = list()
nevalpoints = ncol(solution_disc)
evalpointsvec = seq(1:nevalpoints)
ntpoints = length(tpoints)

for (i in 1:nevalpoints){
  for (j in ntpoints){
    #tpoint, evalpoint, density
    vec = c(tpoints[j], evalpoinitsvec[])
    #append
    latexplot = rbind(latexplot, )
  }
}
