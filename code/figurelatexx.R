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

results = read.table("")