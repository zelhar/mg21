library(rankFD)
library(nparcomp)
library(tidyverse)
library(multcomp)

sp <- c(3.56,3.41,3.2,3.75,3.58,3.88,3.49,3.18,3.9,3.35,3.12,3.9)
sv <- c(4.00,3.84,3.98,3.9,3.88,3.73,4.41,4.19,4.5,4.2,4.05,3.67)
p <-  c(2.81,2.89,3.75,3.3,3.84,3.58,3.89,3.29,3.45,3.6,3.4,3.3)
v <- c(3.85,2.96,3.75,3.6,3.44,3.29,4.04,3.89,4.2,3.6,3.9,3.6)

#group <- as.factor(rep(c(1:4), each=12))
group <- as.factor(rep(c("sp","p","sv","v"), each=12))
group

score <- c(sp, p, sv, v)
score

mydata <- tibble(score, group)

mymat <- rbind(c(0,0,1,-1),
               c(1,0,-1,0),
               c(0,1,0,-1))

# I wanted to use my own custom contrast matrix which looks like
# this:
colnames(mymat) <- c("sp", "p", "sv", "v")
rownames(mymat) <- c("v:sv", "sv:sp", "v:p")
mymat
# but it didn't work, so I chose to use Tukey, which compares
# amongst others, the tests that we wanted.


nparcomp(score ~ group, data = mydata, asy.method = "mult.t", plot.simci = TRUE, type = "Tukey")
# nparcomp(score ~ group, data = mydata, asy.method = "mult.t", contrast.matrix = mymat, plot.simci = TRUE)
# contrMat(n=c(10,20,30,40), base=1, type="Williams")

mctp(score ~ group, data = mydata, type = "Tukey", asy.method = "mult.t", plot.simci = TRUE)
