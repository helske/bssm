library(bssm)

set.seed(1)
x <- array(0, c(2,3, 1e6))
L <- t(chol(matrix(0.1*c(2,1,1,2), 2,2)))
m <- c(1:2)
for(i in 1:1e6) {
  for(j in 1:3) {
    x[,j,i] <- m + L%*%rnorm(2)
  }
}
rowMeans(x[,1,])
rowMeans(x[,2,])
rowMeans(x[,3,])
apply(x[,1,], 1, var)
apply(x[,2,], 1, var)
apply(x[,3,], 1, var)

mean_x <- matrix(0,2,3)
cov_x <- array(0, c(2,2,3))
for(i in 1:1e6){
bssm:::running_summary(x[,,i], mean_x, cov_x, i-1)
}
mean_x-apply(x, 2, rowMeans)
cov_x[,,1] - cov(t(x[,1,]))
cov_x[,,2] - cov(t(x[,2,]))
cov_x[,,3] - cov(t(x[,3,]))

wmean_x <- matrix(0,2,3)
wcov_x <- array(0, c(2,2,3))
bssm:::running_weighted_summary(x, wmean_x, wcov_x, rep(1,1e6))

wmean_x-mean_x
cov_x - wcov_x
mean_x-apply(x, 2, rowMeans)
cov_x[,,1] - cov(t(x[,1,]))
cov_x[,,2] - cov(t(x[,2,]))
cov_x[,,3] - cov(t(x[,3,]))

###
set.seed(1)
y <- array(runif(1e6,min=-20,max=20), c(1,1,1e6))
w <- dnorm(y)#/dnorm(y,mean=0.5)

wmean_y <- matrix(0,1,1)
wcov_y <- array(0, c(1,1,1))
bssm:::running_weighted_summary(y, wmean_y, wcov_y, w)
wmean_y
wcov_y

