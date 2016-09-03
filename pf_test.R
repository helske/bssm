## testing particle filter

rm(list=ls())
library("bssm")

y <- dget("sp500.txt")
model <- svm(y, ar = 0.98, sd_ar = 0.2, sigma = 1)

### ML-estimointi
obj <- function(pars) {
  -logLik(svm(y, ar = pars[1], sd_ar = pars[2], sigma = pars[3]), 100)
}
opt <- optim(c(0.98, 0.4, 1), obj, method = "L-BFGS-B", lower = c(-0.999, 1e-4, 1e-4), upper = c(0.999,10,10))
pars <- opt$par
model <- svm(y, ar = pars[1], sd_ar = pars[2], sigma = pars[3])

f_pf <- function(n, q){
  pf <- bootstrap_filter_svm(model, nsim = n, q = q)
  w <- exp(pf$V[,5443])
  w <- w/sum(w)
  sum(w*pf$alpha[1,5443,])
}

gpf1 <- replicate(100, f_pf(n = 100, q=0.25))
gpf2 <- replicate(100, f_pf(n = 100, q=0.5))
gpf3 <- replicate(100, f_pf(n = 100, q=0.75))
gpf4 <- replicate(100, f_pf(n = 100, q=1))


mean(gpf1)
mean(gpf2)
mean(gpf3)
mean(gpf4)

sd(gpf1)
sd(gpf2)
sd(gpf3)
sd(gpf4)

gpf1 <- replicate(1000, bootstrap_filter_svm(model, nsim = 100, q=0.25)$logU)
gpf2 <- replicate(1000, bootstrap_filter_svm(model, nsim = 100, q=0.5)$logU)
gpf3 <- replicate(1000, bootstrap_filter_svm(model, nsim = 100, q=0.75)$logU)
gpf4 <- replicate(1000, bootstrap_filter_svm(model, nsim = 100, q=1)$logU)

logLik(model, 1000) #-7334.273

basic_bs <- bootstrap_filter_svm(model, nsim = 100)
bbs <- replicate(100, bootstrap_filter_svm(model, nsim = 100))

stratified_bs <- bootstrap_filter_svm(model, nsim = 100)
sbs <- replicate(100, bootstrap_filter_svm(model, nsim = 100))

approx_pf <- bootstrap_filter_svm(model, nsim = 100)
gpf1 <- replicate(1000, bootstrap_filter_svm(model, nsim = 100)$logU)
gpf2 <- replicate(100, bootstrap_filter_svm(model, nsim = 1000)$logU)


# > sd(unlist(bbs[seq(4,400,by=4)]))
# [1] 8.365088
# > mean(unlist(bbs[seq(4,400,by=4)]))
# [1] -7361.915
# > sd(unlist(sbs[seq(4,400,by=4)]))
# [1] 5.72974
# > mean(unlist(sbs[seq(4,400,by=4)]))
# [1] -7347.404
# > sd(unlist(gpf[seq(4,400,by=4)]))
# [1]  4.649145
# > mean(unlist(gpf[seq(4,400,by=4)]))
# [1] -7346.767

