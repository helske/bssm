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

## IS

imp <- importance_sample(model, nsim = 1000)
w1 <- imp$w/sum(imp$w)

sum(w1*imp$alpha[1,5443,])

#uskottavuus gaussinen approksimaatio + IS
logLik(model, 10)
## PF
pf <- bootstrap_filter_svm(model, nsim = 1000)
pf$logU #uskottavuus

w2 <- exp(pf$V[,5443])
w2 <- w2/sum(w2)
sum(w2*pf$alpha[1,5443,])

prod(exp(pf$V-0.812)[1,])

f_is <- function(n){
  imp <- importance_sample(model, nsim = n)
  w1 <- imp$w/sum(imp$w)
  sum(w1*imp$alpha[1,5443,])
}

f_pf <- function(n){
  pf <- bootstrap_filter_svm(model, nsim = n)
  w2 <- exp(pf$V[,5443])
  w2 <- w2/sum(w2)
  sum(w2*pf$alpha[1,5443,])
}


is_alpha_10 <- replicate(500, f_is(10))
is_alpha_100 <- replicate(500, f_is(100))
sd(is_alpha_10)/sqrt(500)
sd(is_alpha_100)/sqrt(500)
mean(is_alpha_10)
mean(is_alpha_100)



is_alpha_10b <- replicate(500, f_is(10))
is_alpha_100b <- replicate(500, f_is(100))
sd(is_alpha_10b)/sqrt(500)
sd(is_alpha_100b)/sqrt(500)
mean(is_alpha_10b)
mean(is_alpha_100b)


pf_alpha <- replicate(1000, f_pf(100))
sd(pf_alpha);mean(pf_alpha) #0.094, 1.11

ll_is_10b <- replicate(500,logLik(model, nsim = 10, seed = sample(1e8,size=1)))
ll_is_100b <- replicate(500,logLik(model, nsim = 100, seed = sample(1e8,size=1)))

model$T# R-funktio käyttää oletuksena siemenlukua 1 mahdollisen ML-estimoinnin takia
ll_is_10 <- replicate(500,logLik(model, nsim = 10, seed = sample(1e8,size=1)))
ll_is_100 <- replicate(500,logLik(model, nsim = 100, seed = sample(1e8,size=1)))
ll_pf_10 <- replicate(100,bootstrap_filter_svm(model, nsim = 10)$logU)
ll_pf_100 <- replicate(100,bootstrap_filter_svm(model, nsim = 100)$logU)

sd(ll_is_10);mean(ll_is_10)
sd(ll_is_100);mean(ll_is_100)
sd(ll_pf_10);mean(ll_pf_10)
sd(ll_pf_100);mean(ll_pf_100)

#repossa
load("sv_IS_LL.rda")
ts.plot(apply(IS_LL, 2, mean)) #harhaa...



##poisson-tapaus, näyttää toimivan paremmin
set.seed(555)
slope <- cumsum(c(0, rnorm(99, sd = 0.01)))
y <- rpois(100, exp(cumsum(slope + c(0, rnorm(99, sd = 0.1)))))
model <- ng_bsm(y, sd_level = 0.1, sd_slope = 0.01, P1 = diag(2), distribution = "poisson")


pll_is_2 <- replicate(1000, logLik(model, nsim = 2, seed = sample(1e8,size=1)))
pll_is_10 <- replicate(1000, logLik(model, nsim = 10, seed = sample(1e8,size=1)))
pll_is_100 <- replicate(1000, logLik(model, nsim = 100, seed = sample(1e8,size=1)))
pll_is_1000 <- replicate(1000, logLik(model, nsim = 1000, seed = sample(1e8,size=1)))
sd(pll_is_2)
sd(pll_is_10)
sd(pll_is_100)
sd(pll_is_1000)
mean(pll_is_2)
mean(pll_is_10)
mean(pll_is_100)
mean(pll_is_1000)

