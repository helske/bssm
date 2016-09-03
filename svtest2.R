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

app<-gaussian_approx(model)
mod<-gssm(app$y, Z=1,T=model$T,R=model$R,P1=model$P1,H=app$H)

library(KFAS)
kmodel <- SSModel(app$y ~ -1 + SSMcustom(Z=1,T=model$T,R=model$R,Q=1,P1=model$P1,P1inf=0),H=array(app$H^2,c(1,1,5443)))
logLik(kmodel)
logLik(mod)
out<-KFS(kmodel)
# > sum(dnorm(mod$y,app$signal,mod$H,log=TRUE))
# [1] -115777888
# > sum(dnorm(model$y,0,model$sigma*exp(app$signal/2),log=TRUE))
# [1] -6993.279
# > sum(dnorm(model$y,0,model$sigma*exp(app$signal/2),log=TRUE))- sum(dnorm(mod$y,app$signal,mod$H,log=TRUE))
# [1] 115770895

app<-gaussian_approx(model,8)
sum(dnorm(model$y,0,model$sigma*exp(app$signal/2),log=TRUE))

logLik(gssm(app$y, Z=1,T=model$T,R=model$R,P1=model$P1,H=app$H))

+app$scaling_factor
