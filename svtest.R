library(bssm)
y <- scan("sv.dat")
model <- svm(y, ar=0.9731, sd_ar = 0.1726, sigma = 0.6338)
model$upper_prior[1]<-0.9999
model$lower_prior[] <- 0.001

system.time(out_st <- run_mcmc(model, 6e4, nsim_states = 1, method = "st")) #267
system.time(out_DA10 <- run_mcmc(model, 6e4, nsim_states = 10, method = "DA")) #294
system.time(out_DA50 <- run_mcmc(model, 6e4, nsim_states = 50, method = "DA")) #441
system.time(out_st10 <- run_mcmc(model, 6e4, nsim_states = 10, method = "st")) #424
system.time(out_st50 <- run_mcmc(model, 6e4, nsim_states = 50, method = "st")) #948
system.time(out_DA250 <- run_mcmc(model, 6e4, nsim_states = 250, method = "DA")) #1170

mean(coda::effectiveSize(out_st$theta))
mean(coda::effectiveSize(out_st10$theta))/424 #=2.2
mean(coda::effectiveSize(out_st50$theta))/948 #=1.3
mean(coda::effectiveSize(out_DA10$theta))/294 #=1.2
mean(coda::effectiveSize(out_DA50$theta))/441 #=2.8
mean(coda::effectiveSize(out_DA250$theta))/ 1170#=1.0


ess_st <- s_st <- ess_da <- s_da <- matrix(NA, 50, 50)
seeds <- 1:50
for (i in 1:50) {
  for (j in 1:50) {
    s_st[j,i] <-
      system.time(res <- run_mcmc(model, method = "st", n_iter = 5e4,
        nsim_states = i*5, seed = j))[3]
    ess_st[j, i] <- mean(coda::effectiveSize(res$theta))
    s_da[j,i] <-
      system.time(res <- run_mcmc(model, method = "DA", n_iter = 5e4,
        nsim_states = i*5, seed = j))[3]
    ess_da[j, i] <- mean(coda::effectiveSize(res$theta))

  }
  print(i)
}



ll <- s <- matrix(NA, 100, 500)
seeds <- 1:100
for (i in 1:500) {
  for (j in 1:100) {
    s[j,i] <-
      system.time(ll[j, i] <- logLik(model, nsim_states = 1 + i, j))[3]
  }
  print(i)
}


model2 <- model
model2$sigma <- 1
model2$T[]<-0.95
model2$R[]<-0.2
ll2 <- s2 <- matrix(NA, 100, 500)
seeds <- 1:100
for (i in 1:500) {
  for (j in 1:100) {
    s2[j,i] <-
      system.time(ll2[j, i] <- logLik(model2, nsim_states = 1 + i, j))[3]
  }
  print(i)
}


ts.plot(colMeans(ll))
acf(colMeans(ll))
ts.plot(apply(ll,2,sd))
ts.plot(apply(s,2,mean))
plot((apply(ll,2,sd)[1:400]~I(log(1:400))))
summary(lm(apply(s,2,mean)[1:400]~I(1:400)))
plot(apply(ll,2,sd)[1:400] ~ apply(s,2,mean)[1:400])

ts.plot(colMeans(ll2))
acf(colMeans(ll2))
ts.plot(apply(ll2,2,sd))

summary(coda::effectiveSize(coda::mcmc(t(out_st$alpha[,1,]))))
summary(coda::effectiveSize(coda::mcmc(t(out_DA10$alpha[,1,]))))
summary(coda::effectiveSize(coda::mcmc(t(out_DA50$alpha[,1,]))))
326/294
1387/441

summary(out_st$theta)$stat
summary(out_DA10$theta)$stat
summary(out_DA50$theta)$stat

plot(out_st$theta)
plot(out_DA10$theta)
plot(out_DA50$theta)

a<-rowMeans(out_st$alpha[,1,])
ts.plot(cbind(a, t(apply(out_st$alpha[,1,],1,quantile,prob=c(0.05,0.95)))),lty=c(1,2,2))
obj <- function(pars, model, estimate = TRUE) {
  model$T[1] <- pars[1]
  model$R[1] <- pars[2]
  model$P1[1] <- pars[2]^2/(1-pars[1]^2)
  model$sigma <- pars[3]
  if(estimate) {
    -logLik(model,100)
  } else {
    model
  }
}
model <- svm(y, ar=0.9731, sd_ar = 0.1726, sigma = 0.6338)

fit <- optim(fn=obj, par = c(0.7,1,1), model = model, method ="L-BFGS-B",
  lower=c(0,0,0)+1e-4, upper = c(1-1e-4,2,2), hessian = TRUE)
fit

S<-t(chol(solve(fit$hessian)))

out <- run_mcmc(model, 1e4, n_burnin = 0, nsim_states = 1, S = S, method ="st")

plot(out$theta)
x<-logLik(model,0)
range(x$y)
x$llg
x$ll
modg <- gssm(x$y, Z=model$Z, T=model$T,R = model$R, H = sqrt(x$HH), a1 =model$a1, P1=model$P1)
logLik(modg)
smg <- smoother(modg)
ts.plot(smg$alphahat)

library(KFAS)
modgkfas<-SSModel(x$y ~ -1 + SSMcustom(Z=1, T=model$T,R=model$R,Q=1,a1=model$a1,P1=model$P1),H=array(x$HH,c(1,1,945)))
modgkfas$H[1:10]
modg$H[1:10]^2

out1<-KFS(modgkfas)
out2<-smoother(modg)
all.equal(out1$alpha,out2$alpha, check.attributes=FALSE)
all.equal(out1$V,out2$Vt, check.attributes=FALSE)
out1$lo
logLik(modg)
range(x$y)
range(x$HH)
