library(bssm)
set.seed(1)
model <- bsm(rep(NA,100), a1=2, P1=matrix(0,1,1), sd_level=0.001, slope=FALSE, sd_y=0)
y <- rpois(100, exp(sim_smoother(model,1)[,1,1]))
ts.plot(y)

model <- ng_bsm(y, a1=2, P1=matrix(0,1,1), sd_level=0.001, slope=FALSE, distribution="poisson")
###

o_da <- run_mcmc(model,n_iter=1e5, nsim_states = 50, method ="delayed", type = "summary")
o_is <- run_mcmc(model,n_iter=1e5, nsim_states = 50, method ="IS c", type = "summary")
o_bis <- run_mcmc(model,n_iter=1e5, nsim_states = 50, method ="block", type = "summary")
o_bis2 <- run_mcmc(model,n_iter=1e5, nsim_states = 50, method ="IS2", type = "summary") ##

ts.plot(o_da$alphahat, o_is$alphahat, o_bis$alphahat, o_bis2$alpha, col=1:4)
ts.plot(cbind(o_da$Vt, o_is$Vt, o_bis$Vt, o_bis2$Vt), col=1:4)

co_da <- run_mcmc(model,n_iter=1e5, nsim_states = 150, method ="delayed", type = "summary")
co_is <- run_mcmc(model,n_iter=1e5, nsim_states = 150, method ="IS c", type = "summary") #
co_bis <- run_mcmc(model,n_iter=1e5, nsim_states = 150, method ="block", type = "summary") #
co_bis2 <- run_mcmc(model,n_iter=1e5, nsim_states = 150, method ="IS2", type = "summary")
ts.plot(co_da$alphahat, co_is$alphahat, co_bis$alphahat, co_bis2$alpha, col=1:4)
ts.plot(cbind(co_da$Vt, co_is$Vt, co_bis$Vt, co_bis2$Vt), col=1:4)


bo_da <- run_mcmc(model,n_iter=1e6, nsim_states = 10, method ="delayed", type = "summary")
bo_is <- run_mcmc(model,n_iter=1e6, nsim_states = 10, method ="IS c", type = "summary")
bo_bis <- run_mcmc(model,n_iter=1e6, nsim_states = 10, method ="block", type = "summary")
bo_bis2 <- run_mcmc(model,n_iter=1e6, nsim_states = 10, method ="IS2", type = "summary")
ts.plot(bo_da$alphahat, bo_is$alphahat, bo_bis$alphahat, bo_bis2$alpha, col=1:4)
ts.plot(cbind(bo_da$Vt, bo_is$Vt, bo_bis$Vt, bo_bis2$Vt), col=1:4)

do_da <- run_mcmc(model,n_iter=1e6, nsim_states = 50, method ="delayed", type = "summary")
do_is <- run_mcmc(model,n_iter=1e6, nsim_states = 50, method ="IS c", type = "summary")
do_bis <- run_mcmc(model,n_iter=1e6, nsim_states = 50, method ="block", type = "summary")
do_bis2 <- run_mcmc(model,n_iter=1e6, nsim_states = 50, method ="IS2", type = "summary")
ts.plot(do_da$alphahat, do_is$alphahat, do_bis$alphahat, do_bis2$alpha, col=1:4)
ts.plot(cbind(do_da$Vt, do_is$Vt, do_bis$Vt, do_bis2$Vt), col=1:4)




oo_da <- run_mcmc(model,n_iter=1e4, nsim_states = 10, method ="delayed", type = "summary")
oo_is <- run_mcmc(model,n_iter=1e4, nsim_states = 10, method ="IS c", type = "summary")
oo_bis <- run_mcmc(model,n_iter=1e4, nsim_states = 10, method ="block", type = "summary")
oo_bis2 <- run_mcmc(model,n_iter=1e4, nsim_states = 10, method ="IS2", type = "summary")

ts.plot(oo_da$alphahat, oo_is$alphahat, oo_bis$alphahat, oo_bis2$alpha, col=1:4)
ts.plot(cbind(oo_da$Vt, oo_is$Vt, oo_bis$Vt, oo_bis2$Vt), col=1:4)

x1 <- run_mcmc(model,n_iter=500, nsim_states = 10, method ="block", type = "summary",thread=1:4,n_threads=4,seed=1)
x2 <- run_mcmc(model,n_iter=500, nsim_states = 10, method ="block", type = "summary",thread=1:4,n_threads=4,seed=1)
all.equal(x1, x2,tol=0)
x1$w
x2$w
system.time(o_da <- run_mcmc(model,n_iter=1e7, nsim_states = 50, method ="delayed", type = "summary"))
system.time(o_is <- run_mcmc(model,n_iter=1e7, nsim_states = 50, method ="IS c", type = "summary"))
system.time(o_bis <- run_mcmc(model,n_iter=1e7, nsim_states = 50, method ="block", type = "summary"))
system.time(o_bis2 <- run_mcmc(model,n_iter=1e7, nsim_states = 50, method ="IS2", type = "summary"))

summary(o_da$alphahat-o_is$alphahat)
summary(o_da$alphahat-o_bis$alphahat)
summary(o_da$alphahat-o_bis2$alphahat)
ts.plot(o_da$alphahat, o_is$alphahat, o_bis$alphahat, o_bis2$alpha, col=1:4)
all.equal(o_da$alphahat, o_bis2$alphahat)

ts.plot(cbind(o_da$Vt, o_is$Vt, o_bis$Vt, o_bis2$Vt), col=1:4)


system.time(bo_da <- run_mcmc(model,n_iter=1e6, nsim_states = 250, method ="delayed", type = "summary"))
system.time(bo_is <- run_mcmc(model,n_iter=1e6, nsim_states = 250, method ="IS c", type = "summary"))
system.time(bo_bis <- run_mcmc(model,n_iter=1e6, nsim_states = 250, method ="block", type = "summary"))
system.time(bo_bis2 <- run_mcmc(model,n_iter=1e6, nsim_states = 250, method ="IS2", type = "summary"))

system.time(fo_da <- run_mcmc(model,n_iter=1e6, nsim_states = 50, method ="delayed", type = "full"))
system.time(fo_is <- run_mcmc(model,n_iter=1e6, nsim_states = 50, method ="IS c", type = "full"))
system.time(fo_bis <- run_mcmc(model,n_iter=1e6, nsim_states = 50, method ="block", type = "full"))
system.time(fo_bis2 <- run_mcmc(model,n_iter=1e6, nsim_states = 50, method ="IS2", type = "full"))

w1 <- fo_is$weights/sum(fo_is$weights)
w2 <- fo_bis$weights*fo_bis$counts / sum(fo_bis$weights*fo_bis$counts)
w3 <- fo_bis2$weights*fo_bis2$counts / sum(fo_bis2$weights*fo_bis2$counts)

ts.plot(cbind(o_da$Vt, o_is$Vt, o_bis$Vt, o_bis2$Vt,
  apply(fo_da$alpha[,1,], 1, var),
  apply(fo_is$alpha[,1,], 1, function(x) sum(w1*x^2) - sum(w1*x)^2),
  apply(fo_bis$alpha[,1,], 1, function(x) sum(w2*x^2) - sum(w2*x)^2),
  apply(fo_bis2$alpha[,1,], 1, function(x) sum(w3*x^2) - sum(w3*x)^2)), col=1:8)

ts.plot(cbind(o_da$Vt, o_is$Vt, o_bis$Vt, o_bis2$Vt,
  bo_da$Vt, bo_is$Vt, bo_bis$Vt, bo_bis2$Vt), col=1:8)

system.time(fo_st <- run_mcmc(model,n_iter=1e6, nsim_states = 25, method ="standard", type = "full"))

ts.plot(cbind(apply(fo_st$alpha[,1,], 1, var), o_da$Vt, apply(fo_da$alpha[,1,], 1, var)),col = 1:3)


system.time(o_da2 <- run_mcmc(model,n_iter=1e7, nsim_states = 10, method ="delayed", type = "summary"))
ts.plot(cbind(apply(fo_st$alpha[,1,], 1, var), o_da$Vt, apply(fo_da$alpha[,1,], 1, var), o_da2$Vt),col = 1:4)
####

system.time(out_da <- run_mcmc(model,n_iter=1e7, nsim_states = 100, method ="delayed", type = "summary"))
system.time(out_is <- run_mcmc(model,n_iter=1e7, nsim_states = 100, method ="IS c", type = "summary"))
system.time(out_bis <- run_mcmc(model,n_iter=1e7, nsim_states = 100, method ="block", type = "summary"))
system.time(out_bis2 <- run_mcmc(model,n_iter=1e7, nsim_states = 100, method ="IS2", type = "summary"))

ts.plot(out_da$alphahat, out_is$alphahat, out_bis$alphahat, out_bis2$alphahat,
  out_dab$alpha, out_bisb$alpha, col=1:6)

ts.plot(cbind(out_da$Vt, out_is$Vt, out_bis$Vt, out_bis2$Vt, out_dab$Vt,out_bisb$Vt), col=1:6)

ts.plot(cbind(out_da$Vt, out_is$Vt, out_bis$Vt, out_bis2$Vt, out_dab$Vt,out_bisb$Vt)[60:80,], col=1:6)

all.equal(out_bis$Vt,out_bisb$Vt)

all.equal(out_da$alp,out_dab$alp)
all.equal(out_da$alp,out_is$alp)
all.equal(out_da$alp,out_bis$alp)
all.equal(out_da$alp,out_bisb$alp)

all.equal(out_da$Vt,out_dab$Vt)
all.equal(out_da$Vt,out_is$Vt)
all.equal(out_da$Vt,out_bis$Vt)
all.equal(out_da$Vt,out_bisb$Vt)

ts.plot(out_da$alphahat, out_is$alphahat, out_bis$alphahat, out_bis2$alphahat, col=1:4)
system.time(out5 <- run_mcmc(model,n_iter=1e5, nsim_states = 1, method ="sta", type = "summary"))



system.time(out3 <- run_mcmc(model,n_iter=1e5, nsim_states = 50, method ="del", type = "full"))

system.time(out4 <- run_mcmc(model,n_iter=1e5, nsim_states = 1, method ="sta", type = "full"))

system.time(out5 <- run_mcmc(model,n_iter=1e5, nsim_states = 1, method ="sta", type = "summary"))

system.time(out6 <- run_mcmc(model,n_iter=1e6, nsim_states = 100, method ="IS c", type = "summary"))

ts.plot(rowMeans(out3$alpha), out2$alphahat, out$alphahat, out4$alphahat, rowMeans(out5$alpha),
  out6$alphahat, col=1:6)

ts.plot(cbind( out$Vt,  out2$Vt, apply(out3$alpha[,1,],1,var), apply(out4$alpha[,1,],1,var),
  out5$Vt,out6$Vt), col=1:6)

ts.plot(rowMeans(out3$alpha)- out$alphahat, rowMeans(out3$alpha)- out2$alphahat,
  out$alphahat-out2$alphahat, rowMeans(out3$alpha)- out5$alphahat,
    rowMeans(out3$alpha)-  rowMeans(out4$alpha), col=1:5)


ts.plot(cbind( out$Vt,  out2$Vt, apply(out3$alpha[,1,],1,var), apply(out4$alpha[,1,],1,var),
  out5$Vt), col=1:5)


ts.plot(cbind(apply(out3$alpha[,1,],1,var) - out$Vt, apply(out3$alpha[,1,],1,var) - out2$Vt,
  out$Vt-out2$Vt), col=1:3)


###
system.time(out <- run_mcmc(model,n_iter=1e5, nsim_states = 10, method ="delayed", type = "param"))
summary(out$theta)


model <- ng_bsm(y, sd_level=0.05, sd_slope=0.005, distribution="poisson")

#system.time(out1 <- run_mcmc(model,n_iter=2e5, n_burnin = 1e5, nsim_states = 1, method ="sta", type = "full",seed=1))
system.time(out10 <- run_mcmc(model,n_iter=2e5, n_burnin = 1e5, nsim_states = 50, method ="sta", type = "full",seed=1))
#system.time(out2 <- run_mcmc(model,n_iter=2e5, n_burnin = 1e5, nsim_states = 1, method ="sta", type = "summary",seed=1))
system.time(out20 <- run_mcmc(model,n_iter=2e5, n_burnin = 1e5, nsim_states = 50, method ="sta", type = "summary",seed=1))

all.equal(rowMeans(out10b$alpha[,1,]),unclass(out20b$alphahat[,1]),check.attributes = FALSE)

ts.plot(cbind(rowMeans(out1$alpha[,1,]),rowMeans(out10$alpha[,1,]),out2$alphahat[,1],out20$alphahat[,1]),col=1:4)
ts.plot(cbind(apply(out1$alpha[,1,],1,var),apply(out10$alpha[,1,],1,var),out2$Vt[1,1,],out20$Vt[1,1,]),col=1:4)


ts.plot(cbind(rowMeans(out1b$alpha[,1,]),rowMeans(out10b$alpha[,1,]),out2b$alphahat[,1],out20b$alphahat[,1]),col=1:4,lwd=c(2,1,1,1))
ts.plot(cbind(apply(out1b$alpha[,1,],1,var),apply(out10b$alpha[,1,],1,var),out2b$Vt[1,1,],out20b$Vt[1,1,]),col=1:4,lwd=c(2,1,1,1))


ts.plot(cbind(rowMeans(out1$alpha[,1,]),rowMeans(out10$alpha[,1,])
  ,rowMeans(out2$alpha[,1,]),rowMeans(out20$alpha[,1,])),col=1:4)





system.time(out2 <- run_mcmc(model,n_iter=2e5, n_burnin = 1e5, nsim_states = 10, method ="sta", type = "summary"))
ts.plot(cbind(rowMeans(out$alpha[,1,]),rowMeans(out1$alpha[,1,]),out2$alphahat[,1]),col=1:3)

dim(out$alpha)
cov(t(out$alpha[20,,]))
out2$Vt[,,20]
rowMeans(out$alpha[20,,])
out2$alphahat[20,]

ts.plot(cbind(rowMeans(out$alpha[,1,]),out2$alphahat[,1]),col=1:2)
ts.plot(cbind(apply(out$alpha[,1,],1,var),out2$Vt[1,1,]),col=1:2)

KFS(mod,nsim=10)$alphahat[20,]
system.time(outb <- run_mcmc(model,n_iter=1e6, n_burnin = 0, nsim_states = 1, method ="delayed", type = "full"))
logLik(mod)
max(outb$logLik)
outb$theta[which.max(outb$logLik),]
sqrt(diag(mod$Q[,,1]))
model <- ng_bsm(y, sd_level=sqrt(mod$Q[1,1,1]),
  sd_slope=sqrt(mod$Q[2,2,1]), distribution="poisson")
logLik(model,0)
logLik(mod)
