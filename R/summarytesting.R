library(bssm)
set.seed(1)
model <- bsm(rep(NA,100), a1=2, P1=matrix(0,1,1), sd_level=0.001, slope=FALSE, sd_y=0)
y <- rpois(100, exp(sim_smoother(model,1)[,1,1]))
ts.plot(y)

model <- ng_bsm(y, a1=2, P1=matrix(0,1,1), sd_level=0.001, slope=FALSE, distribution="poisson")

outs <-  run_mcmc(model,n_iter=1e6, n_burnin = 1e5, nsim_states = 100, method ="delayed",seed=1)
S <- outs$S

###


out_100 <-  run_mcmc(model,1e6, 100, n_burnin = 1e5, method ="delayed", seed = 1, S = S)
o_da_100 <- run_mcmc(model,1e6, 100, n_burnin = 1e5, method ="delayed", seed = 1, S = S, type = "summary")
o_is1_100 <- run_mcmc(model,1e6, 100, n_burnin = 1e5, method ="IS c", seed = 1, seeds=1, S = S, type = "summary")
o_is2_100 <- run_mcmc(model,1e6, 100, n_burnin = 1e5, method ="block IS", seed = 1, seeds=1, S = S, type = "summary")
o_is3_100 <- run_mcmc(model,1e6, 100, n_burnin = 1e5, method ="IS2", seed = 1, seeds=1, S = S, type = "summary")

out_10 <-  run_mcmc(model,1e6, 10, n_burnin = 1e5, method ="delayed", seed = 1, S = S)
o_da_10 <- run_mcmc(model,1e6, 10, n_burnin = 1e5, method ="delayed", seed = 1, S = S, type = "summary")
o_is1_10 <- run_mcmc(model,1e6, 10, n_burnin = 1e5, method ="IS c", seed = 1, seeds=1, S = S, type = "summary")
o_is2_10 <- run_mcmc(model,1e6, 10, n_burnin = 1e5, method ="block IS", seed = 1, seeds=1, S = S, type = "summary")
o_is3_10 <- run_mcmc(model,1e6, 10, n_burnin = 1e5, method ="IS2", seed = 1, seeds=1, S = S, type = "summary")

out_4 <-  run_mcmc(model,1e6, 4, n_burnin = 1e5, method ="delayed", seed = 1, S = S)
o_da_4 <- run_mcmc(model,1e6, 4, n_burnin = 1e5, method ="delayed", seed = 1, S = S, type = "summary")
o_is1_4 <- run_mcmc(model,1e6, 4, n_burnin = 1e5, method ="IS c", seed = 1, seeds=1, S = S, type = "summary")
o_is2_4 <- run_mcmc(model,1e6, 4, n_burnin = 1e5, method ="block IS", seed = 1, seeds=1, S = S, type = "summary")
o_is3_4 <- run_mcmc(model,1e6, 4, n_burnin = 1e5, method ="IS2", seed = 1, seeds=1, S = S, type = "summary")


alphahat_100 <- rowMeans(out_100$alpha[,1,])
Vt_100 <- apply(out_100$alpha[,1,],1,var)
alphahat_10 <- rowMeans(out_10$alpha[,1,])
Vt_10 <- apply(out_10$alpha[,1,],1,var)
alphahat_4 <- rowMeans(out_4$alpha[,1,])
Vt_4 <- apply(out_4$alpha[,1,],1,var)

ts.plot(cbind(alphahat_100,alphahat_10,alphahat_4),col=1:3)
ts.plot(cbind(Vt_100,Vt_10,Vt_4),col=1:3)

ts.plot(alphahat_100, o_da_100$alphahat, o_is1_100$alphahat, o_is2_100$alphahat, o_is3_100$alphahat, col=1:5)
ts.plot(alphahat_10, o_da_10$alphahat, o_is1_10$alphahat, o_is2_10$alphahat, o_is3_10$alphahat, col=1:5)
ts.plot(alphahat_4, o_da_4$alphahat, o_is1_4$alphahat, o_is2_4$alphahat, o_is3_4$alphahat, col=1:5)

ts.plot(cbind(Vt_100, o_da_100$Vt, o_is1_100$Vt, o_is2_100$Vt, o_is3_100$Vt), col=1:5)
ts.plot(cbind(Vt_10, o_da_10$Vt, o_is1_10$Vt, o_is2_10$Vt, o_is3_10$Vt), col=1:5)
ts.plot(cbind(Vt_4, o_da_4$Vt, o_is1_4$Vt, o_is2_4$Vt, o_is3_4$Vt), col=1:5)

o_is1_4b <- run_mcmc(model,1e6, 4, n_burnin = 1e5, method ="IS c", seed = 1, seeds=1, S = S, type = "summary")
o_is2_4b <- run_mcmc(model,1e6, 4, n_burnin = 1e5, method ="block IS", seed = 1, seeds=1, S = S, type = "summary")
o_is3_4b <- run_mcmc(model,1e6, 4, n_burnin = 1e5, method ="IS2", seed = 1, seeds=1, S = S, type = "summary")
