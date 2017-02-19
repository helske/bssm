## Lotka-Volterra example, taken from:
## http://blogs2.datall-analyse.nl/2016/02/11/rcode_extended_kalman_filter/
source("http://www.datall-analyse.nl/EKF.R")
set.seed(1)

dt <- 1/5000 #time step for Euler integration
tt <- 7 #upper bound of time window
st <- seq(0, tt, by=dt) #lower time bounds of the integration intervals
ns <- length(st) #number of Euler integrations
x <- matrix(0, ncol=2, nrow=ns) #specify matrix for states
x[1,] <- c(400, 200) #respective amounts of prey and predators at t=0
colnames(x) <- c("Prey", "Predators")

#parameters in the Lotka-Volterra model
true<-c(
alpha <- 1,
beta <- 1/300,
delta <- 1/200,
gamma <- 1)

#simulate true states
for (i in 2:ns) {
  #prey population
  x[i,1] <- x[i-1,1] + (alpha*x[i-1,1] - beta*x[i-1,1]*x[i-1,2])*dt
  #predator population
  x[i,2] <- x[i-1,2] + (delta*x[i-1,1]*x[i-1,2] - gamma*x[i-1,2])*dt
}

#phase-space plot of simulated predator-prey system
plot(x[,1], x[,2], type="l", xlab="Prey", ylab="Predators")

#Take measurements with a sample time of .01
dT <- .001 #sample time for measurements
#you may change the value of dT and see how it influences
#the behavior of the extended Kalman filter
nm <- tt/dT #total number of measurements
mt <- seq(dT, tt, dT) #measurement times

#standard deviations for the measurement noise
sigmaPrey <- 7 #prey
sigmaPred <- 10 #predators

#measurements at specified measurement times
yprey <- sapply(1:nm, function(i) x[i*((ns-1)/nm) + 1, 1] + rnorm(1, 0, sigmaPrey))
ypred <- sapply(1:nm, function(i) x[i*((ns-1)/nm) + 1, 2] + rnorm(1, 0, sigmaPred))

#store measurement data
dataEx1 <- cbind(yprey, ypred)

#plot the generated measurements
plot(x[,1], x[,2], type="l", xlab="Prey", ylab="Predators",
  xlim=range(yprey), ylim=range(ypred))
points(yprey, ypred, cex=.5, col="red")

#plot of time against measurements
par(mfrow=c(2, 1))
plot(st, x[,1], type="l", xlab="Time", ylab="Prey",
  ylim=range(yprey))
lines(mt, yprey, col="darkgreen")

plot(st, x[,2], type="l", xlab="Time", ylab="Predators",
  ylim=range(ypred))
lines(mt, ypred, col="brown")
par(mfrow=c(1, 1))

########################
########################
initial_theta <- c(
  alpha = 1,
  beta = 1/300,
  delta = 1/200,
  gamma = 1
)


Rcpp::sourceCpp("model_functions.cpp")
pntrs <- create_xptrs()
library(bssm)
model <- nlg_ssm(y=rbind(NA,dataEx1), a1=pntrs$a1, P1 = pntrs$P1, Z = pntrs$Z_fn, H = pntrs$H_fn,
  T = pntrs$T_fn, R = pntrs$R_fn, Z_gn = pntrs$Z_gn, T_gn = pntrs$T_gn, 
  theta = initial_theta, log_prior_pdf = pntrs$log_prior_pdf, 
  known_params = matrix(1), known_tv_params = matrix(1), 
  n_states = 2, n_etas = 2)


out <- ekf(model)
out_sm <- ekf_smoother(model)
out_iekf <- iekf_smoother(model)
out_sm[1:2,]
smoothState$s[1:2,]
ts.plot(cbind(ext1$m, out$att),col=rep(1:2,each=2))
all.equal(ts(ext1$m), out$att)
ts.plot(cbind(smoothState$s, out_sm, out_iekf),col=rep(1:3,each=2))
all.equal(smoothState$s, out_sm)

#Dynamic model:
#specifying 2 states, namely [amount of prey, amount of predators].
#We will first use the dlm function from the dlm package for specifying the
#dynamic model. The dlm function checks for us if the dimensions
#of our specified matrices are correct, and if the specified covariance matrices
#are positive semidefinite.
library(dlm)
ex1 <- dlm(m0=c(350, 250), #initial state estimates for prey and predators
  #error covariances of the initial state estimates:
  #this matrix reflects the uncertainty in our initial state estimates
  C0=diag(rep(100,2)),
  #observation matrix:
  #we will not use this FF matrix in the extended Kalman filter,
  #so we set the values in this matrix to zero
  FF=matrix(0, nrow=2, ncol=2),
  #measurement noise
  V=diag(c(sigmaPrey^2, sigmaPred^2)),
  #state transition matrix:
  #we will not use this GG matrix in the extended Kalman filter,
  #so we also set the values in this matrix to zero
  GG=matrix(0, nrow=2, ncol=2),
  #process noise
  W=diag(rep(0,2)))

#For the EKF we will use a list-object (instead of the dlm-object above),
#and remove the FF and GG matrix (which we do not need anyway for the EKF).
#Note that the specified initial state estimates (at t=0) below for prey and
#predators deviate from the values that were used for generating the data
#(i.e., 400 for prey and 200 for predators).
#You may change these initial state estimates too and see how they
#influence the behavior of the extended Kalman filter.
ex1 <- list(m0=c(350, 250), #initial state estimates
  #error covariances of the initial state estimates:
  #this matrix reflects the uncertainty in our initial state estimates
  #you may change the values in this matrix and see how they influence
  #the behavior of the Kalman filter
  C0=diag(rep(100,2)),
  #measurement noise
  V=diag(c(sigmaPrey^2, sigmaPred^2)),
  #process noise
  W=diag(rep(0,2)))
#Note that all covariances in the process noise matrix (W) are set to zero.
#This makes sense since the change in the amount of prey and predators at
#each time step is fully explained by our Lotka-Volterra model describing the
#prey-predator interaction.

#Specify the state transition function:
#note that we will use as state functions the difference equations
#given by Euler's forward method. These difference equations will yield valid 
#estimates for the amounts of prey and predators at each time step as long
#as the specified value for dT above is relatively small.
#WARNING: always use arguments x and k when specifying the GGfunction
GGfunction <- function (x, k){
  prey <- x[1]; pred <- x[2]
  c(prey + (alpha*prey - beta*prey*pred)*dT,
    pred + (delta*prey*pred - gamma*pred)*dT)}

#Specify the observation/measurement function
#WARNING: always use arguments x and k when specifying the FFfunction
FFfunction <- function (x, k){
  prey <- x[1]; pred <- x[2]
  c(prey, pred)}



##Compute the filtered (a posteriori) state estimates
ext1 <- dlmExtFilter(y=dataEx1, mod=ex1,
  GGfunction=GGfunction, FFfunction=FFfunction)

#Instead of relying on a numerical method for approximating the Jacobians
#in the EKF, it is also possible to calculate the Jacobians by hand and
#subsequently use these in the EKF.
#WARNING: always use arguments x and k when specifying the GGjacobian
GGjacobian <- function (x, k){
  prey <- x[1]; pred <- x[2]
  c(1 + alpha*dT - beta*pred*dT, -beta*prey*dT,
    delta*pred*dT, 1 + delta*prey*dT - gamma*dT)}

#WARNING: always use arguments x and k when specifying the FFjacobian
FFjacobian <- function (x, k){
  prey <- x[1]; pred <- x[2]
  c(1, 0,
    0, 1)}

#Use these latter Jacobians in the EKF
ext1 <- dlmExtFilter(y=dataEx1, mod=ex1,
  GGfunction=GGfunction, FFfunction=FFfunction,
  GGjacobian=GGjacobian, FFjacobian=FFjacobian)

initial_theta
alpha<-2
beta<-0.005
delta<-0.0025
gamma<-1.5
##
#plot the filtered state estimates
plot(x[,1], x[,2], type="l", lwd=2, xlab="Prey", ylab="Predators",
  xlim=range(yprey), ylim=range(ypred), col=gray(level=.5))
points(yprey, ypred, col="red", cex=.5)
lines(ext1$m[,1], ext1$m[,2], lty=2, col="blue", lwd=2)
legend("topright", pch=c(NA, 1, NA), lty=c(1, NA, 2), lwd=c(2, NA, 2),
  col=c(gray(level=.5), "red", "blue"),
  legend=c("true state", "measured", "filtered state"),
  bty="n", y.intersp=1.2, cex=.7)




#Use these latter Jacobians in the EKF
ext1 <- dlmExtFilter(y=dataEx1, mod=ex1,
  GGfunction=GGfunction, FFfunction=FFfunction,
  GGjacobian=GGjacobian, FFjacobian=FFjacobian)

smoothState <- dlmExtSmooth(ext1)
plot(x[,1], x[,2], type="l", lwd=2, xlab="Prey", ylab="Predators",
  xlim=range(yprey), ylim=range(ypred), col=gray(level=.5))
lines(ext1$m[,1], ext1$m[,2], type="l", lty=2, col="blue")
lines(smoothState$s[,1], smoothState$s[,2], type="l", lty=2, col="darkgreen")
legend("topright", lty=c(1, 2, 2), lwd=c(1, 1, 1),
  col=c("black", "blue", "darkgreen"),
  legend=c("true state", "filtered state", "smoothed state"),
  bty="n", y.intersp=1.2, cex=.7)
