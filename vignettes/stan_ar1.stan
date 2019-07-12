data {
  int<lower=0> n;             // number of data points
  int<lower=0> y[n];          // time series
}

parameters {
  vector[n] state;
  real<lower=0> sigma;
  real<lower=-1, upper=1> rho;
  real mu;
}

transformed parameters {
  real intercept = (1.0 - rho) * mu;
}

model {
  state[1] ~ normal(mu, sigma / sqrt(1.0 - rho * rho));
  for (t in 2:n)
    state[t] ~ normal(intercept + rho * state[t-1], sigma);
  // priors for theta (rho is uniform)
  sigma ~ normal(0, 1);
  mu ~ normal(0, 10);
  // Poisson likelihood
  y ~ poisson_log(state); 
} 
