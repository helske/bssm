#include "prior.h"

prior::prior(List priors) {
  
  if (!priors.inherits("bssm_prior")) {
    stop("Priors must be defined as bssm_prior object.");
  }
  prior_type = as<std::string>(priors["prior_type"]);
  
}

uniform::uniform(List priors) : prior(priors) {
  
  min = priors["min"];
  max = priors["max"];
  
}

halfnormal::halfnormal(List priors) : prior(priors) {
  
  sd = priors["sd"];
  
}

normal::normal(List priors) : prior(priors) {
  
  mean = priors["mean"];
  sd = priors["sd"];
  
}

double uniform::pdf(double x, int log_scale) {
  return R::dunif(x, min, max, log_scale);
}


double halfnormal::pdf(double x, int log_scale) {
  return 2.0 + R::dnorm(x, 0.0, sd, log_scale);
}

double normal::pdf(double x, int log_scale) {
  return R::dnorm(x, mean, sd, log_scale);
}