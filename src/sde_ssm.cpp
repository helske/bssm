#include "sde_ssm.h"
#include "milstein_functions.h"
#include "sample.h"

sde_ssm::sde_ssm(const arma::vec& y, const arma::vec& theta, 
  const double x0, bool positive, const unsigned int seed,
  funcPtr drift_, funcPtr diffusion_, funcPtr ddiffusion_,
  prior_funcPtr log_prior_pdf_, obs_funcPtr log_obs_density_) :
  y(y), theta(theta), x0(x0), n(y.n_elem),
  positive(positive), seed(seed), coarse_engine(seed), engine(seed + 1),
  drift(drift_), diffusion(diffusion_), ddiffusion(ddiffusion_), 
  log_prior_pdf(log_prior_pdf_), log_obs_density(log_obs_density_) {
}

double sde_ssm::bsf_filter(const unsigned int nsim, const unsigned int L, 
  arma::cube& alpha, arma::mat& weights, arma::umat& indices) {
  
  // alpha is  n x 1 x nsim
  for (unsigned int i = 0; i < nsim; i++) {
    alpha(0, 0, i) = milstein(x0, L, 1, theta, drift, diffusion, ddiffusion,
      positive, coarse_engine);
  }

  std::uniform_real_distribution<> unif(0.0, 1.0);
  arma::vec normalized_weights(nsim);
  double loglik = 0.0;
  
  if(arma::is_finite(y(0))) {
    weights.col(0) = log_obs_density(y(0), alpha.tube(0, 0), theta);
    double max_weight = weights.col(0).max();
    weights.col(0) = arma::exp(weights.col(0) - max_weight);
    double sum_weights = arma::accu(weights.col(0));
    
    if(sum_weights > 0.0){
      normalized_weights = weights.col(0) / sum_weights;
    } else {
      return -arma::datum::inf;
    }
    loglik = max_weight + std::log(sum_weights / nsim);
  } else {
    weights.col(0).ones();
    normalized_weights.fill(1.0 / nsim);
  }
  for (unsigned int t = 0; t < (n - 1); t++) {
    
    arma::vec r(nsim);
    for (unsigned int i = 0; i < nsim; i++) {
      r(i) = unif(engine);
    }
    
    indices.col(t) = stratified_sample(normalized_weights, r, nsim);
    
    for (unsigned int i = 0; i < nsim; i++) {
      alpha(0, t + 1, i) = milstein(alpha(0, t, indices(i, t)), L, 1, theta, 
        drift, diffusion, ddiffusion, positive, coarse_engine);
    }
    
    if(arma::is_finite(y(t + 1))) {
      weights.col(t + 1) = log_obs_density(y(t + 1), alpha.tube(0, t + 1), theta);
      
      double max_weight = weights.col(t + 1).max();
      weights.col(t + 1) = arma::exp(weights.col(t + 1) - max_weight);
      double sum_weights = arma::accu(weights.col(t + 1));
      if(sum_weights > 0.0){
        normalized_weights = weights.col(t + 1) / sum_weights;
      } else {
        return -arma::datum::inf;
      }
      loglik += max_weight + std::log(sum_weights / nsim);
    } else {
      weights.col(t + 1).ones();
      normalized_weights.fill(1.0/nsim);
    }
  }
  return loglik;
}

double sde_ssm::coupled_bsf_filter(const unsigned int nsim, 
  const unsigned int L_c, const unsigned int L_f, 
  arma::cube& alpha, arma::mat& weights, arma::umat& indices) {
  
  coarse_engine.seed(seed);
  // alpha is  n x 1 x nsim
  for (unsigned int i = 0; i < nsim; i++) {
    alpha(0, 0, i) = milstein_joint(x0, L_c, L_f, 1, theta, 
      drift, diffusion, ddiffusion, positive, 
      coarse_engine, engine);
  }
  
  std::uniform_real_distribution<> unif(0.0, 1.0);
  arma::vec normalized_weights(nsim);
  double loglik = 0.0;
  
  if(arma::is_finite(y(0))) {
    weights.col(0) = log_obs_density(y(0), alpha.tube(0, 0), theta);
    double max_weight = weights.col(0).max();
    weights.col(0) = arma::exp(weights.col(0) - max_weight);
    double sum_weights = arma::accu(weights.col(0));
    
    if(sum_weights > 0.0){
      normalized_weights = weights.col(0) / sum_weights;
    } else {
      return -arma::datum::inf;
    }
    loglik = max_weight + std::log(sum_weights / nsim);
  } else {
    weights.col(0).ones();
    normalized_weights.fill(1.0 / nsim);
  }
  for (unsigned int t = 0; t < (n - 1); t++) {
    
    arma::vec r(nsim);
    for (unsigned int i = 0; i < nsim; i++) {
      r(i) = unif(engine);
    }
    
    indices.col(t) = stratified_sample(normalized_weights, r, nsim);
    
    for (unsigned int i = 0; i < nsim; i++) {
      alpha(0, t + 1, i) = milstein_joint(alpha(0, t, indices(i, t)), L_c, L_f, 1, theta, 
        drift, diffusion, ddiffusion, positive, coarse_engine, engine);
    }
    
    if(arma::is_finite(y(t + 1))) {
      weights.col(t + 1) = log_obs_density(y(t + 1), alpha.tube(0, t + 1), theta);
      
      double max_weight = weights.col(t + 1).max();
      weights.col(t + 1) = arma::exp(weights.col(t + 1) - max_weight);
      double sum_weights = arma::accu(weights.col(t + 1));
      if(sum_weights > 0.0){
        normalized_weights = weights.col(t + 1) / sum_weights;
      } else {
        return -arma::datum::inf;
      }
      loglik += max_weight + std::log(sum_weights / nsim);
    } else {
      weights.col(t + 1).ones();
      normalized_weights.fill(1.0/nsim);
    }
  }
  return loglik;
}