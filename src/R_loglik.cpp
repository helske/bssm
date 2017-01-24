#include "ugg_ssm.h"
#include "ung_ssm.h"
#include "ugg_bsm.h"
#include "ung_bsm.h"

#include "distr_consts.h"

#include "filter_smoother.h"


// [[Rcpp::export]]
double gaussian_loglik(const Rcpp::List& model_, unsigned int model_type) {
  
  double loglik = 0;
  switch (model_type) {
  case 1: {
    ugg_ssm model(clone(model_), 1);
    loglik = model.log_likelihood();
  } break;
  case 2: {
    ugg_bsm model(clone(model_), 1);
    loglik = model.log_likelihood();
  } break;
  default: loglik = -arma::datum::inf;
  }
  
  return loglik;
}

// [[Rcpp::export]]
double nongaussian_loglik(const Rcpp::List& model_, arma::vec mode_estimate,
  unsigned int nsim_states, unsigned int simulation_method,
  unsigned int seed, unsigned int max_iter, double conv_tol, unsigned int model_type) {
  
  double loglik;
  
  switch (model_type) {
  case 1: {
    ung_ssm model(clone(model_), seed);
    Rcpp::Rcout<<"not yet!";
  } break;
  case 2: {
    ung_bsm model(clone(model_), seed);
    
    if(simulation_method == 3) {
      Rcpp::Rcout<<"not yet!";
    } else {
    ugg_ssm approx_model = model.approximate(mode_estimate, max_iter, conv_tol);
    // compute the log-likelihood of the approximate model
    double gaussian_loglik = approx_model.log_likelihood();
    // compute unnormalized mode-based correction terms 
    // log[g(y_t | ^alpha_t) / ~g(y_t | ^alpha_t)]
    arma::vec scales = model.scaling_factors(approx_model, mode_estimate);
    // compute the constant term
    double const_term = compute_const_term(model, approx_model); 
    // log-likelihood approximation
    double approx_loglik = gaussian_loglik + const_term + arma::accu(scales);
    
    arma::cube alpha(model.m, model.n, nsim_states);
    arma::mat weights(nsim_states, model.n);
    arma::umat indices(nsim_states, model.n - 1);
    
    loglik =  model.psi_filter(approx_model, approx_loglik, scales, 
      nsim_states, alpha, weights, indices);
    }
  } break;
  default: loglik = -arma::datum::inf;
  }
  
  return loglik;
}