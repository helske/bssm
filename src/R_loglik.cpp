#include "ugg_ssm.h"
#include "ung_ssm.h"
#include "ugg_bsm.h"
#include "ung_bsm.h"
#include "ng_loglik.h"

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
    loglik = compute_ung_loglik(model, simulation_method, nsim_states,
      mode_estimate, max_iter, conv_tol);
  } break;
  case 2: {
    ung_bsm model(clone(model_), seed);
    loglik = compute_ung_loglik(model, simulation_method, nsim_states,
      mode_estimate, max_iter, conv_tol);
  } break;
  default: loglik = -arma::datum::inf;
  }
  
  return loglik;
}