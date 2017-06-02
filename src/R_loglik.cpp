#include "mgg_ssm.h"
#include "ugg_ssm.h"
#include "ung_ssm.h"
#include "ugg_bsm.h"
#include "ung_bsm.h"
#include "ung_svm.h"
#include "ng_loglik.h"
#include "nlg_ssm.h"

// [[Rcpp::export]]
double gaussian_loglik(const Rcpp::List& model_, const int model_type) {
  
  double loglik = 0;
  switch (model_type) {
  case -1: {
    mgg_ssm model(clone(model_), 1);
    loglik = model.log_likelihood();
  } break;
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
double nongaussian_loglik(const Rcpp::List& model_, const arma::vec mode_estimate,
  const unsigned int nsim_states, const unsigned int simulation_method,
  const unsigned int seed, const unsigned int max_iter, const double conv_tol,
  const int model_type) {
  
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
  case 3: {
    ung_svm model(clone(model_), seed);
    loglik = compute_ung_loglik(model, simulation_method, nsim_states,
      mode_estimate, max_iter, conv_tol);
  } break;
  default: loglik = -arma::datum::inf;
  }
  
  return loglik;
}


// [[Rcpp::export]]
double nonlinear_loglik(const arma::mat& y, SEXP Z_fn_, SEXP H_fn_, 
  SEXP T_fn_, SEXP R_fn_, SEXP Z_gn_, SEXP T_gn_, SEXP a1_fn_, SEXP P1_fn_, 
  const arma::vec& theta, SEXP log_prior_pdf_, const arma::vec& known_params, 
  const arma::mat& known_tv_params, const unsigned int n_states, 
  const unsigned int n_etas,  const arma::uvec& time_varying,
  const unsigned int nsim_states, 
  const unsigned int seed, const unsigned int max_iter, 
  const double conv_tol, const unsigned int iekf_iter, const unsigned int method) {
  
  nlg_ssm model(y, Z_fn_, H_fn_, T_fn_, R_fn_, Z_gn_, T_gn_, a1_fn_, P1_fn_, 
    theta, log_prior_pdf_, known_params, known_tv_params, n_states, n_etas,
    time_varying, seed);
  
  
  unsigned int m = model.m;
  unsigned n = model.n;
  
  arma::cube alpha(m, n, nsim_states);
  arma::mat weights(nsim_states, n);
  arma::umat indices(nsim_states, n - 1);
  double loglik;
  
  switch (method) {
  case 1: {
    arma::mat mode_estimate(m, n);
    mgg_ssm approx_model = model.approximate(mode_estimate, max_iter, conv_tol, 
      iekf_iter);
    if(!arma::is_finite(mode_estimate)) {
      Rcpp::stop("Approximation did not converge. ");
    }
    double approx_loglik = approx_model.log_likelihood();
    loglik = model.psi_filter(approx_model, approx_loglik,
      nsim_states, alpha, weights, indices);
  } break;
  case 2: {
    loglik = model.bsf_filter(nsim_states, alpha, weights, indices);
    
  } break;
  case 3: {
    loglik = model.aux_filter(nsim_states, alpha, weights, indices);
  } break;
  case 4: {
    loglik = model.aux_filter(nsim_states, alpha, weights, indices);
  } break;
  case 5: {
    arma::mat mode_estimate(m, n);
    mgg_ssm approx_model = model.approximate(mode_estimate, max_iter, conv_tol, 
      iekf_iter);
    if(!arma::is_finite(mode_estimate)) {
      Rcpp::stop("Approximation did not converge. ");
    }
    double approx_loglik = approx_model.log_likelihood();
    loglik = model.df_psi_filter(approx_model, approx_loglik,
      nsim_states, alpha, weights, indices);
  } break;
  default: loglik = -arma::datum::inf;
  }
  
  return loglik;
}

