#include "mgg_ssm.h"
#include "nlg_ssm.h"

#include "filter_smoother.h"


// [[Rcpp::export]]
Rcpp::List psi_filter_nlg(const arma::mat& y, SEXP Z_fn_, SEXP H_fn_, 
  SEXP T_fn_, SEXP R_fn_, SEXP Z_gn_, SEXP T_gn_, SEXP a1_fn_, SEXP P1_fn_, 
  const arma::vec& theta, SEXP log_prior_pdf_, const arma::vec& known_params, 
  const arma::mat& known_tv_params, const unsigned int n_states, 
  const unsigned int n_etas,  const arma::uvec& time_varying,
  const arma::uvec& state_varying, const unsigned int nsim_states, 
  const unsigned int seed, arma::mat initial_mode, 
  const unsigned int max_iter, const double conv_tol) {
  
  
  nlg_ssm model(y, Z_fn_, H_fn_, T_fn_, R_fn_, Z_gn_, T_gn_, a1_fn_, P1_fn_, 
    theta, log_prior_pdf_, known_params, known_tv_params, n_states, n_etas,
    time_varying, state_varying, seed);
  
  unsigned int m = model.m;
  unsigned n = model.n;
  
  mgg_ssm approx_model = model.approximate(initial_mode, max_iter, conv_tol);
  
  double approx_loglik = approx_model.log_likelihood();
  
  arma::vec scales = model.scaling_factors(approx_model, initial_mode);
  arma::cube alpha(m, n, nsim_states);
  arma::mat weights(nsim_states, n);
  arma::umat indices(nsim_states, n - 1);
  double loglik = model.psi_filter(approx_model, approx_loglik, scales,
    nsim_states, alpha, weights, indices);
  filter_smoother(alpha, indices);
  return Rcpp::List::create(Rcpp::Named("alpha") = alpha,
    Rcpp::Named("weights") = weights,
    Rcpp::Named("indices") = indices, Rcpp::Named("logLik") = loglik);
}

