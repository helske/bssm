#include "nlg_ssm.h"

// [[Rcpp::export]]
Rcpp::List ukf_nlg(const arma::mat& y, SEXP Z_fn_, SEXP H_fn_, 
  SEXP T_fn_, SEXP R_fn_, SEXP Z_gn_, SEXP T_gn_, SEXP a1_fn_, SEXP P1_fn_, 
  const arma::vec& theta, SEXP log_prior_pdf_, const arma::vec& known_params, 
  const arma::mat& known_tv_params, const unsigned int n_states, 
  const unsigned int n_etas,  const arma::uvec& time_varying, 
  const double alpha, const double beta, 
  const double kappa) {
  
  nlg_ssm model(y, Z_fn_, H_fn_, T_fn_, R_fn_, Z_gn_, T_gn_, a1_fn_, P1_fn_, 
    theta, log_prior_pdf_, known_params, known_tv_params, n_states, n_etas,
    time_varying, 1);

  arma::mat at(model.m, model.n + 1);
  arma::mat att(model.m, model.n);
  arma::cube Pt(model.m, model.m, model.n + 1);
  arma::cube Ptt(model.m, model.m, model.n);
  
  double logLik = model.ukf(at, att, Pt, Ptt, alpha, beta, kappa);
  
  arma::inplace_trans(at);
  arma::inplace_trans(att);
  
  return Rcpp::List::create(
    Rcpp::Named("at") = at,
    Rcpp::Named("att") = att,
    Rcpp::Named("Pt") = Pt,
    Rcpp::Named("Ptt") = Ptt,
    Rcpp::Named("logLik") = logLik);
}
