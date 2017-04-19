#include "mgg_ssm.h"
#include "ugg_ssm.h"
#include "ung_ssm.h"
#include "ung_bsm.h"
#include "ung_svm.h"
#include "nlg_ssm.h"
#include "distr_consts.h"
#include "filter_smoother.h"
#include "ng_psi_filter.h"
#include "summary.h"
// [[Rcpp::export]]
Rcpp::List psi_smoother(const Rcpp::List& model_, const arma::vec mode_estimate,
  const unsigned int nsim_states, const unsigned int smoothing_type,
  const unsigned int seed, const unsigned int max_iter, const double conv_tol,
  const int model_type) {
  
  
  switch (model_type) {
  case 1: {
  ung_ssm model(clone(model_), seed);
  
  arma::cube alpha(model.m, model.n, nsim_states);
  arma::mat weights(nsim_states, model.n);
  arma::umat indices(nsim_states, model.n - 1);
  
  double loglik = compute_ung_psi_filter(model, nsim_states, 
    mode_estimate, max_iter, conv_tol, alpha, weights, indices);
  
  arma::mat alphahat(model.m, model.n);
  arma::cube Vt(model.m, model.m, model.n);
  
  if (smoothing_type == 1) {
    filter_smoother(alpha, indices);
    running_weighted_summary(alpha, alphahat, Vt, weights.col(model.n - 1));
  } else {
    Rcpp::stop("Forward-backward smoothing with psi-filter is not yet implemented.");
  }
  arma::inplace_trans(alphahat);
  return Rcpp::List::create(
    Rcpp::Named("alphahat") = alphahat, Rcpp::Named("Vt") = Vt, 
    Rcpp::Named("weights") = weights,
    Rcpp::Named("logLik") = loglik, Rcpp::Named("alpha") = alpha);
} break;
  case 2: {
    ung_bsm model(clone(model_), seed);
    arma::cube alpha(model.m, model.n, nsim_states);
    arma::mat weights(nsim_states, model.n);
    arma::umat indices(nsim_states, model.n - 1);
    
    double loglik = compute_ung_psi_filter(model, nsim_states, 
      mode_estimate, max_iter, conv_tol, alpha, weights, indices);
    
    arma::mat alphahat(model.m, model.n);
    arma::cube Vt(model.m, model.m, model.n);
    
    if (smoothing_type == 1) {
      filter_smoother(alpha, indices);
      running_weighted_summary(alpha, alphahat, Vt, weights.col(model.n - 1));
    } else {
      Rcpp::stop("Forward-backward smoothing with psi-filter is not yet implemented.");
    }
    arma::inplace_trans(alphahat);
    return Rcpp::List::create(
      Rcpp::Named("alphahat") = alphahat, Rcpp::Named("Vt") = Vt, 
      Rcpp::Named("weights") = weights,
      Rcpp::Named("logLik") = loglik, Rcpp::Named("alpha") = alpha);
  } break;
  case 3: {
    ung_svm model(clone(model_), seed);
    arma::cube alpha(model.m, model.n, nsim_states);
    arma::mat weights(nsim_states, model.n);
    arma::umat indices(nsim_states, model.n - 1);
    
    double loglik = compute_ung_psi_filter(model, nsim_states, 
      mode_estimate, max_iter, conv_tol, alpha, weights, indices);
    
    arma::mat alphahat(model.m, model.n);
    arma::cube Vt(model.m, model.m, model.n);
    
    if (smoothing_type == 1) {
      filter_smoother(alpha, indices);
      running_weighted_summary(alpha, alphahat, Vt, weights.col(model.n - 1));
    } else {
      Rcpp::stop("Forward-backward smoothing with psi-filter is not yet implemented.");
    }
    arma::inplace_trans(alphahat);
    return Rcpp::List::create(
      Rcpp::Named("alphahat") = alphahat, Rcpp::Named("Vt") = Vt, 
      Rcpp::Named("weights") = weights,
      Rcpp::Named("logLik") = loglik, Rcpp::Named("alpha") = alpha);
  } break;
  }
  return Rcpp::List::create(Rcpp::Named("error") = 0);
}
// [[Rcpp::export]]
Rcpp::List psi_smoother_nlg(const arma::mat& y, SEXP Z_fn_, SEXP H_fn_, 
  SEXP T_fn_, SEXP R_fn_, SEXP Z_gn_, SEXP T_gn_, SEXP a1_fn_, SEXP P1_fn_, 
  const arma::vec& theta, SEXP log_prior_pdf_, const arma::vec& known_params, 
  const arma::mat& known_tv_params, const unsigned int n_states, 
  const unsigned int n_etas,  const arma::uvec& time_varying,
  const arma::uvec& state_varying, const unsigned int nsim_states, 
  const unsigned int seed, const unsigned int max_iter, const double conv_tol) {
  
  
  nlg_ssm model(y, Z_fn_, H_fn_, T_fn_, R_fn_, Z_gn_, T_gn_, a1_fn_, P1_fn_, 
    theta, log_prior_pdf_, known_params, known_tv_params, n_states, n_etas,
    time_varying, state_varying, seed);
  
  unsigned int m = model.m;
  unsigned n = model.n;
  
  arma::mat mode_estimate(m, n);
  mgg_ssm approx_model = model.approximate(mode_estimate, max_iter, conv_tol);

  double approx_loglik = approx_model.log_likelihood();

  arma::cube alpha(m, n, nsim_states);
  arma::mat weights(nsim_states, n);
  arma::umat indices(nsim_states, n - 1);
  double loglik = model.psi_filter(approx_model, approx_loglik,
    nsim_states, alpha, weights, indices);

  arma::mat alphahat(model.m, model.n);
  arma::cube Vt(model.m, model.m, model.n);
  
//  if (smoothing_type == 1) {
    filter_smoother(alpha, indices);
    running_weighted_summary(alpha, alphahat, Vt, weights.col(n - 1));
  //} else {
   // Rcpp::stop("Forward-backward smoothing with psi-filter is not yet implemented.");
//  }
  arma::inplace_trans(alphahat);
  return Rcpp::List::create(
    Rcpp::Named("alphahat") = alphahat, Rcpp::Named("Vt") = Vt, 
    Rcpp::Named("weights") = weights,
    Rcpp::Named("logLik") = loglik, Rcpp::Named("alpha") = alpha);
}

