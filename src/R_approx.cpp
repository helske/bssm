#include "ugg_ssm.h"
#include "ung_ssm.h"
#include "ugg_bsm.h"
#include "ung_bsm.h"
#include "ung_svm.h"
#include "mgg_ssm.h"
#include "nlg_ssm.h"

// [[Rcpp::export]]
Rcpp::List gaussian_approx_model(const Rcpp::List& model_, 
  arma::vec mode_estimate, const unsigned int max_iter, 
  const double conv_tol, const int model_type) {
  
  switch (model_type) {
  case 1: {
    ung_ssm model(clone(model_), 1);
    ugg_ssm approx_model = model.approximate(mode_estimate, max_iter, conv_tol);
    return Rcpp::List::create(Rcpp::Named("y") = approx_model.y,
      Rcpp::Named("H") = approx_model.H);
  } break;
  case 2: {
    ung_bsm model(clone(model_), 1);
    ugg_ssm approx_model = model.approximate(mode_estimate, max_iter, conv_tol);
    return Rcpp::List::create(Rcpp::Named("y") = approx_model.y,
      Rcpp::Named("H") = approx_model.H);
  } break;
  case 3: {
    ung_svm model(clone(model_), 1);
    ugg_ssm approx_model = model.approximate(mode_estimate, max_iter, conv_tol);
    return Rcpp::List::create(Rcpp::Named("y") = approx_model.y,
      Rcpp::Named("H") = approx_model.H);
  } break;
  default: 
    return Rcpp::List::create(Rcpp::Named("y") = 0, Rcpp::Named("H") = 0);
  }
}

// [[Rcpp::export]]
Rcpp::List gaussian_approx_model_nlg(const arma::mat& y, SEXP Z_fn_, SEXP H_fn_,
  SEXP T_fn_, SEXP R_fn_, SEXP Z_gn_, SEXP T_gn_, SEXP a1_fn_, SEXP P1_fn_,
  const arma::vec& theta, SEXP log_prior_pdf_, const arma::vec& known_params,
  const arma::mat& known_tv_params, const unsigned int n_states,
  const unsigned int n_etas,  const arma::uvec& time_varying,
  const arma::uvec& state_varying, const unsigned int max_iter, 
  const double conv_tol, const unsigned int iekf_iter) {

  nlg_ssm model(y, Z_fn_, H_fn_, T_fn_, R_fn_, Z_gn_, T_gn_, a1_fn_, P1_fn_,
    theta, log_prior_pdf_, known_params, known_tv_params, n_states, n_etas,
    time_varying, state_varying, 1);
  arma::mat mode_estimate(model.m, model.n);
  mgg_ssm approx_model = model.approximate(mode_estimate, max_iter, 
    conv_tol, iekf_iter);
  if(!arma::is_finite(mode_estimate)) {
    Rcpp::warning("Approximation did not converge. ");
  }
  return Rcpp::List::create(Rcpp::Named("y") = approx_model.y,
    Rcpp::Named("D") = approx_model.D,
    Rcpp::Named("Z") = approx_model.Z, Rcpp::Named("H") = approx_model.H,
    Rcpp::Named("C") = approx_model.C, Rcpp::Named("T") = approx_model.T,
    Rcpp::Named("R") = approx_model.R, Rcpp::Named("a1") = approx_model.a1,
    Rcpp::Named("P1") = approx_model.P1);
}