#include "ugg_ssm.h"
#include "ung_ssm.h"
#include "ugg_bsm.h"
#include "ung_bsm.h"
#include "ung_svm.h"
#include "ung_ar1.h"
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
  case 4: {
    ung_ar1 model(clone(model_), 1);
    ugg_ssm approx_model = model.approximate(mode_estimate, max_iter, conv_tol);
    return Rcpp::List::create(Rcpp::Named("y") = approx_model.y,
      Rcpp::Named("H") = approx_model.H);
  } break;
  default: 
    return Rcpp::List::create(Rcpp::Named("y") = 0, Rcpp::Named("H") = 0);
  }
}

// [[Rcpp::export]]
Rcpp::List gaussian_approx_model_nlg(const arma::mat& y, SEXP Z, SEXP H,
  SEXP T, SEXP R, SEXP Zg, SEXP Tg, SEXP a1, SEXP P1,
  const arma::vec& theta, SEXP log_prior_pdf, const arma::vec& known_params,
  const arma::mat& known_tv_params, const unsigned int n_states,
  const unsigned int n_etas,  const arma::uvec& time_varying,
  const unsigned int max_iter, 
  const double conv_tol, const unsigned int iekf_iter) {

  Rcpp::XPtr<nvec_fnPtr> xpfun_Z(Z);
  Rcpp::XPtr<nmat_fnPtr> xpfun_H(H);
  Rcpp::XPtr<nvec_fnPtr> xpfun_T(T);
  Rcpp::XPtr<nmat_fnPtr> xpfun_R(R);
  Rcpp::XPtr<nmat_fnPtr> xpfun_Zg(Zg);
  Rcpp::XPtr<nmat_fnPtr> xpfun_Tg(Tg);
  Rcpp::XPtr<a1_fnPtr> xpfun_a1(a1);
  Rcpp::XPtr<P1_fnPtr> xpfun_P1(P1);
  Rcpp::XPtr<prior_fnPtr> xpfun_prior(log_prior_pdf);
  
  nlg_ssm model(y, *xpfun_Z, *xpfun_H, *xpfun_T, *xpfun_R, *xpfun_Zg, *xpfun_Tg, 
    *xpfun_a1, *xpfun_P1,  theta, *xpfun_prior, known_params, known_tv_params, n_states, n_etas,
    time_varying, 1);
  
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