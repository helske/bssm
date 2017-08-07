#include "ugg_ssm.h"
#include "ugg_bsm.h"
#include "lgg_ssm.h"
#include "mgg_ssm.h"

// [[Rcpp::export]]
Rcpp::List gaussian_kfilter(const Rcpp::List& model_, const int model_type) {
  
  arma::vec a1 = Rcpp::as<arma::vec>(model_["a1"]);
  unsigned int m = a1.n_elem;
  unsigned int n;
  
  if(model_type > 0) {
    arma::vec y = Rcpp::as<arma::vec>(model_["y"]);
    n = y.n_elem;
  } else {
    arma::vec y = Rcpp::as<arma::mat>(model_["y"]);
    n = y.n_rows;
  }
  
  arma::mat at(m, n + 1);
  arma::mat att(m, n);
  arma::cube Pt(m, m, n + 1);
  arma::cube Ptt(m, m, n);
  
  double loglik;
  
  switch (model_type) {
  case 1: {
    ugg_ssm model(clone(model_), 1);
    loglik = model.filter(at, att, Pt, Ptt);
  } break;
  case 2: {
    ugg_bsm model(clone(model_), 1);
    loglik = model.filter(at, att, Pt, Ptt);
  } break;
  default: 
    loglik = -arma::datum::inf;
  }
  
  arma::inplace_trans(at);
  arma::inplace_trans(att);
  
  return Rcpp::List::create(
    Rcpp::Named("at") = at,
    Rcpp::Named("att") = att,
    Rcpp::Named("Pt") = Pt,
    Rcpp::Named("Ptt") = Ptt,
    Rcpp::Named("logLik") = loglik);
}

// [[Rcpp::export]]
Rcpp::List general_gaussian_kfilter(const arma::mat& y, SEXP Z_fn_, SEXP H_fn_, 
  SEXP T_fn_, SEXP R_fn_, SEXP a1_fn_, SEXP P1_fn_, 
  const arma::vec& theta, 
  SEXP D_fn_, SEXP C_fn_,
  SEXP log_prior_pdf_, const arma::vec& known_params, 
  const arma::mat& known_tv_params,
  const unsigned int n_states, const unsigned int n_etas) {
  
  lgg_ssm model(y, Z_fn_, H_fn_, T_fn_, R_fn_, a1_fn_, P1_fn_, 
    D_fn_, C_fn_, theta, log_prior_pdf_, known_params, known_tv_params, n_states, n_etas,
    1);
  mgg_ssm mgg_model = model.build_mgg();
  
  unsigned int m = model.m;
  unsigned int n = model.n;
  
  arma::mat at(m, n + 1);
  arma::mat att(m, n);
  arma::cube Pt(m, m, n + 1);
  arma::cube Ptt(m, m, n);
  
  double loglik = mgg_model.filter(at, att, Pt, Ptt);

  arma::inplace_trans(at);
  arma::inplace_trans(att);
  
  return Rcpp::List::create(
    Rcpp::Named("at") = at,
    Rcpp::Named("att") = att,
    Rcpp::Named("Pt") = Pt,
    Rcpp::Named("Ptt") = Ptt,
    Rcpp::Named("logLik") = loglik);
}