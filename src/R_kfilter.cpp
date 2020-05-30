#include "model_ssm_ulg.h"
#include "model_bsm_lg.h"
#include "model_ar1_lg.h"
#include "model_ssm_mlg.h"

// [[Rcpp::export]]
Rcpp::List gaussian_kfilter(const Rcpp::List model_, const unsigned int model_type) {

  arma::vec a1 = Rcpp::as<arma::vec>(model_["a1"]);
  unsigned int m = a1.n_elem;
  unsigned int n;
  
  if(model_type > 0) {
    arma::vec y = Rcpp::as<arma::vec>(model_["y"]);
    n = y.n_elem;
  } else {
    arma::mat y = Rcpp::as<arma::mat>(model_["y"]);
    n = y.n_rows;
  }

  arma::mat at(m, n + 1);
  arma::mat att(m, n);
  arma::cube Pt(m, m, n + 1);
  arma::cube Ptt(m, m, n);

  double loglik;

  switch (model_type) {
  case 0: {
    ssm_mlg model(model_, 1);
    loglik = model.filter(at, att, Pt, Ptt);
  } break;
  case 1: {
    ssm_ulg model(model_, 1);
    loglik = model.filter(at, att, Pt, Ptt);
  } break;
  case 2: {
    bsm_lg model(model_, 1);
    loglik = model.filter(at, att, Pt, Ptt);
  } break;
  case 3: {
    ar1_lg model(model_, 1);
    loglik = model.filter(at, att, Pt, Ptt);
  } break;
  default:
    loglik = -std::numeric_limits<double>::infinity();
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
