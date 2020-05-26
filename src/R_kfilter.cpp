#include "ugg_ssm.h"
#include "ugg_bsm.h"
#include "lgg_ssm.h"
#include "mgg_ssm.h"
#include "ugg_ar1.h"

// [[Rcpp::export]]
Rcpp::List gaussian_kfilter(const Rcpp::List& model_, const int model_type) {

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
  case -1: {
    mgg_ssm model(Rcpp::clone(model_), 1);
    loglik = model.filter(at, att, Pt, Ptt);
  } break;
  case 1: {
    ugg_ssm model(Rcpp::clone(model_), 1);
    loglik = model.filter(at, att, Pt, Ptt);
  } break;
  case 2: {
    ugg_bsm model(Rcpp::clone(model_), 1);
    loglik = model.filter(at, att, Pt, Ptt);
  } break;
  case 3: {
    ugg_ar1 model(Rcpp::clone(model_), 1);
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

// [[Rcpp::export]]
Rcpp::List general_gaussian_kfilter(const arma::mat& y, SEXP Z, SEXP H,
  SEXP T, SEXP R, SEXP a1, SEXP P1,
  const arma::vec& theta,
  SEXP D, SEXP C,
  SEXP log_prior_pdf, const arma::vec& known_params,
  const arma::mat& known_tv_params, const arma::uvec& time_varying,
  const unsigned int n_states, const unsigned int n_etas) {

  Rcpp::XPtr<lmat_fnPtr> xpfun_Z(Z);
  Rcpp::XPtr<lmat_fnPtr> xpfun_H(H);
  Rcpp::XPtr<lmat_fnPtr> xpfun_T(T);
  Rcpp::XPtr<lmat_fnPtr> xpfun_R(R);
  Rcpp::XPtr<a1_fnPtr> xpfun_a1(a1);
  Rcpp::XPtr<P1_fnPtr> xpfun_P1(P1);
  Rcpp::XPtr<lvec_fnPtr> xpfun_D(D);
  Rcpp::XPtr<lvec_fnPtr> xpfun_C(C);
  Rcpp::XPtr<prior_fnPtr> xpfun_prior(log_prior_pdf);

  lgg_ssm model(y, *xpfun_Z, *xpfun_H, *xpfun_T, *xpfun_R, *xpfun_a1, *xpfun_P1,
    *xpfun_D, *xpfun_C, theta, *xpfun_prior, known_params, known_tv_params,
    time_varying, n_states, n_etas, 1);

  unsigned int m = model.m;
  unsigned int n = model.n;

  arma::mat at(m, n + 1);
  arma::mat att(m, n);
  arma::cube Pt(m, m, n + 1);
  arma::cube Ptt(m, m, n);

  double loglik = model.filter(at, att, Pt, Ptt);

  arma::inplace_trans(at);
  arma::inplace_trans(att);

  return Rcpp::List::create(
    Rcpp::Named("at") = at,
    Rcpp::Named("att") = att,
    Rcpp::Named("Pt") = Pt,
    Rcpp::Named("Ptt") = Ptt,
    Rcpp::Named("logLik") = loglik);
}
