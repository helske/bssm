#include "model_ssm_nlg.h"

// [[Rcpp::export]]
Rcpp::List ekf_nlg(const arma::mat& y, SEXP Z, SEXP H,
  SEXP T, SEXP R, SEXP Zg, SEXP Tg, SEXP a1, SEXP P1,
  const arma::vec& theta, SEXP log_prior_pdf, const arma::vec& known_params,
  const arma::mat& known_tv_params, const unsigned int n_states,
  const unsigned int n_etas,  const arma::uvec& time_varying,
  const unsigned int iekf_iter) {

  Rcpp::XPtr<nvec_fnPtr> xpfun_Z(Z);
  Rcpp::XPtr<nmat_fnPtr> xpfun_H(H);
  Rcpp::XPtr<nvec_fnPtr> xpfun_T(T);
  Rcpp::XPtr<nmat_fnPtr> xpfun_R(R);
  Rcpp::XPtr<nmat_fnPtr> xpfun_Zg(Zg);
  Rcpp::XPtr<nmat_fnPtr> xpfun_Tg(Tg);
  Rcpp::XPtr<a1_fnPtr> xpfun_a1(a1);
  Rcpp::XPtr<P1_fnPtr> xpfun_P1(P1);
  Rcpp::XPtr<prior_fnPtr> xpfun_prior(log_prior_pdf);

  ssm_nlg model(y, *xpfun_Z, *xpfun_H, *xpfun_T, *xpfun_R, *xpfun_Zg, 
    *xpfun_Tg, *xpfun_a1, *xpfun_P1,  theta, *xpfun_prior, known_params, 
    known_tv_params, n_states, n_etas, time_varying, 1, iekf_iter);

  arma::mat at(model.m, model.n + 1);
  arma::mat att(model.m, model.n);
  arma::cube Pt(model.m, model.m, model.n + 1);
  arma::cube Ptt(model.m, model.m, model.n);

  double logLik = model.ekf(at, att, Pt, Ptt);
  
  arma::inplace_trans(at);
  arma::inplace_trans(att);

  return Rcpp::List::create(
    Rcpp::Named("at") = at,
    Rcpp::Named("att") = att,
    Rcpp::Named("Pt") = Pt,
    Rcpp::Named("Ptt") = Ptt,
    Rcpp::Named("logLik") = logLik);
}

// [[Rcpp::export]]
Rcpp::List ekf_smoother_nlg(const arma::mat& y, SEXP Z, SEXP H,
  SEXP T, SEXP R, SEXP Zg, SEXP Tg, SEXP a1, SEXP P1,
  const arma::vec& theta, SEXP log_prior_pdf, const arma::vec& known_params,
  const arma::mat& known_tv_params, const unsigned int n_states,
  const unsigned int n_etas,  const arma::uvec& time_varying,
  const unsigned int iekf_iter) {


  Rcpp::XPtr<nvec_fnPtr> xpfun_Z(Z);
  Rcpp::XPtr<nmat_fnPtr> xpfun_H(H);
  Rcpp::XPtr<nvec_fnPtr> xpfun_T(T);
  Rcpp::XPtr<nmat_fnPtr> xpfun_R(R);
  Rcpp::XPtr<nmat_fnPtr> xpfun_Zg(Zg);
  Rcpp::XPtr<nmat_fnPtr> xpfun_Tg(Tg);
  Rcpp::XPtr<a1_fnPtr> xpfun_a1(a1);
  Rcpp::XPtr<P1_fnPtr> xpfun_P1(P1);
  Rcpp::XPtr<prior_fnPtr> xpfun_prior(log_prior_pdf);

  ssm_nlg model(y, *xpfun_Z, *xpfun_H, *xpfun_T, *xpfun_R, *xpfun_Zg, 
    *xpfun_Tg, *xpfun_a1, *xpfun_P1,  theta, *xpfun_prior, known_params, 
    known_tv_params, n_states, n_etas, time_varying, 1, iekf_iter);

  arma::mat alphahat(model.m, model.n + 1);
  arma::cube Vt(model.m, model.m, model.n + 1);
  double loglik = model.ekf_smoother(alphahat, Vt);
  arma::inplace_trans(alphahat);

  return Rcpp::List::create(Rcpp::Named("alphahat") = alphahat, Rcpp::Named("Vt") = Vt,
    Rcpp::Named("logLik") = loglik);
}


// [[Rcpp::export]]
Rcpp::List ekf_fast_smoother_nlg(const arma::mat& y, SEXP Z, SEXP H,
  SEXP T, SEXP R, SEXP Zg, SEXP Tg, SEXP a1, SEXP P1,
  const arma::vec& theta, SEXP log_prior_pdf, const arma::vec& known_params,
  const arma::mat& known_tv_params, const unsigned int n_states,
  const unsigned int n_etas,  const arma::uvec& time_varying,
  const unsigned int iekf_iter) {


  Rcpp::XPtr<nvec_fnPtr> xpfun_Z(Z);
  Rcpp::XPtr<nmat_fnPtr> xpfun_H(H);
  Rcpp::XPtr<nvec_fnPtr> xpfun_T(T);
  Rcpp::XPtr<nmat_fnPtr> xpfun_R(R);
  Rcpp::XPtr<nmat_fnPtr> xpfun_Zg(Zg);
  Rcpp::XPtr<nmat_fnPtr> xpfun_Tg(Tg);
  Rcpp::XPtr<a1_fnPtr> xpfun_a1(a1);
  Rcpp::XPtr<P1_fnPtr> xpfun_P1(P1);
  Rcpp::XPtr<prior_fnPtr> xpfun_prior(log_prior_pdf);

  ssm_nlg model(y, *xpfun_Z, *xpfun_H, *xpfun_T, *xpfun_R, *xpfun_Zg, 
    *xpfun_Tg, *xpfun_a1, *xpfun_P1,  theta, *xpfun_prior, known_params, 
    known_tv_params, n_states, n_etas, time_varying, 1, iekf_iter);

  arma::mat alphahat(model.m, model.n + 1);
  double loglik = model.ekf_fast_smoother(alphahat);
  arma::inplace_trans(alphahat);

  return Rcpp::List::create(Rcpp::Named("alphahat") = alphahat,
    Rcpp::Named("logLik") = loglik);
}
