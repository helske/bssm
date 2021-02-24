#include "model_ssm_mlg.h"
#include "model_ssm_ulg.h"
#include "model_ssm_ung.h"
#include "model_bsm_lg.h"
#include "model_ar1_lg.h"
#include "model_bsm_ng.h"
#include "model_svm.h"
#include "model_ar1_ng.h"
#include "model_ssm_mng.h"
#include "model_ssm_nlg.h"

// [[Rcpp::export]]
double gaussian_loglik(const Rcpp::List model_, const int model_type) {
  
  double loglik = 0;
  switch (model_type) {
  case 0: {
    ssm_mlg model(model_, 1);
    loglik = model.log_likelihood();
  } break;
  case 1: {
    ssm_ulg model(model_, 1);
    loglik = model.log_likelihood();
  } break;
  case 2: {
    bsm_lg model(model_, 1);
    loglik = model.log_likelihood();
  } break;
  case 3: {
    ar1_lg model(model_, 1);
    loglik = model.log_likelihood();
  } break;
  default: loglik = -std::numeric_limits<double>::infinity();
  }
  
  return loglik;
}


// [[Rcpp::export]]
double nongaussian_loglik(const Rcpp::List model_,
  const unsigned int nsim, const unsigned int sampling_method,
  const unsigned int seed, const int model_type) {
  
  arma::vec loglik(2);
  loglik.fill(-std::numeric_limits<double>::infinity());

  switch (model_type) {
  case 0: {
    ssm_mng model(model_, seed);
    arma::cube alpha(model.m, model.n + 1, nsim);
    arma::mat weights(nsim, model.n + 1);
    arma::umat indices(nsim, model.n);
    loglik = model.log_likelihood(sampling_method, nsim, alpha, weights, indices);
  } break;
  case 1: {
    ssm_ung model(model_, seed);
    arma::cube alpha(model.m, model.n + 1, nsim);
    arma::mat weights(nsim, model.n + 1);
    arma::umat indices(nsim, model.n);
    loglik = model.log_likelihood(sampling_method, nsim, alpha, weights, indices);
  } break;
  case 2: {
    bsm_ng model(model_, seed);
    arma::cube alpha(model.m, model.n + 1, nsim);
    arma::mat weights(nsim, model.n + 1);
    arma::umat indices(nsim, model.n);
    loglik = model.log_likelihood(sampling_method, nsim, alpha, weights, indices);
  } break;
  case 3: {
    svm model(model_, seed);
    arma::cube alpha(model.m, model.n + 1, nsim);
    arma::mat weights(nsim, model.n + 1);
    arma::umat indices(nsim, model.n);
    loglik = model.log_likelihood(sampling_method, nsim, alpha, weights, indices);
  } break;
  case 4: {
    ar1_ng model(model_, seed);
    arma::cube alpha(model.m, model.n + 1, nsim);
    arma::mat weights(nsim, model.n + 1);
    arma::umat indices(nsim, model.n);
    loglik = model.log_likelihood(sampling_method, nsim, alpha, weights, indices);
  } break;
  }
  
  return loglik(0);
}


// [[Rcpp::export]]
double nonlinear_loglik(const arma::mat& y, SEXP Z, SEXP H,
  SEXP T, SEXP R, SEXP Zg, SEXP Tg, SEXP a1, SEXP P1,
  const arma::vec& theta, SEXP log_prior_pdf, const arma::vec& known_params,
  const arma::mat& known_tv_params, const unsigned int n_states,
  const unsigned int n_etas,  const arma::uvec& time_varying,
  const unsigned int nsim,
  const unsigned int seed, const unsigned int max_iter,
  const double conv_tol, const unsigned int iekf_iter, const unsigned int method) {
  
  
  Rcpp::XPtr<nvec_fnPtr> xpfun_Z(Z);
  Rcpp::XPtr<nmat_fnPtr> xpfun_H(H);
  Rcpp::XPtr<nvec_fnPtr> xpfun_T(T);
  Rcpp::XPtr<nmat_fnPtr> xpfun_R(R);
  Rcpp::XPtr<nmat_fnPtr> xpfun_Zg(Zg);
  Rcpp::XPtr<nmat_fnPtr> xpfun_Tg(Tg);
  Rcpp::XPtr<a1_fnPtr> xpfun_a1(a1);
  Rcpp::XPtr<P1_fnPtr> xpfun_P1(P1);
  Rcpp::XPtr<prior_fnPtr> xpfun_prior(log_prior_pdf);
  
  ssm_nlg model(y, *xpfun_Z, *xpfun_H, *xpfun_T, *xpfun_R, *xpfun_Zg, *xpfun_Tg,
    *xpfun_a1, *xpfun_P1,  theta, *xpfun_prior, known_params, known_tv_params, n_states, n_etas,
    time_varying, seed);
  
  model.max_iter = max_iter;
  model.conv_tol = conv_tol;
  model.iekf_iter = iekf_iter;
  arma::cube alpha(model.m, model.n + 1, nsim);
  arma::mat weights(nsim, model.n + 1);
  arma::umat indices(nsim, model.n);
  arma::vec loglik = model.log_likelihood(method, nsim, alpha, weights, indices);
  
  return loglik(0);
}
