#include "mgg_ssm.h"
#include "ugg_ssm.h"
#include "ung_ssm.h"
#include "ugg_bsm.h"
#include "ugg_ar1.h"
#include "ung_bsm.h"
#include "ung_svm.h"
#include "ung_ar1.h"
#include "mng_ssm.h"
#include "nlg_ssm.h"
#include "lgg_ssm.h"

// [[Rcpp::export]]
double gaussian_loglik(const Rcpp::List& model_, const int model_type) {
  
  double loglik = 0;
  switch (model_type) {
  case -1: {
    mgg_ssm model(Rcpp::clone(model_), 1);
    loglik = model.log_likelihood();
  } break;
  case 1: {
    ugg_ssm model(Rcpp::clone(model_), 1);
    loglik = model.log_likelihood();
  } break;
  case 2: {
    ugg_bsm model(Rcpp::clone(model_), 1);
    loglik = model.log_likelihood();
  } break;
  case 3: {
    ugg_ar1 model(Rcpp::clone(model_), 1);
    loglik = model.log_likelihood();
  } break;
  default: loglik = -std::numeric_limits<double>::infinity();
  }
  
  return loglik;
}


// [[Rcpp::export]]
double general_gaussian_loglik(const arma::mat& y, SEXP Z, SEXP H,
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
  
  return model.mgg_model.log_likelihood();
  
}

// [[Rcpp::export]]
double nongaussian_loglik(const Rcpp::List& model_,
  const unsigned int nsim_states, const unsigned int simulation_method,
  const unsigned int seed, const unsigned int max_iter, const double conv_tol,
  const int model_type) {
  
  arma::vec loglik(2);

  switch (model_type) {
  case 1: {
    ung_ssm model(Rcpp::clone(model_), seed);
    model.max_iter = max_iter;
    model.conv_tol = conv_tol;
    arma::cube alpha(model.m, model.n + 1, nsim_states);
    arma::mat weights(nsim_states, model.n + 1);
    arma::umat indices(nsim_states, model.n);
    loglik = model.log_likelihood(simulation_method, nsim_states, alpha, weights, indices);
  } break;
  case 2: {
    ung_bsm model(Rcpp::clone(model_), seed);
    model.max_iter = max_iter;
    model.conv_tol = conv_tol;
    arma::cube alpha(model.m, model.n + 1, nsim_states);
    arma::mat weights(nsim_states, model.n + 1);
    arma::umat indices(nsim_states, model.n);
    loglik = model.log_likelihood(simulation_method, nsim_states, alpha, weights, indices);
  } break;
  case 3: {
    ung_svm model(Rcpp::clone(model_), seed);
    model.max_iter = max_iter;
    model.conv_tol = conv_tol;
    arma::cube alpha(model.m, model.n + 1, nsim_states);
    arma::mat weights(nsim_states, model.n + 1);
    arma::umat indices(nsim_states, model.n);
    loglik = model.log_likelihood(simulation_method, nsim_states, alpha, weights, indices);
  } break;
  case 4: {
    ung_ar1 model(Rcpp::clone(model_), seed);
    model.max_iter = max_iter;
    model.conv_tol = conv_tol;
    arma::cube alpha(model.m, model.n + 1, nsim_states);
    arma::mat weights(nsim_states, model.n + 1);
    arma::umat indices(nsim_states, model.n);
    loglik = model.log_likelihood(simulation_method, nsim_states, alpha, weights, indices);
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
  const unsigned int nsim_states,
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
  
  nlg_ssm model(y, *xpfun_Z, *xpfun_H, *xpfun_T, *xpfun_R, *xpfun_Zg, *xpfun_Tg,
    *xpfun_a1, *xpfun_P1,  theta, *xpfun_prior, known_params, known_tv_params, n_states, n_etas,
    time_varying, seed);
  
  model.max_iter = max_iter;
  model.conv_tol = conv_tol;
  model.iekf_iter = iekf_iter;
  unsigned int m = model.m;
  unsigned n = model.n;
  arma::cube alpha(model.m, model.n + 1, nsim_states);
  arma::mat weights(nsim_states, model.n + 1);
  arma::umat indices(nsim_states, model.n);
  arma::vec loglik = model.log_likelihood(method, nsim_states, alpha, weights, indices);
  
  return loglik(0);
}
