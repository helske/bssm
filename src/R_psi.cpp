#include "model_ssm_mlg.h"
#include "model_ssm_ulg.h"
#include "model_bsm_lg.h"
#include "model_ar1_lg.h"
#include "model_ssm_mng.h"
#include "model_ssm_ung.h"
#include "model_bsm_ng.h"
#include "model_svm.h"
#include "model_ar1_ng.h"
#include "model_ssm_nlg.h"
#include "distr_consts.h"
#include "filter_smoother.h"
#include "summary.h"

// [[Rcpp::export]]
arma::cube gaussian_psi_smoother(const Rcpp::List model_,
  const unsigned int nsim, const unsigned int seed,
  const int model_type) {
  
  switch (model_type) {
  case 0: {
  ssm_mlg model(model_, seed);
  arma::cube alpha(model.m, model.n + 1, nsim, arma::fill::zeros);
  model.psi_filter(nsim, alpha);
  return alpha;
} break;
  case 1: {
  ssm_ulg model(model_, seed);
  arma::cube alpha(model.m, model.n + 1, nsim, arma::fill::zeros);
  model.psi_filter(nsim, alpha);
  return alpha;
} break;
  case 2: {
    bsm_lg model(model_, seed);
    arma::cube alpha(model.m, model.n + 1, nsim, arma::fill::zeros);
    model.psi_filter(nsim, alpha);
    return alpha;
  } break;
  case 3: {
    ar1_lg model(model_, seed);
    arma::cube alpha(model.m, model.n + 1, nsim, arma::fill::zeros);
    model.psi_filter(nsim, alpha);
    return alpha;
  } break;
  default:
    return arma::cube(0,0,0);
  break;
  }
}

// [[Rcpp::export]]
Rcpp::List psi_smoother(const Rcpp::List model_,
  const unsigned int nsim, const unsigned int seed,
  const int model_type) {
  
  switch (model_type) {
  case 0: {
  ssm_mng model(model_, seed);
  arma::cube alpha(model.m, model.n + 1, nsim, arma::fill::zeros);
  arma::mat weights(nsim, model.n + 1, arma::fill::zeros);
  arma::umat indices(nsim, model.n, arma::fill::zeros);
  
  double loglik = model.psi_filter(nsim, alpha, weights, indices);
  if (!std::isfinite(loglik)) 
    Rcpp::warning("Particle filtering stopped prematurely due to nonfinite log-likelihood.");
  
  arma::mat alphahat(model.m, model.n + 1);
  arma::cube Vt(model.m, model.m, model.n + 1);
  
  filter_smoother(alpha, indices);
  summary(alpha, alphahat, Vt); // weights are uniform due to extra time point
  
  arma::inplace_trans(alphahat);
  return Rcpp::List::create(
    Rcpp::Named("alphahat") = alphahat, Rcpp::Named("Vt") = Vt,
    Rcpp::Named("weights") = weights,
    Rcpp::Named("logLik") = loglik, Rcpp::Named("alpha") = alpha);
} break;
    
  case 1: {
  ssm_ung model(model_, seed);
  
  arma::cube alpha(model.m, model.n + 1, nsim, arma::fill::zeros);
  arma::mat weights(nsim, model.n + 1, arma::fill::zeros);
  arma::umat indices(nsim, model.n, arma::fill::zeros);
  
  double loglik = model.psi_filter(nsim, alpha, weights, indices);
  if (!std::isfinite(loglik)) 
    Rcpp::warning("Particle filtering stopped prematurely due to nonfinite log-likelihood.");
  
  arma::mat alphahat(model.m, model.n + 1);
  arma::cube Vt(model.m, model.m, model.n + 1);
  
  filter_smoother(alpha, indices);
  summary(alpha, alphahat, Vt); // weights are uniform due to extra time point
  
  arma::inplace_trans(alphahat);
  return Rcpp::List::create(
    Rcpp::Named("alphahat") = alphahat, Rcpp::Named("Vt") = Vt,
    Rcpp::Named("weights") = weights,
    Rcpp::Named("logLik") = loglik, Rcpp::Named("alpha") = alpha);
} break;
  case 2: {
    bsm_ng model(model_, seed);
    arma::cube alpha(model.m, model.n + 1, nsim, arma::fill::zeros);
    arma::mat weights(nsim, model.n + 1, arma::fill::zeros);
    arma::umat indices(nsim, model.n, arma::fill::zeros);
    
    double loglik = model.psi_filter(nsim, alpha, weights, indices);
    if (!std::isfinite(loglik)) 
      Rcpp::warning("Particle filtering stopped prematurely due to nonfinite log-likelihood.");
    
    arma::mat alphahat(model.m, model.n + 1);
    arma::cube Vt(model.m, model.m, model.n + 1);
    
    filter_smoother(alpha, indices);
    summary(alpha, alphahat, Vt); // weights are uniform due to extra time point
    
    arma::inplace_trans(alphahat);
    return Rcpp::List::create(
      Rcpp::Named("alphahat") = alphahat, Rcpp::Named("Vt") = Vt,
      Rcpp::Named("weights") = weights,
      Rcpp::Named("logLik") = loglik, Rcpp::Named("alpha") = alpha);
  } break;
  case 3: {
    svm model(model_, seed);
    arma::cube alpha(model.m, model.n + 1, nsim, arma::fill::zeros);
    arma::mat weights(nsim, model.n + 1, arma::fill::zeros);
    arma::umat indices(nsim, model.n, arma::fill::zeros);
    
    double loglik = model.psi_filter(nsim, alpha, weights, indices);
    if (!std::isfinite(loglik)) 
      Rcpp::warning("Particle filtering stopped prematurely due to nonfinite log-likelihood.");
    
    arma::mat alphahat(model.m, model.n + 1);
    arma::cube Vt(model.m, model.m, model.n + 1);
    
    filter_smoother(alpha, indices);
    summary(alpha, alphahat, Vt); // weights are uniform due to extra time point
    
    arma::inplace_trans(alphahat);
    return Rcpp::List::create(
      Rcpp::Named("alphahat") = alphahat, Rcpp::Named("Vt") = Vt,
      Rcpp::Named("weights") = weights,
      Rcpp::Named("logLik") = loglik, Rcpp::Named("alpha") = alpha);
  } break;
  case 4: {
    ar1_ng model(model_, seed);
    arma::cube alpha(model.m, model.n + 1, nsim, arma::fill::zeros);
    arma::mat weights(nsim, model.n + 1, arma::fill::zeros);
    arma::umat indices(nsim, model.n, arma::fill::zeros);
    
    double loglik = model.psi_filter(nsim, alpha, weights, indices);
    if (!std::isfinite(loglik)) 
      Rcpp::warning("Particle filtering stopped prematurely due to nonfinite log-likelihood.");
    
    arma::mat alphahat(model.m, model.n + 1);
    arma::cube Vt(model.m, model.m, model.n + 1);
    
    filter_smoother(alpha, indices);
    summary(alpha, alphahat, Vt); // weights are uniform due to extra time point
    
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
Rcpp::List psi_smoother_nlg(const arma::mat& y, SEXP Z, SEXP H,
  SEXP T, SEXP R, SEXP Zg, SEXP Tg, SEXP a1, SEXP P1,
  const arma::vec& theta, SEXP log_prior_pdf, const arma::vec& known_params,
  const arma::mat& known_tv_params, const unsigned int n_states,
  const unsigned int n_etas,  const arma::uvec& time_varying,
  const unsigned int nsim,
  const unsigned int seed, const unsigned int max_iter,
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

  ssm_nlg model(y, *xpfun_Z, *xpfun_H, *xpfun_T, *xpfun_R, *xpfun_Zg, *xpfun_Tg,
    *xpfun_a1, *xpfun_P1,  theta, *xpfun_prior, known_params, known_tv_params, n_states, n_etas,
    time_varying, seed, iekf_iter, max_iter, conv_tol);

  unsigned int m = model.m;
  unsigned n = model.n;

  model.approximate();
  if(!arma::is_finite(model.mode_estimate)) {
    Rcpp::warning("Approximation did not converge. ");
  }
  arma::cube alpha(m, n + 1, nsim, arma::fill::zeros);
  arma::mat weights(nsim, n + 1, arma::fill::zeros);
  arma::umat indices(nsim, n, arma::fill::zeros);
  double loglik = model.psi_filter(nsim, alpha, weights, indices);
  if (!std::isfinite(loglik)) 
    Rcpp::warning("Particle filtering stopped prematurely due to nonfinite log-likelihood.");
  
  arma::mat alphahat(model.m, model.n + 1);
  arma::cube Vt(model.m, model.m, model.n + 1);

  filter_smoother(alpha, indices);
  summary(alpha, alphahat, Vt); // weights are uniform due to extra time point
  arma::inplace_trans(alphahat);
  return Rcpp::List::create(
    Rcpp::Named("alphahat") = alphahat, Rcpp::Named("Vt") = Vt,
    Rcpp::Named("weights") = weights,
    Rcpp::Named("logLik") = loglik, Rcpp::Named("alpha") = alpha);
}

