#include "mgg_ssm.h"
#include "ugg_ssm.h"
#include "ugg_bsm.h"
#include "ugg_ar1.h"
#include "ung_ssm.h"
#include "ung_bsm.h"
#include "ung_svm.h"
#include "ung_ar1.h"
#include "nlg_ssm.h"
#include "distr_consts.h"
#include "filter_smoother.h"
#include "ng_psi_filter.h"
#include "summary.h"

// [[Rcpp::export]]
arma::cube gaussian_psi_smoother(const Rcpp::List& model_,
  const unsigned int nsim_states, const unsigned int seed,
  const int model_type) {
  
  switch (model_type) {
  case 1: {
  ugg_ssm model(clone(model_), seed);
  arma::cube alpha(model.m, model.n + 1, nsim_states);
  model.psi_filter(nsim_states, alpha);
  return alpha;
} break;
  case 2: {
    ugg_bsm model(clone(model_), seed);
    arma::cube alpha(model.m, model.n + 1, nsim_states);
    model.psi_filter(nsim_states, alpha);
    return alpha;
  } break;
  case 3: {
    ugg_ar1 model(clone(model_), seed);
    arma::cube alpha(model.m, model.n + 1, nsim_states);
    model.psi_filter(nsim_states, alpha);
    return alpha;
  } break;
  default:
    return arma::cube(0,0,0);
  break;
  }
}

// [[Rcpp::export]]
Rcpp::List psi_smoother(const Rcpp::List& model_, const arma::vec mode_estimate,
  const unsigned int nsim_states, const unsigned int seed, 
  const unsigned int max_iter, const double conv_tol,
  const int model_type) {
  
  switch (model_type) {
  case 1: {
  ung_ssm model(clone(model_), seed);
  
  arma::cube alpha(model.m, model.n + 1, nsim_states);
  arma::mat weights(nsim_states, model.n + 1);
  arma::umat indices(nsim_states, model.n);
  
  double loglik = compute_ung_psi_filter(model, nsim_states, 
    mode_estimate, max_iter, conv_tol, alpha, weights, indices);
  
  arma::mat alphahat(model.m, model.n + 1);
  arma::cube Vt(model.m, model.m, model.n + 1);
  
  filter_smoother(alpha, indices);
  weighted_summary(alpha, alphahat, Vt, weights.col(model.n));

  arma::inplace_trans(alphahat);
  return Rcpp::List::create(
    Rcpp::Named("alphahat") = alphahat, Rcpp::Named("Vt") = Vt, 
    Rcpp::Named("weights") = weights,
    Rcpp::Named("logLik") = loglik, Rcpp::Named("alpha") = alpha);
} break;
  case 2: {
    ung_bsm model(clone(model_), seed);
    arma::cube alpha(model.m, model.n + 1, nsim_states);
    arma::mat weights(nsim_states, model.n + 1);
    arma::umat indices(nsim_states, model.n);
    
    double loglik = compute_ung_psi_filter(model, nsim_states, 
      mode_estimate, max_iter, conv_tol, alpha, weights, indices);
    
    arma::mat alphahat(model.m, model.n + 1);
    arma::cube Vt(model.m, model.m, model.n + 1);
    
    filter_smoother(alpha, indices);
    weighted_summary(alpha, alphahat, Vt, weights.col(model.n));
   
    arma::inplace_trans(alphahat);
    return Rcpp::List::create(
      Rcpp::Named("alphahat") = alphahat, Rcpp::Named("Vt") = Vt, 
      Rcpp::Named("weights") = weights,
      Rcpp::Named("logLik") = loglik, Rcpp::Named("alpha") = alpha);
  } break;
  case 3: {
    ung_svm model(clone(model_), seed);
    arma::cube alpha(model.m, model.n + 1, nsim_states);
    arma::mat weights(nsim_states, model.n + 1);
    arma::umat indices(nsim_states, model.n);
    
    double loglik = compute_ung_psi_filter(model, nsim_states, 
      mode_estimate, max_iter, conv_tol, alpha, weights, indices);
    
    arma::mat alphahat(model.m, model.n + 1);
    arma::cube Vt(model.m, model.m, model.n + 1);
    
    filter_smoother(alpha, indices);
    weighted_summary(alpha, alphahat, Vt, weights.col(model.n));
   
    arma::inplace_trans(alphahat);
    return Rcpp::List::create(
      Rcpp::Named("alphahat") = alphahat, Rcpp::Named("Vt") = Vt, 
      Rcpp::Named("weights") = weights,
      Rcpp::Named("logLik") = loglik, Rcpp::Named("alpha") = alpha);
  } break;
  case 4: {
    ung_ar1 model(clone(model_), seed);
    arma::cube alpha(model.m, model.n + 1, nsim_states);
    arma::mat weights(nsim_states, model.n + 1);
    arma::umat indices(nsim_states, model.n);
    
    double loglik = compute_ung_psi_filter(model, nsim_states, 
      mode_estimate, max_iter, conv_tol, alpha, weights, indices);
    
    arma::mat alphahat(model.m, model.n + 1);
    arma::cube Vt(model.m, model.m, model.n + 1);
    
    filter_smoother(alpha, indices);
    weighted_summary(alpha, alphahat, Vt, weights.col(model.n));
    
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
  const unsigned int nsim_states, 
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
  
  nlg_ssm model(y, *xpfun_Z, *xpfun_H, *xpfun_T, *xpfun_R, *xpfun_Zg, *xpfun_Tg, 
    *xpfun_a1, *xpfun_P1,  theta, *xpfun_prior, known_params, known_tv_params, n_states, n_etas,
    time_varying, seed);
  
  unsigned int m = model.m;
  unsigned n = model.n;
  
  arma::mat mode_estimate(m, n);
  mgg_ssm approx_model = model.approximate(mode_estimate, max_iter, conv_tol, 
    iekf_iter);
  if(!arma::is_finite(mode_estimate)) {
    Rcpp::stop("Approximation did not converge. ");
  }
  double approx_loglik = approx_model.log_likelihood();
  
  arma::cube alpha(m, n + 1, nsim_states);
  arma::mat weights(nsim_states, n + 1);
  arma::umat indices(nsim_states, n);
  double loglik = model.psi_filter(approx_model, approx_loglik,
    nsim_states, alpha, weights, indices);

  arma::mat alphahat(model.m, model.n + 1);
  arma::cube Vt(model.m, model.m, model.n + 1);
  
    filter_smoother(alpha, indices);
    weighted_summary(alpha, alphahat, Vt, weights.col(n));
  arma::inplace_trans(alphahat);
  return Rcpp::List::create(
    Rcpp::Named("alphahat") = alphahat, Rcpp::Named("Vt") = Vt, 
    Rcpp::Named("weights") = weights,
    Rcpp::Named("logLik") = loglik, Rcpp::Named("alpha") = alpha);
}

