#include "mgg_ssm.h"
#include "ugg_ssm.h"
#include "ung_ssm.h"
#include "ugg_bsm.h"
#include "ung_bsm.h"
#include "ung_svm.h"
#include "ng_loglik.h"
#include "nlg_ssm.h"
#include "lgg_ssm.h"

// [[Rcpp::export]]
double gaussian_loglik(const Rcpp::List& model_, const int model_type) {
  
  double loglik = 0;
  switch (model_type) {
  case -1: {
    mgg_ssm model(clone(model_), 1);
    loglik = model.log_likelihood();
  } break;
  case 1: {
    ugg_ssm model(clone(model_), 1);
    loglik = model.log_likelihood();
  } break;
  case 2: {
    ugg_bsm model(clone(model_), 1);
    loglik = model.log_likelihood();
  } break;
  default: loglik = -std::numeric_limits<double>::infinity();
  }
  
  return loglik;
}

// [[Rcpp::export]]
double nongaussian_loglik(const Rcpp::List& model_, const arma::vec mode_estimate,
  const unsigned int nsim_states, const unsigned int simulation_method,
  const unsigned int seed, const unsigned int max_iter, const double conv_tol,
  const int model_type) {
  
  double loglik;
  
  switch (model_type) {
  case 1: {
    ung_ssm model(clone(model_), seed);
    loglik = compute_ung_loglik(model, simulation_method, nsim_states,
      mode_estimate, max_iter, conv_tol);
  } break;
  case 2: {
    ung_bsm model(clone(model_), seed);
    loglik = compute_ung_loglik(model, simulation_method, nsim_states,
      mode_estimate, max_iter, conv_tol);
  } break;
  case 3: {
    ung_svm model(clone(model_), seed);
    loglik = compute_ung_loglik(model, simulation_method, nsim_states,
      mode_estimate, max_iter, conv_tol);
  } break;
  default: loglik = -std::numeric_limits<double>::infinity();
  }
  
  return loglik;
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
  
  
  unsigned int m = model.m;
  unsigned n = model.n;
  
  arma::cube alpha(m, n, nsim_states);
  arma::mat weights(nsim_states, n);
  arma::umat indices(nsim_states, n - 1);
  double loglik;
  
  switch (method) {
  case 1: {
    arma::mat mode_estimate(m, n);
    mgg_ssm approx_model = model.approximate(mode_estimate, max_iter, conv_tol, 
      iekf_iter);
    if(!arma::is_finite(mode_estimate)) {
      Rcpp::stop("Approximation did not converge. ");
    }
    double approx_loglik = approx_model.log_likelihood();
    loglik = model.psi_filter(approx_model, approx_loglik,
      nsim_states, alpha, weights, indices);
  } break;
  case 2: {
    loglik = model.bsf_filter(nsim_states, alpha, weights, indices);
    
  } break;
  case 3: {
    loglik = model.aux_filter(nsim_states, alpha, weights, indices);
  } break;
  case 4: {
    loglik = model.aux_filter(nsim_states, alpha, weights, indices);
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
  const arma::mat& known_tv_params,
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
    *xpfun_D, *xpfun_C, theta, *xpfun_prior, known_params, known_tv_params, n_states, n_etas,
    1);
  Rcpp::Rcout<<"lgg_ssm ok"<<std::endl;
  mgg_ssm mgg_model = model.build_mgg();
  
  Rcpp::Rcout<<"mgg_ssm ok"<<std::endl;
  return mgg_model.log_likelihood();
  
}
