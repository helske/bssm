#include "mgg_ssm.h"
#include "ugg_ssm.h"
#include "ung_ssm.h"
#include "ugg_bsm.h"
#include "ung_bsm.h"
#include "ung_svm.h"
#include "nlg_ssm.h"

#include "filter_smoother.h"

// [[Rcpp::export]]
Rcpp::List bsf(const Rcpp::List& model_,
  const unsigned int nsim_states, const unsigned int seed, 
  bool gaussian, const int model_type) {
  
  if (gaussian) {
    switch (model_type) {
    case 1: {
      ugg_ssm model(clone(model_), seed);
      unsigned int m = model.m;
      unsigned n = model.n;
      
      arma::cube alpha(m, n, nsim_states);
      arma::mat weights(nsim_states, n);
      arma::umat indices(nsim_states, n - 1);
      double loglik = model.bsf_filter(nsim_states, alpha, weights, indices);
      
      return Rcpp::List::create(Rcpp::Named("alpha") = alpha,
        Rcpp::Named("weights") = weights,
        Rcpp::Named("indices") = indices, Rcpp::Named("logLik") = loglik);
    } break;
    case 2: {
        ugg_bsm model(clone(model_), seed);
        unsigned int m = model.m;
        unsigned n = model.n;
        
        arma::cube alpha(m, n, nsim_states);
        arma::mat weights(nsim_states, n);
        arma::umat indices(nsim_states, n - 1);
        double loglik = model.bsf_filter(nsim_states, alpha, weights, indices);
        return Rcpp::List::create(Rcpp::Named("alpha") = alpha,
          Rcpp::Named("weights") = weights,
          Rcpp::Named("indices") = indices, Rcpp::Named("logLik") = loglik);
        
      } break;
    }
  } else {
    switch (model_type) {
    case 1: {
    ung_ssm model(clone(model_), seed);
    unsigned int m = model.m;
    unsigned n = model.n;
    
    arma::cube alpha(m, n, nsim_states);
    arma::mat weights(nsim_states, n);
    arma::umat indices(nsim_states, n - 1);
    double loglik = model.bsf_filter(nsim_states, alpha, weights, indices);
    
    return Rcpp::List::create(Rcpp::Named("alpha") = alpha,
      Rcpp::Named("weights") = weights,
      Rcpp::Named("indices") = indices, Rcpp::Named("logLik") = loglik);
  } break;
    case 2: {
      ung_bsm model(clone(model_), seed);
      unsigned int m = model.m;
      unsigned n = model.n;
      
      arma::cube alpha(m, n, nsim_states);
      arma::mat weights(nsim_states, n);
      arma::umat indices(nsim_states, n - 1);
      double loglik = model.bsf_filter(nsim_states, alpha, weights, indices);
      return Rcpp::List::create(Rcpp::Named("alpha") = alpha,
        Rcpp::Named("weights") = weights,
        Rcpp::Named("indices") = indices, Rcpp::Named("logLik") = loglik);
      
    } break;
    case 3: {
      ung_svm model(clone(model_), seed);
      unsigned int m = model.m;
      unsigned n = model.n;
      
      arma::cube alpha(m, n, nsim_states);
      arma::mat weights(nsim_states, n);
      arma::umat indices(nsim_states, n - 1);
      double loglik = model.bsf_filter(nsim_states, alpha, weights, indices);
      return Rcpp::List::create(Rcpp::Named("alpha") = alpha,
        Rcpp::Named("weights") = weights,
        Rcpp::Named("indices") = indices, Rcpp::Named("logLik") = loglik);
      
    } break;
    }
  }
  return Rcpp::List::create(Rcpp::Named("error") = 0);
}


// [[Rcpp::export]]
Rcpp::List bsf_smoother(const Rcpp::List& model_,
  const unsigned int nsim_states, const unsigned int seed, const int model_type) {
  
  switch (model_type) {
  case 1: {
  ung_ssm model(clone(model_), seed);
  unsigned int m = model.m;
  unsigned n = model.n;
  
  arma::cube alpha(m, n, nsim_states);
  arma::mat weights(nsim_states, n);
  arma::umat indices(nsim_states, n - 1);
  double loglik = model.bsf_filter(nsim_states, alpha, weights, indices);
  filter_smoother(alpha, indices);
  
  return Rcpp::List::create(Rcpp::Named("alpha") = alpha,
    Rcpp::Named("weights") = weights,
    Rcpp::Named("indices") = indices, Rcpp::Named("logLik") = loglik);
} break;
  case 2: {
    ung_bsm model(clone(model_), seed);
    unsigned int m = model.m;
    unsigned n = model.n;
    
    arma::cube alpha(m, n, nsim_states);
    arma::mat weights(nsim_states, n);
    arma::umat indices(nsim_states, n - 1);
    double loglik = model.bsf_filter(nsim_states, alpha, weights, indices);
    filter_smoother(alpha, indices);
    
    return Rcpp::List::create(Rcpp::Named("alpha") = alpha,
      Rcpp::Named("weights") = weights,
      Rcpp::Named("indices") = indices, Rcpp::Named("logLik") = loglik);
    
  } break;
  }
  
  return Rcpp::List::create(Rcpp::Named("error") = 0);
}


// [[Rcpp::export]]
Rcpp::List bsf_nlg(const arma::mat& y, SEXP Z_fn_, SEXP H_fn_, 
  SEXP T_fn_, SEXP R_fn_, SEXP Z_gn_, SEXP T_gn_, SEXP a1_fn_, SEXP P1_fn_, 
  const arma::vec& theta, SEXP log_prior_pdf_, const arma::vec& known_params, 
  const arma::mat& known_tv_params, const unsigned int n_states, 
  const unsigned int n_etas,  const arma::uvec& time_varying,
  const unsigned int nsim_states, const unsigned int seed) {
  
  
  nlg_ssm model(y, Z_fn_, H_fn_, T_fn_, R_fn_, Z_gn_, T_gn_, a1_fn_, P1_fn_, 
    theta, log_prior_pdf_, known_params, known_tv_params, n_states, n_etas,
    time_varying, seed);
  
  unsigned int m = model.m;
  unsigned n = model.n;
  
  arma::cube alpha(m, n, nsim_states);
  arma::mat weights(nsim_states, n);
  arma::umat indices(nsim_states, n - 1);
  double loglik = model.bsf_filter(nsim_states, alpha, weights, indices);
  
  return Rcpp::List::create(Rcpp::Named("alpha") = alpha,
    Rcpp::Named("weights") = weights,
    Rcpp::Named("indices") = indices, Rcpp::Named("logLik") = loglik);
}
