#include "ugg_ssm.h"
#include "ugg_bsm.h"
#include "ung_ssm.h"
#include "ung_bsm.h"
#include "ung_svm.h"
#include "nlg_ssm.h"

// [[Rcpp::export]]
Rcpp::List gaussian_predict(const Rcpp::List& model_,
  const arma::vec& probs, const arma::mat theta, const arma::mat alpha, 
  const arma::uvec& counts, const unsigned int predict_type,
  const bool intervals, const unsigned int seed, const int model_type) {
  
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
  
  switch (model_type) {
  case 1: {
    ugg_ssm model(clone(model_), seed, 0, 0, 0, 0);
    if (intervals) {
      return model.predict_interval(probs, theta, alpha, counts, predict_type);
    } else {
      return Rcpp::List::create(model.predict_sample(theta, alpha, counts,
        predict_type < 3));
    }
  } break;
  case 2: {
    ugg_bsm model(clone(model_), seed);
    if (intervals) {
      return model.predict_interval(probs, theta, alpha, counts, predict_type);
    } else {
      return Rcpp::List::create(model.predict_sample(theta, alpha, counts, 
        predict_type < 3));
    }
  } break;
  }
  return Rcpp::List::create(Rcpp::Named("error") = arma::datum::inf);
}

// [[Rcpp::export]]
arma::cube nongaussian_predict(const Rcpp::List& model_,
  const arma::vec& probs, const arma::mat& theta, const arma::mat& alpha, 
  const arma::uvec& counts, const unsigned int predict_type, 
  const unsigned int seed, const int model_type) {
  
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
  
  switch (model_type) {
  case 1: {
    ung_ssm model(clone(model_), seed, 0, 0, 0);
    return model.predict_sample(theta, alpha, counts, predict_type);
  } break;
  case 2: {
    ung_bsm model(clone(model_), seed);
    return model.predict_sample(theta, alpha, counts, predict_type);
  } break;
  case 3: {
    ung_svm model(clone(model_), seed);
    return model.predict_sample(theta, alpha, counts, predict_type);
  } break;
  }
  return arma::cube(0,0,0);
}

// [[Rcpp::export]]
arma::cube nonlinear_predict(const arma::mat& y, SEXP Z_fn_, SEXP H_fn_, 
  SEXP T_fn_, SEXP R_fn_, SEXP Z_gn_, SEXP T_gn_, SEXP a1_fn_, SEXP P1_fn_, 
  SEXP log_prior_pdf_, const arma::vec& known_params, 
  const arma::mat& known_tv_params, const arma::uvec& time_varying, 
  const unsigned int n_states, const unsigned int n_etas,
  const arma::vec& probs, const arma::mat& theta, const arma::mat& alpha, 
  const arma::uvec& counts, const unsigned int predict_type, const unsigned int seed) {
  
  nlg_ssm model(y, Z_fn_, H_fn_, T_fn_, R_fn_, Z_gn_, T_gn_, a1_fn_, P1_fn_, 
    theta.col(0), log_prior_pdf_, known_params, known_tv_params, n_states, n_etas,
    time_varying, seed);
  return model.predict_sample(theta, alpha, counts, predict_type);
}

