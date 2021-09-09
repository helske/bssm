#include "model_ssm_ulg.h"
#include "model_bsm_lg.h"
#include "model_ssm_ung.h"
#include "model_ssm_mlg.h"
#include "model_ssm_mng.h"
#include "model_bsm_ng.h"
#include "model_svm.h"
#include "model_ssm_nlg.h"
#include "model_ar1_ng.h"
#include "model_ar1_lg.h"

// [[Rcpp::export]]
arma::cube gaussian_predict(const Rcpp::List model_,
   const arma::mat theta, const arma::mat alpha,
  const unsigned int predict_type, const unsigned int seed, 
  const int model_type) {
  
  // needs a deep copy via cloning in ar1 and bsm case, I don't understand why
  switch (model_type) {
    case 0: {
    ssm_mlg model(model_, seed);
    return model.predict_sample(theta, alpha, predict_type, model_["update_fn"]);
    } break;
    case 1: {
      ssm_ulg model(model_, seed);
      return model.predict_sample(theta, alpha, predict_type, model_["update_fn"]);
    } break;
    case 2: {
      bsm_lg model(Rcpp::clone(model_), seed);
      return model.predict_sample(theta, alpha, predict_type);
    } break;
    case 3: {
      ar1_lg model(Rcpp::clone(model_), seed);
      return model.predict_sample(theta, alpha, predict_type);
    } break;
  }
  return arma::cube(0,0,0);
}

// [[Rcpp::export]]
arma::cube nongaussian_predict(const Rcpp::List model_,
  const arma::mat& theta, const arma::mat& alpha,
  const unsigned int predict_type,const unsigned int seed, 
  const unsigned int model_type) {
  
  // needs a deep copy via cloning in ar1, bsm and svm cases, I don't understand why
  switch (model_type) {
  case 0: {
  ssm_mng model(model_, seed);
  return model.predict_sample(theta, alpha, predict_type, model_["update_fn"]);
} break;
  case 1: {
  ssm_ung model(model_, seed);
  return model.predict_sample(theta, alpha, predict_type, model_["update_fn"]);
} break;
  case 2: {
    bsm_ng model(Rcpp::clone(model_), seed);
    return model.predict_sample(theta, alpha, predict_type);
  } break;
  case 3: {
    svm model(Rcpp::clone(model_), seed);
    return model.predict_sample(theta, alpha, predict_type);
  } break;
  case 4: {
    ar1_ng model(Rcpp::clone(model_), seed);
    return model.predict_sample(theta, alpha, predict_type);
  } break;
  }
  return arma::cube(0,0,0);
}

// [[Rcpp::export]]
arma::cube nonlinear_predict(const arma::mat& y, SEXP Z, SEXP H,
  SEXP T, SEXP R, SEXP Zg, SEXP Tg, SEXP a1, SEXP P1,
  SEXP log_prior_pdf, const arma::vec& known_params,
  const arma::mat& known_tv_params, const arma::uvec& time_varying,
  const unsigned int n_states, const unsigned int n_etas,
  const arma::mat& theta, const arma::mat& alpha,
  const unsigned int predict_type,
  const unsigned int seed) {
  
  
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
    *xpfun_a1, *xpfun_P1, theta.col(0), *xpfun_prior, known_params, known_tv_params, 
    n_states, n_etas, time_varying, seed);
  
  return model.predict_sample(theta, alpha, predict_type);
  
}
