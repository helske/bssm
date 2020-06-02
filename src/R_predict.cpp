#include "model_ssm_ulg.h"
#include "model_bsm_lg.h"
#include "model_ssm_ung.h"
#include "model_bsm_ng.h"
#include "model_svm.h"
#include "model_ssm_nlg.h"
#include "model_ar1_ng.h"
#include "model_ar1_lg.h"

// [[Rcpp::export]]
Rcpp::List gaussian_predict(const Rcpp::List model_,
  const arma::vec& probs, const arma::mat theta, const arma::mat alpha,
  const arma::uvec& counts, const unsigned int predict_type,
  const bool intervals, const unsigned int seed, const int model_type,
  const unsigned int nsim) {

  switch (model_type) {
  case 1: {
  ssm_ulg model(model_, seed);
  if (intervals) {
    return model.predict_interval(probs, theta, alpha, counts, predict_type);
  } else {
    return Rcpp::List::create(model.predict_sample(theta, alpha, counts,
      predict_type, nsim));
  }
} break;
  case 2: {
    bsm_lg model(model_, seed);
    if (intervals) {
      return model.predict_interval(probs, theta, alpha, counts, predict_type);
    } else {
      return Rcpp::List::create(model.predict_sample(theta, alpha, counts,
        predict_type, nsim));
    }
  } break;
  case 3: {
    ar1_lg model(model_, seed);
    if (intervals) {
      return model.predict_interval(probs, theta, alpha, counts, predict_type);
    } else {
      return Rcpp::List::create(model.predict_sample(theta, alpha, counts,
        predict_type, nsim));
    }
  } break;
  }
  return Rcpp::List::create(Rcpp::Named("error") = std::numeric_limits<double>::infinity());
}

// [[Rcpp::export]]
arma::cube nongaussian_predict(const Rcpp::List model_,
  const arma::vec& probs, const arma::mat& theta, const arma::mat& alpha,
  const arma::uvec& counts, const unsigned int predict_type,
  const unsigned int seed, const unsigned int model_type, const unsigned int nsim) {

  switch (model_type) {
  case 1: {
  ssm_ung model(model_, seed);

  return model.predict_sample(theta, alpha, counts, predict_type, nsim);
} break;
  case 2: {
    bsm_ng model(model_, seed);
    return model.predict_sample(theta, alpha, counts, predict_type, nsim);
  } break;
  case 3: {
    svm model(model_, seed);
    return model.predict_sample(theta, alpha, counts, predict_type, nsim);
  } break;
  case 4: {
    ar1_ng model(model_, seed);
    return model.predict_sample(theta, alpha, counts, predict_type, nsim);
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
  const arma::vec& probs, const arma::mat& theta, const arma::mat& alpha,
  const arma::uvec& counts, const unsigned int predict_type,
  const unsigned int seed, const unsigned int nsim) {


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
    *xpfun_a1, *xpfun_P1, theta.col(0), *xpfun_prior, known_params, known_tv_params, n_states, n_etas,
    time_varying, seed);

  return model.predict_sample(theta, alpha, counts, predict_type, nsim);

}

// [[Rcpp::export]]
Rcpp::List nonlinear_predict_ekf(const arma::mat& y, SEXP Z, SEXP H,
  SEXP T, SEXP R, SEXP Zg, SEXP Tg, SEXP a1, SEXP P1,
  SEXP log_prior_pdf, const arma::vec& known_params,
  const arma::mat& known_tv_params, const arma::uvec& time_varying,
  const unsigned int n_states, const unsigned int n_etas,
  const arma::vec& probs, const arma::mat& theta, const arma::mat& alpha_last, const arma::cube P_last,
  const arma::uvec& counts, const unsigned int predict_type) {

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
    *xpfun_a1, *xpfun_P1, theta.col(0), *xpfun_prior, known_params, known_tv_params, n_states, n_etas,
    time_varying, 1);
  return model.predict_interval(probs, theta,
    alpha_last, P_last, counts, predict_type);
}
