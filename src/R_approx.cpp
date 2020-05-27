#include "model_ung_ssm.h"
#include "model_ung_bsm.h"
#include "model_ung_svm.h"
#include "model_ung_ar1.h"
#include "model_mng_ssm.h"
#include "model_nlg_ssm.h"

// [[Rcpp::export]]
Rcpp::List gaussian_approx_model(const Rcpp::List& model_, const int model_type) {

  switch (model_type) {
  case 1: {
    ung_ssm model(Rcpp::clone(model_), 1);
    model.approximate();
    return Rcpp::List::create(Rcpp::Named("y") = model.approx_model.y,
      Rcpp::Named("H") = model.approx_model.H);
  } break;
  case 2: {
    ung_bsm model(Rcpp::clone(model_), 1);
    model.approximate();
    return Rcpp::List::create(Rcpp::Named("y") = model.approx_model.y,
      Rcpp::Named("H") = model.approx_model.H);
  } break;
  case 3: {
    ung_svm model(Rcpp::clone(model_), 1);
    model.approximate();
    return Rcpp::List::create(Rcpp::Named("y") = model.approx_model.y,
      Rcpp::Named("H") = model.approx_model.H);
  } break;
  case 4: {
    ung_ar1 model(Rcpp::clone(model_), 1);
    model.approximate();
    return Rcpp::List::create(Rcpp::Named("y") = model.approx_model.y,
      Rcpp::Named("H") = model.approx_model.H);
  } break;
  default:
    return Rcpp::List::create(Rcpp::Named("y") = 0, Rcpp::Named("H") = 0);
  }
}

// [[Rcpp::export]]
Rcpp::List gaussian_approx_model_nlg(const arma::mat& y, SEXP Z, SEXP H,
  SEXP T, SEXP R, SEXP Zg, SEXP Tg, SEXP a1, SEXP P1,
  const arma::vec& theta, SEXP log_prior_pdf, const arma::vec& known_params,
  const arma::mat& known_tv_params, const unsigned int n_states,
  const unsigned int n_etas,  const arma::uvec& time_varying,
  const unsigned int max_iter,
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
    time_varying, 1, iekf_iter, max_iter, conv_tol);

  model.approximate();
  if(!arma::is_finite(model.mode_estimate)) {
    Rcpp::warning("Approximation did not converge. ");
  }
  return Rcpp::List::create(Rcpp::Named("y") = model.approx_model.y,
    Rcpp::Named("D") = model.approx_model.D,
    Rcpp::Named("Z") = model.approx_model.Z, Rcpp::Named("H") = model.approx_model.H,
    Rcpp::Named("C") = model.approx_model.C, Rcpp::Named("T") = model.approx_model.T,
    Rcpp::Named("R") = model.approx_model.R, Rcpp::Named("a1") = model.approx_model.a1,
    Rcpp::Named("P1") = model.approx_model.P1);
}
