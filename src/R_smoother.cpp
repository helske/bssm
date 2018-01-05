#include "ugg_ssm.h"
#include "ugg_bsm.h"
#include "mgg_ssm.h"
#include "lgg_ssm.h"

// [[Rcpp::export]]
Rcpp::List gaussian_smoother(const Rcpp::List& model_, const int model_type) {
  
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
  
  arma::mat alphahat(m, n + 1);
  arma::cube Vt(m, m, n + 1);
  
  switch (model_type) {
  case -1: {
    mgg_ssm model(clone(model_), 1);
    model.smoother(alphahat, Vt);
  } break;
  case 1: {
    ugg_ssm model(clone(model_), 1);
    model.smoother(alphahat, Vt);
  } break;
  case 2: {
    ugg_bsm model(clone(model_), 1);
    model.smoother(alphahat, Vt);
  } break;
  }
  
  arma::inplace_trans(alphahat);
  
  return Rcpp::List::create(
    Rcpp::Named("alphahat") = alphahat,
    Rcpp::Named("Vt") = Vt);
}

// [[Rcpp::export]]
Rcpp::List general_gaussian_smoother(const arma::mat& y, SEXP Z, SEXP H, 
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
  mgg_ssm mgg_model = model.build_mgg();
  
  unsigned int m = model.m;
  unsigned int n = model.n;
  
  arma::mat alphahat(m, n + 1);
  arma::cube Vt(m, m, n + 1);
  
  mgg_model.smoother(alphahat, Vt);
  
  arma::inplace_trans(alphahat);
  
  return Rcpp::List::create(
    Rcpp::Named("alphahat") = alphahat,
    Rcpp::Named("Vt") = Vt);
}
  
// [[Rcpp::export]]
Rcpp::List gaussian_ccov_smoother(const Rcpp::List& model_, const int model_type) {
  
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
  
  arma::mat alphahat(m, n + 1);
  arma::cube Vt(m, m, n + 1);
  arma::cube Ct(m, m, n + 1);
  switch (model_type) {
  case -1: {
    mgg_ssm model(clone(model_), 1);
    Rcpp::Rcout<<"not yet"<<std::endl;
    // model.smoother(alphahat, Vt);
  } break;
  case 1: {
    ugg_ssm model(clone(model_), 1);
    model.smoother_ccov(alphahat, Vt, Ct);
  } break;
  case 2: {
    ugg_bsm model(clone(model_), 1);
    model.smoother_ccov(alphahat, Vt, Ct);
  } break;
  }
  
  arma::inplace_trans(alphahat);
  
  return Rcpp::List::create(
    Rcpp::Named("alphahat") = alphahat,
    Rcpp::Named("Vt") = Vt,
    Rcpp::Named("Ct") = Ct);
}



// [[Rcpp::export]]
arma::mat gaussian_fast_smoother(const Rcpp::List& model_, const int model_type) {
  
  switch (model_type) {
  case -1: {
  mgg_ssm model(clone(model_), 1);
  return model.fast_smoother().t();
} break;
  case 1: {
    ugg_ssm model(clone(model_), 1);
    return model.fast_smoother().t();
  } break;
  case 2: {
    ugg_bsm model(clone(model_), 1);
    return model.fast_smoother().t();
  } break;
  default:
    return arma::mat(0,0);
  break;
  }
}

// [[Rcpp::export]]
arma::cube gaussian_sim_smoother(const Rcpp::List& model_, const unsigned int nsim, 
  bool use_antithetic, const unsigned int seed, const int model_type) {
  
  switch (model_type) {
  case 1: {
  ugg_ssm model(clone(model_), seed);
  return model.simulate_states(nsim, use_antithetic);
} break;
  case 2: {
    ugg_bsm model(clone(model_), seed);
    return model.simulate_states(nsim, use_antithetic);
  } break;
  default:
    return arma::cube(0,0,0);
  break;
  }
}

// [[Rcpp::export]]
arma::cube general_gaussian_sim_smoother(const arma::mat& y, SEXP Z, SEXP H, 
  SEXP T, SEXP R, SEXP a1, SEXP P1, 
  const arma::vec& theta, 
  SEXP D, SEXP C,
  SEXP log_prior_pdf, const arma::vec& known_params, 
  const arma::mat& known_tv_params, const arma::uvec& time_varying,
  const unsigned int n_states, const unsigned int n_etas, const unsigned int nsim, 
  bool use_antithetic, const unsigned int seed) {
  
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
    time_varying, n_states, n_etas, seed);
  mgg_ssm mgg_model = model.build_mgg();
  
  return mgg_model.simulate_states();
}