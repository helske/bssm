#include "model_ssm_ulg.h"
#include "model_bsm_lg.h"
#include "model_ar1_lg.h"
#include "model_ssm_mlg.h"

// [[Rcpp::export]]
Rcpp::List gaussian_smoother(const Rcpp::List model_, const int model_type) {

  arma::vec a1 = Rcpp::as<arma::vec>(model_["a1"]);
  unsigned int m = a1.n_elem;
  unsigned int n;

  if(model_type > 0) {
    arma::vec y = Rcpp::as<arma::vec>(model_["y"]);
    n = y.n_elem;
  } else {
    arma::mat y = Rcpp::as<arma::mat>(model_["y"]);
    n = y.n_rows;
  }

  arma::mat alphahat(m, n + 1);
  arma::cube Vt(m, m, n + 1);

  switch (model_type) {
  case 0: {
    ssm_mlg model(model_, 1);
    model.smoother(alphahat, Vt);
  } break;
  case 1: {
    ssm_ulg model(model_, 1);
    model.smoother(alphahat, Vt);
  } break;
  case 2: {
    bsm_lg model(model_, 1);
    model.smoother(alphahat, Vt);
  } break;
  case 3: {
    ar1_lg model(model_, 1);
    model.smoother(alphahat, Vt);
  } break;
  }

  arma::inplace_trans(alphahat);

  return Rcpp::List::create(
    Rcpp::Named("alphahat") = alphahat,
    Rcpp::Named("Vt") = Vt);
}

// [[Rcpp::export]]
Rcpp::List gaussian_ccov_smoother(const Rcpp::List model_, const int model_type) {

  arma::vec a1 = Rcpp::as<arma::vec>(model_["a1"]);
  unsigned int m = a1.n_elem;
  unsigned int n;

  if(model_type > 0) {
    arma::vec y = Rcpp::as<arma::vec>(model_["y"]);
    n = y.n_elem;
  } else {
    arma::mat y = Rcpp::as<arma::mat>(model_["y"]);
    n = y.n_rows;
  }

  arma::mat alphahat(m, n + 1);
  arma::cube Vt(m, m, n + 1);
  arma::cube Ct(m, m, n + 1);
  switch (model_type) {
  case 0: {
    ssm_mlg model(model_, 1);
  } break;
  case 1: {
    ssm_ulg model(model_, 1);
    model.smoother_ccov(alphahat, Vt, Ct);
  } break;
  case 2: {
    bsm_lg model(model_, 1);
    model.smoother_ccov(alphahat, Vt, Ct);
  } break;
  case 3: {
    ar1_lg model(model_, 1);
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
arma::mat gaussian_fast_smoother(const Rcpp::List model_, const int model_type) {

  switch (model_type) {
  case 0: {
  ssm_mlg model(model_, 1);
  return model.fast_smoother().t();
} break;
  case 1: {
    ssm_ulg model(model_, 1);
    return model.fast_smoother().t();
  } break;
  case 2: {
    bsm_lg model(model_, 1);
    return model.fast_smoother().t();
  } break;
  case 3: {
    ar1_lg model(model_, 1);
    return model.fast_smoother().t();
  } break;
  default:
    return arma::mat(0,0);
  break;
  }
}

// [[Rcpp::export]]
arma::cube gaussian_sim_smoother(const Rcpp::List model_, const unsigned int nsim,
  bool use_antithetic, const unsigned int seed, const int model_type) {

  switch (model_type) {
  case 0: {
  ssm_mlg model(model_, seed);
  return model.simulate_states(nsim);
} break;
  case 1: {
  ssm_ulg model(model_, seed);
  return model.simulate_states(nsim, use_antithetic);
} break;
  case 2: {
    bsm_lg model(model_, seed);
    return model.simulate_states(nsim, use_antithetic);
  } break;
  case 3: {
    ar1_lg model(model_, seed);
    return model.simulate_states(nsim, use_antithetic);
  } break;
  default:
    return arma::cube(0,0,0);
  break;
  }
}

