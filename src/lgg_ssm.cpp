#include "lgg_ssm.h"
#include "mgg_ssm.h"

lgg_ssm::lgg_ssm(const arma::mat& y_, SEXP Z_fn_, SEXP H_fn_, SEXP T_fn_, SEXP R_fn_, 
  SEXP a1_fn_, SEXP P1_fn_,
  const arma::vec& theta, SEXP log_prior_pdf_, const arma::vec& known_params,
  const arma::mat& known_tv_params, const unsigned int m, const unsigned int k,
  const unsigned int seed) :
  y(y_), Z_fn(mat_fn2(Z_fn_)), H_fn(mat_varfn(H_fn_)), T_fn(mat_fn2(T_fn_)), 
  R_fn(mat_varfn(R_fn_)),
  a1_fn(vec_initfn(a1_fn_)), P1_fn(mat_initfn(P1_fn_)), theta(theta), 
  log_prior_pdf(log_prior_pdf_), known_params(known_params), 
  known_tv_params(known_tv_params), m(m), k(k), n(y.n_cols),  p(y.n_rows),
  seed(seed), engine(seed), zero_tol(1e-8) {
}

mgg_ssm lgg_ssm::build_mgg() {
  
  arma::vec a1 = a1_fn.eval(theta, known_params);
  arma::mat P1 = P1_fn.eval(theta, known_params);
  arma::cube Z(p, m, n);
  for (unsigned int t = 0; t < n; t++) {
    Z.slice(t) = Z_fn.eval(t, theta, known_params, known_tv_params);
  }
  arma::cube H(p, p, n);
  for (unsigned int t = 0; t < n; t++) {
    H.slice(t) = H_fn.eval(t, theta, known_params, known_tv_params); //at??
  }
  arma::cube T(m, m, n);
  for (unsigned int t = 0; t < n; t++) {
    T.slice(t) = T_fn.eval(t, theta, known_params, known_tv_params);
  }
  
  arma::cube R(m, k, n);
  for (unsigned int t = 0; t < n; t++) {
    R.slice(t) = R_fn.eval(t, theta, known_params, known_tv_params);
  }
  arma::mat D(p, n,arma::fill::zeros);
  arma::mat C(m, n,arma::fill::zeros);
  
  mgg_ssm mgg_model = mgg_ssm(y, Z, H, T, R, a1, P1, arma::cube(0,0,0),
    arma::mat(0,0), D, C, seed);
  mgg_model.engine = engine;
  return mgg_model;
}