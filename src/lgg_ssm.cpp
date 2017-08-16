#include "lgg_ssm.h"
#include "mgg_ssm.h"

lgg_ssm::lgg_ssm(const arma::mat& y_, SEXP Z_fn_, SEXP H_fn_, SEXP T_fn_, SEXP R_fn_, 
  SEXP a1_fn_, SEXP P1_fn_, SEXP D_fn_, SEXP C_fn_, 
  const arma::vec& theta, SEXP log_prior_pdf_, const arma::vec& known_params,
  const arma::mat& known_tv_params, const unsigned int m, const unsigned int k,
  const unsigned int seed) :
  y(y_), Z_fn(mat_fn2(Z_fn_)), H_fn(mat_varfn(H_fn_)), T_fn(mat_fn2(T_fn_)), 
  R_fn(mat_varfn(R_fn_)), a1_fn(vec_initfn(a1_fn_)), P1_fn(mat_initfn(P1_fn_)), 
  theta(theta), D_fn(vec_fn2(D_fn_)), C_fn(vec_fn2(C_fn_)),
  log_prior_pdf(log_prior_pdf_), known_params(known_params), 
  known_tv_params(known_tv_params), m(m), k(k), n(y.n_cols),  p(y.n_rows),
  seed(seed), engine(seed), zero_tol(1e-8) {
}

mgg_ssm lgg_ssm::build_mgg() {
  
  arma::vec a1 = a1_fn.eval(theta, known_params);
  arma::mat P1 = P1_fn.eval(theta, known_params);
  arma::cube Z(p, m, n);
  arma::cube H(p, p, n);
  arma::cube T(m, m, n);
  arma::cube R(m, k, n);
  arma::mat D(p, n);
  arma::mat C(m, n);
  
  for (unsigned int t = 0; t < n; t++) {
    Z.slice(t) = Z_fn.eval(t, theta, known_params, known_tv_params);
    H.slice(t) = H_fn.eval(t, theta, known_params, known_tv_params); 
    T.slice(t) = T_fn.eval(t, theta, known_params, known_tv_params);
    R.slice(t) = R_fn.eval(t, theta, known_params, known_tv_params);
    D.col(t) = D_fn.eval(t, theta, known_params, known_tv_params);
    C.col(t) = C_fn.eval(t, theta, known_params, known_tv_params);
  }
  
  mgg_ssm mgg_model = mgg_ssm(y, Z, H, T, R, a1, P1, arma::cube(0,0,0),
    arma::mat(0,0), D, C, seed);
  mgg_model.engine = engine;
  return mgg_model;
}

void lgg_ssm::update_mgg(mgg_ssm& model) {
  
  for (unsigned int t = 0; t < n; t++) {
    model.Z.slice(t) = Z_fn.eval(t, theta, known_params, known_tv_params);
    model.H.slice(t) = H_fn.eval(t, theta, known_params, known_tv_params); 
    model.T.slice(t) = T_fn.eval(t, theta, known_params, known_tv_params);
    model.R.slice(t) = R_fn.eval(t, theta, known_params, known_tv_params);
    model.D.col(t) = D_fn.eval(t, theta, known_params, known_tv_params);
    model.C.col(t) = C_fn.eval(t, theta, known_params, known_tv_params);
  }
  model.compute_HH();
  model.compute_RR();
}