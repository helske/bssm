#include "lgg_ssm.h"
#include "mgg_ssm.h"

lgg_ssm::lgg_ssm(const arma::mat& y_, lmat_fnPtr Z_fn_, lmat_fnPtr H_fn_, 
  lmat_fnPtr T_fn_, lmat_fnPtr R_fn_, 
  a1_fnPtr a1_fn_, P1_fnPtr P1_fn_, lvec_fnPtr D_fn_, lvec_fnPtr C_fn_, 
  const arma::vec& theta, prior_fnPtr log_prior_pdf_, const arma::vec& known_params,
  const arma::mat& known_tv_params, const unsigned int m, const unsigned int k,
  const unsigned int seed) :
  y(y_), Z_fn(Z_fn_), H_fn(H_fn_), T_fn(T_fn_), 
  R_fn(R_fn_), a1_fn(a1_fn_), P1_fn(P1_fn_), 
  D_fn(D_fn_), C_fn(C_fn_), theta(theta), 
  log_prior_pdf(log_prior_pdf_), known_params(known_params), 
  known_tv_params(known_tv_params), m(m), k(k), n(y.n_cols),  p(y.n_rows),
  seed(seed), engine(seed), zero_tol(1e-8) {
}

mgg_ssm lgg_ssm::build_mgg() {
  
  arma::vec a1 = a1_fn(theta, known_params);
  arma::mat P1 = P1_fn(theta, known_params);
  arma::cube Z(p, m, n);
  arma::cube H(p, p, n);
  arma::cube T(m, m, n);
  arma::cube R(m, k, n);
  arma::mat D(p, n);
  arma::mat C(m, n);
  
  for (unsigned int t = 0; t < n; t++) {
    Z.slice(t) = Z_fn(t, theta, known_params, known_tv_params);
    H.slice(t) = H_fn(t, theta, known_params, known_tv_params); 
    T.slice(t) = T_fn(t, theta, known_params, known_tv_params);
    R.slice(t) = R_fn(t, theta, known_params, known_tv_params);
    D.col(t) = D_fn(t, theta, known_params, known_tv_params);
    C.col(t) = C_fn(t, theta, known_params, known_tv_params);
  }
  
  mgg_ssm mgg_model = mgg_ssm(y, Z, H, T, R, a1, P1, arma::cube(0,0,0),
    arma::mat(0,0), D, C, seed);
  mgg_model.engine = engine;
  return mgg_model;
}

void lgg_ssm::update_mgg(mgg_ssm& model) {
  
  for (unsigned int t = 0; t < n; t++) {
    model.Z.slice(t) = Z_fn(t, theta, known_params, known_tv_params);
    model.H.slice(t) = H_fn(t, theta, known_params, known_tv_params); 
    model.T.slice(t) = T_fn(t, theta, known_params, known_tv_params);
    model.R.slice(t) = R_fn(t, theta, known_params, known_tv_params);
    model.D.col(t) = D_fn(t, theta, known_params, known_tv_params);
    model.C.col(t) = C_fn(t, theta, known_params, known_tv_params);
  }
  model.compute_HH();
  model.compute_RR();
}