#include "lgg_ssm.h"
#include "mgg_ssm.h"

lgg_ssm::lgg_ssm(const arma::mat& y_, lmat_fnPtr Z_fn_, lmat_fnPtr H_fn_, 
  lmat_fnPtr T_fn_, lmat_fnPtr R_fn_, 
  a1_fnPtr a1_fn_, P1_fnPtr P1_fn_, lvec_fnPtr D_fn_, lvec_fnPtr C_fn_, 
  const arma::vec& theta, prior_fnPtr log_prior_pdf_, const arma::vec& known_params,
  const arma::mat& known_tv_params, 
  const arma::uvec& time_varying, const unsigned int m, const unsigned int k,
  const unsigned int seed) :
  y(y_), Z_fn(Z_fn_), H_fn(H_fn_), T_fn(T_fn_), 
  R_fn(R_fn_), a1_fn(a1_fn_), P1_fn(P1_fn_), 
  D_fn(D_fn_), C_fn(C_fn_), theta(theta), 
  log_prior_pdf(log_prior_pdf_), known_params(known_params), 
  known_tv_params(known_tv_params), time_varying(time_varying), 
  m(m), k(k), n(y.n_cols),  p(y.n_rows),
  engine(seed), zero_tol(1e-8) {
}

mgg_ssm lgg_ssm::build_mgg() {
  
  arma::vec a1 = a1_fn(theta, known_params);
  arma::mat P1 = P1_fn(theta, known_params);
  arma::cube Z(p, m, (n - 1) * time_varying(0) + 1);
  arma::cube H(p, p, (n - 1) * time_varying(1) + 1);
  arma::cube T(m, m, (n - 1) * time_varying(2) + 1);
  arma::cube R(m, k, (n - 1) * time_varying(3) + 1);
  arma::mat D(p, (n - 1) * time_varying(4) + 1);
  arma::mat C(m, (n - 1) * time_varying(5) + 1);
  
  for (unsigned int t = 0; t < Z.n_slices; t++) {
    Z.slice(t) = Z_fn(t, theta, known_params, known_tv_params);
  }
  for (unsigned int t = 0; t < H.n_slices; t++) {
    H.slice(t) = H_fn(t, theta, known_params, known_tv_params);
  }
  for (unsigned int t = 0; t < T.n_slices; t++) {
    T.slice(t) = T_fn(t, theta, known_params, known_tv_params);
  }
  for (unsigned int t = 0; t < R.n_slices; t++) {
    R.slice(t) = R_fn(t, theta, known_params, known_tv_params);
  }
  for (unsigned int t = 0; t < D.n_cols; t++) {
    D.col(t) = D_fn(t, theta, known_params, known_tv_params);
  }
  for (unsigned int t = 0; t < C.n_cols; t++) {
    C.col(t) = C_fn(t, theta, known_params, known_tv_params);
  }
 
 // set seed for new RNG stream based on the original model
 std::uniform_int_distribution<> unif(0, std::numeric_limits<int>::max());
  const unsigned int new_seed = unif(engine);
  mgg_ssm mgg_model = mgg_ssm(y, Z, H, T, R, a1, P1, arma::cube(0,0,0),
    arma::mat(0,0), D, C, new_seed);
  
  return mgg_model;
}

void lgg_ssm::update_mgg(mgg_ssm& model) {
  
  for (unsigned int t = 0; t < model.Z.n_slices; t++) {
    model.Z.slice(t) = Z_fn(t, theta, known_params, known_tv_params);
  }
  for (unsigned int t = 0; t < model.H.n_slices; t++) {
    model.H.slice(t) = H_fn(t, theta, known_params, known_tv_params);
  }
  for (unsigned int t = 0; t < model.T.n_slices; t++) {
    model.T.slice(t) = T_fn(t, theta, known_params, known_tv_params);
  }
  for (unsigned int t = 0; t < model.R.n_slices; t++) {
    model.R.slice(t) = R_fn(t, theta, known_params, known_tv_params);
  }
  for (unsigned int t = 0; t < model.D.n_cols; t++) {
    model.D.col(t) = D_fn(t, theta, known_params, known_tv_params);
  }
  for (unsigned int t = 0; t < model.C.n_cols; t++) {
    model.C.col(t) = C_fn(t, theta, known_params, known_tv_params);
  }
  model.compute_HH();
  model.compute_RR();
}