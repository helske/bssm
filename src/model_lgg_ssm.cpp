#include "model_lgg_ssm.h"

lgg_ssm::lgg_ssm(
  const arma::mat& y_, 
  lmat_fnPtr Z_fn_, 
  lmat_fnPtr H_fn_, 
  lmat_fnPtr T_fn_, 
  lmat_fnPtr R_fn_, 
  a1_fnPtr a1_fn_, 
  P1_fnPtr P1_fn_, 
  lvec_fnPtr D_fn_, 
  lvec_fnPtr C_fn_, 
  const arma::vec& theta, 
  prior_fnPtr log_prior_pdf_, 
  const arma::vec& known_params,
  const arma::mat& known_tv_params, 
  const arma::uvec& time_varying, 
  const unsigned int m, 
  const unsigned int k,
  const unsigned int seed) :
  y(y_), Z_fn(Z_fn_), H_fn(H_fn_), T_fn(T_fn_), 
  R_fn(R_fn_), a1_fn(a1_fn_), P1_fn(P1_fn_), 
  D_fn(D_fn_), C_fn(C_fn_), theta(theta), 
  log_prior_pdf(log_prior_pdf_), known_params(known_params), 
  known_tv_params(known_tv_params), time_varying(time_varying), 
  m(m), k(k), n(y.n_cols), p(y.n_rows),
  engine(seed), zero_tol(1e-8), 
  mgg_model(y, 
    arma::cube(p, m, (n - 1) * time_varying(0) + 1), 
    arma::cube(p, p, (n - 1) * time_varying(1) + 1), 
    arma::cube(m, m, (n - 1) * time_varying(2) + 1), 
    arma::cube(m, k, (n - 1) * time_varying(3) + 1), 
    a1_fn(theta, known_params), 
    P1_fn(theta, known_params), 
    arma::mat(p, (n - 1) * time_varying(4) + 1), 
    arma::mat(m, (n - 1) * time_varying(5) + 1), seed + 1) {
  
  update_model(theta);
}

// void lgg_ssm::build_mgg() {
//   
//   // set seed for new RNG stream based on the original model
//   std::uniform_int_distribution<> unif(0, std::numeric_limits<int>::max());
//   mgg_model = mgg_ssm(y, 
//     arma::cube(p, m, (n - 1) * time_varying(0) + 1), 
//     arma::cube(p, p, (n - 1) * time_varying(1) + 1), 
//     arma::cube(m, m, (n - 1) * time_varying(2) + 1), 
//     arma::cube(m, k, (n - 1) * time_varying(3) + 1), 
//     a1_fn(theta, known_params), 
//     P1_fn(theta, known_params), 
//     arma::mat(p, (n - 1) * time_varying(4) + 1), 
//     arma::mat(m, (n - 1) * time_varying(5) + 1), unif(engine));
//   
//   // arma::vec a1 = a1_fn(theta, known_params);
//   // arma::mat P1 = P1_fn(theta, known_params);
//   // arma::cube Z(p, m, (n - 1) * time_varying(0) + 1);
//   // arma::cube H(p, p, (n - 1) * time_varying(1) + 1);
//   // arma::cube T(m, m, (n - 1) * time_varying(2) + 1);
//   // arma::cube R(m, k, (n - 1) * time_varying(3) + 1);
//   // arma::mat D(p, (n - 1) * time_varying(4) + 1);
//   // arma::mat C(m, (n - 1) * time_varying(5) + 1);
//   
//   for (unsigned int t = 0; t < mgg_model.Z.n_slices; t++) {
//     mgg_model.Z.slice(t) = Z_fn(t, theta, known_params, known_tv_params);
//   }
//   for (unsigned int t = 0; t < mgg_model.H.n_slices; t++) {
//     mgg_model.H.slice(t) = H_fn(t, theta, known_params, known_tv_params);
//   }
//   for (unsigned int t = 0; t < mgg_model.T.n_slices; t++) {
//     mgg_model.T.slice(t) = T_fn(t, theta, known_params, known_tv_params);
//   }
//   for (unsigned int t = 0; t < mgg_model.R.n_slices; t++) {
//     mgg_model.R.slice(t) = R_fn(t, theta, known_params, known_tv_params);
//   }
//   for (unsigned int t = 0; t < mgg_model.D.n_cols; t++) {
//     mgg_model.D.col(t) = D_fn(t, theta, known_params, known_tv_params);
//   }
//   for (unsigned int t = 0; t < mgg_model.C.n_cols; t++) {
//     mgg_model.C.col(t) = C_fn(t, theta, known_params, known_tv_params);
//   }
//   mgg_model.compute_HH();
//   mgg_model.compute_RR();
//   // // set seed for new RNG stream based on the original model
//   // std::uniform_int_distribution<> unif(0, std::numeric_limits<int>::max());
//   // const unsigned int new_seed = unif(engine);
//   // mgg_model = mgg_ssm(y, Z, H, T, R, a1, P1, D, C, new_seed);
//   
// }

void lgg_ssm::update_model(const arma::vec& new_theta) {
  
  for (unsigned int t = 0; t < mgg_model.Z.n_slices; t++) {
    mgg_model.Z.slice(t) = Z_fn(t, theta, known_params, known_tv_params);
  }
  for (unsigned int t = 0; t < mgg_model.H.n_slices; t++) {
    mgg_model.H.slice(t) = H_fn(t, theta, known_params, known_tv_params);
  }
  for (unsigned int t = 0; t < mgg_model.T.n_slices; t++) {
    mgg_model.T.slice(t) = T_fn(t, theta, known_params, known_tv_params);
  }
  for (unsigned int t = 0; t < mgg_model.R.n_slices; t++) {
    mgg_model.R.slice(t) = R_fn(t, theta, known_params, known_tv_params);
  }
  for (unsigned int t = 0; t < mgg_model.D.n_cols; t++) {
    mgg_model.D.col(t) = D_fn(t, theta, known_params, known_tv_params);
  }
  for (unsigned int t = 0; t < mgg_model.C.n_cols; t++) {
    mgg_model.C.col(t) = C_fn(t, theta, known_params, known_tv_params);
  }
  mgg_model.compute_HH();
  mgg_model.compute_RR();
}

// compute the log-likelihood using Kalman filter
double lgg_ssm::log_likelihood() const {
  return mgg_model.log_likelihood();
}
// simulate states from smoothing distribution
arma::cube lgg_ssm::simulate_states(const unsigned int nsim_states) {
  return mgg_model.simulate_states(nsim_states);
}

double lgg_ssm::filter(arma::mat& at, arma::mat& att, arma::cube& Pt, arma::cube& Ptt) const {
  return mgg_model.filter(at, att, Pt, Ptt);
}

void lgg_ssm::smoother(arma::mat& alphahat, arma::cube& Vt) const {
  mgg_model.smoother(alphahat, Vt);
} 
// perform fast state smoothing
arma::mat lgg_ssm::fast_smoother() const {
  return mgg_model.fast_smoother();
}

// smoothing which also returns covariances cov(alpha_t, alpha_t-1)
void lgg_ssm::smoother_ccov(arma::mat& alphahat, arma::cube& Vt, arma::cube& ccov) const {
  mgg_model.smoother_ccov(alphahat, Vt, ccov);
}
