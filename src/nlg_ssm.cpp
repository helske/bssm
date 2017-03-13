#include "nlg_ssm.h"
#include "mgg_ssm.h"
#include "sample.h"
#include "dmvnorm.h"
#include "conditional_dist.h"
#include "function_pointers.h"
#include "rep_mat.h"
#include "psd_chol.h"

nlg_ssm::nlg_ssm(const arma::mat& y, SEXP Z_fn_, SEXP H_fn_, SEXP T_fn_, SEXP R_fn_, 
  SEXP Z_gn_, SEXP T_gn_, SEXP a1_fn_, SEXP P1_fn_,
  const arma::vec& theta, SEXP log_prior_pdf_, const arma::vec& known_params,
  const arma::mat& known_tv_params, const unsigned int m, const unsigned int k,
  const arma::uvec& time_varying, const arma::uvec& state_varying, 
  const unsigned int seed) :
  y(y), Z_fn(vec_fn(Z_fn_)), H_fn(mat_fn(H_fn_)), T_fn(vec_fn(T_fn_)), 
  R_fn(mat_fn(R_fn_)), Z_gn(mat_fn(Z_gn_)), T_gn(mat_fn(T_gn_)),
  a1_fn(vec_initfn(a1_fn_)), P1_fn(mat_initfn(P1_fn_)), theta(theta), 
  log_prior_pdf(log_prior_pdf_), known_params(known_params), 
  known_tv_params(known_tv_params), m(m), k(k), n(y.n_cols),  p(y.n_rows),
  Zgtv(time_varying(0)), Tgtv(time_varying(1)), Htv(time_varying(2)),
  Rtv(time_varying(3)), Hsv(state_varying(0)), Rsv(state_varying(1)), seed(seed), 
  engine(seed), zero_tol(1e-8) {
}
mgg_ssm nlg_ssm::approximate(arma::mat& mode_estimate, const unsigned int max_iter,
  const double conv_tol) const {
  
  double loglik = ekf_fast_smoother(mode_estimate);
  
  unsigned int iekf_iter = 0;
  double diff = conv_tol + 1.0; 
  while(iekf_iter < max_iter && diff > conv_tol) {
    iekf_iter++;
    // compute new guess of mode by EKF
    arma::mat mode_estimate_new(m, n);
    double loglik_new  = iekf_smoother(mode_estimate, mode_estimate_new);
    diff = std::abs(loglik_new - loglik) / (0.1 + loglik_new);
    mode_estimate = mode_estimate_new;
    loglik = loglik_new;
  }
  
  arma::vec a1 = a1_fn.eval(theta, known_params);
  arma::mat P1 = P1_fn.eval(theta, known_params);
  arma::cube Z(p, m, (n - 1) * Zgtv + 1);
  for (unsigned int t = 0; t < Z.n_slices; t++) {
    Z.slice(t) = Z_gn.eval(t, mode_estimate.col(t), theta, known_params, known_tv_params);
  }
  arma::cube H(p, p, (n - 1) * Htv + 1);
  for (unsigned int t = 0; t < H.n_slices; t++) {
    H.slice(t) = H_fn.eval(t, mode_estimate.col(t), theta, known_params, known_tv_params);
  }
  arma::cube T(m, m, (n - 1) * Tgtv + 1);
  for (unsigned int t = 0; t < T.n_slices; t++) {
    T.slice(t) = T_gn.eval(t, mode_estimate.col(t), theta, known_params, known_tv_params);
  }
  
  arma::cube R(m, k, (n - 1) * Rtv + 1);
  for (unsigned int t = 0; t < R.n_slices; t++) {
    R.slice(t) = R_fn.eval(t, mode_estimate.col(t), theta, known_params, known_tv_params);
  }
  arma::mat D(p, n);
  arma::mat C(m, n);
  for (unsigned int t = 0; t < n; t++) {
    D.col(t) = Z_fn.eval(t, mode_estimate.col(t), theta, known_params, known_tv_params) -
      Z.slice(t * Zgtv) * mode_estimate.col(t);
    C.col(t) =  T_fn.eval(t, mode_estimate.col(t), theta, known_params, known_tv_params) -
      T.slice(t * Tgtv) * mode_estimate.col(t);
  }
  
  mgg_ssm approx_model(y, Z, H, T, R, a1, P1, arma::cube(0,0,0),
    arma::mat(0,0), D, C, seed);
  return approx_model;
}

void nlg_ssm::approximate(mgg_ssm& approx_model, arma::mat& mode_estimate,
  const unsigned int max_iter, const double conv_tol) const {
  
  double loglik = ekf_fast_smoother(mode_estimate);
  
  unsigned int iekf_iter = 0;
  double diff = conv_tol + 1.0; 
  while(iekf_iter < max_iter && diff > conv_tol) {
    iekf_iter++;
    // compute new guess of mode by EKF
    arma::mat mode_estimate_new(m, n);
    double loglik_new  = iekf_smoother(mode_estimate, mode_estimate_new);
    diff = std::abs(loglik_new - loglik) / (0.1 + loglik_new);
    mode_estimate = mode_estimate_new;
    loglik = loglik_new;
  }
  
  approx_model.a1 = a1_fn.eval(theta, known_params);
  approx_model.P1 = P1_fn.eval(theta, known_params);
  
  for (unsigned int t = 0; t < approx_model.Z.n_slices; t++) {
    approx_model.Z.slice(t) = Z_gn.eval(t, mode_estimate.col(t), theta, known_params, known_tv_params);
  }
  for (unsigned int t = 0; t < approx_model.H.n_slices; t++) {
    approx_model.H.slice(t) = H_fn.eval(t, mode_estimate.col(t), theta, known_params, known_tv_params);
  }
  for (unsigned int t = 0; t < approx_model.T.n_slices; t++) {
    approx_model.T.slice(t) = T_gn.eval(t, mode_estimate.col(t), theta, known_params, known_tv_params);
  }
  for (unsigned int t = 0; t < approx_model.R.n_slices; t++) {
    approx_model.R.slice(t) = R_fn.eval(t, mode_estimate.col(t), theta, known_params, known_tv_params);
  }
  for (unsigned int t = 0; t < n; t++) {
    approx_model.D.col(t) = Z_fn.eval(t, mode_estimate.col(t), theta, known_params, known_tv_params) -
      approx_model.Z.slice(t * Zgtv) * mode_estimate.col(t);
    approx_model.C.col(t) =  T_fn.eval(t, mode_estimate.col(t), theta, known_params, known_tv_params) -
      approx_model.T.slice(t * Tgtv) * mode_estimate.col(t);
  }
  approx_model.compute_HH();
  approx_model.compute_RR();
}
// 
// mgg_ssm nlg_ssm::approximate(arma::mat& mode_estimate, const unsigned int max_iter,
//   const double conv_tol) const {
// 
//   double loglik = ekf_smoother(mode_estimate);
//   
//   arma::vec a1 = a1_fn.eval(theta, known_params);
//   arma::mat P1 = P1_fn.eval(theta, known_params);
//   arma::cube Z(p, m, (n - 1) * Zgtv + 1);
//   for (unsigned int t = 0; t < Z.n_slices; t++) {
//     Z.slice(t) = Z_gn.eval(t, mode_estimate.col(t), theta, known_params, known_tv_params);
//   }
//   arma::cube H(p, p, (n - 1) * Htv + 1);
//   for (unsigned int t = 0; t < H.n_slices; t++) {
//     H.slice(t) = H_fn.eval(t, mode_estimate.col(t), theta, known_params, known_tv_params);
//   }
//   arma::cube T(m, m, (n - 1) * Tgtv + 1);
//   for (unsigned int t = 0; t < T.n_slices; t++) {
//     T.slice(t) = T_gn.eval(t, mode_estimate.col(t), theta, known_params, known_tv_params);
//   }
// 
//   arma::cube R(m, k, (n - 1) * Rtv + 1);
//   for (unsigned int t = 0; t < R.n_slices; t++) {
//     R.slice(t) = R_fn.eval(t, mode_estimate.col(t), theta, known_params, known_tv_params);
//   }
//   arma::mat D(p, n);
//   arma::mat C(m, n);
//   for (unsigned int t = 0; t < n; t++) {
//     D.col(t) = Z_fn.eval(t, mode_estimate.col(t), theta, known_params, known_tv_params) -
//       Z.slice(t * Zgtv) * mode_estimate.col(t);
//     C.col(t) =  T_fn.eval(t, mode_estimate.col(t), theta, known_params, known_tv_params) -
//       T.slice(t * Tgtv) * mode_estimate.col(t);
//   }
// 
//   mgg_ssm approx_model(y, Z, H, T, R, a1, P1, arma::cube(0,0,0),
//     arma::mat(0,0), D, C, seed);
// 
//   unsigned int i = 0;
//   double diff = conv_tol + 1;
//   while(i < max_iter && diff > conv_tol) {
//     i++;
//     // compute new guess of mode
//     arma::mat mode_estimate_new = approx_model.fast_smoother();
//     diff = arma::accu(arma::square(mode_estimate_new - mode_estimate)) / (m * n);
//     mode_estimate = mode_estimate_new;
// 
//     for (unsigned int t = 0; t < approx_model.Z.n_slices; t++) {
//       approx_model.Z.slice(t) = Z_gn.eval(t, mode_estimate.col(t), theta, known_params, known_tv_params);
//     }
//     for (unsigned int t = 0; t < approx_model.H.n_slices; t++) {
//       approx_model.H.slice(t) = H_fn.eval(t, mode_estimate.col(t), theta, known_params, known_tv_params);
//     }
//     for (unsigned int t = 0; t < approx_model.T.n_slices; t++) {
//       approx_model.T.slice(t) = T_gn.eval(t, mode_estimate.col(t), theta, known_params, known_tv_params);
//     }
//     for (unsigned int t = 0; t < approx_model.R.n_slices; t++) {
//       approx_model.R.slice(t) = R_fn.eval(t, mode_estimate.col(t), theta, known_params, known_tv_params);
//     }
//     for (unsigned int t = 0; t < n; t++) {
//       approx_model.D.col(t) = Z_fn.eval(t, mode_estimate.col(t), theta, known_params, known_tv_params) -
//         approx_model.Z.slice(t * Zgtv) * mode_estimate.col(t);
//       approx_model.C.col(t) =  T_fn.eval(t, mode_estimate.col(t), theta, known_params, known_tv_params) -
//         approx_model.T.slice(t * Tgtv) * mode_estimate.col(t);
//     }
// 
//     approx_model.compute_HH();
//     approx_model.compute_RR();
// 
//   }
// 
//   return approx_model;
// }
// 
// void nlg_ssm::approximate(mgg_ssm& approx_model, arma::mat& mode_estimate,
//   const unsigned int max_iter, const double conv_tol) const {
// 
//   arma::vec a1 = a1_fn.eval(theta, known_params);
//   arma::mat P1 = P1_fn.eval(theta, known_params);
// 
//   arma::cube Z(p, m, (n - 1) * Zgtv + 1);
//   for (unsigned int t = 0; t < Z.n_slices; t++) {
//     Z.slice(t) = Z_gn.eval(t, mode_estimate.col(t), theta, known_params, known_tv_params);
//   }
//   arma::cube H(p, p, (n - 1) * Htv + 1);
//   for (unsigned int t = 0; t < H.n_slices; t++) {
//     H.slice(t) = H_fn.eval(t, mode_estimate.col(t), theta, known_params, known_tv_params);
//   }
//   arma::cube T(m, m, (n - 1) * Tgtv + 1);
//   for (unsigned int t = 0; t < T.n_slices; t++) {
//     T.slice(t) = T_gn.eval(t, mode_estimate.col(t), theta, known_params, known_tv_params);
//   }
//   arma::cube R(m, k, (n - 1) * Rtv + 1);
//   for (unsigned int t = 0; t < R.n_slices; t++) {
//     R.slice(t) = R_fn.eval(t, mode_estimate.col(t), theta, known_params, known_tv_params);
//   }
//   arma::mat C(m, n);
//   arma::mat D(p, n);
//   for (unsigned int t = 0; t < n; t++) {
//     D.col(t) = Z_fn.eval(t, mode_estimate.col(t), theta, known_params, known_tv_params) -
//       Z.slice(t * Zgtv) * mode_estimate.col(t);
//     C.col(t) =  T_fn.eval(t, mode_estimate.col(t), theta, known_params, known_tv_params) -
//       T.slice(t * Tgtv) * mode_estimate.col(t);
//   }
// 
//   approx_model.compute_HH();
//   approx_model.compute_RR();
// 
// 
//   unsigned int i = 0;
//   double diff = conv_tol + 1;
// 
//   while(i < max_iter && diff > conv_tol) {
//     i++;
//     // compute new guess of mode
//     arma::mat mode_estimate_new = approx_model.fast_smoother();
//     diff = arma::accu(arma::square(mode_estimate_new - mode_estimate)) / (m * n);
//     mode_estimate = mode_estimate_new;
// 
//     for (unsigned int t = 0; t < approx_model.Z.n_slices; t++) {
//       approx_model.Z.slice(t) = Z_gn.eval(t, mode_estimate.col(t), theta, known_params, known_tv_params);
//     }
//     for (unsigned int t = 0; t < approx_model.H.n_slices; t++) {
//       approx_model.H.slice(t) = H_fn.eval(t, mode_estimate.col(t), theta, known_params, known_tv_params);
//     }
//     for (unsigned int t = 0; t < approx_model.T.n_slices; t++) {
//       approx_model.T.slice(t) = T_gn.eval(t, mode_estimate.col(t), theta, known_params, known_tv_params);
//     }
//     for (unsigned int t = 0; t < approx_model.R.n_slices; t++) {
//       approx_model.R.slice(t) = R_fn.eval(t, mode_estimate.col(t), theta, known_params, known_tv_params);
//     }
//     for (unsigned int t = 0; t < n; t++) {
//       approx_model.D.col(t) = Z_fn.eval(t, mode_estimate.col(t), theta, known_params, known_tv_params) -
//         approx_model.Z.slice(t * Zgtv) * mode_estimate.col(t);
//       approx_model.C.col(t) =  T_fn.eval(t, mode_estimate.col(t), theta, known_params, known_tv_params) -
//         approx_model.T.slice(t * Tgtv) * mode_estimate.col(t);
//     }
//     approx_model.compute_HH();
//     approx_model.compute_RR();
//   }
// }

// apart from using mgg_ssm, identical with ung_ssm::psi_filter
double nlg_ssm::psi_filter(const mgg_ssm& approx_model,
  const double approx_loglik, const arma::vec& scales,
  const unsigned int nsim, arma::cube& alpha, arma::mat& weights,
  arma::umat& indices) {
  
  arma::mat alphahat(m, n);
  arma::cube Vt(m, m, n);
  arma::cube Ct(m, m, n);
  approx_model.smoother_ccov(alphahat, Vt, Ct);
  if (!Vt.is_finite() || !Ct.is_finite()) {
    return -arma::datum::inf;
  }
  conditional_cov(Vt, Ct);
  std::normal_distribution<> normal(0.0, 1.0);
  
  for (unsigned int i = 0; i < nsim; i++) {
    arma::vec um(m);
    for(unsigned int j = 0; j < m; j++) {
      um(j) = normal(engine);
    }
    alpha.slice(i).col(0) = alphahat.col(0) + Vt.slice(0) * um;
  }
  
  std::uniform_real_distribution<> unif(0.0, 1.0);
  arma::vec normalized_weights(nsim);
  double loglik = 0.0;
  if(arma::is_finite(y(0))) {
    weights.col(0) = exp(log_weights(approx_model, 0, alpha) - scales(0));
    double sum_weights = arma::sum(weights.col(0));
    if(sum_weights > 0.0){
      normalized_weights = weights.col(0) / sum_weights;
    } else {
      return -arma::datum::inf;
    }
    loglik = approx_loglik + log(sum_weights / nsim);
  } else {
    weights.col(0).ones();
    normalized_weights.fill(1.0 / nsim);
    loglik = approx_loglik;
  }
  
  for (unsigned int t = 0; t < (n - 1); t++) {
    arma::vec r(nsim);
    for (unsigned int i = 0; i < nsim; i++) {
      r(i) = unif(engine);
    }
    indices.col(t) = stratified_sample(normalized_weights, r, nsim);
    
    arma::mat alphatmp(m, nsim);
    
    for (unsigned int i = 0; i < nsim; i++) {
      alphatmp.col(i) = alpha.slice(indices(i, t)).col(t);
    }
    for (unsigned int i = 0; i < nsim; i++) {
      arma::vec um(m);
      for(unsigned int j = 0; j < m; j++) {
        um(j) = normal(engine);
      }
      alpha.slice(i).col(t + 1) = alphahat.col(t + 1) +
        Ct.slice(t + 1) * (alphatmp.col(i) - alphahat.col(t)) + Vt.slice(t + 1) * um;
    }
    
    if(arma::is_finite(y(t + 1))) {
      weights.col(t + 1) = 
        exp(log_weights(approx_model, t + 1, alpha) - scales(t + 1));
      double sum_weights = arma::sum(weights.col(t + 1));
      if(sum_weights > 0.0){
        normalized_weights = weights.col(t + 1) / sum_weights;
      } else {
        return -arma::datum::inf;
      }
      loglik += log(sum_weights / nsim);
    } else {
      weights.col(t + 1).ones();
      normalized_weights.fill(1.0 / nsim);
    }
  }
  
  return loglik;
}

// Logarithms of _normalized_ importance weights g(y_t | alpha_t) / ~g(~y_t | alpha_t)
/*
 * approx_model:  Gaussian approximation of the original model
 * t:             Time point where the weights are computed
 * alpha:         Simulated particles
 */
arma::vec nlg_ssm::log_weights(const mgg_ssm& approx_model, 
  const unsigned int t, const arma::cube& alpha) const {
  
  arma::vec weights(alpha.n_slices, arma::fill::zeros);
  
  if (arma::is_finite(y(t))) {
    if(Hsv) {
      for (unsigned int i = 0; i < alpha.n_slices; i++) {
        weights(i) = 
          dmvnorm(y.col(t), Z_fn.eval(t, alpha.slice(i).col(t), theta, known_params, known_tv_params), 
            H_fn.eval(t, alpha.slice(i).col(t), theta, known_params, known_tv_params), true, true);
      }
    } else {
      arma::mat H = H_fn.eval(t, alpha.slice(0).col(t), theta, known_params, known_tv_params);
      arma::uvec nonzero = arma::find(H.diag() > (arma::datum::eps * H.n_cols * H.diag().max()));
      arma::mat Linv(nonzero.n_elem, nonzero.n_elem);
      double constant = precompute_dmvnorm(H, Linv, nonzero);
      
      for (unsigned int i = 0; i < alpha.n_slices; i++) {
        weights(i) = fast_dmvnorm(y.col(t), Z_fn.eval(t, alpha.slice(i).col(t), 
          theta, known_params, known_tv_params), Linv, nonzero, constant);
      }
    }
    unsigned int Ztv = approx_model.Ztv;
    unsigned int Htv = approx_model.Htv;
    if(Htv == 1) {
      for (unsigned int i = 0; i < alpha.n_slices; i++) {
        weights(i) -= dmvnorm(y.col(t), approx_model.Z.slice(t * Ztv) * alpha.slice(i).col(t),  
          approx_model.H.slice(t * Htv), true, true);
      }
    } else {
      arma::mat H = approx_model.H.slice(0);
      arma::uvec nonzero = arma::find(H.diag() > (arma::datum::eps * H.n_cols * H.diag().max()));
      arma::mat Linv(nonzero.n_elem, nonzero.n_elem);
      double constant = precompute_dmvnorm(H, Linv, nonzero);
      for (unsigned int i = 0; i < alpha.n_slices; i++) {
        weights(i) -= fast_dmvnorm(y.col(t), approx_model.Z.slice(t * Ztv) * alpha.slice(i).col(t),  
          Linv, nonzero, constant);
      }
      
    }
  }
  return weights;
}

// compute _normalized_ mode-based scaling terms
// log[g(y_t | ^alpha_t) / ~g(y_t | ^alpha_t)]
arma::vec nlg_ssm::scaling_factors(const mgg_ssm& approx_model,
  const arma::mat& mode_estimate) const {
  
  arma::vec weights(n, arma::fill::zeros);
  unsigned int Ztv = approx_model.Ztv;
  unsigned int Htv = approx_model.Htv;
  
  for(unsigned int t = 0; t < n; t++) {
    if (arma::is_finite(y(t))) {
      weights(t) =  dmvnorm(y.col(t), Z_fn.eval(t, mode_estimate.col(t), theta, known_params, known_tv_params), 
        H_fn.eval(t, mode_estimate.col(t), theta, known_params, known_tv_params), true, true) -
          dmvnorm(y.col(t), approx_model.Z.slice(t * Ztv) * mode_estimate.col(t),  
            approx_model.H.slice(t * Htv), true, true);
    }
  }
  return weights;
}

// Logarithms of _normalized_ densities g(y_t | alpha_t)
/*
 * t:             Time point where the densities are computed
 * alpha:         Simulated particles
 */
arma::vec nlg_ssm::log_obs_density(const unsigned int t, 
  const arma::cube& alpha) const {
  
  arma::vec weights(alpha.n_slices, arma::fill::zeros);
  
  if (arma::is_finite(y(t))) {
    for (unsigned int i = 0; i < alpha.n_slices; i++) {
      weights(i) = dmvnorm(y.col(t), Z_fn.eval(t, alpha.slice(i).col(t), theta, known_params, known_tv_params), 
        H_fn.eval(t, alpha.slice(i).col(t), theta, known_params, known_tv_params), true, true);
    }
  }
  return weights;
}

double nlg_ssm::bsf_filter(const unsigned int nsim, arma::cube& alpha,
  arma::mat& weights, arma::umat& indices) {
  
  arma::vec a1 = a1_fn.eval(theta, known_params);
  arma::mat P1 = P1_fn.eval(theta, known_params);
  arma::uvec nonzero = arma::find(P1.diag() > 0);
  arma::mat L_P1(m, m, arma::fill::zeros);
  if (nonzero.n_elem > 0) {
    L_P1.submat(nonzero, nonzero) =
      arma::chol(P1.submat(nonzero, nonzero), "lower");
  }
  std::normal_distribution<> normal(0.0, 1.0);
  for (unsigned int i = 0; i < nsim; i++) {
    arma::vec um(m);
    for(unsigned int j = 0; j < m; j++) {
      um(j) = normal(engine);
    }
    alpha.slice(i).col(0) = a1 + L_P1 * um;
    
  }
  
  std::uniform_real_distribution<> unif(0.0, 1.0);
  arma::vec normalized_weights(nsim);
  double loglik = 0.0;
  
  if(arma::is_finite(y(0))) {
    weights.col(0) = log_obs_density(0, alpha);
    double max_weight = weights.col(0).max();
    weights.col(0) = exp(weights.col(0) - max_weight);
    double sum_weights = arma::sum(weights.col(0));
    
    if(sum_weights > 0.0){
      normalized_weights = weights.col(0) / sum_weights;
    } else {
      return -arma::datum::inf;
    }
    loglik = max_weight + log(sum_weights / nsim);
  } else {
    weights.col(0).ones();
    normalized_weights.fill(1.0 / nsim);
  }
  for (unsigned int t = 0; t < (n - 1); t++) {
    
    arma::vec r(nsim);
    for (unsigned int i = 0; i < nsim; i++) {
      r(i) = unif(engine);
    }
    
    indices.col(t) = stratified_sample(normalized_weights, r, nsim);
    
    arma::mat alphatmp(m, nsim);
    
    for (unsigned int i = 0; i < nsim; i++) {
      alphatmp.col(i) = alpha.slice(indices(i, t)).col(t);
    }
    
    for (unsigned int i = 0; i < nsim; i++) {
      arma::vec uk(k);
      for(unsigned int j = 0; j < k; j++) {
        uk(j) = normal(engine);
      }
      alpha.slice(i).col(t + 1) = T_fn.eval(t, alphatmp.col(i), theta, known_params, known_tv_params) + 
        R_fn.eval(t, alphatmp.col(i), theta, known_params, known_tv_params) * uk;
    }
    
    if(arma::is_finite(y(t + 1))) {
      weights.col(t + 1) = log_obs_density(t + 1, alpha);
      
      double max_weight = weights.col(t + 1).max();
      weights.col(t + 1) = exp(weights.col(t + 1) - max_weight);
      double sum_weights = arma::sum(weights.col(t + 1));
      if(sum_weights > 0.0){
        normalized_weights = weights.col(t + 1) / sum_weights;
      } else {
        return -arma::datum::inf;
      }
      loglik += max_weight + log(sum_weights / nsim);
    } else {
      weights.col(t + 1).ones();
      normalized_weights.fill(1.0/nsim);
    }
  }
  return loglik;
}

double nlg_ssm::ekf(arma::mat& at, arma::mat& att, arma::cube& Pt, 
  arma::cube& Ptt) const {
  
  at.col(0) = a1_fn.eval(theta, known_params);
  Pt.slice(0) = P1_fn.eval(theta, known_params);
  
  const double LOG2PI = std::log(2.0 * M_PI);
  double logLik = 0.0;
  
  for (unsigned int t = 0; t < n; t++) {
    
    arma::uvec na_y = arma::find_nonfinite(y.col(t));
    
    if (na_y.n_elem < p) {
      arma::mat Zg = Z_gn.eval(t, at.col(t), theta, known_params, known_tv_params);
      Zg.rows(na_y).zeros();
      arma::mat HHt = H_fn.eval(t, at.col(t), theta, known_params, known_tv_params);
      HHt = HHt * HHt.t();
      HHt.submat(na_y, na_y) = arma::eye(na_y.n_elem, na_y.n_elem);
      
      arma::mat Ft = Zg * Pt.slice(t) * Zg.t() + HHt;
      
      arma::mat cholF(p, p);
      bool chol_ok = arma::chol(cholF,Ft);
      if(!chol_ok) return -arma::datum::inf;
      
      arma::vec vt = y.col(t) - 
        Z_fn.eval(t, at.col(t), theta, known_params, known_tv_params);
      vt.rows(na_y).zeros();
      
      arma::mat inv_cholF = arma::inv(arma::trimatu(cholF));
      arma::mat Kt = Pt.slice(t) * Zg.t() * inv_cholF * inv_cholF.t();
      
      att.col(t) = at.col(t) + Kt * vt;
      Ptt.slice(t) = Pt.slice(t) - Kt * Ft * Kt.t();
      
      arma::vec Fv = inv_cholF.t() * vt; 
      logLik -= 0.5 * arma::as_scalar((p - na_y.n_elem) * LOG2PI + 
        2.0 * arma::sum(log(arma::diagvec(cholF))) + Fv.t() * Fv);
    } else {
      att.col(t) = at.col(t);
      Ptt.slice(t) = Pt.slice(t);
    } 
    
    at.col(t + 1) = T_fn.eval(t, att.col(t), theta, known_params, known_tv_params);
    arma::mat Tg = T_gn.eval(t, att.col(t), theta, known_params, known_tv_params);
    arma::mat Rt = R_fn.eval(t, att.col(t), theta, known_params, known_tv_params);
    Pt.slice(t + 1) = Tg * Ptt.slice(t) * Tg.t() + Rt * Rt.t();
  }
  return logLik;
}

double nlg_ssm::ekf_loglik() const {
  
  
  arma::vec at = a1_fn.eval(theta, known_params);
  arma::mat Pt = P1_fn.eval(theta, known_params);
  
  const double LOG2PI = std::log(2.0 * M_PI);
  double logLik = 0.0;
  
  for (unsigned int t = 0; t < n; t++) {
    
    arma::uvec na_y = arma::find_nonfinite(y.col(t));
    
    arma::vec att = at;
    arma::mat Ptt = Pt;
    if (na_y.n_elem < p) {
      arma::mat Zg = Z_gn.eval(t, at, theta, known_params, known_tv_params);
      Zg.rows(na_y).zeros();
      arma::mat HHt = H_fn.eval(t, at, theta, known_params, known_tv_params);
      HHt = HHt * HHt.t();
      HHt.submat(na_y, na_y) = arma::eye(na_y.n_elem, na_y.n_elem);
      
      arma::mat Ft = Zg * Pt * Zg.t() + HHt;
      arma::mat cholF(p, p);
      bool chol_ok = arma::chol(cholF,Ft);
      if(!chol_ok) return -arma::datum::inf;
      
      arma::vec vt = y.col(t) - 
        Z_fn.eval(t, at, theta, known_params, known_tv_params);
      vt.rows(na_y).zeros();
      
      arma::mat inv_cholF = arma::inv(arma::trimatu(cholF));
      arma::mat Kt = Pt * Zg.t() * inv_cholF * inv_cholF.t();
      
      att += Kt * vt;
      Ptt -= Kt * Ft * Kt.t();
      
      arma::vec Fv = inv_cholF.t() * vt; 
      logLik -= 0.5 * arma::as_scalar((p - na_y.n_elem) * LOG2PI + 
        2.0 * arma::sum(log(arma::diagvec(cholF))) + Fv.t() * Fv);
    }
    
    at = T_fn.eval(t, att, theta, known_params, known_tv_params);
    
    arma::mat Tg = T_gn.eval(t, att, theta, known_params, known_tv_params);
    arma::mat Rt = R_fn.eval(t, att, theta, known_params, known_tv_params);
    Pt = Tg * Ptt * Tg.t() + Rt * Rt.t();
    
  }
  return logLik;
}

double nlg_ssm::ekf_smoother(arma::mat& at, arma::cube& Pt) const {
  
  at.col(0) = a1_fn.eval(theta, known_params);
  
  Pt.slice(0) = P1_fn.eval(theta, known_params);
  
  arma::mat vt(p, n,arma::fill::zeros);
  arma::cube ZFinv(m, p, n,arma::fill::zeros);
  arma::cube Kt(m, p, n,arma::fill::zeros);
  
  arma::uvec obs(n, arma::fill::ones);
  
  arma::mat att(m, n);
  
  const double LOG2PI = std::log(2.0 * M_PI);
  double logLik = 0.0;
  
  for (unsigned int t = 0; t < (n - 1); t++) {
    
    arma::uvec na_y = arma::find_nonfinite(y.col(t));
    arma::mat Ptt = Pt.slice(t);
    
    if (na_y.n_elem < p) {
      
      arma::mat Zg = Z_gn.eval(t, at.col(t), theta, known_params, known_tv_params);
      Zg.rows(na_y).zeros();
      arma::mat HHt = H_fn.eval(t, at.col(t), theta, known_params, known_tv_params);
      HHt = HHt * HHt.t();
      HHt.submat(na_y, na_y) = arma::eye(na_y.n_elem, na_y.n_elem);
      
      arma::mat Ft = Zg * Pt.slice(t) * Zg.t() + HHt;
      arma::mat cholF(p, p);
      bool chol_ok = arma::chol(cholF,Ft);
      if(!chol_ok) return -arma::datum::inf;
      
      vt.col(t) = y.col(t) - 
        Z_fn.eval(t, at.col(t), theta, known_params, known_tv_params);
      vt.rows(na_y).zeros();
      
      arma::mat inv_cholF = arma::inv(arma::trimatu(cholF));
      ZFinv.slice(t) = Zg.t() * inv_cholF * inv_cholF.t();
      Kt.slice(t) = Pt.slice(t) * ZFinv.slice(t);
      
      att.col(t) = at.col(t) + Kt.slice(t) * vt.col(t);
      Ptt = Pt.slice(t) - Kt.slice(t) * Ft * Kt.slice(t).t();
      
      arma::vec Fv = inv_cholF.t() * vt.col(t); 
      logLik -= 0.5 * arma::as_scalar((p - na_y.n_elem) * LOG2PI + 
        2.0 * arma::sum(log(arma::diagvec(cholF))) + Fv.t() * Fv);
    } else {
      att.col(t) = at.col(t);
      obs(t) = 0;
    }
    at.col(t + 1) = T_fn.eval(t, att.col(t), theta, known_params, known_tv_params);
    arma::mat Tg = T_gn.eval(t, att.col(t), theta, known_params, known_tv_params);
    arma::mat Rt = R_fn.eval(t, att.col(t), theta, known_params, known_tv_params);
    Pt.slice(t + 1) = Tg * Ptt * Tg.t() + Rt * Rt.t();
  }
  
  unsigned int t = n - 1;
  arma::uvec na_y = arma::find_nonfinite(y.col(t));
  
  if (na_y.n_elem < p) {
    arma::mat Zg = Z_gn.eval(t, at.col(t), theta, known_params, known_tv_params);
    Zg.rows(na_y).zeros();
    arma::mat HHt = H_fn.eval(t, at.col(t), theta, known_params, known_tv_params);
    HHt = HHt * HHt.t();
    HHt.submat(na_y, na_y) = arma::eye(na_y.n_elem, na_y.n_elem);
    
    arma::mat Ft = Zg * Pt.slice(t) * Zg.t() + HHt;
    arma::mat cholF(p, p);
    bool chol_ok = arma::chol(cholF,Ft);
    if(!chol_ok) return -arma::datum::inf;
    vt.col(t) = y.col(t) - 
      Z_fn.eval(t, at.col(t), theta, known_params, known_tv_params);
    vt.rows(na_y).zeros();
    
    arma::mat inv_cholF = arma::inv(arma::trimatu(cholF));
    ZFinv.slice(t) = Zg.t() * inv_cholF * inv_cholF.t();
    Kt.slice(t) = Pt.slice(t) * ZFinv.slice(t);
    att.col(t) = at.col(t) + Kt.slice(t) * vt.col(t);
    arma::vec Fv = inv_cholF.t() * vt.col(t); 
    logLik -= 0.5 * arma::as_scalar((p - na_y.n_elem) * LOG2PI + 
      2.0 * arma::sum(log(arma::diagvec(cholF))) + Fv.t() * Fv);
  } else {
    att.col(t) = at.col(t);
    obs(t) = 0;
  }
  
  arma::vec rt(m, arma::fill::zeros);
  arma::mat Nt(m, m, arma::fill::zeros);
  for (int t = (n - 1); t >= 0; t--) {
    arma::mat Tg = T_gn.eval(t, att.col(t), theta, known_params, known_tv_params);
    if (obs(t)) {
      arma::mat Zg = Z_gn.eval(t, at.col(t), theta, known_params, known_tv_params);
      arma::mat L = Tg * (arma::eye(m, m) - Kt.slice(t) * Zg);
      rt = ZFinv.slice(t) * vt.col(t) + L.t() * rt;
      Nt = arma::symmatu(ZFinv.slice(t) * Zg + L.t() * Nt * L);
    } else {
      rt = Tg.t() * rt;
      Nt = Tg.t() * Nt * Tg;
    }
    at.col(t) += Pt.slice(t) * rt;
    Pt.slice(t) -= Pt.slice(t) * Nt * Pt.slice(t);
  }
  return logLik;
}

double nlg_ssm::ekf_fast_smoother(arma::mat& at) const {
  
  at.col(0) = a1_fn.eval(theta, known_params);
  
  arma::cube Pt(m, m, n);
  
  Pt.slice(0) = P1_fn.eval(theta, known_params);
  
  arma::mat vt(p, n,arma::fill::zeros);
  arma::cube ZFinv(m, p, n,arma::fill::zeros);
  arma::cube Kt(m, p, n,arma::fill::zeros);
  
  arma::uvec obs(n, arma::fill::ones);
  
  arma::mat att(m, n);
  
  const double LOG2PI = std::log(2.0 * M_PI);
  double logLik = 0.0;
  
  for (unsigned int t = 0; t < (n - 1); t++) {
    
    arma::uvec na_y = arma::find_nonfinite(y.col(t));
    arma::mat Ptt = Pt.slice(t);
    
    if (na_y.n_elem < p) {
      
      arma::mat Zg = Z_gn.eval(t, at.col(t), theta, known_params, known_tv_params);
      Zg.rows(na_y).zeros();
      arma::mat HHt = H_fn.eval(t, at.col(t), theta, known_params, known_tv_params);
      HHt = HHt * HHt.t();
      HHt.submat(na_y, na_y) = arma::eye(na_y.n_elem, na_y.n_elem);
      
      arma::mat Ft = Zg * Pt.slice(t) * Zg.t() + HHt;
      arma::mat cholF(p, p);
      bool chol_ok = arma::chol(cholF,Ft);
      if(!chol_ok) return -arma::datum::inf;
      
      vt.col(t) = y.col(t) - 
        Z_fn.eval(t, at.col(t), theta, known_params, known_tv_params);
      vt.rows(na_y).zeros();
      
      arma::mat inv_cholF = arma::inv(arma::trimatu(cholF));
      ZFinv.slice(t) = Zg.t() * inv_cholF * inv_cholF.t();
      Kt.slice(t) = Pt.slice(t) * ZFinv.slice(t);
      
      att.col(t) = at.col(t) + Kt.slice(t) * vt.col(t);
      Ptt = Pt.slice(t) - Kt.slice(t) * Ft * Kt.slice(t).t();
      
      arma::vec Fv = inv_cholF.t() * vt.col(t); 
      logLik -= 0.5 * arma::as_scalar((p - na_y.n_elem) * LOG2PI + 
        2.0 * arma::sum(log(arma::diagvec(cholF))) + Fv.t() * Fv);
    } else {
      att.col(t) = at.col(t);
      obs(t) = 0;
    }
    at.col(t + 1) = T_fn.eval(t, att.col(t), theta, known_params, known_tv_params);
    arma::mat Tg = T_gn.eval(t, att.col(t), theta, known_params, known_tv_params);
    arma::mat Rt = R_fn.eval(t, att.col(t), theta, known_params, known_tv_params);
    Pt.slice(t + 1) = Tg * Ptt * Tg.t() + Rt * Rt.t();
  }
  
  unsigned int t = n - 1;
  arma::uvec na_y = arma::find_nonfinite(y.col(t));
  
  if (na_y.n_elem < p) {
    arma::mat Zg = Z_gn.eval(t, at.col(t), theta, known_params, known_tv_params);
    Zg.rows(na_y).zeros();
    arma::mat HHt = H_fn.eval(t, at.col(t), theta, known_params, known_tv_params);
    HHt = HHt * HHt.t();
    HHt.submat(na_y, na_y) = arma::eye(na_y.n_elem, na_y.n_elem);
    
    arma::mat Ft = Zg * Pt.slice(t) * Zg.t() + HHt;
    arma::mat cholF(p, p);
    bool chol_ok = arma::chol(cholF,Ft);
    if(!chol_ok) return -arma::datum::inf;
    vt.col(t) = y.col(t) - 
      Z_fn.eval(t, at.col(t), theta, known_params, known_tv_params);
    vt.rows(na_y).zeros();
    
    arma::mat inv_cholF = arma::inv(arma::trimatu(cholF));
    ZFinv.slice(t) = Zg.t() * inv_cholF * inv_cholF.t();
    Kt.slice(t) = Pt.slice(t) * ZFinv.slice(t);
    att.col(t) = at.col(t) + Kt.slice(t) * vt.col(t);
    arma::vec Fv = inv_cholF.t() * vt.col(t); 
    logLik -= 0.5 * arma::as_scalar((p - na_y.n_elem) * LOG2PI + 
      2.0 * arma::sum(log(arma::diagvec(cholF))) + Fv.t() * Fv);
  } else {
    att.col(t) = at.col(t);
    obs(t) = 0;
  }
  
  arma::vec rt(m, arma::fill::zeros);
  for (int t = (n - 1); t >= 0; t--) {
    arma::mat Tg = T_gn.eval(t, att.col(t), theta, known_params, known_tv_params);
    if (obs(t)) {
      arma::mat Zg = Z_gn.eval(t, at.col(t), theta, known_params, known_tv_params);
      arma::mat L = Tg * (arma::eye(m, m) - Kt.slice(t) * Zg);
      rt = ZFinv.slice(t) * vt.col(t) + L.t() * rt;
    } else {
      rt = Tg.t() * rt;
    }
    at.col(t) += Pt.slice(t) * rt;
  }
  return logLik;
  
  // arma::mat rt(m, n);
  // rt.col(n - 1).zeros();
  // for (int t = (n - 1); t > 0; t--) {
  //   arma::mat Tg = T_gn.eval(t, att.col(t), theta, known_params, known_tv_params);
  //   if (obs(t)) {
  //     arma::mat Zg = Z_gn.eval(t, at.col(t), theta, known_params, known_tv_params);
  //     arma::mat L = Tg * (arma::eye(m, m) - Kt.slice(t) * Zg);
  //     rt.col(t - 1) = ZFinv.slice(t) * vt.col(t) + L.t() * rt.col(t);
  //   } else {
  //     rt.col(t - 1) = Tg.t() * rt.col(t);
  //   }
  // }
  // if (obs(0)){
  //   arma::mat Tg = T_gn.eval(0, att.col(0), theta, known_params, known_tv_params);
  //   arma::mat Zg = Z_gn.eval(0, at.col(0), theta, known_params, known_tv_params);
  //   arma::mat L = Tg * (arma::eye(m, m) - Kt.slice(0) * Zg);
  //   at.col(0) += Pt.slice(0) * (ZFinv.slice(0) * vt.col(0) + L.t() * rt.col(0));
  // } else {
  //   arma::mat Tg = T_gn.eval(0, att.col(t), theta, known_params, known_tv_params);
  //   at.col(0) += Pt.slice(0) * Tg.t() * rt.col(0);
  // }
  // 
  // for (unsigned int t = 0; t < (n - 1); t++) {
  //   //arma::mat Tg = T_gn.eval(t, at.col(t), theta, known_params, known_tv_params);
  //   arma::mat Rt = R_fn.eval(t, at.col(t), theta, known_params, known_tv_params);
  //   at.col(t + 1) = T_fn.eval(t, at.col(t), theta, known_params, known_tv_params) 
  //     + Rt * Rt.t() * rt.col(t);
  // }
  return logLik;
}

// arma::mat nlg_ssm::iekf_smoother(const arma::mat& alphahat) const {
//   
//   
//   arma::mat at(m, n);
//   at.col(0) = a1_fn.eval(theta, known_params);
//   
//   arma::mat Pt = P1_fn.eval(theta, known_params);
//   
//   arma::mat vt(p, n, arma::fill::zeros);
//   arma::cube ZFinv(m, p, n, arma::fill::zeros);
//   arma::cube Kt(m, p, n, arma::fill::zeros);
//   
//   arma::uvec obs(n, arma::fill::ones);
//   
//   arma::mat att(m, n);
//   
//   for (unsigned int t = 0; t < (n - 1); t++) {
//     
//     arma::uvec na_y = arma::find_nonfinite(y.col(t));
//     
//     arma::mat Ptt = Pt;
//     
//     if (na_y.n_elem < p) {
//       
//       arma::mat Zg = Z_gn.eval(t, alphahat.col(t), theta, known_params, known_tv_params);
//       Zg.rows(na_y).zeros();
//       arma::mat HHt = H_fn.eval(t, at.col(t), theta, known_params, known_tv_params);
//       HHt = HHt * HHt.t();
//       HHt.submat(na_y, na_y) = arma::eye(na_y.n_elem, na_y.n_elem);
//       
//       arma::mat Ft = Zg * Pt * Zg.t() + HHt;
//       arma::mat cholF = arma::chol(Ft);
//       
//       vt.col(t) = y.col(t) - 
//         Z_fn.eval(t, at.col(t), theta, known_params, known_tv_params);
//       vt.rows(na_y).zeros();
//       
//       arma::mat inv_cholF = arma::inv(arma::trimatu(cholF));
//       ZFinv.slice(t) = Zg.t() * inv_cholF * inv_cholF.t();
//       Kt.slice(t) = Pt * ZFinv.slice(t);
//       
//       att.col(t) = at.col(t) + Kt.slice(t) * vt.col(t);
//       Ptt = Pt - Kt.slice(t) * Ft * Kt.slice(t).t();
//     } else {
//       att.col(t) = at.col(t);
//       obs(t) = 0;
//     }
//     
//    
//     at.col(t + 1) = T_fn.eval(t, att.col(t), theta, known_params, known_tv_params);
//     arma::mat Tg = T_gn.eval(t, alphahat.col(t), theta, known_params, known_tv_params);
//     arma::mat Rt = R_fn.eval(t, att.col(t), theta, known_params, known_tv_params);
//     Pt = Tg * Ptt * Tg.t() + Rt * Rt.t();
//   }
//   
//   unsigned int t = n - 1;
//   arma::uvec na_y = arma::find_nonfinite(y.col(t));
//   
//   if (na_y.n_elem < p) {
//     arma::mat Zg = Z_gn.eval(t, alphahat.col(t), theta, known_params, known_tv_params);
//     Zg.rows(na_y).zeros();
//     arma::mat HHt = H_fn.eval(t, at.col(t), theta, known_params, known_tv_params);
//     HHt = HHt * HHt.t();
//     HHt.submat(na_y, na_y) = arma::eye(na_y.n_elem, na_y.n_elem);
//     
//     arma::mat Ft = Zg * Pt * Zg.t() + HHt;
//     arma::mat cholF = arma::chol(Ft);
//     vt.col(t) = y.col(t) - Z_fn.eval(t, at.col(t), theta, known_params, known_tv_params);
//     vt.rows(na_y).zeros();
//     
//     arma::mat inv_cholF = arma::inv(arma::trimatu(cholF));
//     ZFinv.slice(t) = Zg.t() * inv_cholF * inv_cholF.t();
//     Kt.slice(t) = Pt * ZFinv.slice(t);
//   } else {
//     obs(0) = 0;
//   }
//   
//   arma::mat rt(m, n);
//   rt.col(n - 1).zeros();
//   for (int t = (n - 1); t > 0; t--) {
//     arma::mat Tg = T_gn.eval(t, alphahat.col(t), theta, known_params, known_tv_params);
//     if (obs(t)) {
//       arma::mat Zg = Z_gn.eval(t, alphahat.col(t), theta, known_params, known_tv_params);
//       arma::mat L = Tg * (arma::eye(m, m) - Kt.slice(t) * Zg);
//       rt.col(t - 1) = ZFinv.slice(t) * vt.col(t) + L.t() * rt.col(t);
//     } else {
//       rt.col(t - 1) = Tg.t() * rt.col(t);
//     }
//   }
//   arma::mat Tg = T_gn.eval(0, alphahat.col(0), theta, known_params, known_tv_params);
//   arma::vec a1 = a1_fn.eval(theta, known_params);
//   arma::mat P1 = P1_fn.eval(theta, known_params);
//   if (obs(0)){
//     arma::mat Zg = Z_gn.eval(0, alphahat.col(0), theta, known_params, known_tv_params);
//     arma::mat L = Tg * (arma::eye(m, m) - Kt.slice(0) * Zg);
//     at.col(0) = a1 + P1 * (ZFinv.slice(0) * vt.col(0) + L.t() * rt.col(0));
//   } else {
//     at.col(0) = a1 + P1 * Tg.t() * rt.col(0);
//   }
//   for (unsigned int t = 0; t < (n - 1); t++) {
//     arma::mat Tg = T_gn.eval(t, alphahat.col(t), theta, known_params, known_tv_params);
//     arma::mat Rt = R_fn.eval(t, att.col(t), theta, known_params, known_tv_params);
//     at.col(t + 1) = T_fn.eval(t, at.col(t), theta, known_params, known_tv_params) +
//       Tg * (at.col(t) - alphahat.col(t)) + Rt * Rt.t() * rt.col(t);
//     //Tg * at.col(t) + Rt * Rt.t() * rt.col(t);
//   }
//   return at;
// }
// 
double nlg_ssm::iekf_smoother(const arma::mat& alphahat, arma::mat& at) const {
  
  arma::mat att(m, n);
  arma::cube Pt(m, m, n + 1);
  arma::cube Ptt(m, m, n);
  at.col(0) = a1_fn.eval(theta, known_params);
  Pt.slice(0) = P1_fn.eval(theta, known_params);
  
  const double LOG2PI = std::log(2.0 * M_PI);
  double logLik = 0.0;
  
  for (unsigned int t = 0; t < n; t++) {
    
    arma::uvec na_y = arma::find_nonfinite(y.col(t));
    
    if (na_y.n_elem < p) {
      arma::mat Zg = Z_gn.eval(t, alphahat.col(t), theta, known_params, known_tv_params);
      Zg.rows(na_y).zeros();
      arma::mat HHt = H_fn.eval(t, at.col(t), theta, known_params, known_tv_params);
      HHt = HHt * HHt.t();
      HHt.submat(na_y, na_y) = arma::eye(na_y.n_elem, na_y.n_elem);
      
      arma::mat Ft = Zg * Pt.slice(t) * Zg.t() + HHt;
      
      arma::mat cholF(p, p);
      bool chol_ok = arma::chol(cholF,Ft);
      if(!chol_ok) return -arma::datum::inf;
      
      arma::vec vt = y.col(t) - 
        Z_fn.eval(t, at.col(t), theta, known_params, known_tv_params);
      vt.rows(na_y).zeros();
      
      arma::mat inv_cholF = arma::inv(arma::trimatu(cholF));
      arma::mat Kt = Pt.slice(t) * Zg.t() * inv_cholF * inv_cholF.t();
      
      att.col(t) = at.col(t) + Kt * vt;
      Ptt.slice(t) = Pt.slice(t) - Kt * Ft * Kt.t();
      
      arma::vec Fv = inv_cholF.t() * vt; 
      logLik -= 0.5 * arma::as_scalar((p - na_y.n_elem) * LOG2PI + 
        2.0 * arma::sum(log(arma::diagvec(cholF))) + Fv.t() * Fv);
    } else {
      att.col(t) = at.col(t);
      Ptt.slice(t) = Pt.slice(t);
    } 
    
    at.col(t + 1) = T_fn.eval(t, att.col(t), theta, known_params, known_tv_params);
    arma::mat Tg = T_gn.eval(t, alphahat.col(t), theta, known_params, known_tv_params);
    arma::mat Rt = R_fn.eval(t, att.col(t), theta, known_params, known_tv_params);
    Pt.slice(t + 1) = Tg * Ptt.slice(t) * Tg.t() + Rt * Rt.t();
  }
  
  for (int t = (n - 2); t >= 0; t--) {
    arma::vec tmp = T_fn.eval(t, att.col(t), theta, known_params, known_tv_params);
    arma::mat tmp2 = T_gn.eval(t, alphahat.col(t), theta, known_params, known_tv_params);
    arma::mat G = Ptt.slice(t) * tmp2.t() * arma::pinv(Pt.slice(t + 1));
    att.col(t) = att.col(t) + G * (att.col(t + 1) - at.col(t + 1));
    Ptt.slice(t) = Ptt.slice(t) + G * (Ptt.slice(t + 1) - Pt.slice(t + 1)) * G.t();
  }
  
  
  return logLik;
}
// 
// double nlg_ssm::iekf_smoother(const arma::mat& alphahat, arma::mat& at) const {
//   
//   at.col(0) = a1_fn.eval(theta, known_params);
//   
//   arma::cube Pt(m, m, n);
//   Pt.slice(0) = P1_fn.eval(theta, known_params);
//   arma::mat vt(p, n, arma::fill::zeros);
//   arma::cube ZFinv(m, p, n, arma::fill::zeros);
//   arma::cube Kt(m, p, n, arma::fill::zeros);
//   
//   arma::uvec obs(n, arma::fill::ones);
//   
//   arma::mat att(m, n);
//   const double LOG2PI = std::log(2.0 * M_PI);
//   double logLik = 0.0;
//   
//   for (unsigned int t = 0; t < (n - 1); t++) {
//     
//     arma::uvec na_y = arma::find_nonfinite(y.col(t));
//     
//     arma::mat Ptt = Pt.slice(t);
//     
//     if (na_y.n_elem < p) {
//       
//       arma::mat Zg = Z_gn.eval(t, alphahat.col(t), theta, known_params, known_tv_params);
//       Zg.rows(na_y).zeros();
//       arma::mat HHt = H_fn.eval(t, alphahat.col(t), theta, known_params, known_tv_params);
//       HHt = HHt * HHt.t();
//       HHt.submat(na_y, na_y) = arma::eye(na_y.n_elem, na_y.n_elem);
//       
//       arma::mat Ft = Zg * Pt.slice(t) * Zg.t() + HHt;
//       arma::mat cholF(p, p);
//       bool chol_ok = arma::chol(cholF,Ft);
//       if(!chol_ok) return -arma::datum::inf;
//       
//       vt.col(t) = y.col(t) - Zg * (at.col(t) - alphahat.col(t)) -
//         Z_fn.eval(t, alphahat.col(t), theta, known_params, known_tv_params);
//       vt.rows(na_y).zeros();
//       
//       arma::mat inv_cholF = arma::inv(arma::trimatu(cholF));
//       ZFinv.slice(t) = Zg.t() * inv_cholF * inv_cholF.t();
//       Kt.slice(t) = Pt.slice(t) * ZFinv.slice(t);
//       
//       att.col(t) = at.col(t) + Kt.slice(t) * vt.col(t);
//       Ptt = Pt.slice(t) - Kt.slice(t) * Ft * Kt.slice(t).t();
//       
//       arma::vec Fv = inv_cholF.t() * vt.col(t); 
//       logLik -= 0.5 * arma::as_scalar((p - na_y.n_elem) * LOG2PI + 
//         2.0 * arma::sum(log(arma::diagvec(cholF))) + Fv.t() * Fv);
//       
//     } else {
//       att.col(t) = at.col(t);
//       obs(t) = 0;
//     }
//     
//     arma::mat Tg = T_gn.eval(t, alphahat.col(t), theta, known_params, known_tv_params);
//     at.col(t + 1) = T_fn.eval(t, alphahat.col(t), theta, known_params, known_tv_params) +
//       Tg * (att.col(t) - alphahat.col(t));
//     arma::mat Rt = R_fn.eval(t, alphahat.col(t), theta, known_params, known_tv_params);
//     Pt.slice(t) = Tg * Ptt * Tg.t() + Rt * Rt.t();
//   }
//   
//   unsigned int t = n - 1;
//   arma::uvec na_y = arma::find_nonfinite(y.col(t));
//   
//   if (na_y.n_elem < p) {
//     arma::mat Zg = Z_gn.eval(t, alphahat.col(t), theta, known_params, known_tv_params);
//     Zg.rows(na_y).zeros();
//     arma::mat HHt = H_fn.eval(t, alphahat.col(t), theta, known_params, known_tv_params);
//     HHt = HHt * HHt.t();
//     HHt.submat(na_y, na_y) = arma::eye(na_y.n_elem, na_y.n_elem);
//     
//     arma::mat Ft = Zg * Pt.slice(t) * Zg.t() + HHt;
//     arma::mat cholF(p, p);
//     bool chol_ok = arma::chol(cholF,Ft);
//     if(!chol_ok) return -arma::datum::inf;
//     vt.col(t) = y.col(t) - Zg * (at.col(t) - alphahat.col(t)) -
//       Z_fn.eval(t, alphahat.col(t), theta, known_params, known_tv_params);
//     vt.rows(na_y).zeros();
//     
//     arma::mat inv_cholF = arma::inv(arma::trimatu(cholF));
//     ZFinv.slice(t) = Zg.t() * inv_cholF * inv_cholF.t();
//     Kt.slice(t) = Pt.slice(t) * ZFinv.slice(t);
//     att.col(t) = at.col(t) + Kt.slice(t) * vt.col(t);
//   } else {
//     att.col(t) = at.col(t);
//     obs(t) = 0;
//   }
//   
//   arma::vec rt(m, arma::fill::zeros);
//   for (int t = (n - 1); t > 0; t--) {
//     arma::mat Tg = T_gn.eval(t, alphahat.col(t), theta, known_params, known_tv_params);
//     if (obs(t)) {
//       arma::mat Zg = Z_gn.eval(t, alphahat.col(t), theta, known_params, known_tv_params);
//       arma::mat L = Tg * (arma::eye(m, m) - Kt.slice(t) * Zg);
//       rt = ZFinv.slice(t) * vt.col(t) + L.t() * rt;
//     } else {
//       rt = Tg.t() * rt;
//     }
//     at.col(t) += Pt.slice(t) * rt;
//   }
//   
//   return logLik;
// }


arma::cube nlg_ssm::predict_sample(const arma::mat& thetasim, 
  const arma::mat& alpha, const arma::uvec& counts, 
  const unsigned int predict_type) {
  
  unsigned int d = 1;
  if (predict_type == 3) d = m;
  
  unsigned int n_samples = thetasim.n_cols;
  arma::cube sample(d, n, n_samples);
  
  theta = thetasim.col(0);
  sample.slice(0) = sample_model(alpha.col(0), predict_type);
  
  for (unsigned int i = 1; i < n_samples; i++) {
    theta = thetasim.col(i);
    sample.slice(i) = sample_model(alpha.col(i), predict_type);
  }
  return rep_cube(sample, counts);
}

arma::mat nlg_ssm::sample_model(const arma::vec& a1_sim,
  const unsigned int predict_type) {
  
  arma::mat alpha(m, n);
  alpha.col(0) = a1_sim;
  std::normal_distribution<> normal(0.0, 1.0);
  
  for (unsigned int t = 0; t < (n - 1); t++) {
    arma::vec uk(k);
    for(unsigned int j = 0; j < k; j++) {
      uk(j) = normal(engine);
    }
    alpha.col(t + 1) = T_fn.eval(t, alpha.col(t), theta, known_params, known_tv_params) +  
      R_fn.eval(t, alpha.col(t), theta, known_params, known_tv_params) * uk;
  }
  
  if (predict_type < 3) {
    arma::mat y(p, n);
    for (unsigned int t = 0; t < n; t++) {
      y.col(t) = Z_fn.eval(t, alpha.col(t), theta, known_params, known_tv_params);
    } 
    if(predict_type == 1) {
      for (unsigned int t = 0; t < n; t++) {
        arma::vec up(p);
        for (unsigned int i = 0; i < p; i++) {
          up(i) = normal(engine);
        }
        y.col(t) += H_fn.eval(t, alpha.col(t), theta, known_params, known_tv_params) * up;
      }
    }
    return y;
  }
  return alpha;
  
}

// Unscented Kalman filter, Särkkä (2013) p.107 (UKF) and
// Note that the initial distribution is given for alpha_1
// so we first do update instead of prediction
double nlg_ssm::ukf(arma::mat& at, arma::mat& att, arma::cube& Pt, 
  arma::cube& Ptt, const double alpha, const double beta, const double kappa) const {
  
  // // Parameters of UKF, currently fixed for simplicity
  // double alpha = 1.0;
  // double beta = 0.0;
  // double kappa = 4.0;
  
  const double LOG2PI = std::log(2.0 * M_PI);
  double logLik = 0.0;
  
  double lambda = alpha * alpha * (m + kappa) - m;
  
  unsigned int n_sigma = 2 * m + 1;
  arma::vec wm(n_sigma);
  wm(0) = lambda / (lambda + m);
  wm.subvec(1, n_sigma - 1).fill(1.0 / (2.0 * (lambda + m)));
  arma::vec wc = wm;
  wc(0) +=  1.0 - alpha * alpha + beta;
  
  
  double sqrt_m_lambda = sqrt(m + lambda);
  
  at.col(0) = a1_fn.eval(theta, known_params);
  Pt.slice(0) = P1_fn.eval(theta, known_params);
  
  for (unsigned int t = 0; t < n; t++) {
    // update step
    
    // compute cholesky of Pt
    arma::mat cholP(m, m);
    cholP = psd_chol(Pt.slice(t));
    
    // form the sigma points
    arma::mat sigma(m, n_sigma);
    sigma.col(0) = at.col(t);
    for (unsigned int i = 1; i <= m; i++) {
      sigma.col(i) = at.col(t) + sqrt_m_lambda * cholP.col(i - 1);
      sigma.col(i + m) = at.col(t) - sqrt_m_lambda * cholP.col(i - 1);
    }
    
    arma::uvec obs_y = arma::find_finite(y.col(t));
    
    if (obs_y.n_elem > 0) {
      
      // propagate sigma points
      arma::mat sigma_y(obs_y.n_elem, n_sigma);
      for (unsigned int i = 0; i < n_sigma; i++) {
        sigma_y.col(i) = Z_fn.eval(t, sigma.col(i), theta, known_params, known_tv_params).rows(obs_y);
      }
      arma::vec pred_mean = sigma_y * wm;
      arma::mat pred_var = H_fn.eval(t, at.col(t), theta, known_params, known_tv_params).submat(obs_y, obs_y);
      arma::mat pred_cov(m, obs_y.n_elem, arma::fill::zeros);
      for (unsigned int i = 0; i < n_sigma; i++) {
        arma::vec tmp = sigma_y.col(i) - pred_mean;
        pred_var += wc(i) * tmp * tmp.t();
        pred_cov += wc(i) * (sigma.col(i) - at.col(t)) * tmp.t();
      }
      // filtered estimates
      arma::vec v = arma::mat(y.rows(obs_y)).col(t) - pred_mean;
      arma::mat K = arma::solve(pred_var, pred_cov.t()).t();
      att.col(t) = at.col(t) + K * v;
      Ptt.slice(t) = Pt.slice(t) - K * pred_var * K.t();
      
      arma::mat cholF = arma::chol(pred_var);
      arma::mat inv_cholF = arma::inv(arma::trimatu(cholF));
      arma::vec Fv = inv_cholF * v; 
      logLik -= 0.5 * arma::as_scalar(obs_y.n_elem * LOG2PI + 
        2.0 * arma::sum(log(arma::diagvec(cholF))) + Fv.t() * Fv);
    } else {
      att.col(t) = at.col(t);
      Ptt.slice(t) = Pt.slice(t);
    }
    
    // prediction
    // compute cholesky of Ptt
    arma::mat cholPtt = psd_chol(Ptt.slice(t));
    
    // form the sigma points and propagate
    sigma.col(0) = T_fn.eval(t, att.col(t), theta, known_params, known_tv_params);
    for (unsigned int i = 1; i <= m; i++) {
      sigma.col(i) = T_fn.eval(t, att.col(t) + sqrt_m_lambda * cholPtt.col(i - 1), 
        theta, known_params, known_tv_params);
      sigma.col(i + m) = T_fn.eval(t, att.col(t) - sqrt_m_lambda * cholPtt.col(i - 1), 
        theta, known_params, known_tv_params);
    }
    
    at.col(t + 1) = sigma * wm;
    
    arma::mat Rt = R_fn.eval(t, att.col(t), theta, known_params, known_tv_params);
    Pt.slice(t + 1) = Rt * Rt.t();
    for (unsigned int i = 0; i < n_sigma; i++) {
      arma::vec tmp = sigma.col(i) - at.col(t + 1);
      Pt.slice(t + 1) += wc(i) * tmp * tmp.t();
    }
  }
  return logLik;
}

// 
// // Unscented Kalman smoother, Särkkä (2013) p.107 (UKF) and
// // Note that the initial distribution is given for alpha_1
// // so we first do update instead of prediction
// double nlg_ssm::ukf_smoother(arma::mat& at, arma::cube& Pt) const {
// 
//   // Parameters of UKF, currently fixed for simplicity
//   double alpha = 1.0;
//   double beta = 0.0;
//   double kappa = 2.0;
// 
//   const double LOG2PI = std::log(2.0 * M_PI);
//   double logLik = 0.0;
// 
//   double lambda = alpha * alpha * (m + kappa) - m;
// 
//   unsigned int n_sigma = 2 * m + 1;
//   arma::vec wm(n_sigma);
//   wm(0) = lambda / (lambda + m);
//   wm.subvec(1, n_sigma - 1).fill(1.0 / (2.0 * (lambda + m)));
//   arma::vec wc = wm;
//   wc(0) +=  1.0 - alpha * alpha + beta;
// 
// 
//   double sqrt_m_lambda = sqrt(m + lambda);
//   arma::cube cholP(m, m, n);
// 
//   at.col(0) = a1_fn.eval(theta, known_params);
//   Pt.slice(0) = P1_fn.eval(theta, known_params);
// 
//   for (unsigned int t = 0; t < n; t++) {
//     // update step
// 
//     // compute cholesky of Pt
//     cholP.slice(t) = psd_chol(Pt.slice(t));
// 
//     // form the sigma points
//     arma::mat sigma(m, n_sigma);
//     sigma.col(0) = at.col(t);
//     for (unsigned int i = 1; i <= m; i++) {
//       sigma.col(i) = at.col(t) + sqrt_m_lambda * cholP.slice(t).col(i - 1);
//       sigma.col(i + m) = at.col(t) - sqrt_m_lambda * cholP.slice(t).col(i - 1);
//     }
// 
//     arma::uvec obs_y = arma::find_finite(y.col(t));
// 
//     if (obs_y.n_elem > 0) {
// 
//       // propagate sigma points
//       arma::mat sigma_y(obs_y.n_elem, n_sigma);
//       for (unsigned int i = 0; i < n_sigma; i++) {
//         sigma_y.col(i) = Z_fn.eval(t, sigma.col(i), theta, known_params, known_tv_params).rows(obs_y);
//       }
//       arma::vec pred_mean = sigma_y * wm;
//       arma::mat pred_var = H_fn.eval(t, at.col(t), theta, known_params, known_tv_params).submat(obs_y, obs_y);
//       arma::mat pred_cov(m, obs_y.n_elem, arma::fill::zeros);
//       for (unsigned int i = 0; i < n_sigma; i++) {
//         arma::vec tmp = sigma_y.col(i) - pred_mean;
//         pred_var += wc(i) * tmp * tmp.t();
//         pred_cov += wc(i) * (sigma.col(i) - at.col(t)) * tmp.t();
//       }
//       // filtered estimates
//       arma::vec v = arma::mat(y.rows(obs_y)).col(t) - pred_mean;
//       arma::mat K = arma::solve(pred_var, pred_cov.t()).t();
//       at.col(t) = at.col(t) + K * v;
//       Pt.slice(t) = Pt.slice(t) - K * pred_var * K.t();
// 
//       arma::mat cholF = arma::chol(pred_var);
//       arma::mat inv_cholF = arma::inv(arma::trimatu(cholF));
//       arma::vec Fv = inv_cholF * v;
//       logLik -= 0.5 * arma::as_scalar(obs_y.n_elem * LOG2PI +
//         2.0 * arma::sum(log(arma::diagvec(cholF))) + Fv.t() * Fv);
//     }
// 
//     // prediction
//     // compute cholesky of Ptt
//     arma::mat cholPtt = psd_chol(Pt.slice(t));
// 
//     // form the sigma points and propagate
//     sigma.col(0) = T_fn.eval(t, at.col(t), theta, known_params, known_tv_params);
//     for (unsigned int i = 1; i <= m; i++) {
//       sigma.col(i) = T_fn.eval(t, at.col(t) + sqrt_m_lambda * cholPtt.col(i - 1),
//         theta, known_params, known_tv_params);
//       sigma.col(i + m) = T_fn.eval(t, at.col(t) - sqrt_m_lambda * cholPtt.col(i - 1),
//         theta, known_params, known_tv_params);
//     }
// 
//     at.col(t + 1) = sigma * wm;
// 
//     arma::mat Rt = R_fn.eval(t, at.col(t), theta, known_params, known_tv_params);
//     Pt.slice(t + 1) = Rt * Rt.t();
//     for (unsigned int i = 0; i < n_sigma; i++) {
//       arma::vec tmp = sigma.col(i) - at.col(t + 1);
//       Pt.slice(t + 1) += wc(i) * tmp * tmp.t();
//     }
//   }
//   
//   // smoothing
//   
//   
//   return logLik;
// }
