#include "nlg_ssm.h"
#include "mgg_ssm.h"
#include "sample.h"
#include "dmvnorm.h"
#include "conditional_dist.h"

nlg_ssm::nlg_ssm(const arma::mat& y, const arma::vec a1, const arma::mat& P1,
  SEXP Z_fn_, SEXP H_fn_, SEXP T_fn_, SEXP R_fn_, 
  SEXP Z_gn_, SEXP H_gn_, SEXP T_gn_, SEXP R_gn_, 
  const arma::vec& theta, const unsigned int seed) :
  y(y), a1(a1), P1(P1), n(y.n_cols), m(a1.n_elem), p(y.n_rows),
  engine(seed), zero_tol(1e-8), seed(seed), theta(theta), 
  Z_fn(nonlinear_fn(Z_fn_)), H_fn(nonlinear_fn(H_fn_)), 
  T_fn(nonlinear_fn(T_fn_)), R_fn(nonlinear_fn(R_fn_)),
  Z_gn(nonlinear_gn(Z_gn_)), H_gn(nonlinear_gn(H_gn_)), 
  T_gn(nonlinear_gn(T_gn_)), R_gn(nonlinear_gn(R_gn_)) {
}

mgg_ssm nlg_ssm::approximate(arma::mat& mode_estimate, const unsigned int max_iter, 
  const double conv_tol) const {
  
  arma::mat D(p, n);
  arma::cube Z(p, m, n);
  arma::cube H(p, p, n);
  arma::mat C(m, n);
  arma::cube T(m, m, n);
  arma::cube R(m, m, n);
  
  for (unsigned int t = 0; t < n; t++) {
    Z.slice(t) = Z_gn.eval(mode_estimate.col(t), theta);
    D.col(t) = Z_fn.eval(mode_estimate.col(t), theta) - Z.slice(t) * mode_estimate.col(t);
    H.slice(t) = H_gn.eval(mode_estimate.col(t), theta);
    T.slice(t) = T_gn.eval(mode_estimate.col(t), theta);
    C.col(t) =  T_fn.eval(mode_estimate.col(t), theta) - T.slice(t) * mode_estimate.col(t);
    R.slice(t) = R_gn.eval(mode_estimate.col(t), theta);
  }
  
  mgg_ssm approx_model(y, Z, H, T, R, a1, P1, arma::cube(0,0,0), arma::mat(0,0), D, C, seed);
  
  unsigned int i = 0;
  double diff = conv_tol + 1; 
  
  while(i < max_iter && diff > conv_tol) {
    i++;
    // compute new guess of mode
    arma::mat mode_estimate_new = approx_model.fast_smoother();
    diff = arma::accu(arma::square(mode_estimate_new - mode_estimate)) / (m * n);
    mode_estimate = mode_estimate_new;
    
    for (unsigned int t = 0; t < n; t++) {
      approx_model.Z.slice(t) = Z_gn.eval(mode_estimate.col(t), theta);
      approx_model.D.col(t) = Z_fn.eval(mode_estimate.col(t), theta) - 
        approx_model.Z.slice(t) * mode_estimate.col(t);
      approx_model.H.slice(t) = H_gn.eval(mode_estimate.col(t), theta);
      approx_model.T.slice(t) = T_gn.eval(mode_estimate.col(t), theta);
      approx_model.C.col(t) =  T_fn.eval(mode_estimate.col(t), theta) - 
        approx_model.T.slice(t) * mode_estimate.col(t);
      approx_model.R.slice(t) = R_gn.eval(mode_estimate.col(t), theta);
      approx_model.compute_HH();
      approx_model.compute_RR();
    }
  }
  
  return approx_model;
}

// apart from using mgg_ssm, identical with ung_ssm::psi_filter
double nlg_ssm::psi_filter(const mgg_ssm& approx_model,
  const double approx_loglik, const arma::vec& scales,
  const unsigned int nsim, arma::cube& alpha, arma::mat& weights,
  arma::umat& indices) {
  
  arma::mat alphahat(m, n);
  arma::cube Vt(m, m, n);
  arma::cube Ct(m, m, n);
  approx_model.smoother_ccov(alphahat, Vt, Ct);
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

// Logarithms of _unnormalized_ importance weights g(y_t | alpha_t) / ~g(~y_t | alpha_t)
/*
 * approx_model:  Gaussian approximation of the original model
 * t:             Time point where the weights are computed
 * alpha:         Simulated particles
 */
arma::vec nlg_ssm::log_weights(const mgg_ssm& approx_model, 
  const unsigned int t, const arma::cube& alpha) const {
  
  arma::vec weights(alpha.n_slices, arma::fill::zeros);
  
  if (arma::is_finite(y(t))) {
    for (unsigned int i = 0; i < alpha.n_slices; i++) {
      weights(i) = 
        dmvnorm(y.col(t), Z_fn.eval(alpha.slice(i).col(t), theta), 
          H_fn.eval(alpha.slice(i).col(t), theta), true, true) -
            dmvnorm(y.col(t), approx_model.Z.slice(t) * alpha.slice(i).col(t),  
              approx_model.H.slice(t), true, true);
    }
  }
  return weights;
}

// compute unnormalized mode-based scaling terms
// log[g(y_t | ^alpha_t) / ~g(y_t | ^alpha_t)]
arma::vec nlg_ssm::scaling_factors(const mgg_ssm& approx_model,
  const arma::mat& mode_estimate) const {
  
  arma::vec weights(n, arma::fill::zeros);
  for(unsigned int t = 0; t < n; t++) {
    if (arma::is_finite(y(t))) {
      weights(t) =  dmvnorm(y.col(t), Z_fn.eval(mode_estimate.col(t), theta), 
        H_fn.eval(mode_estimate.col(t), theta), true, true) -
          dmvnorm(y.col(t), approx_model.Z.slice(t) * mode_estimate.col(t),  
            approx_model.H.slice(t), true, true);
    }
  }
  return weights;
}

// Logarithms of _unnormalized_ densities g(y_t | alpha_t)
/*
 * t:             Time point where the densities are computed
 * alpha:         Simulated particles
 */
arma::vec nlg_ssm::log_obs_density(const unsigned int t, 
  const arma::cube& alpha) const {
  
  arma::vec weights(alpha.n_slices, arma::fill::zeros);
  if (arma::is_finite(y(t))) {
    for (unsigned int i = 0; i < alpha.n_slices; i++) {
      weights(i) = dmvnorm(y.col(t), Z_fn.eval(alpha.slice(i).col(t), theta), 
        H_fn.eval(alpha.slice(i).col(t), theta), true, true);
    }
  }
  return weights;
}

double nlg_ssm::bsf_filter(const unsigned int nsim, arma::cube& alpha,
  arma::mat& weights, arma::umat& indices) {
  
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
      arma::vec um(m);
      for(unsigned int j = 0; j < m; j++) {
        um(j) = normal(engine);
      }
      alpha.slice(i).col(t + 1) = T_fn.eval(alphatmp.col(i), theta) + 
        R_fn.eval(alphatmp.col(i), theta) * um;
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
  // constant part of the log-likelihood omitted.??..
  return loglik;
}
