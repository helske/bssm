#include "ung_ssm.h"
#include "ugg_ssm.h"
#include "conditional_dist.h"
#include "distr_consts.h"
#include "sample.h"
#include "rep_mat.h"

// General constructor of ung_ssm object from Rcpp::List
// with parameter indices
ung_ssm::ung_ssm(const Rcpp::List& model, const unsigned int seed, 
  const arma::uvec& Z_ind, const arma::uvec& T_ind, const arma::uvec& R_ind) :
  y(Rcpp::as<arma::vec>(model["y"])), Z(Rcpp::as<arma::mat>(model["Z"])),
  T(Rcpp::as<arma::cube>(model["T"])), R(Rcpp::as<arma::cube>(model["R"])), 
  a1(Rcpp::as<arma::vec>(model["a1"])), P1(Rcpp::as<arma::mat>(model["P1"])), 
  xreg(Rcpp::as<arma::mat>(model["xreg"])), beta(Rcpp::as<arma::vec>(model["coefs"])), 
  D(Rcpp::as<arma::vec>(model["obs_intercept"])), 
  C(Rcpp::as<arma::mat>(model["state_intercept"])),
  Ztv(Z.n_cols > 1), Ttv(T.n_slices > 1), Rtv(R.n_slices > 1), Dtv(D.n_elem > 1), 
  Ctv(C.n_cols > 1),
  n(y.n_elem), m(a1.n_elem), k(R.n_cols), RR(arma::cube(m, m, Rtv * (n - 1) + 1)), 
  xbeta(arma::vec(n, arma::fill::zeros)), engine(seed), zero_tol(1e-8),
  Z_ind(Z_ind), T_ind(T_ind), R_ind(R_ind), phi(model["phi"]), 
  u(Rcpp::as<arma::vec>(model["u"])), distribution(model["distribution"]), 
  phi_est(Rcpp::as<bool>(model["phi_est"])), max_iter(100), conv_tol(1.0e-8), 
  seed(seed) {
  
  if(xreg.n_cols > 0) {
    compute_xbeta();
  }
  compute_RR();
}

void ung_ssm::compute_RR(){
  for (unsigned int t = 0; t < R.n_slices; t++) {
    RR.slice(t) = R.slice(t * Rtv) * R.slice(t * Rtv).t();
  }
}

// update system matrices given theta
void ung_ssm::set_theta(const arma::vec& theta) {
  
  if (Z_ind.n_elem > 0) {
    Z.elem(Z_ind) = theta.subvec(0, Z_ind.n_elem - 1);
  }
  if (T_ind.n_elem > 0) {
    T.elem(T_ind) = theta.subvec(Z_ind.n_elem, Z_ind.n_elem + T_ind.n_elem - 1);
  }
  if (R_ind.n_elem > 0) {
    R.elem(R_ind) = theta.subvec(Z_ind.n_elem + T_ind.n_elem,
      Z_ind.n_elem + T_ind.n_elem + R_ind.n_elem - 1);
  }
  
  if (R_ind.n_elem  > 0) {
    compute_RR();
  }
  
  if(phi_est) {
    phi = theta(Z_ind.n_elem + T_ind.n_elem + R_ind.n_elem);
  }
  if(xreg.n_cols > 0) {
    beta = theta.subvec(theta.n_elem - xreg.n_cols, theta.n_elem - 1);
    compute_xbeta();
  }
  
}

// pick up theta from system matrices
arma::vec ung_ssm::get_theta(void) const {
  
  // !! add phi when adding other distributions !!
  arma::vec theta(Z_ind.n_elem + T_ind.n_elem + R_ind.n_elem + (distribution == 3));
  
  if (Z_ind.n_elem > 0) {
    theta.subvec(0, Z_ind.n_elem - 1) = Z.elem(Z_ind);
  }
  if (T_ind.n_elem > 0) {
    theta.subvec(Z_ind.n_elem,  Z_ind.n_elem + T_ind.n_elem - 1) = T.elem(T_ind);
  }
  if (R_ind.n_elem > 0) {
    theta.subvec(Z_ind.n_elem + T_ind.n_elem,
      Z_ind.n_elem + T_ind.n_elem + R_ind.n_elem - 1) = R.elem(R_ind);
  }
  
  if(phi_est) {
    theta(Z_ind.n_elem + T_ind.n_elem + R_ind.n_elem) = phi;
  }
  if(xreg.n_cols > 0) {
    theta.subvec(theta.n_elem - xreg.n_cols,
      theta.n_elem - 1) = beta;
  }
  
  
  return theta;
}

// given the current guess of mode, compute new values of y and H of 
// approximate model 
/* distribution:
 * 0 = Stochastic volatility model
 * 1 = Poisson
 * 2 = Binomial
 * 3 = Negative binomial
 */
///// TODO: Change to enums later!!!!
void ung_ssm::laplace_iter(const arma::vec& signal, arma::vec& approx_y, 
  arma::vec& approx_H) const {
  
  //note: using the variable approx_H to store approx_HH first
  
  switch(distribution) {
  case 0: {
    arma::vec tmp = y;
    // avoid dividing by zero
    tmp(arma::find(abs(tmp) < 1e-4)).fill(1e-4);
    approx_H = 2.0 * exp(signal) / pow(tmp/phi, 2);
    approx_y = signal + 1.0 - 0.5 * approx_H;
} break;
  case 1: {
    arma::vec tmp = signal + xbeta;
    approx_H = 1.0 / (exp(tmp) % u);
    approx_y = y % approx_H + tmp - 1.0;
  } break;
  case 2: {
    arma::vec exptmp = exp(signal + xbeta);
    approx_H = pow(1.0 + exptmp, 2) / (u % exptmp);
    approx_y = y % approx_H + signal + xbeta - 1.0 - exptmp;
  } break;
  case 3: {
    arma::vec exptmp = 1.0 / (exp(signal + xbeta) % u);
    approx_H = 1.0 / phi + exptmp;
    approx_y = signal + xbeta + y % exptmp - 1.0;
  } break;
  }
  approx_H = sqrt(approx_H);
}

// construct an approximating Gaussian model
// Note the difference to previous versions, the convergence is assessed only
// by checking the changes in mode, not the actual function values. This is 
// slightly faster and sufficient as the approximation doesn't need to be accurate.
// Using function values would be safer though, as we could use line search etc
// in case of potential divergence etc...
ugg_ssm ung_ssm::approximate(arma::vec& mode_estimate, const unsigned int max_iter, 
  const double conv_tol) const {
  //Construct y and H for the Gaussian model
  arma::vec approx_y(n);
  arma::vec approx_H(n);
  ugg_ssm approx_model(approx_y, Z, approx_H, T, R, a1, P1, xreg, beta, D, C, seed);
  
  unsigned int i = 0;
  double diff = conv_tol + 1; 
  while(i < max_iter && diff > conv_tol) {
    i++;
    //Construct y and H for the Gaussian model
    laplace_iter(mode_estimate, approx_model.y, approx_model.H);
    approx_model.compute_HH();
    // compute new guess of mode
    arma::vec mode_estimate_new(n);
    if (distribution == 0) {
      mode_estimate_new = arma::vectorise(approx_model.fast_smoother());
    } else {
      arma::mat alpha = approx_model.fast_smoother();
      for (unsigned int t = 0; t < n; t++) {
        mode_estimate_new(t) = arma::as_scalar(Z.col(Ztv * t).t() * alpha.col(t));
      }
    }
    diff = arma::mean(arma::square(mode_estimate_new - mode_estimate));
    mode_estimate = mode_estimate_new;
  }

  return approx_model;
}

//update previously obtained approximation
void ung_ssm::approximate(ugg_ssm& approx_model, arma::vec& mode_estimate, 
  const unsigned int max_iter, const double conv_tol) const {

  //update model
  approx_model.Z = Z;
  approx_model.T = T;
  approx_model.R = R;
  approx_model.a1 = a1;
  approx_model.P1 = P1;
  approx_model.beta = beta;
  approx_model.D = D;
  approx_model.C = C;
  approx_model.RR = RR;
  approx_model.xbeta = xbeta;
  
  unsigned int i = 0;
  double diff = conv_tol + 1; 
  while(i < max_iter && diff > conv_tol) {
    i++;
    //Construct y and H for the Gaussian model
    laplace_iter(mode_estimate, approx_model.y, approx_model.H);
    approx_model.compute_HH();
    // compute new guess of mode
    arma::vec mode_estimate_new(n);
    if (distribution == 0) {
      mode_estimate_new = arma::vectorise(approx_model.fast_smoother());
    } else {
      arma::mat alpha = approx_model.fast_smoother();
      for (unsigned int t = 0; t < n; t++) {
        mode_estimate_new(t) = arma::as_scalar(Z.col(Ztv * t).t() * alpha.col(t));
      }
    }
    diff = arma::mean(arma::square(mode_estimate_new - mode_estimate));
    mode_estimate = mode_estimate_new;
  }
 
}


// psi particle filter using Gaussian approximation //

/* 
 * approx_model:  Gaussian approximation of the original model
 * approx_loglik: approximate log-likelihood
 *                sum(log[g(y_t | ^alpha_t) / ~g(~y_t | ^alpha_t)]) + loglik(approx_model)
 * scales:        log[g_u(y_t | ^alpha_t) / ~g_u(~y_t | ^alpha_t)] for each t,
 *                where g_u and ~g_u are the unnormalized densities
 * nsim:          Number of particles
 * alpha:         Simulated particles
 * weights:       Potentials g(y_t | alpha_t) / ~g(~y_t | alpha_t)
 * indices:       Indices from resampling, alpha.slice(ind(i, t)).col(t) is 
 *                the ancestor of alpha.slice(i).col(t + 1) 
 */

double ung_ssm::psi_filter(const ugg_ssm& approx_model,
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
arma::vec ung_ssm::log_weights(const ugg_ssm& approx_model, 
  const unsigned int t, const arma::cube& alpha) const {
  
  arma::vec weights(alpha.n_slices, arma::fill::zeros);
  
  if (arma::is_finite(y(t))) {
    switch(distribution) {
    case 0  :
      for (unsigned int i = 0; i < alpha.n_slices; i++) {
        double simsignal = alpha(0, t, i);
        weights(i) = -0.5 * (simsignal + pow(y(t) / phi, 2) * exp(-simsignal)) +
          0.5 * std::pow(approx_model.y(t) - simsignal, 2) / approx_model.HH(t);
      }
      break;
    case 1  :
      for (unsigned int i = 0; i < alpha.n_slices; i++) {
        double simsignal = arma::as_scalar(Z.col(t * Ztv).t() *
          alpha.slice(i).col(t) + xbeta(t));
        weights(i) = y(t) * simsignal  - u(t) * exp(simsignal) +
          0.5 * std::pow(approx_model.y(t) - simsignal, 2) / approx_model.HH(t);
      }
      break;
    case 2  :
      for (unsigned int i = 0; i < alpha.n_slices; i++) {
        double simsignal = arma::as_scalar(Z.col(t * Ztv).t() *
          alpha.slice(i).col(t) + xbeta(t));
        weights(i) = y(t) * simsignal - u(t) * log1p(exp(simsignal)) +
          0.5 * std::pow(approx_model.y(t) - simsignal, 2) / approx_model.HH(t);
      }
      break;
    case 3  :
      for (unsigned int i = 0; i < alpha.n_slices; i++) {
        double simsignal = arma::as_scalar(Z.col(t * Ztv).t() *
          alpha.slice(i).col(t) + xbeta(t));
        weights(i) = y(t) * simsignal - (y(t) + phi) *
          log(phi + u(t) * exp(simsignal)) +
          0.5 * std::pow(approx_model.y(t) - simsignal, 2) / approx_model.HH(t);
      }
      break;
    }
  }
  return weights;
}

// compute unnormalized mode-based scaling terms
// log[g(y_t | ^alpha_t) / ~g(y_t | ^alpha_t)]
arma::vec ung_ssm::scaling_factors(const ugg_ssm& approx_model,
  const arma::vec& mode_estimate) const {
  
  arma::vec weights(n, arma::fill::zeros);
  
  switch(distribution) {
  case 0  :
    for(unsigned int t = 0; t < n; t++) {
      if (arma::is_finite(y(t))) {
        weights(t) = -0.5 * (mode_estimate(t) + pow(y(t) / phi, 2) *
          exp(-mode_estimate(t))) +
          0.5 * std::pow(approx_model.y(t) - mode_estimate(t), 2) / approx_model.HH(t);
      }
    }
    break;
  case 1  :
    for(unsigned int t = 0; t < n; t++) {
      if (arma::is_finite(y(t))) {
        weights(t) = y(t) * (mode_estimate(t) + xbeta(t)) -
          u(t) * exp(mode_estimate(t) + xbeta(t)) +
          0.5 * std::pow(approx_model.y(t) - (mode_estimate(t) + xbeta(t)), 2) /
            approx_model.HH(t);
      }
    }
    break;
  case 2  :
    for(unsigned int t = 0; t < n; t++) {
      if (arma::is_finite(y(t))) {
        weights(t) = y(t) * (mode_estimate(t) + xbeta(t)) -
          u(t) * log1p(exp(mode_estimate(t) + xbeta(t))) +
          0.5 * std::pow(approx_model.y(t) - (mode_estimate(t) + xbeta(t)), 2) /
            approx_model.HH(t);
      }
    }
    break;
  case 3  :
    for(unsigned int t = 0; t < n; t++) {
      if (arma::is_finite(y(t))) {
        weights(t) = y(t) * (mode_estimate(t) + xbeta(t)) -
          (y(t) + phi) *
          log(phi + u(t) * exp(mode_estimate(t) + xbeta(t))) +
          0.5 * std::pow(approx_model.y(t) - (mode_estimate(t) + xbeta(t)), 2) /
            approx_model.HH(t);
      }
    }
    break;
  }
  
  return weights;
}

// Logarithms of _unnormalized_ densities g(y_t | alpha_t)
/*
 * t:             Time point where the densities are computed
 * alpha:         Simulated particles
 */
arma::vec ung_ssm::log_obs_density(const unsigned int t, 
  const arma::cube& alpha) const {
  
  arma::vec weights(alpha.n_slices, arma::fill::zeros);
  
  if (arma::is_finite(y(t))) {
    switch(distribution) {
    case 0  :
      for (unsigned int i = 0; i < alpha.n_slices; i++) {
        double simsignal = alpha(0, t, i);
        weights(i) = -0.5 * (simsignal + pow(y(t) / phi, 2) * exp(-simsignal));
      }
      break;
    case 1  :
      for (unsigned int i = 0; i < alpha.n_slices; i++) {
        double simsignal = arma::as_scalar(Z.col(t * Ztv).t() *
          alpha.slice(i).col(t) + xbeta(t));
        weights(i) = y(t) * simsignal  - u(t) * exp(simsignal);
      }
      break;
    case 2  :
      for (unsigned int i = 0; i < alpha.n_slices; i++) {
        double simsignal = arma::as_scalar(Z.col(t * Ztv).t() *
          alpha.slice(i).col(t) + xbeta(t));
        weights(i) = y(t) * simsignal - u(t) * log1p(exp(simsignal));
      }
      break;
    case 3  :
      for (unsigned int i = 0; i < alpha.n_slices; i++) {
        double simsignal = arma::as_scalar(Z.col(t * Ztv).t() *
          alpha.slice(i).col(t) + xbeta(t));
        weights(i) = y(t) * simsignal - (y(t) + phi) * 
          log(phi + u(t) * exp(simsignal));
      }
      break;
    }
  }
  return weights;
}

double ung_ssm::bsf_filter(const unsigned int nsim, arma::cube& alpha,
  arma::mat& weights, arma::umat& indices) {
  
  // arma::mat U(m, m);
  // arma::mat V(m, m);
  // arma::vec s(m);
  // arma::svd_econ(U, s, V, P1, "left");
  // arma::uvec nonzero = arma::find(s > (arma::datum::eps * m * s(0)));
  // arma::mat L = arma::diagmat(1.0 / s(nonzero)) U
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
      alpha.slice(i).col(t + 1) = C.col(t * Ctv) + 
        T.slice(t * Ttv) * alphatmp.col(i) + R.slice(t * Rtv) * uk;
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
  // constant part of the log-likelihood
  switch(distribution) {
  case 0 :
    loglik += arma::uvec(arma::find_finite(y)).n_elem * norm_log_const(phi);
    break;
  case 1 : {
      arma::uvec finite_y(find_finite(y));
      loglik += poisson_log_const(y(finite_y), u(finite_y));
    } break;
  case 2 : {
    arma::uvec finite_y(find_finite(y));
    loglik += binomial_log_const(y(finite_y), u(finite_y));
  } break;
  case 3 : {
    arma::uvec finite_y(find_finite(y));
    loglik += negbin_log_const(y(finite_y), u(finite_y), phi);
  } break;
  }
  return loglik;
}

arma::cube ung_ssm::predict_sample(const arma::mat& theta,
  const arma::mat& alpha, const arma::uvec& counts, 
  const unsigned int predict_type) {
  
  unsigned int d = 1;
  if (predict_type == 3) d = m;
  
  unsigned int n_samples = theta.n_cols;
  arma::cube sample(d, n, n_samples);
 
  arma::vec theta_i = theta.col(0);
  set_theta(theta_i);
  a1 = alpha.col(0);
  sample.slice(0) = sample_model(predict_type);
  
  for (unsigned int i = 1; i < n_samples; i++) {
    arma::vec theta_i = theta.col(i);
    set_theta(theta_i);
    a1 = alpha.col(i);
    sample.slice(i) = sample_model(predict_type);
  }
  
  return rep_cube(sample, counts);
}


arma::mat ung_ssm::sample_model(const unsigned int predict_type) {
  
  arma::mat alpha(m, n);
  alpha.col(0) = a1;
  std::normal_distribution<> normal(0.0, 1.0);
  
  for (unsigned int t = 0; t < (n - 1); t++) {
    arma::vec uk(k);
    for(unsigned int j = 0; j < k; j++) {
      uk(j) = normal(engine);
    }
    alpha.col(t + 1) = C.col(t * Ctv) + T.slice(t * Ttv) * alpha.col(t) + 
      R.slice(t * Rtv) * uk;
  }
  
  if (predict_type < 3) {
    
    arma::mat y(1, n);
    
    switch(distribution) {
    case 0: 
      y.zeros(); 
      break;
    case 1: 
      for (unsigned int t = 0; t < n; t++) {
        y(0, t) = exp(xbeta(t) + D(t * Dtv) + 
          arma::as_scalar(Z.col(t * Ztv).t() * alpha.col(t)));
      }
      break;
    case 2: 
      for (unsigned int t = 0; t < n; t++) {
        double tmp = exp(xbeta(t) + D(t * Dtv) + 
          arma::as_scalar(Z.col(t * Ztv).t() * alpha.col(t)));
        y(0, t) = tmp / (1.0 + tmp);
      }
      break;
    case 3:
      for (unsigned int t = 0; t < n; t++) {
        y(0, t) = exp(xbeta(t) + D(t * Dtv) + 
          arma::as_scalar(Z.col(t * Ztv).t() * alpha.col(t)));
      }
      break;
    }
    
    if (predict_type == 1) {
      
      switch(distribution) {
      case 0:
        break;
      case 1:
        for (unsigned int t = 0; t < n; t++) {
          std::poisson_distribution<> poisson(u(t) * y(0, t));
          if ((u(t) * y(0, t)) < poisson.max()) {
            y(0, t) = poisson(engine);
          } else {
            y(0, t) = arma::datum::nan;
          }
        } 
        break;
      case 2: 
        for (unsigned int t = 0; t < n; t++) {
          std::binomial_distribution<> binomial(u(t), y(0, t));
          y(0, t) = binomial(engine);
        }
        break;
      case 3: 
        for (unsigned int t = 0; t < n; t++) {
          std::negative_binomial_distribution<> 
          negative_binomial(phi, phi / (phi + u(t) * y(0, t)));
          y(0, t) = negative_binomial(engine);
        }
        break;
      }
    } 
    return y;
  }
  return alpha;
}

//
//
// /////////////////////////////////
// //compute log-weights
// arma::vec ung_ssm::weights_t(const unsigned int t, const arma::cube& alphasim) {
//   
//   arma::vec weights(alphasim.n_slices, arma::fill::zeros);
//   if (arma::is_finite(ng_y(t))) {
//     
//     switch(distribution) {
//     case 0  :
//       for (unsigned int i = 0; i < alphasim.n_slices; i++) {
//         double simsignal = alphasim(0, t, i);
//         weights(i) = -0.5 * (simsignal +
//           pow((ng_y(t) - xbeta(t)) / phi, 2) * exp(-simsignal)) +
//           0.5 * std::pow(y(t) - simsignal, 2) / HH(t);
//       }
//       break;
//     case 1  :
//       for (unsigned int i = 0; i < alphasim.n_slices; i++) {
//         double simsignal = arma::as_scalar(Z.col(t * Ztv).t() *
//           alphasim.slice(i).col(t) + xbeta(t));
//         weights(i) = ng_y(t) * simsignal  - ut(t) * exp(simsignal) +
//           0.5 * std::pow(y(t) - simsignal, 2) / HH(t);
//       }
//       break;
//     case 2  :
//       for (unsigned int i = 0; i < alphasim.n_slices; i++) {
//         double simsignal = arma::as_scalar(Z.col(t * Ztv).t() *
//           alphasim.slice(i).col(t) + xbeta(t));
//         weights(i) = ng_y(t) * simsignal - ut(t) * log1p(exp(simsignal)) +
//           0.5 * std::pow(y(t) - simsignal, 2) / HH(t);
//       }
//       break;
//     case 3  :
//       for (unsigned int i = 0; i < alphasim.n_slices; i++) {
//         double simsignal = arma::as_scalar(Z.col(t * Ztv).t() *
//           alphasim.slice(i).col(t) + xbeta(t));
//         weights(i) = ng_y(t) * simsignal - (ng_y(t) + phi) *
//           log(phi + ut(t) * exp(simsignal)) +
//           0.5 * std::pow(y(t) - simsignal, 2) / HH(t);
//       }
//       break;
//     }
//   }
//   return weights;
// }
// 
// 
// //////////////////////////////////////////////////////////////////////////7
// ///////////////////////////////////////////////////////////////////////////
// 
// arma::mat ung_ssm::predict2(const arma::uvec& prior_types,
//   const arma::mat& prior_pars, unsigned int n_iter, unsigned int nsim_states,
//   unsigned int n_burnin, unsigned int n_thin, double gamma,
//   double target_acceptance, arma::mat S, unsigned int n_ahead,
//   unsigned int interval, arma::vec init_signal) {
//   
//   unsigned int n_samples = floor((n_iter - n_burnin) / n_thin);
//   
//   arma::mat pred_store(n_ahead, nsim_states * n_samples);
//   
//   unsigned int npar = prior_types.n_elem;
//   arma::vec theta = get_theta();
//   double prior = prior_pdf(theta, prior_types, prior_pars);
//   arma::vec signal = init_signal;
//   double ll_approx = approx(signal, max_iter, conv_tol); // log[p(y_ng|alphahat)/g(y|alphahat)]
//   double ll = ll_approx + log_likelihood(distribution != 0);
//   
//   arma::cube alpha_pred(m, n_ahead, nsim_states);
//   double ll_w = 0;
//   if (nsim_states > 1) {
//     arma::cube alpha = sim_smoother(nsim_states, distribution != 0);
//     arma::vec weights = exp(importance_weights(alpha) - scaling_factor(signal));
//     ll_w = log(sum(weights) / nsim_states);
//     // sample from p(alpha | y)
//     std::discrete_distribution<> sample(weights.begin(), weights.end());
//     for (unsigned int ii = 0; ii < nsim_states; ii++) {
//       alpha_pred.slice(ii) = alpha.slice(sample(engine)).cols(n - n_ahead, n - 1);
//     }
//   } else {
//     alpha_pred = sim_smoother(nsim_states, distribution != 0).tube(0, n - n_ahead, m - 1,  n - 1);
//   }
//   
//   unsigned int j = 0;
//   
//   if (n_burnin == 0){
//     for (unsigned int ii = 0; ii < nsim_states; ii++) {
//       for (unsigned int t = 0; t < n_ahead; t++) {
//         pred_store(t, ii) = arma::as_scalar(Z.col(Ztv * (n - n_ahead + t)).t() * alpha_pred.slice(ii).col(t));
//       }
//     }
//     if(xreg.n_cols > 0) {
//       for (unsigned int ii = 0; ii < nsim_states; ii++) {
//         pred_store.col(ii) +=  xbeta.subvec(n - n_ahead, n - 1);
//       }
//     }
//     
//     if (interval == 1) {
//       switch(distribution) {
//       case 1  :
//         for (unsigned int ii = 0; ii < nsim_states; ii++) {
//           pred_store.col(ii) = exp(pred_store.col(ii)) % ut.subvec(n - n_ahead, n - 1);
//         }
//         break;
//       case 2  :
//         for (unsigned int ii = 0; ii < nsim_states; ii++) {
//           pred_store.col(ii) = exp(pred_store.col(ii)) / (1.0 + exp(pred_store.col(ii)));
//         }
//         break;
//       case 3  :
//         for (unsigned int ii = 0; ii < nsim_states; ii++) {
//           pred_store.col(ii) = exp(pred_store.col(ii)) % ut.subvec(n - n_ahead, n - 1);
//         }
//         break;
//       }
//     } else {
//       switch(distribution) {
//       case 1  :
//         for (unsigned int ii = 0; ii < nsim_states; ii++) {
//           for (unsigned int t = 0; t < n_ahead; t++) {
//             pred_store(t, ii) = R::rpois(exp(pred_store(t, ii)) * ut(n - n_ahead + t));
//           }
//         }
//         break;
//       case 2  :
//         for (unsigned int ii = 0; ii < nsim_states; ii++) {
//           for (unsigned int t = 0; t < n_ahead; t++) {
//             pred_store(t, ii) = R::rbinom(ut(n - n_ahead + t), exp(pred_store(t, ii)) /
//               (1.0 + exp(pred_store(t, ii))));
//           }
//         }
//         break;
//       case 3  :
//         for (unsigned int ii = 0; ii < nsim_states; ii++) {
//           for (unsigned int t = 0; t < n_ahead; t++) {
//             pred_store(t, ii) = R::rnbinom(phi, exp(pred_store(t, ii)) *
//               ut(n - n_ahead + t));
//           }
//         }
//         break;
//       }
//     }
//     j++;
//   }
//   
//   double accept_prob = 0;
//   double ll_prop = 0;
//   std::normal_distribution<> normal(0.0, 1.0);
//   std::uniform_real_distribution<> unif(0.0, 1.0);
//   for (unsigned int i = 1; i < n_iter; i++) {
//     // sample from standard normal distribution
//     //arma::vec u = rnorm(npar);
//     arma::vec u(npar);
//     for(unsigned int ii = 0; ii < npar; ii++) {
//       u(ii) = normal(engine);
//     }
//     // propose new theta
//     arma::vec theta_prop = theta + S * u;
//     // check prior
//     double prior_prop = prior_pdf(theta_prop, prior_types, prior_pars);
//     
//     if (prior_prop > -arma::datum::inf) {
//       // update parameters
//       update_model(theta_prop);
//       // compute approximate log-likelihood with proposed theta
//       signal = init_signal;
//       ll_approx = approx(signal, max_iter, conv_tol);
//       ll_prop = ll_approx + log_likelihood(distribution != 0);
//       //compute the acceptance probability
//       // use explicit min(...) as we need this value later
//       double q = proposal(theta, theta_prop);
//       accept_prob = std::min(1.0, exp(ll_prop - ll + prior_prop - prior + q));
//     } else accept_prob = 0;
//     
//     //accept
//     if (unif(engine) < accept_prob) {
//       if (nsim_states > 1) {
//         arma::cube alpha = sim_smoother(nsim_states, distribution != 0);
//         arma::vec weights = exp(importance_weights(alpha) - scaling_factor(signal));
//         double ll_w_prop = log(sum(weights) / nsim_states);
//         double pp = std::min(1.0, exp(ll_w_prop - ll_w));
//         //accept_prob *= pp;
//         
//         if (unif(engine) < pp) {
//           ll = ll_prop;
//           ll_w = ll_w_prop;
//           theta = theta_prop;
//           std::discrete_distribution<> sample(weights.begin(), weights.end());
//           for (unsigned int ii = 0; ii < nsim_states; ii++) {
//             alpha_pred.slice(ii) = alpha.slice(sample(engine)).cols(n - n_ahead, n - 1);
//           }
//         }
//       } else {
//         ll = ll_prop;
//         theta = theta_prop;
//         alpha_pred = sim_smoother(nsim_states, distribution != 0).tube(0, n - n_ahead, m - 1,  n - 1);
//       }
//     }
//     if ((i >= n_burnin) && (i % n_thin == 0)) {
//       update_model(theta);
//       for (unsigned int ii = j * nsim_states; ii < (j + 1) * nsim_states; ii++) {
//         for (unsigned int t = 0; t < n_ahead; t++) {
//           pred_store(t, ii) = arma::as_scalar(Z.col(Ztv * (n - n_ahead + t)).t() *
//             alpha_pred.slice(ii - j * nsim_states).col(t));
//         }
//       }
//       if(xreg.n_cols > 0) {
//         for (unsigned int ii = j * nsim_states; ii < (j + 1) * nsim_states; ii++) {
//           pred_store.col(ii) += xbeta.subvec(n - n_ahead, n - 1);
//         }
//       }
//       if (interval == 1) {
//         switch(distribution) {
//         case 1  :
//           for (unsigned int ii = j * nsim_states; ii < (j + 1) * nsim_states; ii++) {
//             pred_store.col(ii) = exp(pred_store.col(ii)) % ut.subvec(n - n_ahead, n - 1);
//           }
//           break;
//         case 2  :
//           for (unsigned int ii = j * nsim_states; ii < (j + 1) * nsim_states; ii++) {
//             pred_store.col(ii) = exp(pred_store.col(ii)) / (1.0 + exp(pred_store.col(ii)));
//           }
//           break;
//         case 3  :
//           for (unsigned int ii = j * nsim_states; ii < (j + 1) * nsim_states; ii++) {
//             pred_store.col(ii) = exp(pred_store.col(ii)) % ut.subvec(n - n_ahead, n - 1);
//           }
//           break;
//         }
//       } else {
//         switch(distribution) {
//         case 1  :
//           for (unsigned int ii = j * nsim_states; ii < (j + 1) * nsim_states; ii++) {
//             for (unsigned int t = 0; t < n_ahead; t++) {
//               pred_store(t, ii) = R::rpois(exp(pred_store(t, ii)) * ut(n - n_ahead + t));
//             }
//           }
//           break;
//         case 2  :
//           for (unsigned int ii = j * nsim_states; ii < (j + 1) * nsim_states; ii++) {
//             for (unsigned int t = 0; t < n_ahead; t++) {
//               pred_store(t, ii) = R::rbinom(ut(n - n_ahead + t), exp(pred_store(t, ii)) / (1.0 + exp(pred_store(t, ii))));
//             }
//           }
//           break;
//         case 3  :
//           for (unsigned int ii = j * nsim_states; ii < (j + 1) * nsim_states; ii++) {
//             for (unsigned int t = 0; t < n_ahead; t++) {
//               pred_store(t, ii) = R::rnbinom(phi, exp(pred_store(t, ii)) *
//                 ut(n - n_ahead + t));
//             }
//           }
//           break;
//         }
//       }
//       j++;
//     }
//     
//     ramcmc::adapt_S(S, u, accept_prob, target_acceptance, i, gamma);
//     
//   }
//   
//   return pred_store;
//   
// }
// //compute log-weights
// arma::vec ung_ssm::importance_weights(const arma::cube& alphasim) {
//   
//   arma::vec weights(alphasim.n_slices, arma::fill::zeros);
//   
//   switch(distribution) {
//   case 0  :
//     for (unsigned int i = 0; i < alphasim.n_slices; i++) {
//       for (unsigned int t = 0; t < n; t++) {
//         if (arma::is_finite(ng_y(t))) {
//           double simsignal = alphasim(0, t, i);
//           weights(i) += -0.5 * (simsignal +
//             pow((ng_y(t) - xbeta(t)) / phi, 2) * exp(-simsignal)) +
//             0.5 * std::pow(y(t) - simsignal, 2) / HH(t);
//         }
//       }
//     }
//     break;
//   case 1  :
//     for (unsigned int i = 0; i < alphasim.n_slices; i++) {
//       for (unsigned int t = 0; t < n; t++) {
//         if (arma::is_finite(ng_y(t))) {
//           double simsignal = arma::as_scalar(Z.col(t * Ztv).t() *
//             alphasim.slice(i).col(t) + xbeta(t));
//           weights(i) += ng_y(t) * simsignal  - ut(t) * exp(simsignal) +
//             0.5 * std::pow(y(t) - simsignal, 2) / HH(t);
//         }
//       }
//     }
//     break;
//   case 2  :
//     for (unsigned int i = 0; i < alphasim.n_slices; i++) {
//       for (unsigned int t = 0; t < n; t++) {
//         if (arma::is_finite(ng_y(t))) {
//           double simsignal = arma::as_scalar(Z.col(t * Ztv).t() *
//             alphasim.slice(i).col(t) + xbeta(t));
//           weights(i) += ng_y(t) * simsignal - ut(t) * log1p(exp(simsignal)) +
//             0.5 * std::pow(y(t) - simsignal, 2) / HH(t);
//         }
//       }
//     }
//     break;
//   case 3  :
//     for (unsigned int i = 0; i < alphasim.n_slices; i++) {
//       for (unsigned int t = 0; t < n; t++) {
//         if (arma::is_finite(ng_y(t))) {
//           double simsignal = arma::as_scalar(Z.col(t * Ztv).t() *
//             alphasim.slice(i).col(t) + xbeta(t));
//           weights(i) += ng_y(t) * simsignal - (ng_y(t) + phi) *
//             log(phi + ut(t) * exp(simsignal)) +
//             0.5 * std::pow(y(t) - simsignal, 2) / HH(t);
//         }
//       }
//     }
//     break;
//   }
//   
//   return weights;
// }
// //compute log-weights
// arma::vec ung_ssm::importance_weights_t(const unsigned int t, const arma::cube& alphasim) {
//   
//   arma::vec weights(alphasim.n_slices, arma::fill::zeros);
//   if (arma::is_finite(ng_y(t))) {
//     
//     switch(distribution) {
//     case 0  :
//       for (unsigned int i = 0; i < alphasim.n_slices; i++) {
//         double simsignal = alphasim(0, t, i);
//         weights(i) = -0.5 * (simsignal +
//           pow((ng_y(t) - xbeta(t)) / phi, 2) * exp(-simsignal)) +
//           0.5 * std::pow(y(t) - simsignal, 2) / HH(t);
//       }
//       break;
//     case 1  :
//       for (unsigned int i = 0; i < alphasim.n_slices; i++) {
//         double simsignal = arma::as_scalar(Z.col(t * Ztv).t() *
//           alphasim.slice(i).col(t) + xbeta(t));
//         weights(i) = ng_y(t) * simsignal  - ut(t) * exp(simsignal) +
//           0.5 * std::pow(y(t) - simsignal, 2) / HH(t);
//       }
//       break;
//     case 2  :
//       for (unsigned int i = 0; i < alphasim.n_slices; i++) {
//         double simsignal = arma::as_scalar(Z.col(t * Ztv).t() *
//           alphasim.slice(i).col(t) + xbeta(t));
//         weights(i) = ng_y(t) * simsignal - ut(t) * log1p(exp(simsignal)) +
//           0.5 * std::pow(y(t) - simsignal, 2) / HH(t);
//       }
//       break;
//     case 3  :
//       for (unsigned int i = 0; i < alphasim.n_slices; i++) {
//         double simsignal = arma::as_scalar(Z.col(t * Ztv).t() *
//           alphasim.slice(i).col(t) + xbeta(t));
//         weights(i) = ng_y(t) * simsignal - (ng_y(t) + phi) *
//           log(phi + ut(t) * exp(simsignal)) +
//           0.5 * std::pow(y(t) - simsignal, 2) / HH(t);
//       }
//       break;
//     }
//   }
//   return weights;
// }
// //compute log-weights
// arma::vec ung_ssm::importance_weights_t(const unsigned int t, const arma::mat& alphasim) {
//   
//   arma::vec weights(alphasim.n_cols);
//   if (arma::is_finite(ng_y(t))) {
//     
//     switch(distribution) {
//     case 0  :
//       for (unsigned int i = 0; i < alphasim.n_cols; i++) {
//         double simsignal = alphasim(0, i);
//         weights(i) = -0.5 * (simsignal +
//           pow((ng_y(t) - xbeta(t)) / phi, 2) * exp(-simsignal)) +
//           0.5 * std::pow(y(t) - simsignal, 2) / HH(t);
//       }
//       break;
//     case 1  :
//       for (unsigned int i = 0; i < alphasim.n_cols; i++) {
//         double simsignal = arma::as_scalar(Z.col(t * Ztv).t() *
//           alphasim.col(i) + xbeta(t));
//         weights(i) = ng_y(t) * simsignal  - ut(t) * exp(simsignal) +
//           0.5 * std::pow(y(t) - simsignal, 2) / HH(t);
//       }
//       break;
//     case 2  :
//       for (unsigned int i = 0; i < alphasim.n_cols; i++) {
//         double simsignal = arma::as_scalar(Z.col(t * Ztv).t() *
//           alphasim.col(i) + xbeta(t));
//         weights(i) = ng_y(t) * simsignal - ut(t) * log1p(exp(simsignal)) +
//           0.5 * std::pow(y(t) - simsignal, 2) / HH(t);
//       }
//       break;
//     case 3  :
//       for (unsigned int i = 0; i < alphasim.n_cols; i++) {
//         double simsignal = arma::as_scalar(Z.col(t * Ztv).t() *
//           alphasim.col(i) + xbeta(t));
//         weights(i) = ng_y(t) * simsignal - (ng_y(t) + phi) *
//           log(phi + ut(t) * exp(simsignal)) +
//           0.5 * std::pow(y(t) - simsignal, 2) / HH(t);
//       }
//       break;
//     }
//   }
//   return weights;
// }
// //compute log[p(y|alphahat)/g(y|alphahat)] without constants
// double ung_ssm::scaling_factor(const arma::vec& signal) {
//   
//   double ll_approx_u = 0.0;
//   switch(distribution) {
//   case 0  :
//     for (unsigned int t = 0; t < n; t++) {
//       if (arma::is_finite(ng_y(t))) {
//         ll_approx_u += -0.5 * (signal(t) +
//           pow((ng_y(t) - xbeta(t)) / phi, 2) * exp(-signal(t))) +
//           0.5 * pow(y(t) - signal(t), 2) / HH(t);
//       }
//     }
//     break;
//   case 1  :
//     for (unsigned int t = 0; t < n; t++) {
//       if (arma::is_finite(ng_y(t))) {
//         ll_approx_u += ng_y(t) * (signal(t) + xbeta(t)) -
//           ut(t) * exp(signal(t) + xbeta(t)) +
//           0.5 * pow(y(t) - signal(t) - xbeta(t), 2) / HH(t);
//       }
//     }
//     break;
//   case 2  :
//     for (unsigned int t = 0; t < n; t++) {
//       if (arma::is_finite(ng_y(t))) {
//         ll_approx_u += ng_y(t) * (signal(t) + xbeta(t)) -
//           ut(t) * log1p(exp(signal(t) + xbeta(t))) +
//           0.5 * std::pow(y(t) - signal(t) - xbeta(t), 2) / HH(t);
//       }
//     }
//     break;
//   case 3  :
//     for (unsigned int t = 0; t < n; t++) {
//       if (arma::is_finite(ng_y(t))) {
//         ll_approx_u += ng_y(t) * (signal(t) + xbeta(t)) -
//           (ng_y(t) + phi) * log(phi + ut(t) * exp(signal(t) + xbeta(t))) +
//           0.5 * std::pow(y(t) - signal(t) - xbeta(t), 2) / HH(t);
//       }
//     }
//     break;
//   }
//   return ll_approx_u;
// }
// 
// //compute log[p(y|alphahat)/g(y|alphahat)] without constants
// arma::vec ung_ssm::scaling_factor_vec(const arma::vec& signal) {
//   
//   arma::vec ll_approx_u(n, arma::fill::zeros);
//   switch(distribution) {
//   case 0  :
//     for (unsigned int t = 0; t < n; t++) {
//       if (arma::is_finite(ng_y(t))) {
//         ll_approx_u(t) = -0.5 * (signal(t) +
//           pow((ng_y(t) - xbeta(t)) / phi, 2) * exp(-signal(t))) +
//           0.5 * pow(y(t) - signal(t), 2) / HH(t);
//       }
//     }
//     break;
//   case 1  :
//     for (unsigned int t = 0; t < n; t++) {
//       if (arma::is_finite(ng_y(t))) {
//         ll_approx_u(t) = ng_y(t) * (signal(t) + xbeta(t)) -
//           ut(t) * exp(signal(t) + xbeta(t)) +
//           0.5 * pow(y(t) - signal(t) - xbeta(t), 2) / HH(t);
//       }
//     }
//     break;
//   case 2  :
//     for (unsigned int t = 0; t < n; t++) {
//       if (arma::is_finite(ng_y(t))) {
//         ll_approx_u(t) = ng_y(t) * (signal(t) + xbeta(t)) -
//           ut(t) * log1p(exp(signal(t) + xbeta(t))) +
//           0.5 * std::pow(y(t) - signal(t) - xbeta(t), 2) / HH(t);
//       }
//     }
//     break;
//   case 3  :
//     for (unsigned int t = 0; t < n; t++) {
//       if (arma::is_finite(ng_y(t))) {
//         ll_approx_u(t) = ng_y(t) * (signal(t) + xbeta(t)) -
//           (ng_y(t) + phi) * log(phi + ut(t) * exp(signal(t) + xbeta(t))) +
//           0.5 * std::pow(y(t) - signal(t) - xbeta(t), 2) / HH(t);
//       }
//     }
//     break;
//   }
//   return ll_approx_u;
// }
// 
// 
// 
// double ung_ssm::mcmc_approx(const arma::uvec& prior_types, const arma::mat& prior_pars,
//   unsigned int n_iter, unsigned int nsim_states, unsigned int n_burnin,
//   unsigned int n_thin, double gamma, double target_acceptance, arma::mat& S,
//   const arma::vec init_signal, arma::mat& theta_store, arma::vec& ll_store,
//   arma::vec& prior_store,
//   arma::mat& y_store, arma::mat& H_store, arma::mat& ll_approx_u_store,
//   arma::uvec& counts, bool end_ram, bool adapt_approx) {
//   
//   unsigned int npar = prior_types.n_elem;
//   
//   double acceptance_rate = 0.0;
//   arma::vec theta = get_theta();
//   double prior = prior_pdf(theta, prior_types, prior_pars);
//   arma::vec signal = init_signal;
//   double ll_approx = approx(signal, max_iter, conv_tol); // log[p(y_ng|alphahat)/g(y|alphahat)]
//   double ll = ll_approx + log_likelihood(distribution != 0);
//   if (!std::isfinite(ll)) {
//     Rcpp::stop("Non-finite log-likelihood from initial values. ");
//   }
//   
//   double accept_prob = 0.0;
//   
//   std::normal_distribution<> normal(0.0, 1.0);
//   std::uniform_real_distribution<> unif(0.0, 1.0);
//   for(unsigned int i = 0; i < n_burnin; i++) {
//     
//     if (i % 16 == 0) {
//       Rcpp::checkUserInterrupt();
//     }
//     
//     // sample from standard normal distribution
//     // arma::vec u = rnorm(npar);
//     arma::vec u(npar);
//     for(unsigned int ii = 0; ii < npar; ii++) {
//       u(ii) = normal(engine);
//     }
//     // propose new theta
//     arma::vec theta_prop = theta + S * u;
//     // compute prior
//     double prior_prop = prior_pdf(theta_prop, prior_types, prior_pars);
//     
//     if (prior_prop > -arma::datum::inf) {
//       // update parameters
//       update_model(theta_prop);
//       // compute approximate log-likelihood with proposed theta
//       if (adapt_approx) {
//         signal = init_signal;
//         ll_approx = approx(signal, max_iter, conv_tol);
//       }
//       double ll_prop = ll_approx + log_likelihood(distribution != 0);
//       //compute the acceptance probability
//       // use explicit min(...) as we need this value later
//       if(!std::isfinite(ll_prop)) {
//         accept_prob = 0.0;
//       } else {
//         double q = proposal(theta, theta_prop);
//         accept_prob = std::min(1.0, exp(ll_prop - ll + prior_prop - prior + q));
//       }
//       
//       if (unif(engine) < accept_prob) {
//         ll = ll_prop;
//         prior = prior_prop;
//         theta = theta_prop;
//       }
//     } else accept_prob = 0.0;
//     ramcmc::adapt_S(S, u, accept_prob, target_acceptance, i, gamma);
//     
//   }
//   
//   
//   update_model(theta);
//   prior = prior_pdf(theta, prior_types, prior_pars);
//   
//   theta_store.col(0) = theta;
//   ll_store(0) = ll;
//   prior_store(0) = prior;
//   if (adapt_approx) {
//     // compute approximate log-likelihood with proposed theta
//     signal = init_signal;
//     ll_approx = approx(signal, max_iter, conv_tol);
//   }
//   y_store.col(0) = y;
//   H_store.col(0) = H;
//   arma::vec ll_approx_u = scaling_factor_vec(signal);
//   ll_approx_u_store.col(0) = ll_approx_u;
//   counts(0) = 1;
//   unsigned int n_unique = 0;
//   arma::vec y_tmp(n);
//   arma::vec H_tmp(n);
//   
//   for (unsigned int i = n_burnin  + 1; i < n_iter; i++) {
//     
//     if (i % 16 == 0) {
//       Rcpp::checkUserInterrupt();
//     }
//     
//     // sample from standard normal distribution
//     arma::vec u(npar);
//     for(unsigned int ii = 0; ii < npar; ii++) {
//       u(ii) = normal(engine);
//     }
//     // propose new theta
//     arma::vec theta_prop = theta + S * u;
//     double prior_prop = prior_pdf(theta_prop, prior_types, prior_pars);
//     
//     if (prior_prop > -arma::datum::inf) {
//       y_tmp = y;
//       H_tmp = H;
//       // update parameters
//       update_model(theta_prop);
//       // compute approximate log-likelihood with proposed theta
//       if (adapt_approx) {
//         signal = init_signal;
//         ll_approx = approx(signal, max_iter, conv_tol);
//       }
//       double ll_prop = ll_approx + log_likelihood(distribution != 0);
//       //compute the acceptance probability
//       // use explicit min(...) as we need this value later
//       double q = proposal(theta, theta_prop);
//       accept_prob = std::min(1.0, exp(ll_prop - ll + prior_prop - prior + q));
//       
//       if (unif(engine) < accept_prob) {
//         ll = ll_prop;
//         prior = prior_prop;
//         theta = theta_prop;
//         if (adapt_approx) {
//           ll_approx_u = scaling_factor_vec(signal);
//         }
//         n_unique++;
//         acceptance_rate++;
//         counts(n_unique) = 1;
//         ll_store(n_unique) = ll;
//         prior_store(n_unique) = prior;
//         theta_store.col(n_unique) = theta;
//         y_store.col(n_unique) = y;
//         H_store.col(n_unique) = H;
//         ll_approx_u_store.col(n_unique) = ll_approx_u;
//         
//       } else {
//         y = y_tmp;
//         H = H_tmp;
//         counts(n_unique) = counts(n_unique) + 1;
//       }
//     } else {
//       counts(n_unique) = counts(n_unique) + 1;
//       accept_prob = 0.0;
//     }
//     
//     if (!end_ram) {
//       ramcmc::adapt_S(S, u, accept_prob, target_acceptance, i, gamma);
//     }
//     
//   }
//   theta_store.resize(npar, n_unique + 1);
//   ll_store.resize(n_unique + 1);
//   prior_store.resize(n_unique + 1);
//   counts.resize(n_unique + 1);
//   y_store.resize(n, n_unique + 1);
//   H_store.resize(n, n_unique + 1);
//   ll_approx_u_store.resize(n, n_unique + 1);
//   
//   return acceptance_rate / (n_iter - n_burnin);
//   
// }
// 
// double ung_ssm::run_mcmc(const arma::uvec& prior_types, const arma::mat& prior_pars,
//   unsigned int n_iter, unsigned int nsim_states, unsigned int n_burnin,
//   unsigned int n_thin, double gamma, double target_acceptance, arma::mat& S,
//   const arma::vec init_signal, bool end_ram, bool adapt_approx, bool da,
//   arma::mat& theta_store, arma::vec& posterior_store,
//   arma::cube& alpha_store) {
//   
//   if(nsim_states == 1) {
//     da = false;
//   }
//   unsigned int npar = prior_types.n_elem;
//   
//   unsigned int n_samples = floor((n_iter - n_burnin) / n_thin);
//   double acceptance_rate = 0.0;
//   arma::vec theta = get_theta();
//   double prior = prior_pdf(theta, prior_types, prior_pars);
//   arma::vec signal = init_signal;
//   double ll_approx = approx(signal, max_iter, conv_tol); // log[p(y_ng|alphahat)/g(y|alphahat)]
//   double ll_approx_u = scaling_factor(signal);
//   double ll = ll_approx + log_likelihood(distribution != 0);
//   
//   if (!std::isfinite(ll)) {
//     Rcpp::stop("Non-finite log-likelihood from initial values. ");
//   }
//   arma::cube alpha = sim_smoother(nsim_states, distribution != 0);
//   unsigned int ind = 0;
//   double ll_w = 0.0;
//   if (nsim_states > 1) {
//     arma::vec weights = exp(importance_weights(alpha) - ll_approx_u);
//     std::discrete_distribution<> sample(weights.begin(), weights.end());
//     ind = sample(engine);
//     ll_w = log(sum(weights) / nsim_states);
//   }
//   
//   unsigned int j = 0;
//   
//   if (n_burnin == 0) {
//     theta_store.col(0) = theta;
//     posterior_store(0) = ll + ll_w + prior;
//     alpha_store.slice(0) = alpha.slice(ind).t();
//     acceptance_rate++;
//     j++;
//   }
//   
//   double accept_prob = 0.0;
//   unsigned int ind_prop = 0;
//   double ll_w_prop = 0.0;
//   std::normal_distribution<> normal(0.0, 1.0);
//   std::uniform_real_distribution<> unif(0.0, 1.0);
//   
//   for (unsigned int i = 1; i < n_iter; i++) {
//     
//     if (i % 16 == 0) {
//       Rcpp::checkUserInterrupt();
//     }
//     
//     // sample from standard normal distribution
//     arma::vec u(npar);
//     for(unsigned int ii = 0; ii < npar; ii++) {
//       u(ii) = normal(engine);
//     }
//     // propose new theta
//     arma::vec theta_prop = theta + S * u;
//     
//     // compute prior
//     
//     double prior_prop = prior_pdf(theta_prop, prior_types, prior_pars);
//     
//     
//     if (prior_prop > -arma::datum::inf) {
//       // update parameters
//       update_model(theta_prop);
//       // compute approximate log-likelihood with proposed theta
//       if (adapt_approx) {
//         signal = init_signal;
//         ll_approx = approx(signal, max_iter, conv_tol);
//         ll_approx_u = scaling_factor(signal);
//       }
//       double ll_prop = ll_approx + log_likelihood(distribution != 0);
//       //compute the acceptance probability
//       // use explicit min(...) as we need this value later
//       if(!std::isfinite(ll_prop)) {
//         accept_prob = 0.0;
//       } else {
//         double q = proposal(theta, theta_prop);
//         //used in RAM and DA
//         accept_prob = std::min(1.0, exp(ll_prop - ll + prior_prop - prior + q));
//         // initial acceptance based on hat_p(theta, alpha | y)
//         
//         if (da) {
//           if (unif(engine) < accept_prob) {
//             // simulate states
//             
//             arma::cube alpha_prop = sim_smoother(nsim_states, distribution != 0);
//             arma::vec weights = exp(importance_weights(alpha_prop) - ll_approx_u);
//             
//             ll_w_prop = log(sum(weights) / nsim_states);
//             // delayed acceptance ratio
//             double pp = 0.0;
//             if(std::isfinite(ll_w_prop)) {
//               pp = std::min(1.0, exp(ll_w_prop - ll_w));
//             }
//             if (unif(engine) < pp) {
//               if (i >= n_burnin) {
//                 acceptance_rate++;
//               }
//               ll = ll_prop;
//               prior = prior_prop;
//               ll_w = ll_w_prop;
//               theta = theta_prop;
//               std::discrete_distribution<> sample(weights.begin(), weights.end());
//               ind = sample(engine);
//               alpha = alpha_prop;
//             }
//           }
//         } else {
//           // if nsim_states = 1, target hat_p(theta, alpha | y)
//           arma::cube alpha_prop = sim_smoother(nsim_states, distribution != 0);
//           if (nsim_states > 1) {
//             arma::vec weights = exp(importance_weights(alpha_prop) - ll_approx_u);
//             ll_w_prop = log(sum(weights) / nsim_states);
//             std::discrete_distribution<> sample(weights.begin(), weights.end());
//             ind_prop = sample(engine);
//           }
//           double pp = std::min(1.0, exp(ll_prop - ll + ll_w_prop - ll_w +
//             prior_prop - prior + q));
//           
//           if (unif(engine) < pp) {
//             if (i >= n_burnin) {
//               acceptance_rate++;
//             }
//             ll = ll_prop;
//             ll_w = ll_w_prop;
//             prior = prior_prop;
//             theta = theta_prop;
//             alpha = alpha_prop;
//             ind = ind_prop;
//           }
//         }
//       }
//     } else accept_prob = 0.0;
//     
//     //store
//     if ((i >= n_burnin) && (i % n_thin == 0) && j < n_samples) {
//       posterior_store(j) = ll + ll_w + prior;
//       theta_store.col(j) = theta;
//       alpha_store.slice(j) = alpha.slice(ind).t();
//       j++;
//     }
//     
//     if (!end_ram || i < n_burnin) {
//       ramcmc::adapt_S(S, u, accept_prob, target_acceptance, i, gamma);
//     }
//     
//   }
//   
//   return acceptance_rate / (n_iter - n_burnin);
// }
// 
// double ung_ssm::run_mcmc_pf(const arma::uvec& prior_types, const arma::mat& prior_pars,
//   unsigned int n_iter, unsigned int nsim_states, unsigned int n_burnin,
//   unsigned int n_thin, double gamma, double target_acceptance, arma::mat& S,
//   const arma::vec init_signal, bool end_ram, bool adapt_approx, bool da,
//   arma::mat& theta_store, arma::vec& posterior_store,
//   arma::cube& alpha_store, bool bf) {
//   
//   unsigned int npar = prior_types.n_elem;
//   unsigned int n_samples = floor((n_iter - n_burnin) / n_thin);
//   double acceptance_rate = 0.0;
//   
//   arma::vec theta = get_theta();
//   double prior = prior_pdf(theta, prior_types, prior_pars);
//   
//   
//   double ll_g = 0.0;
//   double ll_approx = 0.0;
//   arma::vec ll_approx_u(n, arma::fill::zeros);
//   if(da || !bf) {
//     arma::vec signal = init_signal;
//     ll_approx = approx(signal, max_iter, conv_tol); // log[p(y_ng|alphahat)/g(y|alphahat)]
//     ll_g = log_likelihood(distribution != 0);
//     ll_approx_u = scaling_factor_vec(signal);
//   }
//   double ll_init = ll_g + ll_approx;
//   arma::cube alpha(m, n, nsim_states);
//   arma::mat w(nsim_states, n);
//   arma::umat omega(nsim_states, n - 1);
//   double ll;
//   if (bf){
//     ll = particle_filter(nsim_states, alpha, w, omega);
//   } else {
//     ll = psi_filter(nsim_states, alpha, w, omega, ll_init, ll_approx_u);
//   }
//   backtrack_pf(alpha, omega);
//   if (!std::isfinite(ll)) {
//     Rcpp::stop("Non-finite log-likelihood from initial values. ");
//   }
//   
//   unsigned int ind = 0;
//   
//   arma::vec weights = w.col(n - 1);
//   std::discrete_distribution<> sample(weights.begin(), weights.end());
//   ind = sample(engine);
//   unsigned int j = 0;
//   if (n_burnin == 0) {
//     theta_store.col(0) = theta;
//     posterior_store(0) = ll + prior;
//     alpha_store.slice(0) = alpha.slice(ind).t();
//     acceptance_rate++;
//     j++;
//   }
//   
//   double accept_prob = 0.0;
//   std::normal_distribution<> normal(0.0, 1.0);
//   std::uniform_real_distribution<> unif(0.0, 1.0);
//   for (unsigned int i = 1; i < n_iter; i++) {
//     
//     if (i % 16 == 0) {
//       Rcpp::checkUserInterrupt();
//     }
//     
//     // sample from standard normal distribution
//     arma::vec u(npar);
//     for(unsigned int ii = 0; ii < npar; ii++) {
//       u(ii) = normal(engine);
//     }
//     // propose new theta
//     arma::vec theta_prop = theta + S * u;
//     // compute prior
//     double prior_prop = prior_pdf(theta_prop, prior_types, prior_pars);
//     
//     if (prior_prop > -arma::datum::inf) {
//       // update parameters
//       update_model(theta_prop);
//       
//       if(da){
//         arma::vec signal = init_signal;
//         if (adapt_approx) {
//           ll_approx = approx(signal, max_iter, conv_tol);
//         }
//         double ll_init_prop = ll_approx + log_likelihood(distribution != 0);
//         //compute the acceptance probability
//         // use explicit min(...) as we need this value later
//         
//         if(!std::isfinite(ll_init_prop)) {
//           accept_prob = 0.0;
//         } else {
//           double q = proposal(theta, theta_prop);
//           //used in RAM and DA
//           accept_prob = std::min(1.0, exp(ll_init_prop + prior_prop - ll_init  - prior + q));
//           // initial acceptance based on hat_p(theta, alpha | y)
//           if (unif(engine) < accept_prob) {
//             // simulate states
//             arma::cube alpha_prop(m, n, nsim_states);
//             double ll_prop;
//             if(bf) {
//               ll_prop = particle_filter(nsim_states, alpha_prop, w, omega);
//             } else {
//               if(adapt_approx) {
//                 ll_approx_u = scaling_factor_vec(signal);
//               }
//               ll_prop = psi_filter(nsim_states, alpha_prop, w, omega, ll_init_prop, ll_approx_u);
//             }
//             // delayed acceptance ratio
//             double pp = 0.0;
//             if(std::isfinite(ll_prop)) {
//               pp = std::min(1.0, exp(ll_prop + ll_init - ll - ll_init_prop));
//             }
//             if (unif(engine) < pp) {
//               if (i >= n_burnin) {
//                 acceptance_rate++;
//               }
//               ll = ll_prop;
//               ll_init = ll_init_prop;
//               prior = prior_prop;
//               theta = theta_prop;
//               arma::vec weights = w.col(n-1);
//               std::discrete_distribution<> sample(weights.begin(), weights.end());
//               ind = sample(engine);
//               alpha = alpha_prop;
//               backtrack_pf(alpha, omega);
//             }
//           }
//         }
//       } else {
//         // simulate states
//         arma::cube alpha_prop(m, n, nsim_states);
//         double ll_prop = 0.0;
//         double pp = 0.0;
//         double ll_init_prop = 0.0;
//         if(bf) {
//           
//           ll_prop = particle_filter(nsim_states, alpha_prop, w, omega);
//           if(std::isfinite(ll_prop)) {
//             double q = proposal(theta, theta_prop);
//             accept_prob = std::min(1.0, exp(ll_prop - ll + prior_prop - prior + q));
//           } else accept_prob = 0.0;
//           pp = accept_prob;
//         } else {
//           
//           arma::vec signal = init_signal;
//           if (adapt_approx) {
//             ll_approx = approx(signal, max_iter, conv_tol);
//             
//           }
//           ll_init_prop = ll_approx + log_likelihood(distribution != 0);
//           if(std::isfinite(ll_init_prop)) {
//             double q = proposal(theta, theta_prop);
//             //accept_prob used in RAM
//             if(adapt_approx){
//               ll_approx_u = scaling_factor_vec(signal);
//             }
//             accept_prob = std::min(1.0, exp(ll_init_prop - ll_init + prior_prop - prior + q));
//             ll_prop = psi_filter(nsim_states, alpha_prop, w, omega, ll_init_prop, ll_approx_u);
//             pp = std::min(1.0, exp(ll_prop - ll + prior_prop - prior + q));
//           } else {
//             accept_prob = 0.0;
//             pp = 0.0;
//           }
//           
//         }
//         
//         if (unif(engine) < pp) {
//           if (i >= n_burnin) {
//             acceptance_rate++;
//           }
//           ll = ll_prop;
//           ll_init = ll_init_prop;
//           prior = prior_prop;
//           theta = theta_prop;
//           arma::vec weights = w.col(n-1);
//           std::discrete_distribution<> sample(weights.begin(), weights.end());
//           ind = sample(engine);
//           alpha = alpha_prop;
//           backtrack_pf(alpha, omega);
//         }
//       }
//     } else accept_prob = 0.0;
//     
//     //store
//     if ((i >= n_burnin) && (i % n_thin == 0) && j < n_samples) {
//       posterior_store(j) = ll  + prior;
//       theta_store.col(j) = theta;
//       alpha_store.slice(j) = alpha.slice(ind).t();
//       j++;
//     }
//     
//     if (!end_ram || i < n_burnin) {
//       ramcmc::adapt_S(S, u, accept_prob, target_acceptance, i, gamma);
//     }
//     
//   }
//   
//   return acceptance_rate / (n_iter - n_burnin);
// }
// 
// double ung_ssm::run_mcmc_summary(const arma::uvec& prior_types, const arma::mat& prior_pars,
//   unsigned int n_iter, unsigned int nsim_states, unsigned int n_burnin,
//   unsigned int n_thin, double gamma, double target_acceptance, arma::mat& S,
//   const arma::vec init_signal, bool end_ram, bool adapt_approx, bool da,
//   arma::mat& theta_store, arma::vec& posterior_store,
//   arma::mat& alphahat, arma::cube& Vt, arma::mat& mu, arma::cube& Vmu) {
//   
//   
//   unsigned int npar = prior_types.n_elem;
//   unsigned int n_samples = floor((n_iter - n_burnin) / n_thin);
//   double acceptance_rate = 0.0;
//   
//   arma::vec theta = get_theta();
//   double prior = prior_pdf(theta, prior_types, prior_pars);
//   arma::vec signal = init_signal;
//   double ll_approx = approx(signal, max_iter, conv_tol); // log[p(y_ng|alphahat)/g(y|alphahat)]
//   double ll_approx_u = scaling_factor(signal);
//   double ll = ll_approx + log_likelihood(distribution != 0);
//   
//   if (!std::isfinite(ll)) {
//     Rcpp::stop("Non-finite log-likelihood from initial values. ");
//   }
//   
//   arma::cube alpha = sim_smoother(nsim_states, distribution != 0);
//   arma::vec weights(nsim_states, arma::fill::ones);
//   double ll_w = 0.0;
//   if (nsim_states > 1) {
//     weights = exp(importance_weights(alpha) - ll_approx_u);
//     ll_w = log(sum(weights) / nsim_states);
//   }
//   
//   unsigned int j = 0;
//   
//   arma::cube Vt2(m, m, n, arma::fill::zeros);
//   arma::cube Vmu2(1, 1, n, arma::fill::zeros);
//   
//   if (n_burnin == 0) {
//     summary_iter(j, alpha, weights, alphahat, Vt, Vt2, mu, Vmu, Vmu2);
//     theta_store.col(0) = theta;
//     posterior_store(0) = ll + ll_w + prior;
//     acceptance_rate++;
//     j++;
//   }
//   
//   double accept_prob = 0.0;
//   arma::cube alpha_prop = alpha;
//   double ll_w_prop = 0.0;
//   std::normal_distribution<> normal(0.0, 1.0);
//   std::uniform_real_distribution<> unif(0.0, 1.0);
//   
//   for (unsigned int i = 1; i < n_iter; i++) {
//     
//     if (i % 16 == 0) {
//       Rcpp::checkUserInterrupt();
//     }
//     
//     // sample from standard normal distribution
//     arma::vec u(npar);
//     for(unsigned int ii = 0; ii < npar; ii++) {
//       u(ii) = normal(engine);
//     }
//     // propose new theta
//     arma::vec theta_prop = theta + S * u;
//     // compute prior
//     double prior_prop = prior_pdf(theta_prop, prior_types, prior_pars);
//     
//     if (prior_prop > -arma::datum::inf) {
//       // update parameters
//       update_model(theta_prop);
//       // compute approximate log-likelihood with proposed theta
//       if (adapt_approx) {
//         signal = init_signal;
//         ll_approx = approx(signal, max_iter, conv_tol);
//         ll_approx_u = scaling_factor(signal);
//       }
//       double ll_prop = ll_approx + log_likelihood(distribution != 0);
//       //compute the acceptance probability
//       // use explicit min(...) as we need this value later
//       
//       if(!std::isfinite(ll_prop)) {
//         accept_prob = 0.0;
//       } else {
//         double q = proposal(theta, theta_prop);
//         //used in RAM and DA
//         accept_prob = std::min(1.0, exp(ll_prop - ll + prior_prop - prior + q));
//         
//         // initial acceptance based on hat_p(theta, alpha | y)
//         
//         if (da) {
//           if (unif(engine) < accept_prob) {
//             // simulate states
//             alpha_prop = sim_smoother(nsim_states, distribution != 0);
//             arma::vec weights_prop = exp(importance_weights(alpha_prop) - ll_approx_u);
//             ll_w_prop = log(sum(weights_prop) / nsim_states);
//             // delayed acceptance ratio
//             double pp = 0;
//             if(std::isfinite(ll_w_prop)) {
//               pp = std::min(1.0, exp(ll_w_prop - ll_w));
//             }
//             if (unif(engine) < pp) {
//               if (i >= n_burnin) {
//                 acceptance_rate++;
//               }
//               ll = ll_prop;
//               prior = prior_prop;
//               ll_w = ll_w_prop;
//               theta = theta_prop;
//               alpha = alpha_prop;
//               weights = weights_prop;
//             }
//           }
//         } else {
//           // if nsim_states = 1, target hat_p(theta, alpha | y)
//           alpha_prop = sim_smoother(nsim_states, distribution != 0);
//           arma::vec weights_prop = weights;
//           if (nsim_states > 1) {
//             weights_prop = exp(importance_weights(alpha_prop) - ll_approx_u);
//             ll_w_prop = log(sum(weights_prop) / nsim_states);
//             
//           }
//           double pp = std::min(1.0, exp(ll_prop - ll + ll_w_prop - ll_w +
//             prior_prop - prior + q));
//           
//           if (unif(engine) < pp) {
//             if (i >= n_burnin) {
//               acceptance_rate++;
//             }
//             ll = ll_prop;
//             ll_w = ll_w_prop;
//             prior = prior_prop;
//             theta = theta_prop;
//             alpha = alpha_prop;
//             weights = weights_prop;
//           }
//         }
//       }
//     } else accept_prob = 0.0;
//     
//     //store
//     if ((i >= n_burnin) && (i % n_thin == 0) && j < n_samples) {
//       update_model(theta);
//       summary_iter(j, alpha, weights, alphahat, Vt, Vt2, mu, Vmu, Vmu2);
//       
//       posterior_store(j) = ll + ll_w + prior;
//       theta_store.col(j) = theta;
//       j++;
//     }
//     
//     if (!end_ram || i < n_burnin) {
//       ramcmc::adapt_S(S, u, accept_prob, target_acceptance, i, gamma);
//     }
//     
//   }
//   
//   
//   Vt = Vt + Vt2;
//   Vmu = Vmu + Vmu2;
//   return acceptance_rate / (n_iter - n_burnin);
//   
// }
// 
// 
// double ung_ssm::run_mcmc_summary_pf(const arma::uvec& prior_types, const arma::mat& prior_pars,
//   unsigned int n_iter, unsigned int nsim_states, unsigned int n_burnin,
//   unsigned int n_thin, double gamma, double target_acceptance, arma::mat& S,
//   const arma::vec init_signal, bool end_ram, bool adapt_approx, bool da,
//   arma::mat& theta_store, arma::vec& posterior_store,
//   arma::mat& alphahat, arma::cube& Vt, arma::mat& mu, arma::cube& Vmu, bool bf) {
//   
//   unsigned int npar = prior_types.n_elem;
//   unsigned int n_samples = floor((n_iter - n_burnin) / n_thin);
//   double acceptance_rate = 0.0;
//   
//   arma::vec theta = get_theta();
//   double prior = prior_pdf(theta, prior_types, prior_pars);
//   
//   
//   double ll_g = 0.0;
//   double ll_approx = 0.0;
//   arma::vec ll_approx_u(n, arma::fill::zeros);
//   if(da || !bf) {
//     arma::vec signal = init_signal;
//     ll_approx = approx(signal, max_iter, conv_tol); // log[p(y_ng|alphahat)/g(y|alphahat)]
//     ll_g = log_likelihood(distribution != 0);
//     ll_approx_u = scaling_factor_vec(signal);
//   }
//   double ll_init = ll_g + ll_approx;
//   arma::cube alpha(m, n, nsim_states);
//   arma::mat w(nsim_states, n);
//   arma::umat omega(nsim_states, n - 1);
//   double ll;
//   if (bf){
//     ll = particle_filter(nsim_states, alpha, w, omega);
//   } else {
//     ll = psi_filter(nsim_states, alpha, w, omega, ll_init, ll_approx_u);
//   }
//   backtrack_pf(alpha, omega);
//   if (!std::isfinite(ll)) {
//     Rcpp::stop("Non-finite log-likelihood from initial values. ");
//   }
//   
//   unsigned int j = 0;
//   arma::cube Vt2(m, m, n, arma::fill::zeros);
//   arma::cube Vmu2(1, 1, n, arma::fill::zeros);
//   arma::vec weights = w.col(n - 1);
//   if (n_burnin == 0) {
//     
//     summary_iter(j, alpha, weights, alphahat, Vt, Vt2, mu, Vmu, Vmu2);
//     theta_store.col(0) = theta;
//     posterior_store(0) = ll + prior;
//     acceptance_rate++;
//     j++;
//   }
//   
//   double accept_prob = 0.0;
//   std::normal_distribution<> normal(0.0, 1.0);
//   std::uniform_real_distribution<> unif(0.0, 1.0);
//   for (unsigned int i = 1; i < n_iter; i++) {
//     
//     if (i % 16 == 0) {
//       Rcpp::checkUserInterrupt();
//     }
//     
//     // sample from standard normal distribution
//     arma::vec u(npar);
//     for(unsigned int ii = 0; ii < npar; ii++) {
//       u(ii) = normal(engine);
//     }
//     // propose new theta
//     arma::vec theta_prop = theta + S * u;
//     // compute prior
//     double prior_prop = prior_pdf(theta_prop, prior_types, prior_pars);
//     
//     if (prior_prop > -arma::datum::inf) {
//       // update parameters
//       update_model(theta_prop);
//       
//       if(da){
//         arma::vec signal = init_signal;
//         if (adapt_approx) {
//           ll_approx = approx(signal, max_iter, conv_tol);
//         }
//         double ll_init_prop = ll_approx + log_likelihood(distribution != 0);
//         //compute the acceptance probability
//         // use explicit min(...) as we need this value later
//         
//         if(!std::isfinite(ll_init_prop)) {
//           accept_prob = 0.0;
//         } else {
//           double q = proposal(theta, theta_prop);
//           //used in RAM and DA
//           accept_prob = std::min(1.0, exp(ll_init_prop + prior_prop - ll_init  - prior + q));
//           // initial acceptance based on hat_p(theta, alpha | y)
//           if (unif(engine) < accept_prob) {
//             // simulate states
//             arma::cube alpha_prop(m, n, nsim_states);
//             double ll_prop;
//             if(bf) {
//               ll_prop = particle_filter(nsim_states, alpha_prop, w, omega);
//             } else {
//               if(adapt_approx) {
//                 ll_approx_u = scaling_factor_vec(signal);
//               }
//               ll_prop = psi_filter(nsim_states, alpha_prop, w, omega, ll_init_prop, ll_approx_u);
//             }
//             // delayed acceptance ratio
//             double pp = 0.0;
//             if(std::isfinite(ll_prop)) {
//               pp = std::min(1.0, exp(ll_prop + ll_init - ll - ll_init_prop));
//             }
//             if (unif(engine) < pp) {
//               if (i >= n_burnin) {
//                 acceptance_rate++;
//               }
//               ll = ll_prop;
//               ll_init = ll_init_prop;
//               prior = prior_prop;
//               theta = theta_prop;
//               weights = w.col(n-1);
//               alpha = alpha_prop;
//               backtrack_pf(alpha, omega);
//             }
//           }
//         }
//       } else {
//         // simulate states
//         arma::cube alpha_prop(m, n, nsim_states);
//         double ll_prop = 0.0;
//         double pp = 0.0;
//         double ll_init_prop = 0.0;
//         if(bf) {
//           
//           ll_prop = particle_filter(nsim_states, alpha_prop, w, omega);
//           if(std::isfinite(ll_prop)) {
//             double q = proposal(theta, theta_prop);
//             accept_prob = std::min(1.0, exp(ll_prop - ll + prior_prop - prior + q));
//           } else accept_prob = 0.0;
//           pp = accept_prob;
//         } else {
//           
//           arma::vec signal = init_signal;
//           if (adapt_approx) {
//             ll_approx = approx(signal, max_iter, conv_tol);
//             
//           }
//           ll_init_prop = ll_approx + log_likelihood(distribution != 0);
//           if(std::isfinite(ll_init_prop)) {
//             double q = proposal(theta, theta_prop);
//             //accept_prob used in RAM
//             if(adapt_approx){
//               ll_approx_u = scaling_factor_vec(signal);
//             }
//             accept_prob = std::min(1.0, exp(ll_init_prop - ll_init + prior_prop - prior + q));
//             ll_prop = psi_filter(nsim_states, alpha_prop, w, omega, ll_init_prop, ll_approx_u);
//             pp = std::min(1.0, exp(ll_prop - ll + prior_prop - prior + q));
//           } else {
//             accept_prob = 0.0;
//             pp = 0.0;
//           }
//           
//         }
//         
//         if (unif(engine) < pp) {
//           if (i >= n_burnin) {
//             acceptance_rate++;
//           }
//           ll = ll_prop;
//           ll_init = ll_init_prop;
//           prior = prior_prop;
//           theta = theta_prop;
//           weights = w.col(n-1);
//           alpha = alpha_prop;
//           backtrack_pf(alpha, omega);
//         }
//       }
//     } else accept_prob = 0.0;
//     
//     //store
//     if ((i >= n_burnin) && (i % n_thin == 0) && j < n_samples) {
//       
//       summary_iter(j, alpha, weights, alphahat, Vt, Vt2, mu, Vmu, Vmu2);
//       posterior_store(j) = ll  + prior;
//       theta_store.col(j) = theta;
//       j++;
//     }
//     
//     if (!end_ram || i < n_burnin) {
//       ramcmc::adapt_S(S, u, accept_prob, target_acceptance, i, gamma);
//     }
//     
//   }
//   Vt = Vt + Vt2;
//   Vmu = Vmu + Vmu2;
//   return acceptance_rate / (n_iter - n_burnin);
// }
// 
// arma::cube ung_ssm::invlink(const arma::cube& alpha) {
//   
//   unsigned int nsim = alpha.n_slices;
//   arma::cube y_mean(1, n, nsim);
//   switch(distribution) {
//   case 0  :
//     if(xreg.n_cols > 0) {
//       for (unsigned int i = 0; i < nsim; i++) {
//         for (unsigned int t = 0; t < n; t++) {
//           y_mean(0, t, i) = xbeta(t);
//         }
//       }
//     } else {
//       y_mean.zeros();
//     }
//     break;
//   case 1  :
//     if(xreg.n_cols > 0) {
//       for (unsigned int i = 0; i < nsim; i++) {
//         for (unsigned int t = 0; t < n; t++) {
//           y_mean(0, t, i) = arma::as_scalar(
//             exp(xbeta(t) + Z.col(Ztv * t).t() * alpha.slice(i).col(t)));
//         }
//       }
//     } else {
//       for (unsigned int i = 0; i < nsim; i++) {
//         for (unsigned int t = 0; t < n; t++) {
//           y_mean(0, t, i) = arma::as_scalar(
//             exp(Z.col(Ztv * t).t() * alpha.slice(i).col(t)));
//         }
//       }
//     }
//     break;
//   case 2  :
//     if(xreg.n_cols > 0) {
//       for (unsigned int i = 0; i < nsim; i++) {
//         for (unsigned int t = 0; t < n; t++) {
//           double tmp = arma::as_scalar(exp(xbeta(t) + Z.col(Ztv * t).t() * alpha.slice(i).col(t)));
//           y_mean(0, t, i) = tmp / (1.0 + tmp);
//         }
//       }
//     } else {
//       for (unsigned int i = 0; i < nsim; i++) {
//         for (unsigned int t = 0; t < n; t++) {
//           double tmp = arma::as_scalar(exp(Z.col(Ztv * t).t() * alpha.slice(i).col(t)));
//           y_mean(0, t, i) = tmp / (1.0 + tmp);
//         }
//       }
//     }
//     break;
//   case 3  :
//     if(xreg.n_cols > 0) {
//       for (unsigned int i = 0; i < nsim; i++) {
//         for (unsigned int t = 0; t < n; t++) {
//           y_mean(0, t, i) = arma::as_scalar(exp(xbeta(t) + Z.col(Ztv * t).t() * alpha.slice(i).col(t)));
//         }
//       }
//     } else {
//       for (unsigned int i = 0; i < nsim; i++) {
//         for (unsigned int t = 0; t < n; t++) {
//           y_mean(0, t, i) = arma::as_scalar(exp(Z.col(Ztv * t).t() * alpha.slice(i).col(t)));
//         }
//       }
//     }
//     break;
//   }
//   return y_mean;
// }
// 
// 
// void ung_ssm::summary_iter(const unsigned int j, const arma::cube& alpha, const arma::vec& weights,
//   arma::mat& alphahat, arma::cube& Vt, arma::cube& Valpha, arma::mat& mu,
//   arma::cube& Vmu, arma::cube& Vmu2) {
//   
//   arma::mat alphahat_i(m, n);
//   arma::cube Vt_i(m, m, n);
//   running_weighted_summary(alpha, alphahat_i, Vt_i, weights);
//   Vt += (Vt_i - Vt) / (j + 1);
//   running_summary(alphahat_i, alphahat, Valpha, j);
//   
//   arma::mat mu_i(1, n);
//   arma::cube Vmu_i(1, 1, n);
//   running_weighted_summary(invlink(alpha), mu_i, Vmu_i, weights);
//   Vmu += (Vmu_i - Vmu) / (j + 1);
//   running_summary(mu_i, mu, Vmu2, j);
// }
// 
// //compute p(y_t| xbeta_t, Z_t alpha_t)
// arma::vec ung_ssm::pyt(const unsigned int t, const arma::cube& alphasim) {
//   
//   arma::vec weights(alphasim.n_slices);
//   
//   switch(distribution) {
//   case 0  : {
//     double x = pow((ng_y(t) - xbeta(t)) / phi, 2);
//     for (unsigned int i = 0; i < alphasim.n_slices; i++) {
//       double simsignal = alphasim(0, t, i);
//       weights(i) = -0.5 * (simsignal + x * exp(-simsignal));
//     }
//   } break;
//   case 1  :
//     for (unsigned int i = 0; i < alphasim.n_slices; i++) {
//       double simsignal = arma::as_scalar(Z.col(t * Ztv).t() *
//         alphasim.slice(i).col(t) + xbeta(t));
//       weights(i) = ng_y(t) * simsignal  - ut(t) * exp(simsignal);
//     }
//     break;
//   case 2  :
//     for (unsigned int i = 0; i < alphasim.n_slices; i++) {
//       double simsignal = arma::as_scalar(Z.col(t * Ztv).t() *
//         alphasim.slice(i).col(t) + xbeta(t));
//       weights(i) = ng_y(t) * simsignal - ut(t) * log1p(exp(simsignal));
//     }
//     break;
//   case 3  :
//     for (unsigned int i = 0; i < alphasim.n_slices; i++) {
//       double simsignal = arma::as_scalar(Z.col(t * Ztv).t() *
//         alphasim.slice(i).col(t) + xbeta(t));
//       weights(i) = ng_y(t) * simsignal - (ng_y(t) + phi) *
//         log(phi + ut(t) * exp(simsignal));
//     }
//     break;
//   }
//   
//   return weights;
// }
// 
// //compute p(y_t| xbeta_t, Z_t alpha_t)
// arma::vec ung_ssm::pyt(const unsigned int t, const arma::mat& alphasim) {
//   
//   arma::vec weights(alphasim.n_cols);
//   
//   if (arma::is_finite(ng_y(t))) {
//     
//     switch(distribution) {
//     case 0  : {
//     double x = pow((ng_y(t) - xbeta(t)) / phi, 2);
//     for (unsigned int i = 0; i < alphasim.n_cols; i++) {
//       double simsignal = alphasim(0, i);
//       weights(i) = -0.5 * (simsignal + x * exp(-simsignal));
//     }
//   } break;
//     case 1  :
//       for (unsigned int i = 0; i < alphasim.n_cols; i++) {
//         double simsignal = arma::as_scalar(Z.col(t * Ztv).t() *
//           alphasim.col(i) + xbeta(t));
//         weights(i) = ng_y(t) * simsignal  - ut(t) * exp(simsignal);
//       }
//       break;
//     case 2  :
//       for (unsigned int i = 0; i < alphasim.n_cols; i++) {
//         double simsignal = arma::as_scalar(Z.col(t * Ztv).t() *
//           alphasim.col(i) + xbeta(t));
//         weights(i) = ng_y(t) * simsignal - ut(t) * log1p(exp(simsignal));
//       }
//       break;
//     case 3  :
//       for (unsigned int i = 0; i < alphasim.n_cols; i++) {
//         double simsignal = arma::as_scalar(Z.col(t * Ztv).t() *
//           alphasim.col(i) + xbeta(t));
//         weights(i) = ng_y(t) * simsignal - (ng_y(t) + phi) *
//           log(phi + ut(t) * exp(simsignal));
//       }
//       break;
//     }
//   }
//   return weights;
// }
// 
// 
// //particle filter
// double ung_ssm::particle_filter(unsigned int nsim, arma::cube& alphasim, arma::mat& w, arma::umat& ind) {
//   
//   std::normal_distribution<> normal(0.0, 1.0);
//   std::uniform_real_distribution<> unif(0.0, 1.0);
//   
//   arma::uvec nonzero = arma::find(P1.diag() > 0);
//   arma::mat L_P1(m, m, arma::fill::zeros);
//   if (nonzero.n_elem > 0) {
//     L_P1.submat(nonzero, nonzero) =
//       arma::chol(P1.submat(nonzero, nonzero), "lower");
//   }
//   for (unsigned int i = 0; i < nsim; i++) {
//     arma::vec um(m);
//     for(unsigned int j = 0; j < m; j++) {
//       um(j) = normal(engine);
//     }
//     alphasim.slice(i).col(0) = a1 + L_P1 * um;
//   }
//   arma::vec normalized_weights(nsim);
//   double ll = 0.0;
//   double const_term = 0.0;
//   switch(distribution) {
//   case 0 :
//     const_term = arma::uvec(arma::find_finite(ng_y)).n_elem * norm_log_const(phi);
//     break;
//   case 1 : {
//       arma::uvec y_ind(find_finite(ng_y));
//       const_term = poisson_log_const(ng_y(y_ind), ut(y_ind));
//     } break;
//   case 2 : {
//     arma::uvec y_ind(find_finite(ng_y));
//     const_term = binomial_log_const(ng_y(y_ind), ut(y_ind));
//   } break;
//   case 3 : {
//     arma::uvec y_ind(find_finite(ng_y));
//     const_term = negbin_log_const(ng_y(y_ind), ut(y_ind), phi);
//   } break;
//   }
//   
//   if(arma::is_finite(ng_y(0))) {
//     w.col(0) = pyt(0, alphasim);
//     double max_weight = w.col(0).max();
//     w.col(0) = exp(w.col(0) - max_weight);
//     double sumw = arma::sum(w.col(0));
//     if(sumw > 0.0){
//       normalized_weights = w.col(0) / sumw;
//     } else {
//       return -arma::datum::inf;
//     }
//     ll = max_weight + log(sumw / nsim);
//   } else {
//     w.col(0).ones();
//     normalized_weights.fill(1.0/nsim);
//   }
//   for (unsigned int t = 0; t < (n - 1); t++) {
//     
//     arma::vec r(nsim);
//     for (unsigned int i = 0; i < nsim; i++) {
//       r(i) = unif(engine);
//     }
//     
//     ind.col(t) = stratified_sample(normalized_weights, r, nsim);
//     
//     arma::mat alphatmp(m, nsim);
//     
//     for (unsigned int i = 0; i < nsim; i++) {
//       alphatmp.col(i) = alphasim.slice(ind(i, t)).col(t);
//     }
//     
//     for (unsigned int i = 0; i < nsim; i++) {
//       arma::vec uk(k);
//       for(unsigned int j = 0; j < k; j++) {
//         uk(j) = normal(engine);
//       }
//       alphasim.slice(i).col(t + 1) = C.col(t * Ctv) + T.slice(t * Ttv) * alphatmp.col(i) + R.slice(t * Rtv) * uk;
//     }
//     
//     if(arma::is_finite(ng_y(t + 1))) {
//       w.col(t + 1) = pyt(t + 1, alphasim);
//       
//       double max_weight = w.col(t + 1).max();
//       w.col(t + 1) = exp(w.col(t + 1) - max_weight);
//       double sumw = arma::sum(w.col(t + 1));
//       if(sumw > 0.0){
//         normalized_weights = w.col(t + 1) / sumw;
//       } else {
//         return -arma::datum::inf;
//       }
//       ll += max_weight + log(sumw / nsim);
//     } else {
//       w.col(t + 1).ones();
//       normalized_weights.fill(1.0/nsim);
//     }
//     
//     
//   }
//   return ll + const_term;
// }
// 
// 
// 
// //psi-auxiliary particle filter
// double ung_ssm::psi_filter(unsigned int nsim, arma::cube& alphasim, arma::mat& w,
//   arma::umat& ind, const double ll_g, const arma::vec& ll_approx_u) {
//   
//   arma::mat alphahat(m, n);
//   arma::cube Vt(m, m, n);
//   arma::cube Ct(m, m, n);
//   smoother_ccov(alphahat, Vt, Ct, distribution != 0);
//   conditional_dist_helper(Vt, Ct);
//   std::normal_distribution<> normal(0.0, 1.0);
//   std::uniform_real_distribution<> unif(0.0, 1.0);
//   for (unsigned int i = 0; i < nsim; i++) {
//     arma::vec um(m);
//     for(unsigned int j = 0; j < m; j++) {
//       um(j) = normal(engine);
//     }
//     alphasim.slice(i).col(0) = alphahat.col(0) + Vt.slice(0) * um;
//   }
//   
//   arma::vec normalized_weights(nsim);
//   double ll = 0.0;
//   if(arma::is_finite(ng_y(0))) {
//     //don't add gaussian likelihood to weights
//     w.col(0) = exp(importance_weights_t(0, alphasim) - ll_approx_u(0));
//     double sumw = arma::sum(w.col(0));
//     if(sumw > 0.0){
//       normalized_weights = w.col(0) / sumw;
//     } else {
//       return -arma::datum::inf;
//     }
//     ll = ll_g + log(sumw / nsim);
//   } else {
//     w.col(0).ones();
//     normalized_weights.fill(1.0/nsim);
//     ll = ll_g;
//   }
//   for (unsigned int t = 0; t < (n - 1); t++) {
//     
//     arma::vec r(nsim);
//     for (unsigned int i = 0; i < nsim; i++) {
//       r(i) = unif(engine);
//     }
//     
//     ind.col(t) = stratified_sample(normalized_weights, r, nsim);
//     
//     arma::mat alphatmp(m, nsim);
//     
//     for (unsigned int i = 0; i < nsim; i++) {
//       alphatmp.col(i) = alphasim.slice(ind(i, t)).col(t);
//     }
//     for (unsigned int i = 0; i < nsim; i++) {
//       arma::vec um(m);
//       for(unsigned int j = 0; j < m; j++) {
//         um(j) = normal(engine);
//       }
//       alphasim.slice(i).col(t + 1) = alphahat.col(t + 1) +
//         Ct.slice(t + 1) * (alphatmp.col(i) - alphahat.col(t)) + Vt.slice(t + 1) * um;
//     }
//     
//     if(arma::is_finite(ng_y(t + 1))) {
//       w.col(t + 1) = exp(importance_weights_t(t + 1, alphasim)- ll_approx_u(t + 1));
//       double sumw = arma::sum(w.col(t + 1));
//       if(sumw > 0.0){
//         normalized_weights = w.col(t + 1) / sumw;
//       } else {
//         return -arma::datum::inf;
//       }
//       ll += log(sumw / nsim);
//     } else {
//       w.col(t + 1).ones();
//       normalized_weights.fill(1.0/nsim);
//     }
//     
//     
//   }
//   
//   return ll;
// }
// 
// //psi-auxiliary particle filter
// double ung_ssm::psi_loglik(unsigned int nsim, const double ll_g, const arma::vec& ll_approx_u) {
//   
//   arma::mat alphahat(m, n);
//   arma::cube Vt(m, m, n);
//   arma::cube Ct(m, m, n);
//   smoother_ccov(alphahat, Vt, Ct, distribution != 0);
//   conditional_dist_helper(Vt, Ct);
//   arma::mat alphasim(m, nsim);
//   std::normal_distribution<> normal(0.0, 1.0);
//   std::uniform_real_distribution<> unif(0.0, 1.0);
//   for (unsigned int i = 0; i < nsim; i++) {
//     arma::vec um(m);
//     for(unsigned int j = 0; j < m; j++) {
//       um(j) = normal(engine);
//     }
//     alphasim.col(i) = alphahat.col(0) + Vt.slice(0) * um;
//   }
//   
//   arma::vec w(nsim);
//   arma::vec normalized_weights(nsim);
//   double ll = 0.0;
//   if(arma::is_finite(ng_y(0))) {
//     //don't add gaussian likelihood to weights
//     w = exp(importance_weights_t(0, alphasim) - ll_approx_u(0));
//     double sumw = arma::sum(w);
//     if(sumw > 0.0){
//       normalized_weights = w / sumw;
//     } else {
//       return -arma::datum::inf;
//     }
//     ll = ll_g + log(sumw / nsim);
//   } else {
//     normalized_weights.fill(1.0/nsim);
//     ll = ll_g;
//   }
//   
//   for (unsigned int t = 0; t < (n - 1); t++) {
//     
//     arma::vec r(nsim);
//     for (unsigned int i = 0; i < nsim; i++) {
//       r(i) = unif(engine);
//     }
//     
//     arma::uvec ind = stratified_sample(normalized_weights, r, nsim);
//     
//     arma::mat alphatmp(m, nsim);
//     
//     for (unsigned int i = 0; i < nsim; i++) {
//       alphatmp.col(i) = alphasim.col(ind(i));
//     }
//     for (unsigned int i = 0; i < nsim; i++) {
//       arma::vec um(m);
//       for(unsigned int j = 0; j < m; j++) {
//         um(j) = normal(engine);
//       }
//       alphasim.col(i) = alphahat.col(t + 1) +
//         Ct.slice(t + 1) * (alphatmp.col(i) - alphahat.col(t)) + Vt.slice(t + 1) * um;
//     }
//     
//     if(arma::is_finite(ng_y(t + 1))) {
//       w = exp(importance_weights_t(t + 1, alphasim)- ll_approx_u(t + 1));
//       double sumw = arma::sum(w);
//       if(sumw > 0.0){
//         normalized_weights = w / sumw;
//       } else {
//         return -arma::datum::inf;
//       }
//       ll += log(sumw / nsim);
//     } else {
//       normalized_weights.fill(1.0/nsim);
//     }
//   }
//   
//   return ll;
// }
// 
// double ung_ssm::bsf_loglik(unsigned int nsim) {
//   
//   std::normal_distribution<> normal(0.0, 1.0);
//   std::uniform_real_distribution<> unif(0.0, 1.0);
//   
//   arma::uvec nonzero = arma::find(P1.diag() > 0);
//   arma::mat L_P1(m, m, arma::fill::zeros);
//   arma::mat alphasim(m, nsim);
//   if (nonzero.n_elem > 0) {
//     L_P1.submat(nonzero, nonzero) =
//       arma::chol(P1.submat(nonzero, nonzero), "lower");
//   }
//   for (unsigned int i = 0; i < nsim; i++) {
//     arma::vec um(m);
//     for(unsigned int j = 0; j < m; j++) {
//       um(j) = normal(engine);
//     }
//     alphasim.col(i) = a1 + L_P1 * um;
//   }
//   
//   arma::vec w(nsim);
//   arma::vec normalized_weights(nsim);
//   double ll = 0.0;
//   double const_term = 0.0;
//   switch(distribution) {
//   case 0 :
//     const_term = arma::uvec(arma::find_finite(ng_y)).n_elem * norm_log_const(phi);
//     break;
//   case 1 : {
//       arma::uvec y_ind(find_finite(ng_y));
//       const_term = poisson_log_const(ng_y(y_ind), ut(y_ind));
//     } break;
//   case 2 : {
//     arma::uvec y_ind(find_finite(ng_y));
//     const_term = binomial_log_const(ng_y(y_ind), ut(y_ind));
//   } break;
//   case 3 : {
//     arma::uvec y_ind(find_finite(ng_y));
//     const_term = negbin_log_const(ng_y(y_ind), ut(y_ind), phi);
//   } break;
//   }
//   if(arma::is_finite(ng_y(0))) {
//     w = pyt(0, alphasim);
//     double max_weight = w.max();
//     w = exp(w - max_weight);
//     double sumw = arma::sum(w);
//     if(sumw > 0.0){
//       normalized_weights = w / sumw;
//     } else {
//       return -arma::datum::inf;
//     }
//     ll = max_weight + log(sumw / nsim);
//   } else {
//     normalized_weights.fill(1.0/nsim);
//   }
//   for (unsigned int t = 0; t < (n - 1); t++) {
//     
//     arma::vec r(nsim);
//     for (unsigned int i = 0; i < nsim; i++) {
//       r(i) = unif(engine);
//     }
//     
//     arma::uvec ind = stratified_sample(normalized_weights, r, nsim);
//     
//     arma::mat alphatmp(m, nsim);
//     
//     
//     for (unsigned int i = 0; i < nsim; i++) {
//       alphatmp.col(i) = alphasim.col(ind(i));
//     }
//     
//     for (unsigned int i = 0; i < nsim; i++) {
//       arma::vec uk(k);
//       for(unsigned int j = 0; j < k; j++) {
//         uk(j) = normal(engine);
//       }
//       alphasim.col(i) = C.col(t * Ctv) + T.slice(t * Ttv) * alphatmp.col(i) + R.slice(t * Rtv) * uk;
//     }
//     
//     if(arma::is_finite(ng_y(t + 1))) {
//       w = pyt(t + 1, alphasim);
//       
//       double max_weight = w.max();
//       w = exp(w - max_weight);
//       double sumw = arma::sum(w);
//       if(sumw > 0.0){
//         normalized_weights = w / sumw;
//       } else {
//         return -arma::datum::inf;
//       }
//       ll += max_weight + log(sumw / nsim);
//     } else {
//       w.ones();
//       normalized_weights.fill(1.0/nsim);
//     }
//     
//     
//   }
//   return ll + const_term;
// }
