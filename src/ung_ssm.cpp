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
  phi(model["phi"]),
  u(Rcpp::as<arma::vec>(model["u"])), distribution(model["distribution"]),
  phi_est(Rcpp::as<bool>(model["phi_est"])), max_iter(100), conv_tol(1.0e-8),
  theta(Rcpp::as<arma::vec>(model["theta"])), 
  prior_distributions(Rcpp::as<arma::uvec>(model["prior_distributions"])), 
  prior_parameters(Rcpp::as<arma::mat>(model["prior_parameters"])),
  Z_ind(Z_ind), T_ind(T_ind), R_ind(R_ind) {
  
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
void ung_ssm::update_model(const arma::vec& new_theta) {
  
  if (Z_ind.n_elem > 0) {
    Z.elem(Z_ind) = new_theta.subvec(0, Z_ind.n_elem - 1);
  }
  if (T_ind.n_elem > 0) {
    T.elem(T_ind) = new_theta.subvec(Z_ind.n_elem, Z_ind.n_elem + T_ind.n_elem - 1);
  }
  if (R_ind.n_elem > 0) {
    R.elem(R_ind) = new_theta.subvec(Z_ind.n_elem + T_ind.n_elem,
      Z_ind.n_elem + T_ind.n_elem + R_ind.n_elem - 1);
  }
  
  if (R_ind.n_elem  > 0) {
    compute_RR();
  }
  
  if(phi_est) {
    phi = new_theta(Z_ind.n_elem + T_ind.n_elem + R_ind.n_elem);
  }
  if(xreg.n_cols > 0) {
    beta = new_theta.subvec(new_theta.n_elem - xreg.n_cols, new_theta.n_elem - 1);
    compute_xbeta();
  }
  theta = new_theta;
}

double ung_ssm::log_prior_pdf(const arma::vec& x) const {
  
  double log_prior = 0.0;
  
  for(unsigned int i = 0; i < x.n_elem; i++) {
    switch(prior_distributions(i)) {
    case 0  :
      if (x(i) < prior_parameters(0, i) || x(i) > prior_parameters(1, i)) {
        return -std::numeric_limits<double>::infinity(); 
      }
      break;
    case 1  :
      if (x(i) < 0) {
        return -std::numeric_limits<double>::infinity();
      } else {
        log_prior -= 0.5 * std::pow(x(i) / prior_parameters(0, i), 2);
      }
      break;
    case 2  :
      log_prior -= 0.5 * std::pow((x(i) - prior_parameters(0, i)) / prior_parameters(1, i), 2);
      break;
    }
  }
  return log_prior;
}

double ung_ssm::log_proposal_ratio(const arma::vec& new_theta, const arma::vec& old_theta) const {
  return 0.0;
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
  tmp(arma::find(arma::abs(tmp) < 1e-4)).fill(1e-4);
  approx_H = 2.0 * arma::exp(signal) / arma::square(tmp/phi);
  approx_y = signal + 1.0 - 0.5 * approx_H;
} break;
  case 1: {
    arma::vec tmp = signal + xbeta;
    approx_H = 1.0 / (arma::exp(tmp) % u);
    approx_y = y % approx_H + tmp - 1.0;
  } break;
  case 2: {
    arma::vec exptmp = arma::exp(signal + xbeta);
    approx_H = arma::square(1.0 + exptmp) / (u % exptmp);
    approx_y = y % approx_H + signal + xbeta - 1.0 - exptmp;
  } break;
  case 3: {
    arma::vec exptmp = 1.0 / (arma::exp(signal + xbeta) % u);
    approx_H = 1.0 / phi + exptmp;
    approx_y = signal + xbeta + y % exptmp - 1.0;
  } break;
  }
  approx_H = arma::sqrt(approx_H);
}

// construct an approximating Gaussian model
// Note the difference to previous versions, the convergence is assessed only
// by checking the changes in mode, not the actual function values. This is
// slightly faster and sufficient as the approximation doesn't need to be accurate.
// Using function values would be safer though, as we could use line search etc
// in case of potential divergence etc...
ugg_ssm ung_ssm::approximate(arma::vec& mode_estimate, const unsigned int max_iter,
  const double conv_tol) {
  
  //Construct y and H for the Gaussian model
  arma::vec approx_y(n, arma::fill::zeros);
  arma::vec approx_H(n, arma::fill::zeros);
  
  // RNG of approximate model is only used in basic IS sampling
  // set seed for new RNG stream based on the original model
  std::uniform_int_distribution<> unif(0, std::numeric_limits<int>::max());
  const unsigned int new_seed = unif(engine);
  ugg_ssm approx_model(approx_y, Z, approx_H, T, R, a1, P1, xreg, beta, D, C, new_seed);
  
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
      mode_estimate_new = arma::vectorise(approx_model.fast_smoother().head_cols(n));
    } else {
      arma::mat alpha = approx_model.fast_smoother().head_cols(n);
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
  
  if(max_iter == 0 && mode_estimate.n_elem == n) {
    if (distribution == 0) {
      mode_estimate = arma::vectorise(approx_model.fast_smoother().head_cols(n));
    } else {
      arma::mat alpha = approx_model.fast_smoother().head_cols(n);
      for (unsigned int t = 0; t < n; t++) {
        mode_estimate(t) = arma::as_scalar(Z.col(Ztv * t).t() * alpha.col(t));
      }
    }
  }
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
      mode_estimate_new = arma::vectorise(approx_model.fast_smoother().head_cols(n));
    } else {
      arma::mat alpha = approx_model.fast_smoother().head_cols(n);
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
  
  arma::mat alphahat(m, n + 1);
  arma::cube Vt(m, m, n + 1);
  arma::cube Ct(m, m, n + 1);
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
    weights.col(0) = arma::exp(log_weights(approx_model, 0, alpha) - scales(0));
    double sum_weights = arma::accu(weights.col(0));
    if(sum_weights > 0.0){
      normalized_weights = weights.col(0) / sum_weights;
    } else {
      return -std::numeric_limits<double>::infinity();
    }
    loglik = approx_loglik + std::log(sum_weights / nsim);
  } else {
    weights.col(0).ones();
    normalized_weights.fill(1.0 / nsim);
    loglik = approx_loglik;
  }
  
  for (unsigned int t = 0; t < n; t++) {
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
    
    if ((t < (n - 1)) && arma::is_finite(y(t + 1))) {
      weights.col(t + 1) =
        arma::exp(log_weights(approx_model, t + 1, alpha) - scales(t + 1));
      double sum_weights = arma::accu(weights.col(t + 1));
      if(sum_weights > 0.0){
        normalized_weights = weights.col(t + 1) / sum_weights;
      } else {
        return -std::numeric_limits<double>::infinity();
      }
      loglik += std::log(sum_weights / nsim);
    } else {
      weights.col(t + 1).ones();
      normalized_weights.fill(1.0 / nsim);
    }
  }
  return loglik;
}

arma::vec ung_ssm::importance_weights(const ugg_ssm& approx_model,
  const arma::cube& alpha) const {
  arma::vec weights(alpha.n_slices, arma::fill::zeros);
  for(unsigned int t = 0; t < n; t++) {
    weights += log_weights(approx_model, t, alpha);
  }
  return weights;
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
        weights(i) = -0.5 * (simsignal + std::pow(y(t) / phi, 2.0) * std::exp(-simsignal)) +
          0.5 * std::pow((approx_model.y(t) - simsignal) / approx_model.H(t), 2.0);
      }
      break;
    case 1  :
      for (unsigned int i = 0; i < alpha.n_slices; i++) {
        double simsignal = arma::as_scalar(Z.col(t * Ztv).t() *
          alpha.slice(i).col(t) + xbeta(t));
        weights(i) = y(t) * simsignal  - u(t) * std::exp(simsignal) +
          0.5 * std::pow((approx_model.y(t) - simsignal) / approx_model.H(t), 2.0);
      }
      break;
    case 2  :
      for (unsigned int i = 0; i < alpha.n_slices; i++) {
        double simsignal = arma::as_scalar(Z.col(t * Ztv).t() *
          alpha.slice(i).col(t) + xbeta(t));
        weights(i) = y(t) * simsignal - u(t) * std::log1p(std::exp(simsignal)) +
          0.5 * std::pow((approx_model.y(t) - simsignal) / approx_model.H(t), 2.0);
      }
      break;
    case 3  :
      for (unsigned int i = 0; i < alpha.n_slices; i++) {
        double simsignal = arma::as_scalar(Z.col(t * Ztv).t() *
          alpha.slice(i).col(t) + xbeta(t));
        weights(i) = y(t) * simsignal - (y(t) + phi) *
          std::log(phi + u(t) * std::exp(simsignal)) +
          0.5 * std::pow((approx_model.y(t) - simsignal) / approx_model.H(t), 2.0);
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
        weights(t) = -0.5 * (mode_estimate(t) + std::pow(y(t) / phi, 2.0) *
          std::exp(-mode_estimate(t))) +
          0.5 * std::pow((approx_model.y(t) - mode_estimate(t)) / approx_model.H(t), 2.0);
      }
    }
    break;
  case 1  :
    for(unsigned int t = 0; t < n; t++) {
      if (arma::is_finite(y(t))) {
        weights(t) = y(t) * (mode_estimate(t) + xbeta(t)) -
          u(t) * std::exp(mode_estimate(t) + xbeta(t)) +
          0.5 * std::pow((approx_model.y(t) - (mode_estimate(t) + xbeta(t))) / approx_model.H(t), 2.0);
      }
    }
    break;
  case 2  :
    for(unsigned int t = 0; t < n; t++) {
      if (arma::is_finite(y(t))) {
        weights(t) = y(t) * (mode_estimate(t) + xbeta(t)) -
          u(t) * std::log1p(std::exp(mode_estimate(t) + xbeta(t))) +
          0.5 * std::pow((approx_model.y(t) - (mode_estimate(t) + xbeta(t))) / approx_model.H(t), 2.0);
      }
    }
    break;
  case 3  :
    for(unsigned int t = 0; t < n; t++) {
      if (arma::is_finite(y(t))) {
        weights(t) = y(t) * (mode_estimate(t) + xbeta(t)) -
          (y(t) + phi) *
          std::log(phi + u(t) * std::exp(mode_estimate(t) + xbeta(t))) +
          0.5 * std::pow((approx_model.y(t) - (mode_estimate(t) + xbeta(t))) / approx_model.H(t), 2.0);
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
        weights(i) = -0.5 * (simsignal + std::pow(y(t) / phi, 2.0) * std::exp(-simsignal));
      }
      break;
    case 1  :
      for (unsigned int i = 0; i < alpha.n_slices; i++) {
        double simsignal = arma::as_scalar(Z.col(t * Ztv).t() *
          alpha.slice(i).col(t) + xbeta(t));
        weights(i) = y(t) * simsignal  - u(t) * std::exp(simsignal);
      }
      break;
    case 2  :
      for (unsigned int i = 0; i < alpha.n_slices; i++) {
        double simsignal = arma::as_scalar(Z.col(t * Ztv).t() *
          alpha.slice(i).col(t) + xbeta(t));
        weights(i) = y(t) * simsignal - u(t) * std::log1p(std::exp(simsignal));
      }
      break;
    case 3  :
      for (unsigned int i = 0; i < alpha.n_slices; i++) {
        double simsignal = arma::as_scalar(Z.col(t * Ztv).t() *
          alpha.slice(i).col(t) + xbeta(t));
        weights(i) = y(t) * simsignal - (y(t) + phi) *
          std::log(phi + u(t) * std::exp(simsignal));
      }
      break;
    }
  }
  return weights;
}

double ung_ssm::bsf_filter(const unsigned int nsim, arma::cube& alpha,
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
    weights.col(0) = arma::exp(weights.col(0) - max_weight);
    double sum_weights = arma::accu(weights.col(0));
    if(sum_weights > 0.0){
      normalized_weights = weights.col(0) / sum_weights;
    } else {
      return -std::numeric_limits<double>::infinity();
    }
    loglik = max_weight + std::log(sum_weights / nsim);
  } else {
    weights.col(0).ones();
    normalized_weights.fill(1.0 / nsim);
  }
  for (unsigned int t = 0; t < n; t++) {
    
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
    
    if ((t < (n - 1)) && arma::is_finite(y(t + 1))) {
      weights.col(t + 1) = log_obs_density(t + 1, alpha);
      
      double max_weight = weights.col(t + 1).max();
      weights.col(t + 1) = arma::exp(weights.col(t + 1) - max_weight);
      double sum_weights = arma::accu(weights.col(t + 1));
      if(sum_weights > 0.0){
        normalized_weights = weights.col(t + 1) / sum_weights;
      } else {
        return -std::numeric_limits<double>::infinity();
      }
      loglik += max_weight + std::log(sum_weights / nsim);
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

arma::cube ung_ssm::predict_sample(const arma::mat& theta_posterior,
  const arma::mat& alpha, const arma::uvec& counts,
  const unsigned int predict_type, const unsigned int nsim) {
  
  unsigned int d = 1;
  if (predict_type == 3) d = m;
  
  arma::mat expanded_theta = rep_mat(theta_posterior, counts);
  arma::mat expanded_alpha = rep_mat(alpha, counts);
  unsigned int n_samples = expanded_theta.n_cols;
  arma::cube sample(d, n, nsim * n_samples);
  for (unsigned int i = 0; i < n_samples; i++) {
    update_model(expanded_theta.col(i));
    a1 = expanded_alpha.col(i);
    sample.slices(i * nsim, (i + 1) * nsim - 1) =
      sample_model(predict_type, nsim);
    
  }
  return sample;
}


arma::mat ung_ssm::sample_model(const unsigned int predict_type,
  const unsigned int nsim) {
  
  arma::cube alpha(m, n, nsim);
  std::normal_distribution<> normal(0.0, 1.0);
  
  for (unsigned int i = 0; i < nsim; i++) {
    
    alpha.slice(i).col(0) = a1;
    
    for (unsigned int t = 0; t < (n - 1); t++) {
      arma::vec uk(k);
      for(unsigned int j = 0; j < k; j++) {
        uk(j) = normal(engine);
      }
      alpha.slice(i).col(t + 1) =
        C.col(t * Ctv) + T.slice(t * Ttv) * alpha.slice(i).col(t) +
        R.slice(t * Rtv) * uk;
    }
  }
  if (predict_type < 3) {
    
    arma::cube y(1, n, nsim);
    
    switch(distribution) {
    case 0:
      y.zeros();
      break;
    case 1:
      for (unsigned int i = 0; i < nsim; i++) {
        for (unsigned int t = 0; t < n; t++) {
          y(0, t, i) = std::exp(xbeta(t) + D(t * Dtv) +
            arma::as_scalar(Z.col(t * Ztv).t() * alpha.slice(i).col(t)));
        }
      }
      break;
    case 2:
      for (unsigned int i = 0; i < nsim; i++) {
        for (unsigned int t = 0; t < n; t++) {
          double tmp = std::exp(xbeta(t) + D(t * Dtv) +
            arma::as_scalar(Z.col(t * Ztv).t() * alpha.slice(i).col(t)));
          y(0, t, i) = tmp / (1.0 + tmp);
        }
      }
      break;
    case 3:
      for (unsigned int i = 0; i < nsim; i++) {
        for (unsigned int t = 0; t < n; t++) {
          y(0, t, i) = std::exp(xbeta(t) + D(t * Dtv) +
            arma::as_scalar(Z.col(t * Ztv).t() * alpha.slice(i).col(t)));
        }
      }
      break;
    }
    
    if (predict_type == 1) {
      
      switch(distribution) {
      case 0:
        break;
      case 1:
        for (unsigned int i = 0; i < nsim; i++) {
          for (unsigned int t = 0; t < n; t++) {
            std::poisson_distribution<> poisson(u(t) * y(0, t, i));
            if ((u(t) * y(0, t, i)) < poisson.max()) {
              y(0, t, i) = poisson(engine);
            } else {
              y(0, t, i) = std::numeric_limits<double>::quiet_NaN();
            }
          }
        }
        break;
      case 2:
        for (unsigned int i = 0; i < nsim; i++) {
          for (unsigned int t = 0; t < n; t++) {
            std::binomial_distribution<> binomial(u(t), y(0, t, i));
            y(0, t, i) = binomial(engine);
          }
        }
        break;
      case 3:
        for (unsigned int i = 0; i < nsim; i++) {
          for (unsigned int t = 0; t < n; t++) {
            std::negative_binomial_distribution<>
            negative_binomial(phi, phi / (phi + u(t) * y(0, t, i)));
            y(0, t, i) = negative_binomial(engine);
          }
        }
        break;
      }
    }
    return y;
  }
  return alpha;
}
