#include "mng_ssm.h"
#include "conditional_dist.h"
#include "distr_consts.h"
#include "sample.h"
#include "rep_mat.h"

mng_ssm::mng_ssm(const arma::mat& y, lmat_fnPtr Z_fn_, lmat_fnPtr T_fn_,
  lmat_fnPtr R_fn_, a1_fnPtr a1_fn_, P1_fnPtr P1_fn_,
  const arma::vec& theta, prior_fnPtr log_prior_pdf_, const arma::vec& known_params,
  const arma::mat& known_tv_params, const unsigned int m, const unsigned int k,
  const arma::uvec& time_varying, const arma::vec& phi, const arma::mat& u,
  const arma::uvec& distribution, const arma::uvec& phi_est,
  const arma::mat& initial_mode, const bool local_approx, 
  const unsigned int seed, const double zero_tol) :
  y(y), Z_fn(Z_fn_), T_fn(T_fn_), R_fn(R_fn_),
  a1_fn(a1_fn_), P1_fn(P1_fn_), theta(theta), 
  log_prior_pdf(log_prior_pdf_), known_params(known_params),
  known_tv_params(known_tv_params), m(m), k(k), n(y.n_cols), p(y.n_rows),
  Ztv(time_varying(0)), Ttv(time_varying(1)), Rtv(time_varying(2)), 
  phi(phi), u(u), distribution(distribution), phi_est(phi_est),
  engine(seed), zero_tol(zero_tol), 
  initial_mode(initial_mode), mode_estimate(initial_mode), 
  max_iter(100), conv_tol(1.0e-8), local_approx(local_approx), approx_state(-1),
  approx_loglik(0.0), scales(arma::vec(n, arma::fill::zeros)),
  approx_model(y, 
    arma::cube(p, m, (n - 1) * Ztv + 1),
    arma::cube(p, p, n),
    arma::cube(m, m, (n - 1) * Ttv + 1),
    arma::cube(m, k, (n - 1) * Rtv + 1),
    a1_fn(theta, known_params),
    P1_fn(theta, known_params), 
    arma::mat(p, 1, arma::fill::zeros),
    arma::mat(m, 1, arma::fill::zeros), seed + 1){
}

// update the approximating Gaussian model
// Note that the convergence is assessed only
// by checking the changes in mode, not the actual function values
void mng_ssm::approximate() {
  
  // check if there is need to update the approximation
  if (approx_state < 1) {
    //update model
    
    arma::vec a1 = a1_fn(theta, known_params);
    arma::mat P1 = P1_fn(theta, known_params);
    
    for (unsigned int t = 0; t < approx_model.Z.n_slices; t++) {
      approx_model.Z.slice(t) = Z_fn(t, theta, known_params, known_tv_params);
    }
    for (unsigned int t = 0; t < approx_model.T.n_slices; t++) {
      approx_model.T.slice(t) = T_fn(t, theta, known_params, known_tv_params);
    }
    for (unsigned int t = 0; t < approx_model.R.n_slices; t++) {
      approx_model.R.slice(t) = R_fn(t, theta, known_params, known_tv_params);
    }
    
    // don't update y and H if using global approximation and we have updated them already
    if(!local_approx & (approx_state == 0)) {
      arma::mat alpha = approx_model.fast_smoother().head_cols(n);
      for (unsigned int t = 0; t < n; t++) {
        mode_estimate.col(t) = approx_model.Z.slice(Ztv * t) * alpha.col(t);
      }
      
    } else {
      unsigned int i = 0;
      double diff = conv_tol + 1;
      while(i < max_iter && diff > conv_tol) {
        i++;
        //Construct y and H for the Gaussian model
        laplace_iter(mode_estimate, approx_model.y, approx_model.H);
        approx_model.compute_HH();
        // compute new guess of mode
        arma::vec mode_estimate_new(n);
        arma::mat alpha = approx_model.fast_smoother().head_cols(n);
        for (unsigned int t = 0; t < n; t++) {
          mode_estimate_new.col(t) = approx_model.Z.slice(Ztv * t) * alpha.col(t);
        }
        diff = arma::mean(arma::square(mode_estimate_new - mode_estimate));
        mode_estimate = mode_estimate_new;
      }
    }
    approx_state = 1;
  }
}


// method = 1 psi-APF, 2 = BSF, 3 = SPDK (not applicable), 4 = IEKF (not applicable)
arma::vec mng_ssm::log_likelihood(
    const unsigned int method, 
    const unsigned int nsim_states, 
    arma::cube& alpha, 
    arma::mat& weights, 
    arma::umat& indices) {
  
  arma::vec loglik(2);
  
  if (nsim_states > 0) {
    // bootstrap filter
    if(method == 2) {
      loglik(0) = bsf_filter(nsim_states, alpha, weights, indices);
      loglik(1) = loglik(0);
    } else {
      // check that approx_model matches theta
      if(approx_state != 1) {
        mode_estimate = initial_mode;
        approximate(); 
        
        // compute the log-likelihood of the approximate model
        double gaussian_loglik = approx_model.log_likelihood();
        // compute unnormalized mode-based correction terms 
        // log[g(y_t | ^alpha_t) / ~g(y_t | ^alpha_t)]
        update_scales();
        // compute the constant term
        double const_term = compute_const_term(); 
        // log-likelihood approximation
        approx_loglik = gaussian_loglik + const_term + arma::accu(scales);
      }
      // psi-PF
      if (method == 1) {
        loglik(0) = psi_filter(nsim_states, alpha, weights, indices);
      } else {
        //SPDK
        alpha = approx_model.simulate_states(nsim_states);
        arma::vec w(n, arma::fill::zeros);
        for (unsigned int t = 0; t < n; t++) {
          w += log_weights(t, alpha);
        }
        w -= arma::accu(scales);
        double maxw = w.max();
        w = arma::exp(w - maxw);
        weights.col(n) = w; // we need these for sampling etc
        loglik(0) = approx_loglik + std::log(arma::mean(w)) + maxw;
      }
      loglik(1) = approx_loglik;
    } 
  } else {
    // check that approx_model matches theta
    if(approx_state != 1) {
      mode_estimate = initial_mode;
      approximate(); 
      // compute the log-likelihood of the approximate model
      double gaussian_loglik = approx_model.log_likelihood();
      // compute unnormalized mode-based correction terms 
      // log[g(y_t | ^alpha_t) / ~g(y_t | ^alpha_t)]
      update_scales();
      // compute the constant term
      double const_term = compute_const_term(); 
      // log-likelihood approximation
      approx_loglik = gaussian_loglik + const_term + arma::accu(scales);
    }
    loglik(0) = approx_loglik;
    loglik(1) = loglik(0);
  }
  return loglik;
}


// compute unnormalized mode-based scaling terms
// log[g(y_t | ^alpha_t) / ~g(y_t | ^alpha_t)]
void mng_ssm::update_scales() {
  
  for(unsigned int t = 0; t < n; t++) {
    for(unsigned int i = 0; i < p; i++) {
      if (arma::is_finite(y(i, t))) {
        switch(distribution(i)) {
        case 0  :
          scales(t) += -0.5 * (mode_estimate(i, t) + 
            std::pow(y(i, t) / phi(i), 2.0) * std::exp(-mode_estimate(i, t)));
          break;
        case 1  :
          scales(t) += y(i, t) * mode_estimate(i, t) -
            u(i, t) * std::exp(mode_estimate(i, t));
          break;
        case 2  :
          scales(t) += y(i, t) * mode_estimate(t) - 
            u(i, t) * std::log1p(std::exp(mode_estimate(i, t)));
          break;
        case 3  :
          scales(t) += y(i, t) * mode_estimate(i, t) -(y(i, t) + phi(i)) *
            std::log(phi(i) + u(i, t) * std::exp(mode_estimate(i, t)));
          break;
        }
        scales(t) +=
          0.5 * std::pow((approx_model.y(i, t) - mode_estimate(i, t)) / approx_model.H(i, i, t), 2.0);
      }
    }
  }
}

// given the current guess of mode, compute new values of y and H of
// approximate model
/* distribution:
 * 0 = Stochastic volatility model
 * 1 = Poisson
 * 2 = Binomial
 * 3 = Negative binomial
 */
void mng_ssm::laplace_iter(const arma::mat& signal, arma::mat& approx_y,
  arma::cube& approx_H) const {
  
  //note: using the variable approx_H to store approx_HH first
  for(unsigned int i = 0; i < p; i++) {
    arma::rowvec Hvec = approx_H.tube(i, i);
    switch(distribution(i)) {
    case 0: {
      arma::rowvec tmp = y.row(i);
      // avoid dividing by zero
      tmp(arma::find(arma::abs(tmp) < 1e-4)).fill(1e-4);
      approx_H.tube(i, i) = 2.0 * arma::exp(signal.row(i)) / arma::square(tmp/phi(i));
      approx_y.row(i) = signal.row(i) + 1.0 - 0.5 * Hvec;
    } break;
    case 1: {
      approx_H.tube(i, i) = 1.0 / (arma::exp(signal.row(i)) % u.row(i));
      approx_y.row(i) = y.row(i) % Hvec + signal.row(i) - 1.0;
    } break;
    case 2: {
      arma::rowvec exptmp = arma::exp(signal.row(i));
      approx_H.tube(i, i) = arma::square(1.0 + exptmp) / (u.row(i) % exptmp);
      approx_y.row(i) = y.row(i) % Hvec + signal.row(i) - 1.0 - exptmp;
    } break;
    case 3: {
      arma::vec exptmp = 1.0 / (arma::exp(signal.row(i)) % u.row(i));
      approx_H.tube(i, i) = 1.0 / phi(i) + exptmp;
      approx_y.row(i) = signal.row(i) + y.row(i) % exptmp - 1.0;
    } break;
    }
  }
  approx_H = arma::sqrt(approx_H);
}
// these are really not constant in all cases (note phi)
double mng_ssm::compute_const_term() const {
  
  double const_term = 0.0;
  for(unsigned int i = 0; i < p; i++) {
    arma::uvec y_ind(find_finite(y.row(i)));
    switch(distribution(i)) {
    case 0 :
      const_term += y_ind.n_elem * norm_log_const(phi(i));
      break;
    case 1 : 
      for(unsigned int t = 0; t < y_ind.n_elem; t++) {
        const_term += poisson_log_const(y(i, y_ind(t)), u(i, y_ind(t)));
      }
      break;
    case 2 : 
      for(unsigned int t = 0; t < y_ind.n_elem; t++) {
        const_term += binomial_log_const(y(i, y_ind(t)), u(i, y_ind(t)));
      }
      break;
    case 3 : 
      for(unsigned int t = 0; t < y_ind.n_elem; t++) {
        const_term += negbin_log_const(y(i, y_ind(t)), u(i, y_ind(t)), phi(i));
      }
      break;
    }
    for(unsigned int t = 0; t < y_ind.n_elem; t++) {
      const_term -= norm_log_const(approx_model.H(i, i, y_ind(t)));
    }
  }
  return const_term;
}

arma::vec mng_ssm::log_weights(const unsigned int t,  const arma::cube& alpha) const {
  
  arma::vec weights(alpha.n_slices, arma::fill::zeros);
  
  arma::mat Z = Z_fn(t, theta, known_params, known_tv_params);
  for (unsigned int i = 0; i < alpha.n_slices; i++) {
    arma::vec simsignal = Z * alpha.slice(i).col(t);
    for(unsigned int j = 0; j < p; j++) {
      if (arma::is_finite(y(j, t))) {
        switch(distribution(j)) {
        case 0  :
          weights(i) += -0.5 * (simsignal(j) + std::pow(y(j,t) / phi(j), 2.0) * 
            std::exp(-simsignal(j)));
          break;
        case 1  :
          weights(i) += y(j,t) * simsignal(j) - u(j,t) * std::exp(simsignal(j));
          break;
        case 2  :
          weights(i) += y(j,t) * simsignal(j) - u(j,t) * std::log1p(std::exp(simsignal(j)));
          break;
        case 3  :
          weights(i) += y(j,t) * simsignal(j) - (y(j,t) + phi(j)) *
            std::log(phi(j) + u(j,t) * std::exp(simsignal(j)));
          break;
        }
        weights += 
          0.5 * std::pow((approx_model.y(j,t) - simsignal(j)) / approx_model.H(j,j,t), 2.0);
      }
    }
  }
  return weights;
}

// Logarithms of _normalized_ densities g(y_t | alpha_t)
/*
 * t:             Time point where the densities are computed
 * alpha:         Simulated particles
 */
arma::vec mng_ssm::log_obs_density(const unsigned int t, 
  const arma::cube& alpha) const {
  
  arma::vec weights(alpha.n_slices, arma::fill::zeros);
  
  arma::mat Z = Z_fn(t, theta, known_params, known_tv_params);
  for (unsigned int i = 0; i < alpha.n_slices; i++) {
    arma::vec simsignal = Z * alpha.slice(i).col(t);
    for(unsigned int j = 0; j < p; j++) {
      if (arma::is_finite(y(j, t))) {
        switch(distribution(j)) {
        case 0  :
          weights(i) += -0.5 * (simsignal(j) + std::pow(y(j,t) / phi(j), 2.0) * 
            std::exp(-simsignal(j)));
          break;
        case 1  :
          weights(i) += y(j,t) * simsignal(j) - u(j,t) * std::exp(simsignal(j));
          break;
        case 2  :
          weights(i) += y(j,t) * simsignal(j) - u(j,t) * std::log1p(std::exp(simsignal(j)));
          break;
        case 3  :
          weights(i) += y(j,t) * simsignal(j) - (y(j,t) + phi(j)) *
            std::log(phi(j) + u(j,t) * std::exp(simsignal(j)));
          break;
        }
      }
    }
  }
  return weights;
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

double mng_ssm::psi_filter(const unsigned int nsim, arma::cube& alpha, 
  arma::mat& weights, arma::umat& indices) {
  
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
  arma::uvec na_y = arma::find_nonfinite(y.col(0));
  if (na_y.n_elem < p) {
    weights.col(0) = arma::exp(log_weights(0, alpha) - scales(0));
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
    
    if ((t < (n - 1)) && arma::uvec(arma::find_nonfinite(y.col(t + 1))).n_elem < p) {
      weights.col(t + 1) =
        arma::exp(log_weights(t + 1, alpha) - scales(t + 1));
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


double mng_ssm::bsf_filter(const unsigned int nsim, arma::cube& alpha,
  arma::mat& weights, arma::umat& indices) {
  
  arma::mat P1 = P1_fn(theta, known_params);
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
    alpha.slice(i).col(0) = a1_fn(theta, known_params) + L_P1 * um;
  }
  
  std::uniform_real_distribution<> unif(0.0, 1.0);
  arma::vec normalized_weights(nsim);
  double loglik = 0.0;
  
  arma::uvec na_y = arma::find_nonfinite(y.col(0));
  if (na_y.n_elem < p) {
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
      alpha.slice(i).col(t + 1) = 
        T_fn(t, theta, known_params, known_tv_params) * alphatmp.col(i) + 
        R_fn(t, theta, known_params, known_tv_params) * uk;
    }
    
    if ((t < (n - 1)) && arma::uvec(arma::find_nonfinite(y.col(t + 1))).n_elem < p) {
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
  for(unsigned int i = 0; i < p; i++) {
    arma::uvec y_ind(find_finite(y.row(i)));
    // constant part of the log-likelihood
    switch(distribution(i)) {
    case 0 :
      loglik += y_ind.n_elem * norm_log_const(phi(i));
      break;
    case 1 : 
      for(unsigned int t = 0; t < y_ind.n_elem; t++) {
        loglik += poisson_log_const(y(i, y_ind(t)), u(i, y_ind(t)));
      }
      break;
    case 2 :
      for(unsigned int t = 0; t < y_ind.n_elem; t++) {
        loglik += binomial_log_const(y(i,y_ind(t)), u(i,y_ind(t)));
      }
      break;
    case 3 :
      for(unsigned int t = 0; t < y_ind.n_elem; t++) {
        loglik += negbin_log_const(y(i,y_ind(t)), u(i,y_ind(t)), phi(i));
      }
      break;
    }
  }
  return loglik;
}
