#include "model_ssm_ung.h"
#include "model_ssm_ulg.h"
#include "conditional_dist.h"
#include "distr_consts.h"
#include "sample.h"
#include "rep_mat.h"

// General constructor of ssm_ung object from Rcpp::List
ssm_ung::ssm_ung(const Rcpp::List model, const unsigned int seed, const double zero_tol) 
  : y(Rcpp::as<arma::vec>(model["y"])), Z(Rcpp::as<arma::mat>(model["Z"])),
    T(Rcpp::as<arma::cube>(model["T"])), R(Rcpp::as<arma::cube>(model["R"])),
    a1(Rcpp::as<arma::vec>(model["a1"])), P1(Rcpp::as<arma::mat>(model["P1"])),
    D(Rcpp::as<arma::vec>(model["D"])), C(Rcpp::as<arma::mat>(model["C"])),
    xreg(Rcpp::as<arma::mat>(model["xreg"])), beta(Rcpp::as<arma::vec>(model["beta"])),
    n(y.n_elem), m(a1.n_elem), k(R.n_cols), Ztv(Z.n_cols > 1), Ttv(T.n_slices > 1), 
    Rtv(R.n_slices > 1), Dtv(D.n_elem > 1),  Ctv(C.n_cols > 1),
    theta(Rcpp::as<arma::vec>(model["theta"])), 
    phi(model["phi"]),
    u(Rcpp::as<arma::vec>(model["u"])), 
    distribution(model["distribution"]),
    max_iter(model["max_iter"]), conv_tol(model["conv_tol"]), 
    local_approx(model["local_approx"]),
    initial_mode((Rcpp::as<arma::mat>(model["initial_mode"])).t()),
    mode_estimate(initial_mode),
    approx_state(-1),
    approx_loglik(0.0), scales(arma::vec(n, arma::fill::zeros)),
    engine(seed), zero_tol(zero_tol),
    RR(arma::cube(m, m, Rtv * (n - 1) + 1)),
    xbeta(arma::vec(n, arma::fill::zeros)),
    approx_model(arma::vec(n, arma::fill::zeros),
      Z, arma::vec(n, arma::fill::zeros),
      T, R, a1, P1, D, C, xreg, beta, theta, seed + 1) {
  
  if(xreg.n_cols > 0) {
    compute_xbeta();
  }
  compute_RR();
  
}

void ssm_ung::update_model(const arma::vec& new_theta, const Rcpp::Function update_fn) {
  
  Rcpp::List model_list =
    update_fn(Rcpp::NumericVector(new_theta.begin(), new_theta.end()));
  
  if (model_list.containsElementNamed("Z")) {
    Z = Rcpp::as<arma::mat>(model_list["Z"]);
  }
  if (model_list.containsElementNamed("T")) {
    T = Rcpp::as<arma::cube>(model_list["T"]);
  }
  if (model_list.containsElementNamed("R")) {
    R = Rcpp::as<arma::cube>(model_list["R"]);
    compute_RR();
  }
  if (model_list.containsElementNamed("a1")) {
    a1 = Rcpp::as<arma::vec>(model_list["a1"]);
  }
  if (model_list.containsElementNamed("P1")) {
    P1 = Rcpp::as<arma::mat>(model_list["P1"]);
  }
  if (model_list.containsElementNamed("D")) {
    D = Rcpp::as<arma::vec>(model_list["D"]);
  }
  if (model_list.containsElementNamed("C")) {
    C = Rcpp::as<arma::mat>(model_list["C"]);
  }
  if (model_list.containsElementNamed("phi")) {
    phi = Rcpp::as<double>(model_list["phi"]);
  }
  if (model_list.containsElementNamed("beta")) {
    beta = Rcpp::as<arma::vec>(model_list["beta"]);
    compute_xbeta();
  }
  theta = new_theta;
  // approximation does not match theta anymore (keep as -1 if so)
  if (approx_state > 0) approx_state = 0;
}

double ssm_ung::log_prior_pdf(const arma::vec& x, const Rcpp::Function prior_fn) const {
  return Rcpp::as<double>(prior_fn(Rcpp::NumericVector(x.begin(), x.end())));
}

// update the approximating Gaussian model
// Note that the convergence is assessed only
// by checking the changes in mode, not the actual function values
void ssm_ung::approximate() {
  
  // check if there is need to update the approximation
  if (approx_state < 1) {
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
    
    // don't update y and H if using global approximation and we have updated them already
    if(!local_approx & (approx_state == 0)) {
      if (distribution == 0) {
        mode_estimate = approx_model.fast_smoother().head_cols(n);
      } else {
        arma::mat alpha = approx_model.fast_smoother();
        for (unsigned int t = 0; t < n; t++) {
          mode_estimate.col(t) = xbeta(t) + D(Dtv * t) + 
            Z.col(Ztv * t).t() * alpha.col(t);
        }
      }
    } else {
      unsigned int i = 0;
      double diff = conv_tol + 1;
      while(i < max_iter && diff > conv_tol) {
        i++;
        //Construct y and H for the Gaussian model
        laplace_iter(arma::vectorise(mode_estimate));
        
        // compute new guess of mode
        arma::mat mode_estimate_new(1, n);
        if (distribution == 0) {
          mode_estimate_new = approx_model.fast_smoother().head_cols(n);
        } else {
          arma::mat alpha = approx_model.fast_smoother();
          for (unsigned int t = 0; t < n; t++) {
            mode_estimate_new.col(t) = xbeta(t) + 
              D(Dtv * t) + Z.col(Ztv * t).t() * alpha.col(t);
          }
        }
        
        diff = arma::accu(arma::square(mode_estimate_new - mode_estimate)) / n;
        mode_estimate = mode_estimate_new;
      }
    }
    approx_state = 1;
  }
}
// construct approximating model from fixed mode estimate, no iterations
// used in IS-correction
void ssm_ung::approximate_for_is(const arma::mat& mode_estimate_) {
  
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
  //Construct y and H for the Gaussian model
  mode_estimate = mode_estimate_;
  laplace_iter(arma::vectorise(mode_estimate));
  update_scales();
  approx_loglik = 0.0;
  approx_state = 2;
}

// method = 1 psi-APF, 2 = BSF, 3 = SPDK, 4 = IEKF (not applicable)
arma::vec ssm_ung::log_likelihood(
    const unsigned int method, 
    const unsigned int nsim, 
    arma::cube& alpha, 
    arma::mat& weights, 
    arma::umat& indices) {
  
  arma::vec loglik(2);
  
  if (nsim > 0) {
    // bootstrap filter
    if(method == 2) {
      loglik(0) = bsf_filter(nsim, alpha, weights, indices);
      loglik(1) = loglik(0);
    } else {
      // check that approx_model matches theta
      if(approx_state < 2) {
        if (approx_state < 1) {
          mode_estimate = initial_mode;
          approximate(); 
        }
        // compute the log-likelihood of the approximate model
        double gaussian_loglik = approx_model.log_likelihood();
        // compute unnormalized mode-based correction terms 
        // log[g(y_t | ^alpha_t) / ~g(y_t | ^alpha_t)]
        update_scales();
        // compute the constant term
        double const_term = compute_const_term(); 
        // log-likelihood approximation
        approx_loglik = gaussian_loglik + const_term + arma::accu(scales);
        approx_state = 2;
      }
      // psi-PF
      if (method == 1) {
        loglik(0) = psi_filter(nsim, alpha, weights, indices);
      } else {
        //SPDK
        alpha = approx_model.simulate_states(nsim, true);
        arma::vec w(nsim, arma::fill::zeros);
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
    if(approx_state < 2) {
      if (approx_state < 1) {
        mode_estimate = initial_mode;
        approximate(); 
      }
      // compute the log-likelihood of the approximate model
      double gaussian_loglik = approx_model.log_likelihood();
      // compute unnormalized mode-based correction terms 
      // log[g(y_t | ^alpha_t) / ~g(y_t | ^alpha_t)]
      update_scales();
      // compute the constant term
      double const_term = compute_const_term(); 
      // log-likelihood approximation
      approx_loglik = gaussian_loglik + const_term + arma::accu(scales);
      approx_state = 2;
    }
    loglik(0) = approx_loglik;
    loglik(1) = loglik(0);
  }
  return loglik;
}


// compute unnormalized mode-based scaling terms
// log[g(y_t | ^alpha_t) / ~g(y_t | ^alpha_t)]
void ssm_ung::update_scales() {
  
  scales.zeros();
  
  switch(distribution) {
  case 0  :
    for(unsigned int t = 0; t < n; t++) {
      if (arma::is_finite(y(t))) {
        scales(t) = -0.5 * (mode_estimate(t) + std::pow(y(t) / phi, 2.0) *
          std::exp(-mode_estimate(t))) +
          0.5 * std::pow((approx_model.y(t) - mode_estimate(t)) / approx_model.H(t), 2.0);
      }
    }
    break;
  case 1  :
    for(unsigned int t = 0; t < n; t++) {
      if (arma::is_finite(y(t))) {
        scales(t) = y(t) * mode_estimate(t) -
          u(t) * std::exp(mode_estimate(t)) +
          0.5 * std::pow((approx_model.y(t) - mode_estimate(t)) / approx_model.H(t), 2.0);
      }
    }
    break;
  case 2  :
    for(unsigned int t = 0; t < n; t++) {
      if (arma::is_finite(y(t))) {
        scales(t) = y(t) * mode_estimate(t) -
          u(t) * std::log1p(std::exp(mode_estimate(t))) +
          0.5 * std::pow((approx_model.y(t) - mode_estimate(t)) / approx_model.H(t), 2.0);
      }
    }
    break;
  case 3  :
    for(unsigned int t = 0; t < n; t++) {
      if (arma::is_finite(y(t))) {
        scales(t) = y(t) * mode_estimate(t) -
          (y(t) + phi) *
          std::log(phi + u(t) * std::exp(mode_estimate(t))) +
          0.5 * std::pow((approx_model.y(t) - mode_estimate(t)) / approx_model.H(t), 2.0);
      } 
    }
    break;
  case 4  :
    for(unsigned int t = 0; t < n; t++) {
      if (arma::is_finite(y(t))) {
        scales(t) = -phi * (mode_estimate(t) + (y(t) * exp(-mode_estimate(t)) / u(t))) +
          0.5 * std::pow((approx_model.y(t) - mode_estimate(t)) / approx_model.H(t), 2.0);
      } 
    }
    break;
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
void ssm_ung::laplace_iter(const arma::vec& signal) {
  
  
  switch(distribution) {
  case 0: {
  arma::vec tmp = y;
  // avoid dividing by zero
  tmp(arma::find(arma::abs(tmp) < 1e-4)).fill(1e-4);
  approx_model.HH = 2.0 * arma::exp(signal) / arma::square(tmp/phi);
  approx_model.y = signal + 1.0 - 0.5 * approx_model.HH;
} break;
  case 1: {
    arma::vec tmp = signal;
    approx_model.HH = 1.0 / (arma::exp(tmp) % u);
    approx_model.y = y % approx_model.HH + tmp - 1.0;
  } break;
  case 2: {
    arma::vec exptmp = arma::exp(signal);
    approx_model.HH = arma::square(1.0 + exptmp) / (u % exptmp);
    approx_model.y = y % approx_model.HH + signal - 1.0 - exptmp;
  } break;
  case 3: {
    // negative binomial
    arma::vec exptmp = arma::exp(signal) % u;
    approx_model.HH = arma::square(phi + exptmp) / (phi * exptmp % (y + phi));
    approx_model.y = signal + (phi + exptmp) % (y - exptmp) / ((y + phi) % exptmp);
  } break;
  case 4: {
    // gamma
    arma::vec exptmp = arma::exp(signal) % u;
    approx_model.HH = exptmp / (y * phi);
    approx_model.y = signal - exptmp / y + 1;
  } break;
    // case 5: {
    //   // gaussian, not actually used here as univariate gaussian belongs to ulg...
    //   approx_model.HH = phi;
    //   approx_model.y = y;
    // } break;
  }
  approx_model.H = arma::sqrt(approx_model.HH);
}



// these are really not constant in all cases (note phi)
double ssm_ung::compute_const_term() {
  
  double const_term = 0.0;
  
  arma::uvec y_ind(find_finite(y));
  switch(distribution) {
  case 0 :
    const_term = y_ind.n_elem * norm_log_const(phi);
    break;
  case 1 : 
    const_term = poisson_log_const(y(y_ind), u(y_ind));
    break;
  case 2 : 
    const_term = binomial_log_const(y(y_ind), u(y_ind));
    break;
  case 3 :
    const_term = negbin_log_const(y(y_ind), u(y_ind), phi);
    break;  
  case 4 :
    const_term = gamma_log_const(y(y_ind), u(y_ind), phi);
    break;
  }
  return const_term - norm_log_const(approx_model.y(y_ind), approx_model.H(y_ind));
}

arma::vec ssm_ung::importance_weights(const arma::cube& alpha) const {
  arma::vec weights(alpha.n_slices, arma::fill::zeros);
  for(unsigned int t = 0; t < n; t++) {
    weights += log_weights(t, alpha);
  }
  return weights;
}
// Logarithms of _unnormalized_ importance weights g(y_t | alpha_t) / ~g(~y_t | alpha_t)
/*
 * approx_model:  Gaussian approximation of the original model
 * t:             Time point where the weights are computed
 * alpha:         Simulated particles
 */
arma::vec ssm_ung::log_weights(
    const unsigned int t, 
    const arma::cube& alpha) const {
  
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
        double simsignal = arma::as_scalar(D(t * Dtv) + Z.col(t * Ztv).t() *
          alpha.slice(i).col(t) + xbeta(t));
        weights(i) = y(t) * simsignal  - u(t) * std::exp(simsignal) +
          0.5 * std::pow((approx_model.y(t) - simsignal) / approx_model.H(t), 2.0);
      }
      break;
    case 2  :
      for (unsigned int i = 0; i < alpha.n_slices; i++) {
        double simsignal = arma::as_scalar(D(t * Dtv) + Z.col(t * Ztv).t() *
          alpha.slice(i).col(t) + xbeta(t));
        weights(i) = y(t) * simsignal - u(t) * std::log1p(std::exp(simsignal)) +
          0.5 * std::pow((approx_model.y(t) - simsignal) / approx_model.H(t), 2.0);
      }
      break;
    case 3  : //negbin
      for (unsigned int i = 0; i < alpha.n_slices; i++) {
        double simsignal = arma::as_scalar(D(t * Dtv) + Z.col(t * Ztv).t() *
          alpha.slice(i).col(t) + xbeta(t));
        weights(i) = y(t) * simsignal - (y(t) + phi) *
          std::log(phi + u(t) * std::exp(simsignal)) +
          0.5 * std::pow((approx_model.y(t) - simsignal) / approx_model.H(t), 2.0);
      }
      break;
    case 4  : //gamma
      for (unsigned int i = 0; i < alpha.n_slices; i++) {
        double simsignal = arma::as_scalar(D(t * Dtv) + Z.col(t * Ztv).t() *
          alpha.slice(i).col(t) + xbeta(t));
        weights(i) = -phi * (simsignal + (y(t) * exp(-simsignal) / u(t))) +
          0.5 * std::pow((approx_model.y(t) - simsignal) / approx_model.H(t), 2.0);
      }
      break;
      // case 5  :
      //   Rcpp::stop("Impossible thing happened: Univariate non-gaussian model is Gaussian!")
      //   break;
    }
  }
  return weights;
}


// Logarithms of _unnormalized_ densities g(y_t | alpha_t)
/*
 * t:             Time point where the densities are computed
 * alpha:         Simulated particles
 */
arma::vec ssm_ung::log_obs_density(const unsigned int t, 
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
        double simsignal = arma::as_scalar(D(t * Dtv) + Z.col(t * Ztv).t() *
          alpha.slice(i).col(t) + xbeta(t));
        weights(i) = y(t) * simsignal  - u(t) * std::exp(simsignal);
      }
      break;
    case 2  :
      for (unsigned int i = 0; i < alpha.n_slices; i++) {
        double simsignal = arma::as_scalar(D(t * Dtv) + Z.col(t * Ztv).t() *
          alpha.slice(i).col(t) + xbeta(t));
        weights(i) = y(t) * simsignal - u(t) * std::log1p(std::exp(simsignal));
      }
      break;
    case 3  :
      for (unsigned int i = 0; i < alpha.n_slices; i++) {
        double simsignal = arma::as_scalar(D(t * Dtv) + Z.col(t * Ztv).t() *
          alpha.slice(i).col(t) + xbeta(t));
        weights(i) = y(t) * simsignal - (y(t) + phi) *
          std::log(phi + u(t) * std::exp(simsignal));
      }
      break;
    case 4  :
      for (unsigned int i = 0; i < alpha.n_slices; i++) {
        double simsignal = arma::as_scalar(D(t * Dtv) + Z.col(t * Ztv).t() *
          alpha.slice(i).col(t) + xbeta(t));
        weights(i) = -phi * (simsignal + (y(t) * exp(-simsignal) / u(t)));
        
      }
      break;
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

double ssm_ung::psi_filter(const unsigned int nsim, arma::cube& alpha, 
  arma::mat& weights, arma::umat& indices) {
  
  if(approx_state < 2) {
    if (approx_state < 1) {
      mode_estimate = initial_mode;
      approximate(); 
    }
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
    weights.col(0) = log_weights(0, alpha) - scales(0);
    
    double max_weight = weights.col(0).max();
    weights.col(0) = arma::exp(weights.col(0) - max_weight);
    
    double sum_weights = arma::accu(weights.col(0));
    if(sum_weights > 0.0){
      normalized_weights = weights.col(0) / sum_weights;
    } else {
      return -std::numeric_limits<double>::infinity();
    }
    loglik = max_weight + approx_loglik + std::log(sum_weights / nsim);
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
      weights.col(t + 1) = log_weights(t + 1, alpha) - scales(t + 1);
      
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
      normalized_weights.fill(1.0 / nsim);
    }
  }
  
  return loglik;
}

double ssm_ung::bsf_filter(const unsigned int nsim, arma::cube& alpha,
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
  case 4 : {
    arma::uvec finite_y(find_finite(y));
    loglik += gamma_log_const(y(finite_y), u(finite_y), phi);
  } break;
  }
  return loglik;
}

arma::cube ssm_ung::predict_sample(const arma::mat& theta_posterior,
  const arma::mat& alpha, const unsigned int predict_type, const Rcpp::Function update_fn) {
  
  unsigned int d = 1;
  if (predict_type == 3) d = m;
  
  unsigned int n_samples = theta_posterior.n_cols;
  arma::cube sample(d, n,  n_samples);
  
  
  for (unsigned int i = 0; i < n_samples; i++) {
    update_model(theta_posterior.col(i), update_fn);
    a1 = alpha.col(i);
    sample.slice(i) = sample_model(predict_type);
  }
  
  return sample;
}


arma::mat ssm_ung::sample_model(const unsigned int predict_type) {
  
  arma::mat alpha(m, n);
  std::normal_distribution<> normal(0.0, 1.0);
  
  alpha.col(0) = a1;
  
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
        y(0, t) = std::exp(xbeta(t) + D(t * Dtv) +
          arma::as_scalar(Z.col(t * Ztv).t() * alpha.col(t)));
      }
      break;
    case 2:
      for (unsigned int t = 0; t < n; t++) {
        double tmp = std::exp(xbeta(t) + D(t * Dtv) +
          arma::as_scalar(Z.col(t * Ztv).t() * alpha.col(t)));
        y(0, t) = tmp / (1.0 + tmp);
      }
      
      break;
    case 3:
      for (unsigned int t = 0; t < n; t++) {
        y(0, t) = std::exp(xbeta(t) + D(t * Dtv) +
          arma::as_scalar(Z.col(t * Ztv).t() * alpha.col(t)));
      }
      break;
    case 4:
      for (unsigned int t = 0; t < n; t++) {
        y(0, t) = std::exp(xbeta(t) + D(t * Dtv) +
          arma::as_scalar(Z.col(t * Ztv).t() * alpha.col(t)));
      }
      break;
    }
    
    if (predict_type == 1) {
      
      switch(distribution) {
      case 0:
        for (unsigned int t = 0; t < n; t++) {
          y(0, t) = phi * exp(0.5 * alpha(0, t)) * normal(engine);
        }
        break;
      case 1:
        for (unsigned int t = 0; t < n; t++) {
          std::poisson_distribution<> poisson(u(t) * y(0, t));
          if ((u(t) * y(0, t)) < poisson.max()) {
            y(0, t) = poisson(engine);
          } else {
            y(0, t) = std::numeric_limits<double>::quiet_NaN();
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
          // std::negative_binomial_distribution<>
          // negative_binomial(phi, phi / (phi + u(t) * y(0, t)));
          // y(0, t) = negative_binomial(engine);
          double prob = phi / (phi + u(t) * y(0, t));
          std::gamma_distribution<> gamma(phi, (1 - prob) / prob);
          std::poisson_distribution<> poisson(gamma(engine));
          y(0, t) = poisson(engine);
        }
        break;
      case 4:
        for (unsigned int t = 0; t < n; t++) {
          std::gamma_distribution<>
          gamma(phi, u(t) * y(0, t) / phi);
          y(0, t) = gamma(engine);
        }
        break;
      }
    }
    return y;
  }
  return alpha;
}


arma::cube ssm_ung::predict_past(const arma::mat& theta_posterior,
  const arma::cube& alpha, const unsigned int predict_type, const Rcpp::Function update_fn) {
  
  unsigned int n_samples = theta_posterior.n_cols;
  arma::cube samples(p, n, n_samples);
  
  std::normal_distribution<> normal(0.0, 1.0);
  for (unsigned int i = 0; i < n_samples; i++) {
    
    update_model(theta_posterior.col(i), update_fn);
    arma::mat y(1, n);
    switch(distribution) {
      case 0:
        y.zeros();
        break;
      case 1:
        for (unsigned int t = 0; t < n; t++) {
          y(0, t) = std::exp(xbeta(t) + D(t * Dtv) +
            arma::as_scalar(Z.col(t * Ztv).t() * alpha.slice(i).col(t)));
        }
        break;
      case 2:
        for (unsigned int t = 0; t < n; t++) {
          double tmp = std::exp(xbeta(t) + D(t * Dtv) +
            arma::as_scalar(Z.col(t * Ztv).t() * alpha.slice(i).col(t)));
          y(0, t) = tmp / (1.0 + tmp);
        }
        
        break;
      case 3:
        for (unsigned int t = 0; t < n; t++) {
          y(0, t) = std::exp(xbeta(t) + D(t * Dtv) +
            arma::as_scalar(Z.col(t * Ztv).t() * alpha.slice(i).col(t)));
        }
        break;
      case 4:
        for (unsigned int t = 0; t < n; t++) {
          y(0, t) = std::exp(xbeta(t) + D(t * Dtv) +
            arma::as_scalar(Z.col(t * Ztv).t() * alpha.slice(i).col(t)));
        }
        break;
    }
    if (predict_type == 1) {
      
      switch(distribution) {
      case 0:
        for (unsigned int t = 0; t < n; t++) {
          y(0, t) = phi * exp(0.5 * alpha(0, t, i)) * normal(engine);
        }
        break;
      case 1:
        for (unsigned int t = 0; t < n; t++) {
          std::poisson_distribution<> poisson(u(t) * y(0, t));
          if ((u(t) * y(0, t)) < poisson.max()) {
            y(0, t) = poisson(engine);
          } else {
            y(0, t) = std::numeric_limits<double>::quiet_NaN();
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
          double prob = phi / (phi + u(t) * y(0, t));
          std::gamma_distribution<> gamma(phi, (1 - prob) / prob);
          std::poisson_distribution<> poisson(gamma(engine));
          y(0, t) = poisson(engine);
        }
        break;
      case 4:
        for (unsigned int t = 0; t < n; t++) {
          std::gamma_distribution<>
          gamma(phi, u(t) * y(0, t) / phi);
          y(0, t) = gamma(engine);
        }
        break;
      }
    }
    samples.slice(i) = y;
  }
  return samples;
}
