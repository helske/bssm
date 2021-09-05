#include "model_ssm_mng.h"
#include "conditional_dist.h"
#include "distr_consts.h"
#include "sample.h"
#include "rep_mat.h"

ssm_mng::ssm_mng(const Rcpp::List model, const unsigned int seed, const double zero_tol) 
  :  y((Rcpp::as<arma::mat>(model["y"])).t()), Z(Rcpp::as<arma::cube>(model["Z"])),
    T(Rcpp::as<arma::cube>(model["T"])),
    R(Rcpp::as<arma::cube>(model["R"])), a1(Rcpp::as<arma::vec>(model["a1"])),
    P1(Rcpp::as<arma::mat>(model["P1"])), D(Rcpp::as<arma::mat>(model["D"])),
    C(Rcpp::as<arma::mat>(model["C"])), 
    n(y.n_cols), m(a1.n_elem), k(R.n_cols), p(y.n_rows), 
    Ztv(Z.n_slices > 1), Ttv(T.n_slices > 1), Rtv(R.n_slices > 1),
    Dtv(D.n_cols > 1), Ctv(C.n_cols > 1), 
    theta(Rcpp::as<arma::vec>(model["theta"])), 
    phi(Rcpp::as<arma::vec>(model["phi"])), u((Rcpp::as<arma::mat>(model["u"])).t()), 
    distribution(Rcpp::as<arma::uvec>(model["distribution"])),
    max_iter(model["max_iter"]), conv_tol(model["conv_tol"]), 
    local_approx(model["local_approx"]),
    initial_mode((Rcpp::as<arma::mat>(model["initial_mode"])).t()),
    mode_estimate(initial_mode),
    approx_state(-1),
    approx_loglik(0.0), scales(arma::vec(n, arma::fill::zeros)),
    engine(seed), zero_tol(zero_tol),
    RR(arma::cube(m, m, Rtv * (n - 1) + 1)),
    approx_model(y, Z, arma::cube(p, p, n, arma::fill::zeros), T, R, a1, P1, 
      D, C, theta, seed + 1){
  compute_RR();
}


void ssm_mng::update_model(const arma::vec& new_theta, const Rcpp::Function update_fn) {
  Rcpp::List model_list = 
    update_fn(Rcpp::NumericVector(new_theta.begin(), new_theta.end()));
  if (model_list.containsElementNamed("Z")) {
    Z = Rcpp::as<arma::cube>(model_list["Z"]);
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
    D = Rcpp::as<arma::mat>(model_list["D"]);
  }
  if (model_list.containsElementNamed("C")) {
    C = Rcpp::as<arma::mat>(model_list["C"]);
  }
  if (model_list.containsElementNamed("phi")) {
    phi = Rcpp::as<arma::vec>(model_list["phi"]);
  }
  theta = new_theta;
  // approximation does not match theta anymore (keep as -1 if so)
  if (approx_state > 0) approx_state = 0;
}

double ssm_mng::log_prior_pdf(const arma::vec& x, const Rcpp::Function prior_fn) const {
  return Rcpp::as<double>(prior_fn(Rcpp::NumericVector(x.begin(), x.end())));
}

// update the approximating Gaussian model
// Note that the convergence is assessed only
// by checking the changes in mode, not the actual function values
void ssm_mng::approximate() {
  
  // check if there is need to update the approximation
  if (approx_state < 1) {
    //update model
    approx_model.Z = Z;
    approx_model.T = T;
    approx_model.R = R;
    approx_model.a1 = a1;
    approx_model.P1 = P1;
    approx_model.D = D;
    approx_model.C = C;
    approx_model.RR = RR;
    
    // don't update y and H if using global approximation and we have updated them already
    if(!local_approx & (approx_state == 0)) {
      arma::mat alpha = approx_model.fast_smoother();
      for (unsigned int t = 0; t < n; t++) {
        mode_estimate.col(t) = D.col(Dtv * t) + approx_model.Z.slice(Ztv * t) * alpha.col(t);
      }
      
    } else {
      unsigned int i = 0;
      double diff = conv_tol + 1;
      while(i < max_iter && diff > conv_tol) {
        i++;
        //Construct y and H for the Gaussian model
        laplace_iter(mode_estimate);
        // compute new guess of mode
        arma::mat mode_estimate_new(p, n);
        arma::mat alpha = approx_model.fast_smoother();
        for (unsigned int t = 0; t < n; t++) {
          mode_estimate_new.col(t) = 
            D.col(Dtv * t) + Z.slice(Ztv * t) * alpha.col(t);
        }
        
        diff = arma::accu(arma::square(mode_estimate_new - mode_estimate)) / (n * p);
        mode_estimate = mode_estimate_new;
      }
    }
    approx_state = 1; //approx matches theta, approx_loglik does not match
  }
  
}
// construct approximating model from fixed mode estimate, no iterations
// used in IS-correction
void ssm_mng::approximate_for_is(const arma::mat& mode_estimate_) {
  
  approx_model.Z = Z;
  approx_model.T = T;
  approx_model.R = R;
  approx_model.a1 = a1;
  approx_model.P1 = P1;
  approx_model.D = D;
  approx_model.C = C;
  approx_model.RR = RR;
  //Construct y and H for the Gaussian model
  mode_estimate = mode_estimate_;
  laplace_iter(mode_estimate);
  update_scales();
  approx_loglik = 0.0;
  approx_state = 2;
}

// method = 1 psi-APF, 2 = BSF, 3 = SPDK (not applicable), 4 = IEKF (not applicable)
arma::vec ssm_mng::log_likelihood(
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
      // check that approx_model matches theta and approx_loglik
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
        alpha = approx_model.simulate_states(nsim);
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
void ssm_mng::update_scales() {
  
  scales.zeros();
  
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
          scales(t) += y(i, t) * mode_estimate(i, t) - 
            u(i, t) * std::log1p(std::exp(mode_estimate(i, t)));
          break;
        case 3  :
          scales(t) += y(i, t) * mode_estimate(i, t) -(y(i, t) + phi(i)) *
            std::log(phi(i) + u(i, t) * std::exp(mode_estimate(i, t)));
          break;
        case 4  :
          scales(t) += -phi(i) * mode_estimate(i, t) - 
            (y(i, t) * phi(i) * exp(-mode_estimate(i, t)) / u(i, t));
          break;
        case 5  :
          scales(t) += -0.5 * std::pow((y(i, t) - mode_estimate(i, t)) / phi(i), 2.0);
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
 * 4 = Gamma
 * 5 = Gaussian
 */
void ssm_mng::laplace_iter(const arma::mat& signal) {
  
  for(unsigned int i = 0; i < p; i++) {
    switch(distribution(i)) {
    //  case 0: {
    // svm, not actually used in multivariate models...
    // arma::rowvec tmp = y.row(i);
    // // avoid dividing by zero
    // tmp(arma::find(arma::abs(tmp) < 1e-4)).fill(1e-4);
    // approx_model.HH.tube(i, i) = 2.0 * arma::exp(signal.row(i)) / arma::square(tmp/phi(i));
    // arma::rowvec Hvec = approx_model.HH.tube(i, i);
    // approx_model.y.row(i) = signal.row(i) + 1.0 - 0.5 * Hvec;
    //} break;
    case 1: {
      // poisson
      approx_model.HH.tube(i, i) = 1.0 / (arma::exp(signal.row(i)) % u.row(i));
      arma::rowvec Hvec = approx_model.HH.tube(i, i);
      approx_model.y.row(i) = y.row(i) % Hvec + signal.row(i) - 1.0;
    } break;
    case 2: {
      // binomial
      arma::rowvec exptmp = arma::exp(signal.row(i));
      approx_model.HH.tube(i, i) = arma::square(1.0 + exptmp) / (u.row(i) % exptmp);
      arma::rowvec Hvec = approx_model.HH.tube(i, i);
      approx_model.y.row(i) = y.row(i) % Hvec + signal.row(i) - 1.0 - exptmp;
    } break;
    case 3: {
      // negative binomial
      arma::rowvec exptmp = arma::exp(signal.row(i)) % u.row(i);
      approx_model.HH.tube(i, i) = arma::square(phi(i) + exptmp) / 
        (phi(i) * exptmp % (y.row(i) + phi(i)));
      approx_model.y.row(i) = signal.row(i) +
        (phi(i) + exptmp) % (y.row(i) - exptmp) / ((y.row(i) + phi(i)) % exptmp);
    } break;
    case 4: {
      // gamma
      arma::rowvec exptmp = arma::exp(signal.row(i)) % u.row(i);
      approx_model.HH.tube(i, i) = exptmp / (y.row(i) * phi(i));
      approx_model.y.row(i) = signal.row(i) - exptmp / y.row(i) + 1;
    } break;
    case 5: {
      // gaussian
      approx_model.HH.tube(i, i).fill(phi(i) * phi(i));
      approx_model.y.row(i) = y.row(i);
    } break;
    }
  }
  approx_model.H = arma::sqrt(approx_model.HH);  // diagonal
}
// these are really not constant in all cases (note phi)
double ssm_mng::compute_const_term() const {
  
  double const_term = 0.0;
  for(unsigned int i = 0; i < p; i++) {
    arma::uvec y_ind(find_finite(y.row(i)));
    switch(distribution(i)) {
    // case 0 :
    //   const_term += y_ind.n_elem * norm_log_const(phi(i));
    //   break;
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
    case 4 : 
      for(unsigned int t = 0; t < y_ind.n_elem; t++) {
        const_term += gamma_log_const(y(i, y_ind(t)), u(i, y_ind(t)), phi(i));
      }
      break;
    case 5 : 
      const_term += y_ind.n_elem * norm_log_const(phi(i));
      break;
    }
    for(unsigned int t = 0; t < y_ind.n_elem; t++) {
      const_term -= norm_log_const(approx_model.H(i, i, y_ind(t)));
    }
  }
  return const_term;
}

arma::vec ssm_mng::log_weights(const unsigned int t,  const arma::cube& alpha) const {
  
  arma::vec weights(alpha.n_slices, arma::fill::zeros);
  
  for (unsigned int i = 0; i < alpha.n_slices; i++) {
    arma::vec simsignal = D.col(t * Dtv) + Z.slice(t * Ztv) * alpha.slice(i).col(t);
    for(unsigned int j = 0; j < p; j++) {
      if (arma::is_finite(y(j, t))) {
        switch(distribution(j)) {
        // case 0  :
        //   weights(i) += -0.5 * (simsignal(j) + std::pow(y(j,t) / phi(j), 2.0) * 
        //     std::exp(-simsignal(j)));
        //   break;
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
        case 4 :
          weights(i) -= phi(j) * (simsignal(j) + (y(j,t) * exp(-simsignal(j)) / u(j,t)));
          break;
        case 5 :
          weights(i) -= 0.5 * std::pow((y(j,t) - simsignal(j)) / phi(j), 2.0);
          break;
        }
        weights(i) += 
          0.5 * std::pow((approx_model.y(j,t) - simsignal(j)) / approx_model.H(j,j,t), 2.0);
      }
    }
  }
  return weights;
}

arma::vec ssm_mng::importance_weights(const arma::cube& alpha) const {
  arma::vec weights(alpha.n_slices, arma::fill::zeros);
  for(unsigned int t = 0; t < n; t++) {
    weights += log_weights(t, alpha);
  }
  return weights;
}
// Logarithms of _normalized_ densities g(y_t | alpha_t)
/*
 * t:             Time point where the densities are computed
 * alpha:         Simulated particles
 */
arma::vec ssm_mng::log_obs_density(const unsigned int t, 
  const arma::cube& alpha) const {
  
  arma::vec weights(alpha.n_slices, arma::fill::zeros);
  
  for (unsigned int i = 0; i < alpha.n_slices; i++) {
    arma::vec simsignal = D.col(t * Dtv) + Z.slice(t * Ztv) * alpha.slice(i).col(t);
    for(unsigned int j = 0; j < p; j++) {
      if (arma::is_finite(y(j, t))) {
        switch(distribution(j)) {
        // case 0  :
        //   weights(i) += -0.5 * (simsignal(j) + std::pow(y(j,t) / phi(j), 2.0) * 
        //     std::exp(-simsignal(j)));
        //   break;
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
        case 4 :
          weights(i) += -phi(j) * (simsignal(j) + (y(j,t) * exp(-simsignal(j)) / u(j, t)));
          break;
        case 5 :
          weights(i) += -0.5 * std::pow((y(j,t) - simsignal(j)) / phi(j), 2.0);
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

double ssm_mng::psi_filter(const unsigned int nsim, arma::cube& alpha, 
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
  arma::uvec na_y = arma::find_nonfinite(y.col(0));
  if (na_y.n_elem < p) {
    
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
    
    if ((t < (n - 1)) && arma::uvec(arma::find_nonfinite(y.col(t + 1))).n_elem < p) {
      
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


double ssm_mng::bsf_filter(const unsigned int nsim, arma::cube& alpha,
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
      alpha.slice(i).col(t + 1) = C.col(t * Ctv) + 
        T.slice(t * Ttv) * alphatmp.col(i) + R.slice(t * Rtv) * uk;
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
    // case 0 :
    //   loglik += y_ind.n_elem * norm_log_const(phi(i));
    //   break;
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
    case 4 :
      for(unsigned int t = 0; t < y_ind.n_elem; t++) {
        loglik += gamma_log_const(y(i,y_ind(t)), u(i,y_ind(t)), phi(i));
      }
      break;
    case 5 :
      loglik += y_ind.n_elem * norm_log_const(phi(i));
      break;
    }
  }
  return loglik;
}


arma::cube ssm_mng::predict_sample(const arma::mat& theta_posterior,
  const arma::mat& alpha, const unsigned int predict_type, const Rcpp::Function update_fn) {
  
  unsigned int d = p;
  if (predict_type == 3) d = m;
  
  unsigned int n_samples = theta_posterior.n_cols;
  arma::cube sample(d, n, n_samples);
  
  
  for (unsigned int i = 0; i < n_samples; i++) {
    update_model(theta_posterior.col(i), update_fn);
    a1 = alpha.col(i);
    sample.slice(i) = sample_model(predict_type);
  }
  return sample;
}


arma::mat ssm_mng::sample_model(const unsigned int predict_type) {
  
  arma::mat alpha(m, n);
  std::normal_distribution<> normal(0.0, 1.0);
  
  alpha.col(0) = a1;
  
  for (unsigned int t = 0; t < (n - 1); t++) {
    arma::vec uk(k);
    for(unsigned int j = 0; j < k; j++) {
      uk(j) = normal(engine);
    }
    alpha.col(t + 1) =
      C.col(t * Ctv) + T.slice(t * Ttv) * alpha.col(t) +
      R.slice(t * Rtv) * uk;
  }
  if (predict_type < 3) {
    
    arma::mat y(p, n);
    for (unsigned int t = 0; t < n; t++) {
      arma::vec signal = D.col(t * Dtv) + Z.slice(t * Ztv) * alpha.col(t);
      for(unsigned int j = 0; j < p; j++) {
        
        switch(distribution(j)) {
        case 1:
          y(j, t) =  std::exp(signal(j));
          break;
        case 2:
          y(j, t) =  std::exp(signal(j)) / (1.0 +  std::exp(signal(j)));
          break;
        case 3:
          y(j, t) =  std::exp(signal(j));
          break;
        case 4:
          y(j, t) =  std::exp(signal(j));
          break;
        case 5:
          y(j, t) =  signal(j);
        }
        
        if (predict_type == 1) {
          
          switch(distribution(j)) {
          case 1: {
            std::poisson_distribution<> poisson(u(j,t) * y(j, t));
            if ((u(j,t) * y(j, t)) < poisson.max()) {
              y(j, t) = poisson(engine);
            } else {
              y(j, t) = std::numeric_limits<double>::quiet_NaN();
            }
          } 
            break;
          case 2: {
            std::binomial_distribution<> binomial(u(j,t), y(j, t));
            y(j, t) = binomial(engine);
          }
            break;
          case 3: {
            double prob = phi(j) / (phi(j) + u(j,t) * y(j, t));
            std::gamma_distribution<> gamma(phi(j), (1 - prob) / prob);
            std::poisson_distribution<> poisson(gamma(engine));
            y(j, t) = poisson(engine);
          }
            break;
          case 4: {
            std::gamma_distribution<> gamma(phi(j), u(j,t) * y(j, t) / phi(j));
            y(j, t) = gamma(engine);
          }
            break;
          }
        }
      }
    }
    return y;
  }
  return alpha;
}


arma::cube ssm_mng::predict_past(const arma::mat& theta_posterior,
  const arma::cube& alpha, const unsigned int predict_type, const Rcpp::Function update_fn) {
  
  unsigned int n_samples = theta_posterior.n_cols;
  arma::cube samples(p, n, n_samples);
  
  std::normal_distribution<> normal(0.0, 1.0);
  for (unsigned int i = 0; i < n_samples; i++) {
    update_model(theta_posterior.col(i), update_fn);
    arma::mat y(p, n);
    for (unsigned int t = 0; t < n; t++) {
      arma::vec signal = D.col(t * Dtv) + Z.slice(t * Ztv) * alpha.slice(i).col(t);
      
      for(unsigned int j = 0; j < p; j++) {
        switch(distribution(j)) {
        case 1:
          y(j, t) =  std::exp(signal(j));
          break;
        case 2:
          y(j, t) =  std::exp(signal(j)) / (1.0 +  std::exp(signal(j)));
          break;
        case 3:
          y(j, t) =  std::exp(signal(j));
          break;
        case 4:
          y(j, t) =  std::exp(signal(j));
          break;
        case 5:
          y(j, t) =  signal(j);
        }
        
        if (predict_type == 1) {
          
          switch(distribution(j)) {
          case 1: {
          std::poisson_distribution<> poisson(u(j,t) * y(j, t));
          if ((u(j,t) * y(j, t)) < poisson.max()) {
            y(j, t) = poisson(engine);
          } else {
            y(j, t) = std::numeric_limits<double>::quiet_NaN();
          }
        } 
            break;
          case 2: {
            std::binomial_distribution<> binomial(u(j,t), y(j, t));
            y(j, t) = binomial(engine);
          }
            break;
          case 3: {
            double prob = phi(j) / (phi(j) + u(j,t) * y(j, t));
            std::gamma_distribution<> gamma(phi(j), (1 - prob) / prob);
            std::poisson_distribution<> poisson(gamma(engine));
            y(j, t) = poisson(engine);
          }
            break;
          case 4: {
            std::gamma_distribution<> gamma(phi(j), u(j,t) * y(j, t) / phi(j));
            y(j, t) = gamma(engine);
          }
            break;
          }
        }
      }
    }
    samples.slice(i) = y;
  }
  return samples;
}
