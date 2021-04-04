#include "model_ssm_gsv.h"
#include "conditional_dist.h"
#include "distr_consts.h"
#include "dmvnorm.h"
#include "sample.h"
#include "rep_mat.h"

// General constructor of ssm_gsv object from Rcpp::List
ssm_gsv::ssm_gsv(const Rcpp::List model, const unsigned int seed, const double zero_tol) 
  : y(Rcpp::as<arma::vec>(model["y"])), 
    Z_mu(Rcpp::as<arma::mat>(model["Z_mu"])),
    T_mu(Rcpp::as<arma::cube>(model["T_mu"])), R_mu(Rcpp::as<arma::cube>(model["R_mu"])),
  a1_mu(Rcpp::as<arma::vec>(model["a1_mu"])), P1_mu(Rcpp::as<arma::mat>(model["P1_mu"])),
  D_mu(Rcpp::as<arma::vec>(model["D_mu"])), C_mu(Rcpp::as<arma::mat>(model["C_mu"])),
  
  Z_sv(Rcpp::as<arma::mat>(model["Z_sv"])),
  T_sv(Rcpp::as<arma::cube>(model["T_sv"])), R_sv(Rcpp::as<arma::cube>(model["R_sv"])),
  a1_sv(Rcpp::as<arma::vec>(model["a1_sv"])), P1_sv(Rcpp::as<arma::mat>(model["P1_sv"])),
  D_sv(Rcpp::as<arma::vec>(model["D_sv"])), C_sv(Rcpp::as<arma::mat>(model["C_sv"])),
  
  n(y.n_elem), 
  m_mu(a1_mu.n_elem), k_mu(R_mu.n_cols), 
  m_sv(a1_sv.n_elem), k_sv(R_sv.n_cols), 
  m(m_mu + m_sv), k(k_mu + k_sv),
  Ztv_mu(Z_mu.n_cols > 1), 
  Ttv_mu(T_mu.n_slices > 1), 
  Rtv_mu(R_mu.n_slices > 1), 
  Dtv_mu(D_mu.n_elem > 1), 
  Ctv_mu(C_mu.n_cols > 1), 
  Ztv_sv(Z_sv.n_cols > 1), 
  Ttv_sv(T_sv.n_slices > 1), 
  Rtv_sv(R_sv.n_slices > 1), 
  Dtv_sv(D_sv.n_elem > 1), 
  Ctv_sv( C_sv.n_cols > 1),
  
  theta(Rcpp::as<arma::vec>(model["theta"])),
  max_iter(model["max_iter"]), conv_tol(model["conv_tol"]), 
  local_approx(model["local_approx"]),
  initial_mode((Rcpp::as<arma::mat>(model["initial_mode"])).t()),
  mode_estimate(initial_mode),
  approx_state(-1),
  approx_loglik(0.0), scales(arma::vec(n, arma::fill::zeros)),
  engine(seed), zero_tol(zero_tol),
  RR_mu(arma::cube(m_mu, m_mu, Rtv_mu * (n - 1) + 1)),
  RR_sv(arma::cube(m_sv, m_sv, Rtv_sv * (n - 1) + 1)),
  approx_model(arma::mat(2, n, arma::fill::zeros),
    arma::cube(2, m, (Ztv_mu || Ztv_sv) * (n - 1) + 1, arma::fill::zeros), 
    arma::cube(2, 2, n, arma::fill::zeros), 
    arma::cube(m, m, (Ttv_mu || Ttv_sv) * (n - 1) + 1, arma::fill::zeros),
    arma::cube(m, k, (Rtv_mu || Rtv_sv) * (n - 1) + 1, arma::fill::zeros),
    arma::vec(m, arma::fill::zeros), arma::mat(m, m, arma::fill::zeros),
    arma::mat(m, (Dtv_mu || Dtv_sv) * (n - 1) + 1, arma::fill::zeros), 
    arma::mat(2, (Ctv_mu || Ctv_sv) * (n - 1) + 1, arma::fill::zeros),
    theta, seed + 1) {
    compute_RR();
  }

void ssm_gsv::update_model(const arma::vec& new_theta, const Rcpp::Function update_fn) {
  
  Rcpp::List model_list =
    update_fn(Rcpp::NumericVector(new_theta.begin(), new_theta.end()));
  
  bool update_R = false;
  if (model_list.containsElementNamed("Z_mu")) {
    Z_mu = Rcpp::as<arma::mat>(model_list["Z_mu"]);
  }
  if (model_list.containsElementNamed("T_mu")) {
    T_mu = Rcpp::as<arma::cube>(model_list["T_mu"]);
  }
  if (model_list.containsElementNamed("R_mu")) {
    R_mu = Rcpp::as<arma::cube>(model_list["R_mu"]);
    update_R = true;
  }
  if (model_list.containsElementNamed("a1_mu")) {
    a1_mu = Rcpp::as<arma::vec>(model_list["a1_mu"]);
  }
  if (model_list.containsElementNamed("P1_mu")) {
    P1_mu = Rcpp::as<arma::mat>(model_list["P1_mu"]);
  }
  if (model_list.containsElementNamed("D_mu")) {
    D_mu = Rcpp::as<arma::vec>(model_list["D_mu"]);
  }
  if (model_list.containsElementNamed("C_mu")) {
    C_mu = Rcpp::as<arma::mat>(model_list["C_mu"]);
  }
  
  if (model_list.containsElementNamed("Z_sv")) {
    Z_sv = Rcpp::as<arma::mat>(model_list["Z_sv"]);
  }
  if (model_list.containsElementNamed("T_sv")) {
    T_sv = Rcpp::as<arma::cube>(model_list["T_sv"]);
  }
  if (model_list.containsElementNamed("R_sv")) {
    R_sv = Rcpp::as<arma::cube>(model_list["R_sv"]);
    update_R = true;
  }
  if (model_list.containsElementNamed("a1_sv")) {
    a1_sv = Rcpp::as<arma::vec>(model_list["a1_sv"]);
  }
  if (model_list.containsElementNamed("P1_sv")) {
    P1_sv = Rcpp::as<arma::mat>(model_list["P1_sv"]);
  }
  if (model_list.containsElementNamed("D_sv")) {
    D_sv = Rcpp::as<arma::vec>(model_list["D_sv"]);
  }
  if (model_list.containsElementNamed("C_sv")) {
    C_sv = Rcpp::as<arma::mat>(model_list["C_sv"]);
  }
  
  if(update_R) compute_RR();
  theta = new_theta;
  // approximation does not match theta anymore (keep as -1 if so)
  if (approx_state > 0) approx_state = 0;
}

double ssm_gsv::log_prior_pdf(const arma::vec& x, const Rcpp::Function prior_fn) const {
  return Rcpp::as<double>(prior_fn(Rcpp::NumericVector(x.begin(), x.end())));
}

void ssm_gsv::joint_model() {
  //update model, probably not the most efficient way
  for(unsigned int t = 0; t < approx_model.Ztv * (n - 1) + 1; t++) {
    approx_model.Z.slice(t).submat(0, 0, 0, m_mu - 1) = Z_mu.col(Ztv_mu * t);
    approx_model.Z.slice(t).submat(1, m_mu, 1, m - 1) = Z_sv.col(Ztv_sv * t);
  }
  
  for(unsigned int t = 0; t < approx_model.Ttv * (n - 1) + 1; t++) {
    approx_model.T.slice(t).submat(0, 0, m_mu - 1, m_mu - 1) = T_mu.slice(Ttv_mu * t);
    approx_model.T.slice(t).submat(m_mu, m_mu, m - 1, m - 1) = T_sv.slice(Ttv_sv * t);
  }
  
  for(unsigned int t = 0; t < approx_model.Rtv * (n - 1) + 1; t++) {
    approx_model.R.slice(t).submat(0, 0, k_mu - 1, m_mu - 1) = R_mu.slice(Rtv_mu * t);
    approx_model.R.slice(t).submat(k_mu, m_mu, k - 1, m - 1) = R_sv.slice(Rtv_sv * t);
    
    approx_model.RR.slice(t).submat(0, 0, m_mu - 1, m_mu - 1) = RR_mu.slice(Rtv_mu * t);
    approx_model.RR.slice(t).submat(m_mu, m_mu, m - 1, m - 1) = RR_sv.slice(Rtv_sv * t);
  }
  
  for(unsigned int t = 0; t < approx_model.Ctv * (n - 1) + 1; t++) {
    approx_model.C.col(t) = arma::join_cols(C_mu.col(Ctv_mu * t), C_sv.col(Ctv_sv * t));
  }
  
  for(unsigned int t = 0; t < approx_model.Dtv * (n - 1) + 1; t++) {
    approx_model.D.col(t) = arma::join_cols(D_mu.col(Dtv_mu * t), D_sv.col(Dtv_sv * t));
  }
  
  approx_model.a1 = arma::join_cols(a1_mu, a1_sv);
  
  approx_model.P1.submat(0, 0, m_mu - 1, m_mu - 1) = P1_mu;
  approx_model.P1.submat(m_mu, m_mu, m - 1, m_mu  + m_sv - 1) = P1_sv;
  
}
// update the approximating Gaussian model
// Note that the convergence is assessed only
// by checking the changes in mode, not the actual function values
unsigned int ssm_gsv::approximate() {
  
  unsigned int iters_used = 0;
  // check if there is a need to update the approximation
  if (approx_state < 1) {
    
    joint_model();
    
    // don't update y and H if using global approximation and we have updated them already
    if(!local_approx & (approx_state == 0)) {
      arma::mat alpha = approx_model.fast_smoother();
      for (unsigned int t = 0; t < n; t++) {
        mode_estimate.col(t) = approx_model.D.col(approx_model.Dtv * t) + 
          approx_model.Z.slice(approx_model.Ztv * t) * alpha.col(t);
      }
    } else {
      
      unsigned int i = 0;
      double diff = conv_tol + 1;
      while(i < max_iter && diff > conv_tol) {
        i++;
        //Construct y and H for the Gaussian model
        laplace_iter(mode_estimate);
        // compute new guess of mode
        arma::mat mode_estimate_new(2, n);
        arma::mat alpha = approx_model.fast_smoother();
        for (unsigned int t = 0; t < n; t++) {
          mode_estimate_new.col(t) = 
            approx_model.D.col(approx_model.Dtv * t) + 
            approx_model.Z.slice(approx_model.Ztv * t) * alpha.col(t);
        }
        diff = arma::accu(arma::square(mode_estimate_new - mode_estimate)) / (n * p);
        mode_estimate = mode_estimate_new;
      }
      iters_used = i;
    }
    approx_state = 1; //approx matches theta, approx_loglik does not match
  }
  
  return iters_used;
}
// construct approximating model from fixed mode estimate, no iterations
// used in IS-correction
void ssm_gsv::approximate_for_is(const arma::mat& mode_estimate_) {
  
  joint_model();
  
  //Construct y and H for the Gaussian model
  mode_estimate = mode_estimate_;
  laplace_iter(mode_estimate);
  update_scales();
  approx_loglik = 0.0;
  approx_state = 2;
}

// method = 1 psi-APF, these are not implemented (yet): 2 = BSF, 3 = SPDK, 4 = IEKF (not applicable)
arma::vec ssm_gsv::log_likelihood(
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
        Rcpp::stop("Not yet");
        // //SPDK
        // alpha = approx_model.simulate_states(nsim, true);
        // arma::vec w(nsim, arma::fill::zeros);
        // for (unsigned int t = 0; t < n; t++) {
        //   w += log_weights(t, alpha);
        // }
        // w -= arma::accu(scales);
        // double maxw = w.max();
        // w = arma::exp(w - maxw);
        // weights.col(n) = w; // we need these for sampling etc
        // loglik(0) = approx_loglik + std::log(arma::mean(w)) + maxw;
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
void ssm_gsv::update_scales() {
  
  scales.zeros();
  
  for(unsigned int t = 0; t < n; t++) {
    if (arma::is_finite(y(t))) {
      scales(t) = -0.5 * (mode_estimate(1, t) + 
        std::pow(y(t) - mode_estimate(0, t), 2.0) * std::exp(-mode_estimate(1,t))) +
        (0.5 * std::pow((approx_model.y(0, t) - mode_estimate(0, t)) / approx_model.H(0, 0, t), 2.0) +
        0.5 * std::pow((approx_model.y(1, t) - mode_estimate(1, t)) / approx_model.H(1, 1, t), 2.0));
    }
  }
  
}

// given the current guess of mode, compute new values of y and H of
// approximate model
// signal_t is (C^mu_t + Z^mu * mu_t, C^eta_t + Z^eta_t* eta_t)
void ssm_gsv::laplace_iter(const arma::mat& signal) {
  
  arma::rowvec tmp = y.t() - signal.row(0);
  // avoid dividing by zero
  tmp(arma::find(arma::abs(tmp) < 1e-4)).fill(1e-4);
  arma::rowvec tmp2 = arma::exp(signal.row(1));

  approx_model.HH.tube(0, 0) = tmp2;
  approx_model.HH.tube(0, 1).zeros();
  approx_model.HH.tube(1, 0).zeros();
  approx_model.HH.tube(1, 1).fill(2);
  approx_model.H.tube(0,0) = arma::sqrt(tmp2);
  approx_model.H.tube(0, 1).zeros();
  approx_model.H.tube(1, 0).zeros();
  approx_model.H.tube(1, 1).fill(sqrt(2));
  
  approx_model.y.row(0) = y.t();
  approx_model.y.row(1) = signal.row(1) + arma::square(tmp) / tmp2  - 1;
}

double ssm_gsv::compute_const_term() const {
  
  arma::uvec y_ind(find_finite(y));
  double const_term = 0.5 * y_ind.n_elem * std::log(2 * M_PI) +
  arma::accu(log(arma::vec(approx_model.H.tube(0, 0)).elem(y_ind))) + 
  arma::accu(log(arma::vec(approx_model.H.tube(1, 1)).elem(y_ind)));
  return const_term;
}

arma::vec ssm_gsv::importance_weights(const arma::cube& alpha) const {
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

arma::vec ssm_gsv::log_weights(
    const unsigned int t, 
    const arma::cube& alpha) const {
  
  arma::vec weights(alpha.n_slices, arma::fill::zeros);
  
  if (arma::is_finite(y(t))) {
      for (unsigned int i = 0; i < alpha.n_slices; i++) {
        // use joint components from approximate model
        arma::vec simsignal = approx_model.D.col(t * approx_model.Dtv) + 
          approx_model.Z.slice(t * approx_model.Ztv) * alpha.slice(i).col(t);
        
        weights(i) = -0.5 * (simsignal(1) + std::pow(y(t) - simsignal(0), 2.0) * std::exp(-simsignal(1))) +
          (0.5 * std::pow((approx_model.y(0, t) - simsignal(0)) / approx_model.H(0, 0, t), 2.0) +
          0.5 * std::pow((approx_model.y(1, t) - simsignal(1)) / approx_model.H(1, 1, t), 2.0));
      }
  }
  return weights;
}


// Logarithms of _unnormalized_ densities g(y_t | alpha_t)
/*
 * t:             Time point where the densities are computed
 * alpha:         Simulated particles
 */
arma::vec ssm_gsv::log_obs_density(const unsigned int t, 
  const arma::cube& alpha) const {
  
  arma::vec weights(alpha.n_slices, arma::fill::zeros);
  
  if (arma::is_finite(y(t))) {
    for (unsigned int i = 0; i < alpha.n_slices; i++) {
      // use joint components from approximate model
      arma::vec simsignal = approx_model.D.col(t * approx_model.Dtv) + 
        approx_model.Z.slice(t * approx_model.Ztv) * alpha.slice(i).col(t);
      
      weights(i) = -0.5 * (simsignal(1) + std::pow(y(t) - simsignal(0), 2.0) * std::exp(-simsignal(1)));
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

double ssm_gsv::psi_filter(const unsigned int nsim, arma::cube& alpha, 
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


double ssm_gsv::bsf_filter(const unsigned int nsim, arma::cube& alpha,
  arma::mat& weights, arma::umat& indices) {
  
  // use the joint construction from the approximate model
  // for the state equation
  joint_model(); 
  arma::uvec nonzero = arma::find(approx_model.P1.diag() > 0);
  arma::mat L_P1(m, m, arma::fill::zeros);
  if (nonzero.n_elem > 0) {
    L_P1.submat(nonzero, nonzero) =
      arma::chol(approx_model.P1.submat(nonzero, nonzero), "lower");
  }
  std::normal_distribution<> normal(0.0, 1.0);
  for (unsigned int i = 0; i < nsim; i++) {
    arma::vec um(m);
    for(unsigned int j = 0; j < m; j++) {
      um(j) = normal(engine);
    }
    alpha.slice(i).col(0) = approx_model.a1 + L_P1 * um;
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
      alpha.slice(i).col(t + 1) = 
        approx_model.C.col(t * approx_model.Ctv) +
        approx_model.T.slice(t * approx_model.Ttv) * alphatmp.col(i) + 
        approx_model.R.slice(t * approx_model.Rtv) * uk;
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
  loglik += -0.5 * arma::uvec(arma::find_finite(y)).n_elem * std::log(2 * M_PI);
  return loglik;
}
