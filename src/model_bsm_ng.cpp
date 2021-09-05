#include "model_bsm_ng.h"

// from Rcpp::List
bsm_ng::bsm_ng(const Rcpp::List model, const unsigned int seed) :
  ssm_ung(model, seed), 
  prior_distributions(Rcpp::as<arma::uvec>(model["prior_distributions"])), 
  prior_parameters(Rcpp::as<arma::mat>(model["prior_parameters"])),
  slope(Rcpp::as<bool>(model["slope"])),
  seasonal(Rcpp::as<bool>(model["seasonal"])),
  noise(Rcpp::as<bool>(model["noise"])),
  fixed(Rcpp::as<arma::uvec>(model["fixed"])), level_est(fixed(0) == 0),
  slope_est(slope && fixed(1) == 0), seasonal_est(seasonal && fixed(2) == 0),
  phi_est(Rcpp::as<bool>(model["phi_est"])) {
}

// used in parallel regions, does not depend on R
void bsm_ng::update_model(const arma::vec& new_theta) {
  
  if (arma::accu(fixed) < 3 || noise) {
    
    // sd_level
    if (level_est) {
      R(0, 0, 0) = std::exp(new_theta(0));
    }
    // sd_slope
    if (slope_est) {
      R(1, 1, 0) = std::exp(new_theta(level_est));
    }
    // sd_seasonal
    if (seasonal_est) {
      R(1 + slope, 1 + slope, 0) =
        std::exp(new_theta(level_est + slope_est));
    }
    if(noise) {
      R(m - 1, 1 + slope + seasonal, 0) =
        std::exp(new_theta(level_est + slope_est + seasonal_est));
      P1(m - 1, m - 1) = std::pow(R(m - 1, 1 + slope + seasonal, 0), 2.0);
    }
    compute_RR();
  }
  if(phi_est) {
    phi = std::exp(new_theta(level_est + slope_est + seasonal_est + noise));
  }
  if(xreg.n_cols > 0) {
    beta = new_theta.subvec(new_theta.n_elem - xreg.n_cols, new_theta.n_elem - 1);
    compute_xbeta();
  }
  theta = new_theta;
  // approximation does not match theta anymore (keep as -1 if so)
  if (approx_state > 0) approx_state = 0;
}

// used in mcmc, latter argument is not actually used
void bsm_ng::update_model(const arma::vec& new_theta, const Rcpp::Function update_fn) {

  if (arma::accu(fixed) < 3 || noise) {
    
    // sd_level
    if (level_est) {
      R(0, 0, 0) = std::exp(new_theta(0));
    }
    // sd_slope
    if (slope_est) {
      R(1, 1, 0) = std::exp(new_theta(level_est));
    }
    // sd_seasonal
    if (seasonal_est) {
      R(1 + slope, 1 + slope, 0) =
        std::exp(new_theta(level_est + slope_est));
    }
    if(noise) {
      R(m - 1, 1 + slope + seasonal, 0) =
        std::exp(new_theta(level_est + slope_est + seasonal_est));
      P1(m - 1, m - 1) = std::pow(R(m - 1, 1 + slope + seasonal, 0), 2.0);
    }
    compute_RR();
  }
  if(phi_est) {
    phi = std::exp(new_theta(level_est + slope_est + seasonal_est + noise));
  }
  if(xreg.n_cols > 0) {
    beta = new_theta.subvec(new_theta.n_elem - xreg.n_cols, new_theta.n_elem - 1);
    compute_xbeta();
  }
  theta = new_theta;
  // approximation does not match theta anymore (keep as -1 if so)
  if (approx_state > 0) approx_state = 0;
}

double bsm_ng::log_prior_pdf(const arma::vec& x, const Rcpp::Function prior_fn) const {
  
  double log_prior = 0.0;
  arma::vec pars = x;
  if (arma::accu(fixed) < 3 || noise || phi_est) {
    pars.subvec(0, pars.n_elem - xreg.n_cols - 1) = 
      arma::exp(pars.subvec(0, pars.n_elem - xreg.n_cols - 1));
    // add jacobian
    log_prior += arma::accu(x.subvec(0, x.n_elem - xreg.n_cols - 1));
  }
  
  for(unsigned int i = 0; i < pars.n_elem; i++) {
    switch(prior_distributions(i)) {
    case 0  :
      if (pars(i) < prior_parameters(0, i) || pars(i) > prior_parameters(1, i)) {
        return -std::numeric_limits<double>::infinity(); 
      }
      break;
    case 1  :
      if (pars(i) < 0) {
        return -std::numeric_limits<double>::infinity();
      } else {
        log_prior -= 0.5 * std::pow(pars(i) / prior_parameters(0, i), 2);
      }
      break;
    case 2  :
      log_prior -= 0.5 * std::pow((pars(i) - prior_parameters(0, i)) / prior_parameters(1, i), 2);
      break;
    case 3 : // truncated normal
      if (pars(i) < prior_parameters(2, i) || pars(i) > prior_parameters(3, i)) {
        return -std::numeric_limits<double>::infinity(); 
      } else {
        log_prior -= 0.5 * std::pow((pars(i) - prior_parameters(0, i)) / prior_parameters(1, i), 2);
      }
      break;
    case 4 : // gamma
      if (pars(i) < 0) {
        return -std::numeric_limits<double>::infinity(); 
      } else {
        log_prior += (prior_parameters(0, i) - 1) * log(pars(i)) - prior_parameters(1, i) * pars(i);
      }
      break;
    }
    
  }

  return log_prior;
}

