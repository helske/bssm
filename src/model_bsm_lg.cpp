// Gaussian structural time series model

#include "model_bsm_lg.h"

// Construct bsm model from Rcpp::List
bsm_lg::bsm_lg(const Rcpp::List model, const unsigned int seed) :
  ssm_ulg(model, seed),  
  prior_distributions(Rcpp::as<arma::uvec>(model["prior_distributions"])), 
  prior_parameters(Rcpp::as<arma::mat>(model["prior_parameters"])),
  slope(Rcpp::as<bool>(model["slope"])),
  seasonal(Rcpp::as<bool>(model["seasonal"])),
  fixed(Rcpp::as<arma::uvec>(model["fixed"])),
  y_est(fixed(0) == 0),
  level_est(fixed(1) == 0),
  slope_est(slope && fixed(2) == 0),
  seasonal_est(seasonal && fixed(3) == 0)
{
  
}

// update the model given theta
// standard deviation parameters sigma are sampled in a transformed space
// with theta = log(sigma) <=> sigma = exp(theta)
void bsm_lg::update_model(const arma::vec& new_theta) {
  
  if (arma::accu(fixed) < 4) {
    if (y_est) {
      H(0) = std::exp(new_theta(0));
      HH(0) = std::pow(H(0), 2.0);
    }
    // sd_level
    if (level_est) {
      R(0, 0, 0) = std::exp(new_theta(y_est));
    }
    // sd_slope
    if (slope_est) {
      R(1, 1, 0) =  std::exp(new_theta(y_est + level_est));
    }
    // sd_seasonal
    if (seasonal_est) {
      R(1 + slope, 1 + slope, 0) =
        std::exp(new_theta(y_est + level_est + slope_est));
    }
    compute_RR();
  }
  if(xreg.n_cols > 0) {
    beta = new_theta.subvec(new_theta.n_elem - xreg.n_cols, new_theta.n_elem - 1);
    compute_xbeta();
  }
  theta = new_theta;
}
void bsm_lg::update_model(const arma::vec& new_theta, const Rcpp::Function update_fn) {
  
  if (arma::accu(fixed) < 4) {
    if (y_est) {
      H(0) = std::exp(new_theta(0));
      HH(0) = std::pow(H(0), 2.0);
    }
    // sd_level
    if (level_est) {
      R(0, 0, 0) = std::exp(new_theta(y_est));
    }
    // sd_slope
    if (slope_est) {
      R(1, 1, 0) =  std::exp(new_theta(y_est + level_est));
    }
    // sd_seasonal
    if (seasonal_est) {
      R(1 + slope, 1 + slope, 0) =
        std::exp(new_theta(y_est + level_est + slope_est));
    }
    compute_RR();
  }
  if(xreg.n_cols > 0) {
    beta = new_theta.subvec(new_theta.n_elem - xreg.n_cols, new_theta.n_elem - 1);
    compute_xbeta();
  }
  theta = new_theta;
}

double bsm_lg::log_prior_pdf(const arma::vec& x, const Rcpp::Function prior_fn) const {
  
  double log_prior = 0.0;
  arma::vec pars = x;
  if (arma::accu(fixed) < 4) {
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

