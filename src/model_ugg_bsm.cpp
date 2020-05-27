// Gaussian structural time series model

#include "model_ugg_bsm.h"

// Construct bsm model from Rcpp::List
ugg_bsm::ugg_bsm(const Rcpp::List& model, const unsigned int seed) :
  ugg_ssm(model, seed),  
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
void ugg_bsm::update_model(const arma::vec& new_theta) {
  
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

double ugg_bsm::log_prior_pdf(const arma::vec& x) const {
  
  double log_prior = 0.0;
  arma::vec pars = x;
  pars.subvec(0, pars.n_elem - xreg.n_cols - 1) = 
    arma::exp(pars.subvec(0, pars.n_elem - xreg.n_cols - 1));
  
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
    }
  }
  // add jacobian
  log_prior += arma::accu(x.subvec(0, x.n_elem - xreg.n_cols - 1));
  return log_prior;
}

