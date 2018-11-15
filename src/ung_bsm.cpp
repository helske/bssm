#include "ung_bsm.h"

// from Rcpp::List
ung_bsm::ung_bsm(const Rcpp::List& model, const unsigned int seed) :
  ung_ssm(model, seed), slope(Rcpp::as<bool>(model["slope"])),
  seasonal(Rcpp::as<bool>(model["seasonal"])),
  noise(Rcpp::as<bool>(model["noise"])),
  fixed(Rcpp::as<arma::uvec>(model["fixed"])), level_est(fixed(0) == 0),
  slope_est(slope && fixed(1) == 0), seasonal_est(seasonal && fixed(2) == 0) {
}

void ung_bsm::update_model(const arma::vec& new_theta) {

  if (arma::accu(fixed) < 3 || noise || phi_est) {

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
}

double ung_bsm::log_prior_pdf(const arma::vec& x) const {
  
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
    }
  }
  return log_prior;
}

double ung_bsm::log_proposal_ratio(const arma::vec& new_theta, const arma::vec& old_theta) const {
  
  return arma::accu(new_theta.subvec(0, new_theta.n_elem - xreg.n_cols - 1)) -
    arma::accu(old_theta.subvec(0, old_theta.n_elem - xreg.n_cols - 1));
}
