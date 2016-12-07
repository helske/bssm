// Gaussian structural time series model

#include "bsm.h"

// Construct bsm model from Rcpp::List
bsm::bsm(const Rcpp::List& model, unsigned int seed, bool log_space) :
  gssm(model, seed), 
  slope(Rcpp::as<bool>(model["slope"])),
  seasonal(Rcpp::as<bool>(model["seasonal"])),
  fixed(Rcpp::as<arma::uvec>(model["fixed"])), 
  y_est(fixed(0) == 0), 
  level_est(fixed(1) == 0),
  slope_est(slope && fixed(2) == 0), 
  seasonal_est(seasonal && fixed(3) == 0),
  log_space(log_space) {
  
}

// for proposal on log-space, compute the (non-symmetric) ratio of proposals
double bsm::proposal(const arma::vec& theta, const arma::vec& theta_prop) {
  
  double q = 0.0;
  
  if(log_space) {
    if (sum(fixed) < 4) {
      if (y_est) {
        q += theta_prop(0) - theta(0);
      }
      if (level_est) {
        q += theta_prop(y_est) - theta(y_est);
      }
      if (slope_est) {
        q += theta_prop(y_est + level_est) - theta(y_est + level_est);
      }
      if (seasonal_est) {
        q += theta_prop(y_est + level_est + slope_est) - theta(y_est + level_est + slope_est);
      }
    }
  }
  return q;
}

// update the model given theta
void bsm::update_model(arma::vec theta) {
  
  if (log_space) {
    theta.subvec(0, theta.n_elem - xreg.n_cols - 1) =
      exp(theta.subvec(0, theta.n_elem - xreg.n_cols - 1));
  }
  
  if (sum(fixed) < 4) {
    if (y_est) {
      H(0) = theta(0);
      HH(0) = pow(theta(0), 2);
    }
    // sd_level
    if (level_est) {
      R(0, 0, 0) = theta(y_est);
    }
    // sd_slope
    if (slope_est) {
      R(1, 1, 0) = theta(y_est + level_est);
    }
    // sd_seasonal
    if (seasonal_est) {
      R(1 + slope, 1 + slope, 0) =
        theta(y_est + level_est + slope_est);
    }
    compute_RR();
  }
  if(xreg.n_cols > 0) {
    beta = theta.subvec(theta.n_elem - xreg.n_cols, theta.n_elem - 1);
    compute_xbeta();
  }
  
}

// extract the current value of theta from the model
arma::vec bsm::get_theta() {
  
  unsigned int npar = 1 + level_est + slope_est + seasonal_est + xreg.n_cols;
  
  arma::vec theta(npar);
  
  if (sum(fixed) < 4) {
    if(y_est) {
      theta(0) = H(0);
    }
    // sd_level
    if (level_est) {
      theta(y_est) = R(0, 0, 0);
    }
    // sd_slope
    if (slope_est) {
      theta(y_est + level_est) = R(1, 1, 0);
    }
    // sd_seasonal
    if (seasonal_est) {
      theta(y_est + level_est + slope_est) =
        R(1 + slope, 1 + slope, 0);
    }
    if (log_space) {
      theta.subvec(0, theta.n_elem - xreg.n_cols - 1) =
        log(theta.subvec(0, theta.n_elem - xreg.n_cols - 1));
    }
  }
  if(xreg.n_cols > 0) {
    theta.subvec(theta.n_elem - xreg.n_cols, theta.n_elem - 1) = beta;
  }
  return theta;
}
