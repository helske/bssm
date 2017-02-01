// Gaussian structural time series model

#include "ugg_bsm.h"

// Construct bsm model from Rcpp::List
ugg_bsm::ugg_bsm(const Rcpp::List& model, const unsigned int seed) :
  ugg_ssm(model, seed),
  slope(Rcpp::as<bool>(model["slope"])),
  seasonal(Rcpp::as<bool>(model["seasonal"])),
  fixed(Rcpp::as<arma::uvec>(model["fixed"])),
  y_est(fixed(0) == 0),
  level_est(fixed(1) == 0),
  slope_est(slope && fixed(2) == 0),
  seasonal_est(seasonal && fixed(3) == 0) {

}

// update the model given theta
void ugg_bsm::set_theta(const arma::vec& theta) {

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
arma::vec ugg_bsm::get_theta() const {

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
  }
  if(xreg.n_cols > 0) {
    theta.subvec(theta.n_elem - xreg.n_cols, theta.n_elem - 1) = beta;
  }
  return theta;
}
