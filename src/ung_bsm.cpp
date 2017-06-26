#include "ung_bsm.h"

// from Rcpp::List
ung_bsm::ung_bsm(const Rcpp::List& model, const unsigned int seed) :
  ung_ssm(model, seed), slope(Rcpp::as<bool>(model["slope"])),
  seasonal(Rcpp::as<bool>(model["seasonal"])),
  noise(Rcpp::as<bool>(model["noise"])),
  fixed(Rcpp::as<arma::uvec>(model["fixed"])), level_est(fixed(0) == 0),
  slope_est(slope && fixed(1) == 0), seasonal_est(seasonal && fixed(2) == 0) {
}

void ung_bsm::set_theta(const arma::vec& theta) {

  if (arma::accu(fixed) < 3 || noise || phi_est) {

    // sd_level
    if (level_est) {
      R(0, 0, 0) = theta(0);
    }
    // sd_slope
    if (slope_est) {
      R(1, 1, 0) = theta(level_est);
    }
    // sd_seasonal
    if (seasonal_est) {
      R(1 + slope, 1 + slope, 0) =
        theta(level_est + slope_est);
    }
    if(noise) {
      R(m - 1, 1 + slope + seasonal, 0) =
        theta(level_est + slope_est + seasonal_est);
      P1(m - 1, m - 1) = std::pow(theta(level_est + slope_est + seasonal_est), 2.0);
    }
    compute_RR();
  }
  if(phi_est) {
    phi = theta(level_est + slope_est + seasonal_est + noise);
  }

  if(xreg.n_cols > 0) {
    beta = theta.subvec(theta.n_elem - xreg.n_cols, theta.n_elem - 1);
    compute_xbeta();
  }
}

arma::vec ung_bsm::get_theta(void) const {

  unsigned int npar = level_est + slope_est + seasonal_est + noise +
    xreg.n_cols + phi_est;

  arma::vec theta(npar);

  if (arma::accu(fixed) < 3 || noise) {
    // sd_level
    if (level_est) {
      theta(0) = R(0, 0, 0);
    }
    // sd_slope
    if (slope_est) {
      theta(level_est) = R(1, 1, 0);
    }
    // sd_seasonal
    if (seasonal_est) {
      theta(level_est + slope_est) =
        R(1 + slope, 1 + slope, 0);
    }
    if (noise) {
      theta(level_est + slope_est + seasonal_est) =
        R(m - 1, 1 + slope + seasonal, 0);
    }
  }

  if(phi_est) {
    theta(level_est + slope_est + seasonal_est + noise) = phi;
  }

  if(xreg.n_cols > 0) {
    theta.subvec(theta.n_elem - xreg.n_cols, theta.n_elem - 1) = beta;
  }
  return theta;
}
