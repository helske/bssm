#include "ung_bsm.h"

// from Rcpp::List
ung_bsm::ung_bsm(const Rcpp::List& model, const unsigned int seed) :
  ung_ssm(model, seed), slope(Rcpp::as<bool>(model["slope"])),
  seasonal(Rcpp::as<bool>(model["seasonal"])),
  noise(Rcpp::as<bool>(model["noise"])),
  fixed(Rcpp::as<arma::uvec>(model["fixed"])), level_est(fixed(0) == 0),
<<<<<<< HEAD:src/ng_bsm.cpp
  slope_est(slope && fixed(1) == 0), seasonal_est(seasonal && fixed(2) == 0),
  log_space(log_space), noise_const(Rcpp::as<arma::vec>(model["noise_const"])) {
}

double ng_bsm::proposal(const arma::vec& theta, const arma::vec& theta_prop) {
  double q = 0.0;
  if(log_space && (sum(fixed) < 3 || noise)) {
    if (level_est) {
      q += theta_prop(0) - theta(0);
    }
    if (slope_est) {
      q += theta_prop(level_est) - theta(level_est);
    }
    if (seasonal_est) {
      q += theta_prop(level_est + slope_est) - theta(level_est + slope_est);
    }
    if (noise) {
      q += theta_prop(level_est + slope_est + seasonal_est) - theta(level_est + slope_est + seasonal_est);
    }
  }
  return q;
}

void ng_bsm::update_model(arma::vec theta) {
  
  if (sum(fixed) < 3 || noise || phi_est) {
    
    if (log_space) {
      theta.subvec(0, theta.n_elem - xreg.n_cols - phi_est - 1) =
        exp(theta.subvec(0, theta.n_elem - xreg.n_cols - phi_est - 1));
    }
    
    arma::mat R1(m, k, arma::fill::zeros);
=======
  slope_est(slope && fixed(1) == 0), seasonal_est(seasonal && fixed(2) == 0) {
}

void ung_bsm::set_theta(const arma::vec& theta) {

  if (sum(fixed) < 3 || noise || phi_est) {

>>>>>>> redesign:src/ung_bsm.cpp
    // sd_level
    if (level_est) {
      R1(0, 0) = theta(0);
    }
    // sd_slope
    if (slope_est) {
      R1(1, 1) = theta(level_est);
    }
    // sd_seasonal
    if (seasonal_est) {
      R1(1 + slope, 1 + slope) =
        theta(level_est + slope_est);
    }
    // noise
    if(noise) {
      R1(m - 1, 1 + slope + seasonal) = theta(level_est + slope_est + seasonal_est);
      P1(m - 1, m - 1) = std::pow(theta(level_est + slope_est + seasonal_est), 2);
    }
    
    if (noise_const.n_elem == 1) {
      R.slice(0) = R1;
    } else {
      P1(m - 1, m - 1) *= std::pow(noise_const(0), 2);
      R.each_slice() = R1;
      R.tube(m - 1, 1 + slope + seasonal) = noise_const.subvec(1, n) * 
        theta(level_est + slope_est + seasonal_est);
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
<<<<<<< HEAD:src/ng_bsm.cpp
  
  
}

arma::vec ng_bsm::get_theta(void) {
  
=======
}

arma::vec ung_bsm::get_theta(void) const {

>>>>>>> redesign:src/ung_bsm.cpp
  unsigned int npar = level_est + slope_est + seasonal_est + noise +
    xreg.n_cols + phi_est;
  
  arma::vec theta(npar);
  
  if (sum(fixed) < 3 || noise) {
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
      if (noise_const.n_elem > 1) {
        theta(level_est + slope_est + seasonal_est) /= noise_const(0);
      }
    }
  }

  if(phi_est) {
    theta(level_est + slope_est + seasonal_est + noise) = phi;
  }

  if(xreg.n_cols > 0) {
    theta.subvec(theta.n_elem - xreg.n_cols, theta.n_elem - 1) = beta;
  }
<<<<<<< HEAD:src/ng_bsm.cpp
  
  
=======
>>>>>>> redesign:src/ung_bsm.cpp
  return theta;
}
