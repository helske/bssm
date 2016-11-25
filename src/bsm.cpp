#include "bsm.h"
#include "filter.h"

// from Rcpp::List
bsm::bsm(const Rcpp::List& model, unsigned int seed, bool log_space) :
  gssm(model, seed), slope(Rcpp::as<bool>(model["slope"])),
  seasonal(Rcpp::as<bool>(model["seasonal"])),
  fixed(Rcpp::as<arma::uvec>(model["fixed"])), y_est(fixed(0) == 0), level_est(fixed(1) == 0),
  slope_est(slope && fixed(2) == 0), seasonal_est(seasonal && fixed(3) == 0),
  log_space(log_space) {

}

//with log_space
bsm::bsm(arma::vec y, arma::mat Z, arma::vec H, arma::cube T,
  arma::cube R, arma::vec a1, arma::mat P1, bool slope, bool seasonal,
  arma::uvec fixed, arma::mat xreg, arma::vec beta, unsigned int seed, bool log_space) :
  gssm(y, Z, H, T, R, a1, P1, xreg, beta, arma::mat(a1.n_elem, 1, arma::fill::zeros), seed), 
  slope(slope), seasonal(seasonal),
  fixed(fixed), y_est(fixed(0) == 0), level_est(fixed(1) == 0), slope_est(slope && fixed(2) == 0),
  seasonal_est(seasonal && fixed(3) == 0), log_space(log_space) {

}

// without log_space
bsm::bsm(arma::vec y, arma::mat Z, arma::vec H, arma::cube T,
  arma::cube R, arma::vec a1, arma::mat P1, bool slope, bool seasonal,
  arma::uvec fixed, arma::mat xreg, arma::vec beta, unsigned int seed) :
  gssm(y, Z, H, T, R, a1, P1, xreg, beta, arma::mat(a1.n_elem, 1, arma::fill::zeros), seed), 
  slope(slope), seasonal(seasonal),
  fixed(fixed), y_est(fixed(0) == 0), level_est(fixed(1) == 0), slope_est(slope && fixed(2) == 0),
  seasonal_est(seasonal && fixed(3) == 0), log_space(false) {

}

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

arma::vec bsm::get_theta(void) {

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

double bsm::log_likelihood(bool demean) {

  double logLik = 0;
  arma::vec at = a1;
  arma::mat Pt = P1;
  if(demean && xreg.n_cols > 0) {
    for (unsigned int t = 0; t < n; t++) {
      logLik += uv_filter(y(t) - xbeta(t), Z.col(0), HH(0),
        T.slice(0), RR.slice(0), C.col(0), at, Pt, zero_tol);
    }
  } else {
    for (unsigned int t = 0; t < n; t++) {
      logLik += uv_filter(y(t), Z.col(0), HH(0),
        T.slice(0), RR.slice(0), C.col(0), at, Pt, zero_tol);
    }
  }
  return logLik;
}

double bsm::filter(arma::mat& at, arma::mat& att, arma::cube& Pt,
  arma::cube& Ptt, bool demean) {

  double logLik = 0;

  at.col(0) = a1;
  Pt.slice(0) = P1;
  if(demean && xreg.n_cols > 0) {
    for (unsigned int t = 0; t < n; t++) {
      // update
      logLik += uv_filter_update(y(t) - xbeta(t), Z.col(0), HH(0),
        at.col(t), Pt.slice(t), att.col(t), Ptt.slice(t), zero_tol);
      // prediction
      uv_filter_predict(T.slice(0), RR.slice(0), C.col(0), att.col(t),
        Ptt.slice(t), at.col(t + 1),  Pt.slice(t + 1));
    }
  } else {
    for (unsigned int t = 0; t < n; t++) {
      // update
      logLik += uv_filter_update(y(t), Z.col(0), HH(0),
        at.col(t), Pt.slice(t), att.col(t), Ptt.slice(t), zero_tol);
      // prediction
      uv_filter_predict(T.slice(0), RR.slice(0), C.col(0), att.col(t),
        Ptt.slice(t), at.col(t + 1),  Pt.slice(t + 1));
    }
  }
  return logLik;
}
