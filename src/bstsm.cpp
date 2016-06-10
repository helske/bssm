#include "bstsm.h"

//general constructor
bstsm::bstsm(arma::vec y, arma::mat Z, arma::vec H, arma::cube T,
  arma::cube R, arma::vec a1, arma::mat P1, bool slope, bool seasonal,
  arma::uvec fixed, arma::mat xreg, arma::vec beta, unsigned int seed, bool log_space) :
  guvssm(y, Z, H, T, R, a1, P1, xreg, beta, seed), slope(slope), seasonal(seasonal),
  fixed(fixed), level_est(fixed(0) == 0), slope_est(slope && fixed(1) == 0),
  seasonal_est(seasonal && fixed(2) == 0), log_space(log_space) {
}
// without log_space
bstsm::bstsm(arma::vec y, arma::mat Z, arma::vec H, arma::cube T,
  arma::cube R, arma::vec a1, arma::mat P1, bool slope, bool seasonal,
  arma::uvec fixed, arma::mat xreg, arma::vec beta, unsigned int seed) :
  guvssm(y, Z, H, T, R, a1, P1, xreg, beta, seed), slope(slope), seasonal(seasonal),
  fixed(fixed), level_est(fixed(0) == 0), slope_est(slope && fixed(1) == 0),
  seasonal_est(seasonal && fixed(2) == 0), log_space(false) {
}
double bstsm::proposal(const arma::vec& theta, const arma::vec& theta_prop) {

  double q = 0.0;

  if(log_space) {
    q += theta_prop(0) - theta(0);
    if (sum(fixed) < 3) {
      if (level_est) {
        q += theta_prop(1) - theta(1);
      }
      if (slope_est) {
        q += theta_prop(1 + level_est) - theta(1 + level_est);
      }
      if (seasonal_est) {
        q += theta_prop(1 + level_est + slope_est) - theta(1 + level_est + slope_est);
      }
    }
  }
  return q;
}

void bstsm::update_model(arma::vec theta) {

  if (log_space) {
    theta.subvec(0, theta.n_elem - xreg.n_cols - 1) =
      exp(theta.subvec(0, theta.n_elem - xreg.n_cols - 1));
  }
  H(0) = theta(0);
  HH(0) = pow(theta(0), 2);

  if (sum(fixed) < 3) {
    // sd_level
    if (level_est) {
      R(0, 0, 0) = theta(1);
    }
    // sd_slope
    if (slope_est) {
      R(1, level_est, 0) = theta(1 + level_est);
    }
    // sd_seasonal
    if (seasonal_est) {
      R(1 + slope, level_est + slope_est, 0) =
        theta(1 + level_est + slope_est);
    }
    compute_RR();
  }
  if(xreg.n_cols > 0) {
    beta = theta.subvec(theta.n_elem - xreg.n_cols, theta.n_elem - 1);
    compute_xbeta();
  }

}

arma::vec bstsm::get_theta(void) {

  unsigned int npar = 1 + level_est + slope_est + seasonal_est + xreg.n_cols;

  arma::vec theta(npar);

  theta(0) = H(0);

  if (sum(fixed) < 3) {
    // sd_level
    if (level_est) {
      theta(1) = R(0, 0, 0);
    }
    // sd_slope
    if (slope_est) {
      theta(1 + level_est) = R(1, level_est, 0);
    }
    // sd_seasonal
    if (seasonal_est) {
      theta(1 + level_est + slope_est) =
        R(1 + slope, level_est + slope_est, 0);
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

double bstsm::log_likelihood(void) {

  double logLik = 0;
  arma::vec at = a1;
  arma::mat Pt = P1;

  for (unsigned int t = 0; t < n; t++) {
    // update
    logLik += uv_filter(y(t), Z.col(0), HH(0),
      xbeta(t), T.slice(0), RR.slice(0), at, Pt);
  }
  return logLik;
}


double bstsm::filter(arma::mat& at, arma::mat& att, arma::cube& Pt,
  arma::cube& Ptt) {

  double logLik = 0;

  at.col(0) = a1;
  Pt.slice(0) = P1;

  for (unsigned int t = 0; t < n; t++) {
    // update
    logLik += uv_filter_update(y(t), Z.col(0), HH(0),
      xbeta(t), at.col(t), Pt.slice(t), att.col(t), Ptt.slice(t));
    // prediction
    uv_filter_predict(T.slice(0), RR.slice(0), att.col(t),
      Ptt.slice(t), at.col(t + 1),  Pt.slice(t + 1));
  }
  return logLik;

}


// [[Rcpp::plugins(openmp)]]
arma::cube sample_states(bstsm mod, const arma::mat& theta,
  unsigned int nsim_states, unsigned int n_threads, arma::uvec seeds) {

  unsigned n_iter = theta.n_cols;
  arma::cube alpha_store(mod.m, mod.n, nsim_states * n_iter);

#pragma omp parallel num_threads(n_threads) default(none) shared(n_iter, \
  nsim_states, theta, alpha_store, seeds) firstprivate(mod)
  {
    if (seeds.n_elem == 1) {
      mod.engine = std::mt19937(seeds(0));
    } else {
      mod.engine = std::mt19937(seeds(omp_get_thread_num()));
    }

#pragma omp for schedule(static)
    for (int i = 0; i < n_iter; i++) {

      arma::vec theta_i = theta.col(i);
      mod.update_model(theta_i);

      alpha_store.slices(i * nsim_states, (i + 1) * nsim_states - 1) = mod.sim_smoother(nsim_states);

    }
  }
  return alpha_store;
}

