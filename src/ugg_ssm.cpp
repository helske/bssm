#include "ugg_ssm.h"
#include "interval.h"
#include "rep_mat.h"
#include "sample.h"
#include "distr_consts.h"

// General constructor of ugg_ssm object from Rcpp::List
// with parameter indices
ugg_ssm::ugg_ssm(const Rcpp::List& model, const unsigned int seed, 
  const arma::uvec& Z_ind, const arma::uvec& H_ind, 
  const arma::uvec& T_ind, const arma::uvec& R_ind) :
  y(Rcpp::as<arma::vec>(model["y"])), Z(Rcpp::as<arma::mat>(model["Z"])),
  H(Rcpp::as<arma::vec>(model["H"])), T(Rcpp::as<arma::cube>(model["T"])), 
  R(Rcpp::as<arma::cube>(model["R"])), a1(Rcpp::as<arma::vec>(model["a1"])), 
  P1(Rcpp::as<arma::mat>(model["P1"])), xreg(Rcpp::as<arma::mat>(model["xreg"])), 
  beta(Rcpp::as<arma::vec>(model["coefs"])), 
  D(Rcpp::as<arma::vec>(model["obs_intercept"])),
  C(Rcpp::as<arma::mat>(model["state_intercept"])), 
  Ztv(Z.n_cols > 1), Htv(H.n_elem > 1), Ttv(T.n_slices > 1), Rtv(R.n_slices > 1),
  Dtv(D.n_elem > 1), Ctv(C.n_cols > 1), n(y.n_elem), m(a1.n_elem), k(R.n_cols), 
  HH(arma::vec(Htv * (n - 1) + 1)), RR(arma::cube(m, m, Rtv * (n - 1) + 1)), 
  xbeta(arma::vec(n, arma::fill::zeros)), engine(seed), zero_tol(1e-8),
  Z_ind(Z_ind), H_ind(H_ind), T_ind(T_ind), R_ind(R_ind), seed(seed) {
  
  if(xreg.n_cols > 0) {
    compute_xbeta();
  }
  compute_HH();
  compute_RR();
}


// General constructor of ugg_ssm object
// with parameter indices
ugg_ssm::ugg_ssm(const arma::vec& y, const arma::mat& Z, const arma::vec& H, 
  const arma::cube& T, const arma::cube& R, const arma::vec& a1, 
  const arma::mat& P1, const arma::mat& xreg, const arma::vec& beta, 
  const arma::vec& D, const arma::mat& C, const unsigned int seed, const arma::uvec& Z_ind, 
  const arma::uvec& H_ind, const arma::uvec& T_ind, const arma::uvec& R_ind) :
  y(y), Z(Z), H(H), T(T), R(R), a1(a1), P1(P1), xreg(xreg), beta(beta), D(D), C(C),
  Ztv(Z.n_cols > 1), Htv(H.n_elem > 1), Ttv(T.n_slices > 1), Rtv(R.n_slices > 1),
  Dtv(D.n_elem > 1), Ctv(C.n_cols > 1), n(y.n_elem), m(a1.n_elem), k(R.n_cols), 
  HH(arma::vec(Htv * (n - 1) + 1)), RR(arma::cube(m, m, Rtv * (n - 1) + 1)), 
  xbeta(arma::vec(n, arma::fill::zeros)), engine(seed), zero_tol(1e-8),
  Z_ind(Z_ind), H_ind(H_ind), T_ind(T_ind), R_ind(R_ind), seed(seed) {
  
  if(xreg.n_cols > 0) {
    compute_xbeta();
  }
  compute_HH();
  compute_RR();
}

void ugg_ssm::set_theta(const arma::vec& theta) {
  
  if (Z_ind.n_elem > 0) {
    Z.elem(Z_ind) = theta.subvec(0, Z_ind.n_elem - 1);
  }
  if (H_ind.n_elem > 0) {
    H.elem(H_ind) = theta.subvec(Z_ind.n_elem, Z_ind.n_elem + H_ind.n_elem - 1);
  }
  if (T_ind.n_elem > 0) {
    T.elem(T_ind) = theta.subvec(Z_ind.n_elem + H_ind.n_elem,
      Z_ind.n_elem + H_ind.n_elem + T_ind.n_elem - 1);
  }
  if (R_ind.n_elem > 0) {
    R.elem(R_ind) = theta.subvec(Z_ind.n_elem + H_ind.n_elem + T_ind.n_elem,
      Z_ind.n_elem + H_ind.n_elem + T_ind.n_elem + R_ind.n_elem - 1);
  }
  
  if (H_ind.n_elem  > 0) {
    compute_HH();
  }
  if (R_ind.n_elem  > 0) {
    compute_RR();
  }
  if(xreg.n_cols > 0) {
    beta = theta.subvec(theta.n_elem - xreg.n_cols, theta.n_elem - 1);
    compute_xbeta();
  }
}

arma::vec ugg_ssm::get_theta() const {
  
  arma::vec theta(Z_ind.n_elem + H_ind.n_elem + T_ind.n_elem + R_ind.n_elem);
  
  if (Z_ind.n_elem > 0) {
    theta.subvec(0, Z_ind.n_elem - 1) = Z.elem(Z_ind);
  }
  if (H_ind.n_elem > 0) {
    theta.subvec(Z_ind.n_elem, Z_ind.n_elem + H_ind.n_elem - 1) = H.elem(H_ind);
  }
  if (T_ind.n_elem > 0) {
    theta.subvec(Z_ind.n_elem + H_ind.n_elem,
      Z_ind.n_elem + H_ind.n_elem + T_ind.n_elem - 1) = T.elem(T_ind);
  }
  if (R_ind.n_elem > 0) {
    theta.subvec(Z_ind.n_elem + H_ind.n_elem + T_ind.n_elem,
      Z_ind.n_elem + H_ind.n_elem + T_ind.n_elem + R_ind.n_elem - 1) =
        R.elem(R_ind);
  }
  if(xreg.n_cols > 0) {
    theta.subvec(theta.n_elem - xreg.n_cols, theta.n_elem - 1) = beta;
  }
  return theta;
}

void ugg_ssm::compute_RR(){
  for (unsigned int t = 0; t < R.n_slices; t++) {
    RR.slice(t) = R.slice(t * Rtv) * R.slice(t * Rtv).t();
  }
}

double ugg_ssm::log_likelihood() const {
  
  double logLik = 0;
  arma::vec at = a1;
  arma::mat Pt = P1;
  
  arma::vec y_tmp = y;
  if(xreg.n_cols > 0) {
    y_tmp -= xbeta;
  }
  
  const double LOG2PI = std::log(2.0 * M_PI);
  
  for (unsigned int t = 0; t < n; t++) {
    double F = arma::as_scalar(Z.col(t * Ztv).t() * Pt * Z.col(t * Ztv) + HH(t * Htv));
    if (arma::is_finite(y_tmp(t)) && F > zero_tol) {
      double v = arma::as_scalar(y_tmp(t) - D(t * Dtv) - Z.col(t * Ztv).t() * at);
      arma::vec K = Pt * Z.col(t * Ztv) / F;
      at = C.col(t * Ctv) + T.slice(t * Ttv) * (at + K * v);
      Pt = arma::symmatu(T.slice(t * Ttv) * (Pt - K * K.t() * F) * T.slice(t * Ttv).t() + RR.slice(t * Rtv));
      logLik -= 0.5 * (LOG2PI + log(F) + v * v/F);
    } else {
      at = C.col(t * Ctv) + T.slice(t * Ttv) * at;
      Pt = arma::symmatu(T.slice(t * Ttv) * Pt * T.slice(t * Ttv).t() + RR.slice(t * Rtv));
    }
  }
  
  return logLik;
}


arma::cube ugg_ssm::simulate_states(const unsigned int nsim, const bool use_antithetic) {
  
  arma::vec y_tmp = y;
  
  arma::uvec nonzero = arma::find(P1.diag() > 0);
  arma::mat L_P1(m, m, arma::fill::zeros);
  if (nonzero.n_elem > 0) {
    L_P1.submat(nonzero, nonzero) =
      arma::chol(P1.submat(nonzero, nonzero), "lower");
  }
  
  arma::cube asim(m, n, nsim);
  
  std::normal_distribution<> normal(0.0, 1.0);
  
  if (nsim > 1) {
    arma::vec Ft(n);
    arma::mat Kt(m, n);
    arma::cube Lt(m, m, n);
    
    arma::mat alphahat = fast_precomputing_smoother(Ft, Kt, Lt);
    
    
    unsigned int nsim2;
    if(use_antithetic) {
      nsim2 = std::floor(nsim / 2.0);
    } else {
      nsim2 = nsim;
    }
    for(unsigned int i = 0; i < nsim2; i++) {
      arma::mat aplus(m, n);
      
      arma::vec um(m);
      for(unsigned int j = 0; j < m; j++) {
        um(j) = normal(engine);
      }
      aplus.col(0) = a1 + L_P1 * um;
      for (unsigned int t = 0; t < (n - 1); t++) {
        if (arma::is_finite(y(t))) {
          y(t) = xbeta(t) + D(t * Dtv) + 
            arma::as_scalar(Z.col(t * Ztv).t() * aplus.col(t)) + 
            H(t * Htv) * normal(engine);
        }
        arma::vec uk(k);
        for(unsigned int j = 0; j < k; j++) {
          uk(j) = normal(engine);
        }
        aplus.col(t + 1) = C.col(t * Ctv) + T.slice(t * Ttv) * aplus.col(t) + R.slice(t * Rtv) * uk;
      }
      if (arma::is_finite(y(n - 1))) {
        y(n - 1) = xbeta(n - 1) + D((n - 1) * Dtv) + 
          arma::as_scalar(Z.col((n - 1) * Ztv).t() * aplus.col(n - 1)) +
          H((n - 1) * Htv) * normal(engine);
      }
      
      asim.slice(i) = -fast_smoother(Ft, Kt, Lt) + aplus;
      if (use_antithetic){
        asim.slice(i + nsim2) = alphahat - asim.slice(i);
      }
      asim.slice(i) += alphahat;
    }
    if ((2 * nsim2) < nsim) {
      
      arma::mat aplus(m, n);
      
      arma::vec um(m);
      for(unsigned int j = 0; j < m; j++) {
        um(j) = normal(engine);
      }
      aplus.col(0) = a1 + L_P1 * um;
      for (unsigned int t = 0; t < (n - 1); t++) {
        if (arma::is_finite(y(t))) {
          y(t) = xbeta(t) + D(t * Dtv) + 
            arma::as_scalar(Z.col(t * Ztv).t() * aplus.col(t)) +
            H(t * Htv) * normal(engine);
        }
        arma::vec uk(k);
        for(unsigned int j = 0; j < k; j++) {
          uk(j) = normal(engine);
        }
        aplus.col(t + 1) = C.col(t * Ctv) + T.slice(t * Ttv) * aplus.col(t) + 
          R.slice(t * Rtv) * uk;
      }
      if (arma::is_finite(y(n - 1))) {
        y(n - 1) = xbeta(n - 1) + D((n - 1) * Dtv) + 
          arma::as_scalar(Z.col((n - 1) * Ztv).t() * aplus.col(n - 1)) +
          H((n - 1) * Htv) * normal(engine);
      }
      
      asim.slice(nsim - 1) = alphahat - fast_smoother(Ft, Kt, Lt) + aplus;
    }
    
  } else {
    // for _single simulation_ this version is faster:
    //  xbeta, C, D, and a1 set to zero when simulating yplus and aplus
    // (see:
    //  Marek Jarociński 2015: "A note on implementing the Durbin and Koopman simulation
    //  smoother")
    
    arma::vec um(m);
    for(unsigned int j = 0; j < m; j++) {
      um(j) = normal(engine);
    }
    asim.slice(0).col(0) = L_P1 * um;
    for (unsigned int t = 0; t < (n - 1); t++) {
      if (arma::is_finite(y(t))) {
        y(t) -= arma::as_scalar(Z.col(t * Ztv).t() * asim.slice(0).col(t)) +
          H(t * Htv) * normal(engine);
      }
      arma::vec uk(k);
      for(unsigned int j = 0; j < k; j++) {
        uk(j) = normal(engine);
      }
      asim.slice(0).col(t + 1) = T.slice(t * Ttv) * asim.slice(0).col(t) +
        R.slice(t * Rtv) * uk;
    }
    if (arma::is_finite(y(n - 1))) {
      y(n - 1) -= arma::as_scalar(Z.col((n - 1) * Ztv).t() * asim.slice(0).col(n - 1)) +
        H((n - 1) * Htv) * normal(engine);
    }
    
    asim.slice(0) += fast_smoother();
    
  }
  
  y = y_tmp;
  
  return asim;
}

/* Fast state smoothing, only returns smoothed estimates of states
 * which are needed in simulation smoother and Laplace approximation
 */
arma::mat ugg_ssm::fast_smoother() const {
  
  arma::mat at(m, n);
  arma::mat Pt(m, m);
  
  arma::vec vt(n);
  arma::vec Ft(n);
  arma::mat Kt(m, n);
  
  at.col(0) = a1;
  Pt = P1;
  arma::vec y_tmp = y;
  if(xreg.n_cols > 0) {
    y_tmp -= xbeta;
  }
  
  for (unsigned int t = 0; t < (n - 1); t++) {
    Ft(t) = arma::as_scalar(Z.col(t * Ztv).t() * Pt * Z.col(t * Ztv) + HH(t * Htv));
    if (arma::is_finite(y_tmp(t)) && Ft(t) > zero_tol) {
      Kt.col(t) = Pt * Z.col(t * Ztv) / Ft(t);
      vt(t) = arma::as_scalar(y_tmp(t) - D(t * Dtv) - Z.col(t * Ztv).t() * at.col(t));
      at.col(t + 1) = C.col(t * Ctv) + T.slice(t * Ttv) * (at.col(t) + Kt.col(t) * vt(t));
      Pt = arma::symmatu(T.slice(t * Ttv) * (Pt - Kt.col(t) * Kt.col(t).t() * Ft(t)) * T.slice(t * Ttv).t() + RR.slice(t * Rtv));
    } else {
      at.col(t + 1) = C.col(t * Ctv) + T.slice(t * Ttv) * at.col(t);
      Pt = arma::symmatu(T.slice(t * Ttv) * Pt * T.slice(t * Ttv).t() + RR.slice(t * Rtv));
    }
  }
  unsigned int t = n - 1;
  Ft(t) = arma::as_scalar(Z.col(t * Ztv).t() * Pt * Z.col(t * Ztv) + HH(t * Htv));
  if (arma::is_finite(y_tmp(t)) && Ft(t) > zero_tol) {
    vt(t) = arma::as_scalar(y_tmp(t) - D(t * Dtv) - Z.col(t * Ztv).t() * at.col(t));
    Kt.col(t) = Pt * Z.col(t * Ztv) / Ft(t);
  }
  arma::mat rt(m, n);
  rt.col(n - 1).zeros();
  for (int t = (n - 1); t > 0; t--) {
    if (arma::is_finite(y_tmp(t)) && Ft(t) > zero_tol){
      arma::mat L = T.slice(t * Ttv) * (arma::eye(m, m) - Kt.col(t) * Z.col(t * Ztv).t());
      rt.col(t - 1) = Z.col(t * Ztv) / Ft(t) * vt(t) + L.t() * rt.col(t);
    } else {
      rt.col(t - 1) = T.slice(t * Ttv).t() * rt.col(t);
    }
  }
  if (arma::is_finite(y(0)) && Ft(0) > zero_tol){
    arma::mat L = T.slice(0) * (arma::eye(m, m) - Kt.col(0) * Z.col(0).t());
    at.col(0) = a1 + P1 * (Z.col(0) / Ft(0) * vt(0) + L.t() * rt.col(0));
  } else {
    at.col(0) = a1 + P1 * T.slice(0).t() * rt.col(0);
  }
  
  for (unsigned int t = 0; t < (n - 1); t++) {
    at.col(t + 1) = C.col(t * Ctv) + T.slice(t * Ttv) * at.col(t) + RR.slice(t * Rtv) * rt.col(t);
  }
  
  return at;
}

/* Fast state smoothing which returns also Ft, Kt and Lt which can be used
 * in subsequent calls of smoother in simulation smoother.
 */


/* Fast state smoothing which uses precomputed Ft, Kt and Lt.
 */
arma::mat ugg_ssm::fast_smoother(const arma::vec& Ft, const arma::mat& Kt,
  const arma::cube& Lt) const {
  
  arma::mat at(m, n);
  arma::mat Pt(m, m);
  
  arma::vec vt(n);
  
  at.col(0) = a1;
  Pt = P1;
  
  arma::vec y_tmp = y;
  if (xreg.n_cols > 0) {
    y_tmp -= xbeta;
  }
  
  for (unsigned int t = 0; t < (n - 1); t++) {
    if (arma::is_finite(y_tmp(t)) && Ft(t) > zero_tol) {
      vt(t) = arma::as_scalar(y_tmp(t) - D(t * Dtv) - Z.col(t * Ztv).t() * at.col(t));
      at.col(t + 1) = C.col(t * Ctv) + T.slice(t * Ttv) * (at.col(t) + Kt.col(t) * vt(t));
    } else {
      at.col(t + 1) = C.col(t * Ctv) + T.slice(t * Ttv) * at.col(t);
    }
  }
  unsigned int t = n - 1;
  if (arma::is_finite(y_tmp(t)) && Ft(t) > zero_tol) {
    vt(t) = arma::as_scalar(y_tmp(t) - D(t * Dtv) - Z.col(t * Ztv).t() * at.col(t));
  }
  arma::mat rt(m, n);
  rt.col(n - 1).zeros();
  
  for (int t = (n - 1); t > 0; t--) {
    if (arma::is_finite(y_tmp(t)) && Ft(t) > zero_tol){
      rt.col(t - 1) = Z.col(t * Ztv) / Ft(t) * vt(t) + Lt.slice(t).t() * rt.col(t);
    } else {
      rt.col(t - 1) = T.slice(t * Ttv).t() * rt.col(t);
    }
  }
  if (arma::is_finite(y(0)) && Ft(0) > zero_tol){
    arma::mat L = T.slice(0) * (arma::eye(m, m) - Kt.col(0) * Z.col(0).t());
    at.col(0) = a1 + P1 * (Z.col(0) / Ft(0) * vt(0) + L.t() * rt.col(0));
  } else {
    at.col(0) = a1 + P1 * T.slice(0).t() * rt.col(0);
  }
  
  for (unsigned int t = 0; t < (n - 1); t++) {
    at.col(t + 1) = C.col(t * Ctv) + T.slice(t * Ttv) * at.col(t) + RR.slice(t * Rtv) * rt.col(t);
  }
  
  
  return at;
}

arma::mat ugg_ssm::fast_precomputing_smoother(arma::vec& Ft, arma::mat& Kt, 
  arma::cube& Lt) const {
  
  arma::mat at(m, n);
  arma::mat Pt(m, m);
  arma::vec vt(n);
  
  at.col(0) = a1;
  Pt = P1;
  
  arma::vec y_tmp = y;
  if (xreg.n_cols > 0) {
    y_tmp -= xbeta;
  }
  for (unsigned int t = 0; t < (n - 1); t++) {
    Ft(t) = arma::as_scalar(Z.col(t * Ztv).t() * Pt * Z.col(t * Ztv) + HH(t * Htv));
    if (arma::is_finite(y_tmp(t)) && Ft(t) > zero_tol) {
      Kt.col(t) = Pt * Z.col(t * Ztv) / Ft(t);
      vt(t) = arma::as_scalar(y_tmp(t) - D(t * Dtv) - Z.col(t * Ztv).t() * at.col(t));
      at.col(t + 1) = C.col(t * Ctv) + T.slice(t * Ttv) * (at.col(t) + Kt.col(t) * vt(t));
      Pt = arma::symmatu(T.slice(t * Ttv) * (Pt - Kt.col(t) * Kt.col(t).t() * Ft(t)) * T.slice(t * Ttv).t() + RR.slice(t * Rtv));
    } else {
      at.col(t + 1) = C.col(t * Ctv) + T.slice(t * Ttv) * at.col(t);
      Pt = arma::symmatu(T.slice(t * Ttv) * Pt * T.slice(t * Ttv).t() + RR.slice(t * Rtv));
    }
  }
  unsigned int t = n - 1;
  Ft(t) = arma::as_scalar(Z.col(t * Ztv).t() * Pt * Z.col(t * Ztv) + HH(t * Htv));
  if (arma::is_finite(y_tmp(t)) && Ft(t) > zero_tol) {
    vt(t) = arma::as_scalar(y_tmp(t) - D(t * Dtv) - Z.col(t * Ztv).t() * at.col(t));
    Kt.col(t) = Pt * Z.col(t * Ztv) / Ft(t);
  }
  arma::mat rt(m, n);
  rt.col(n - 1).zeros();
  
  for (int t = (n - 1); t > 0; t--) {
    if (arma::is_finite(y_tmp(t)) && Ft(t) > zero_tol){
      Lt.slice(t) = T.slice(t * Ttv) * (arma::eye(m, m) - Kt.col(t) * Z.col(t * Ztv).t());
      rt.col(t - 1) = Z.col(t * Ztv) / Ft(t) * vt(t) + Lt.slice(t).t() * rt.col(t);
    } else {
      rt.col(t - 1) = T.slice(t * Ttv).t() * rt.col(t);
    }
  }
  if (arma::is_finite(y_tmp(0)) && Ft(0) > zero_tol){
    arma::mat L = T.slice(0) * (arma::eye(m, m) - Kt.col(0) * Z.col(0).t());
    at.col(0) = a1 + P1 * (Z.col(0) / Ft(0) * vt(0) + L.t() * rt.col(0));
  } else {
    at.col(0) = a1 + P1 * T.slice(0).t() * rt.col(0);
  }
  for (unsigned int t = 0; t < (n - 1); t++) {
    at.col(t + 1) = C.col(t * Ctv) + T.slice(t * Ttv) * at.col(t) + RR.slice(t * Rtv) * rt.col(t);
  }
  
  return at;
}

// smoother which returns also cov(alpha_t, alpha_t-1)
// used in psi particle filter
void ugg_ssm::smoother_ccov(arma::mat& at, arma::cube& Pt, arma::cube& ccov) const {
  
  at.col(0) = a1;
  Pt.slice(0) = P1;
  arma::vec vt(n);
  arma::vec Ft(n);
  arma::mat Kt(m, n);
  
  arma::vec y_tmp = y;
  if(xreg.n_cols > 0) {
    y_tmp -= xbeta;
  }
  
  for (unsigned int t = 0; t < (n - 1); t++) {
    Ft(t) = arma::as_scalar(Z.col(t * Ztv).t() * Pt.slice(t) * Z.col(t * Ztv) +
      HH(t * Htv));
    if (arma::is_finite(y_tmp(t)) && Ft(t) > zero_tol) {
      Kt.col(t) = Pt.slice(t) * Z.col(t * Ztv) / Ft(t);
      vt(t) = arma::as_scalar(y_tmp(t) - D(t * Dtv) - Z.col(t * Ztv).t() * at.col(t));
      at.col(t + 1) = C.col(t * Ctv) + T.slice(t * Ttv) * (at.col(t) + Kt.col(t) * vt(t));
      Pt.slice(t + 1) = arma::symmatu(T.slice(t * Ttv) * (Pt.slice(t) -
        Kt.col(t) * Kt.col(t).t() * Ft(t)) * T.slice(t * Ttv).t() + RR.slice(t * Rtv));
    } else {
      at.col(t + 1) = C.col(t * Ctv) + T.slice(t * Ttv) * at.col(t);
      Pt.slice(t + 1) = arma::symmatu(T.slice(t * Ttv) * Pt.slice(t) * T.slice(t * Ttv).t() +
        RR.slice(t * Rtv));
    }
    ccov.slice(t) = Pt.slice(t+1); //store for smoothing;
  }
  unsigned int t = n - 1;
  Ft(t) = arma::as_scalar(Z.col(t * Ztv).t() * Pt.slice(t) * Z.col(t * Ztv) +
    HH(t * Htv));
  if (arma::is_finite(y_tmp(t)) && Ft(t) > zero_tol) {
    vt(t) = arma::as_scalar(y_tmp(t) - D(t * Dtv) - Z.col(t * Ztv).t() * at.col(t));
    Kt.col(t) = Pt.slice(t) * Z.col(t * Ztv) / Ft(t);
    ccov.slice(t) = arma::symmatu(T.slice(t * Ttv) * (Pt.slice(t) -
      Kt.col(t) * Kt.col(t).t() * Ft(t)) * T.slice(t * Ttv).t() + RR.slice(t * Rtv));
  } else {
    ccov.slice(t) = arma::symmatu(T.slice(t * Ttv) * Pt.slice(t) * T.slice(t * Ttv).t() +
      RR.slice(t * Rtv));
  }
  
  arma::vec rt(m, arma::fill::zeros);
  arma::mat Nt(m, m, arma::fill::zeros);
  
  for (int t = (n - 1); t >= 0; t--) {
    if (arma::is_finite(y_tmp(t)) && Ft(t) > zero_tol){
      arma::mat L = T.slice(t * Ttv) * (arma::eye(m, m) - Kt.col(t) * Z.col(t * Ztv).t());
      //P[t+1] stored to ccov_t
      ccov.slice(t) = Pt.slice(t) * L.t() * (arma::eye(m, m) - Nt * ccov.slice(t));
      rt = Z.col(t * Ztv) / Ft(t) * vt(t) + L.t() * rt;
      Nt = arma::symmatu(Z.col(t * Ztv) * Z.col(t * Ztv).t() / Ft(t) + L.t() * Nt * L);
    } else {
      ccov.slice(t) = Pt.slice(t) * T.slice(t * Ttv).t() * (arma::eye(m, m) - Nt * ccov.slice(t));
      rt = T.slice(t * Ttv).t() * rt;
      Nt = arma::symmatu(T.slice(t * Ttv).t() * Nt * T.slice(t * Ttv));
      //P[t+1] stored to ccov_t //CHECK THIS
    }
    at.col(t) += Pt.slice(t) * rt;
    Pt.slice(t) -= arma::symmatu(Pt.slice(t) * Nt * Pt.slice(t));
  }
}
double ugg_ssm::filter(arma::mat& at, arma::mat& att, arma::cube& Pt,
  arma::cube& Ptt) const {
  
  double logLik = 0;
  
  at.col(0) = a1;
  Pt.slice(0) = P1;
  
  arma::vec y_tmp = y;
  if(xreg.n_cols > 0) {
    y_tmp -= xbeta;
  }
  const double LOG2PI = std::log(2.0 * M_PI);
  
  for (unsigned int t = 0; t < n; t++) {
    double F = arma::as_scalar(Z.col(t * Ztv).t() * Pt.slice(t) * Z.col(t * Ztv) + HH(t * Htv));
    if (arma::is_finite(y_tmp(t)) && F > zero_tol) {
      double v = arma::as_scalar(y_tmp(t) - D(t * Dtv) - Z.col(t * Ztv).t() * at.col(t));
      arma::vec K = Pt.slice(t) * Z.col(t * Ztv) / F;
      att.col(t) = at.col(t) + K * v;
      at.col(t + 1) = C.col(t * Ctv) + T.slice(t * Ttv) * (att.col(t));
      Ptt.slice(t) = Pt.slice(t) - K * K.t() * F;
      Pt.slice(t + 1) = arma::symmatu(T.slice(t * Ttv) * Ptt.slice(t) * T.slice(t * Ttv).t() + RR.slice(t * Rtv));
      logLik -= 0.5 * (LOG2PI + log(F) + v * v/F);
    } else {
      att.col(t) = at.col(t);
      at.col(t + 1) = C.col(t * Ctv) + T.slice(t * Ttv) * att.col(t);
      Ptt.slice(t) = Pt.slice(t);
      Pt.slice(t + 1) = arma::symmatu(T.slice(t * Ttv) * Ptt.slice(t) * T.slice(t * Ttv).t() + RR.slice(t * Rtv));
    }
  }
  
  return logLik;
}
void ugg_ssm::smoother(arma::mat& at, arma::cube& Pt) const {
  
  at.col(0) = a1;
  Pt.slice(0) = P1;
  arma::vec vt(n);
  arma::vec Ft(n);
  arma::mat Kt(m, n);
  
  arma::vec y_tmp = y;
  if (xreg.n_cols > 0) {
    y_tmp -= xbeta;
  }
  
  for (unsigned int t = 0; t < (n - 1); t++) {
    Ft(t) = arma::as_scalar(Z.col(t * Ztv).t() * Pt.slice(t) * Z.col(t * Ztv) +
      HH(t * Htv));
    if (arma::is_finite(y_tmp(t)) && Ft(t) > zero_tol) {
      Kt.col(t) = Pt.slice(t) * Z.col(t * Ztv) / Ft(t);
      vt(t) = arma::as_scalar(y_tmp(t) - D(t * Dtv) - Z.col(t * Ztv).t() * at.col(t));
      at.col(t + 1) = C.col(t * Ctv) + T.slice(t * Ttv) * (at.col(t) + Kt.col(t) * vt(t));
      Pt.slice(t + 1) = arma::symmatu(T.slice(t * Ttv) * (Pt.slice(t) -
        Kt.col(t) * Kt.col(t).t() * Ft(t)) * T.slice(t * Ttv).t() + RR.slice(t * Rtv));
    } else {
      at.col(t + 1) = C.col(t * Ctv) + T.slice(t * Ttv) * at.col(t);
      Pt.slice(t + 1) = arma::symmatu(T.slice(t * Ttv) * Pt.slice(t) * T.slice(t * Ttv).t() +
        RR.slice(t * Rtv));
    }
  }
  unsigned int t = n - 1;
  Ft(t) = arma::as_scalar(Z.col(t * Ztv).t() * Pt.slice(t) * Z.col(t * Ztv) +
    HH(t * Htv));
  if (arma::is_finite(y_tmp(t)) && Ft(t) > zero_tol) {
    vt(t) = arma::as_scalar(y_tmp(t) - D(t * Dtv) - Z.col(t * Ztv).t() * at.col(t));
    Kt.col(t) = Pt.slice(t) * Z.col(t * Ztv) / Ft(t);
  }
  
  
  arma::vec rt(m, arma::fill::zeros);
  arma::mat Nt(m, m, arma::fill::zeros);
  
  for (int t = (n - 1); t >= 0; t--) {
    if (arma::is_finite(y_tmp(t)) && Ft(t) > zero_tol){
      arma::mat L = T.slice(t * Ttv) * (arma::eye(m, m) - Kt.col(t) * Z.col(t * Ztv).t());
      rt = Z.col(t * Ztv) / Ft(t) * vt(t) + L.t() * rt;
      Nt = arma::symmatu(Z.col(t * Ztv) * Z.col(t * Ztv).t() / Ft(t) + L.t() * Nt * L);
    } else {
      rt = T.slice(t * Ttv).t() * rt;
      Nt = arma::symmatu(T.slice(t * Ttv).t() * Nt * T.slice(t * Ttv));
    }
    at.col(t) += Pt.slice(t) * rt;
    Pt.slice(t) -= arma::symmatu(Pt.slice(t) * Nt * Pt.slice(t));
  }
}

double ugg_ssm::bsf_filter(const unsigned int nsim, arma::cube& alpha,
  arma::mat& weights, arma::umat& indices) {
  
  arma::uvec nonzero = arma::find(P1.diag() > 0);
  arma::mat L_P1(m, m, arma::fill::zeros);
  if (nonzero.n_elem > 0) {
    L_P1.submat(nonzero, nonzero) =
      arma::chol(P1.submat(nonzero, nonzero), "lower");
  }
  std::normal_distribution<> normal(0.0, 1.0);
  for (unsigned int i = 0; i < nsim; i++) {
    arma::vec um(m);
    for(unsigned int j = 0; j < m; j++) {
      um(j) = normal(engine);
    }
    alpha.slice(i).col(0) = a1 + L_P1 * um;
  }
  
  std::uniform_real_distribution<> unif(0.0, 1.0);
  arma::vec normalized_weights(nsim);
  double loglik = 0.0;
  
  if(arma::is_finite(y(0))) {
    
    for (unsigned int i = 0; i < nsim; i++) {
      double mu = arma::as_scalar(Z.col(0).t() * alpha.slice(i).col(0));
      weights(i, 0) = -0.5 * std::pow(y(0) - mu, 2) / HH(0);
    }
    double max_weight = weights.col(0).max();
    weights.col(0) = exp(weights.col(0) - max_weight);
    double sum_weights = arma::sum(weights.col(0));
    if(sum_weights > 0.0){
      normalized_weights = weights.col(0) / sum_weights;
    } else {
      return -arma::datum::inf;
    }
    loglik = max_weight + log(sum_weights / nsim) + norm_log_const(H(0));
  } else {
    weights.col(0).ones();
    normalized_weights.fill(1.0 / nsim);
  }
  for (unsigned int t = 0; t < (n - 1); t++) {
    
    arma::vec r(nsim);
    for (unsigned int i = 0; i < nsim; i++) {
      r(i) = unif(engine);
    }
    
    indices.col(t) = stratified_sample(normalized_weights, r, nsim);
    
    arma::mat alphatmp(m, nsim);
    
    for (unsigned int i = 0; i < nsim; i++) {
      alphatmp.col(i) = alpha.slice(indices(i, t)).col(t);
    }
    
    for (unsigned int i = 0; i < nsim; i++) {
      arma::vec uk(k);
      for(unsigned int j = 0; j < k; j++) {
        uk(j) = normal(engine);
      }
      alpha.slice(i).col(t + 1) = C.col(t * Ctv) + 
        T.slice(t * Ttv) * alphatmp.col(i) + R.slice(t * Rtv) * uk;
    }
    
    if(arma::is_finite(y(t + 1))) {
      for (unsigned int i = 0; i < nsim; i++) {
        double mu = arma::as_scalar(Z.col(Ztv * (t + 1)).t() * 
          alpha.slice(i).col(t + 1));
        weights(i, t + 1) = -0.5 * std::pow(y(t + 1) - mu, 2) / HH(Htv * (t + 1));
      }
      
      double max_weight = weights.col(t + 1).max();
      weights.col(t + 1) = exp(weights.col(t + 1) - max_weight);
      double sum_weights = arma::sum(weights.col(t + 1));
      if(sum_weights > 0.0){
        normalized_weights = weights.col(t + 1) / sum_weights;
      } else {
        return -arma::datum::inf;
      }
      loglik += max_weight + log(sum_weights / nsim) +
        norm_log_const(H(Htv * (t + 1)));
    } else {
      weights.col(t + 1).ones();
      normalized_weights.fill(1.0/nsim);
    }
  }
  
  return loglik;
}
Rcpp::List ugg_ssm::predict_interval(const arma::vec& probs, const arma::mat& theta,
  const arma::mat& alpha, const arma::uvec& counts, const unsigned int predict_type) {
  
  arma::vec theta_i = theta.col(0);
  set_theta(theta_i);
  a1 = alpha.col(0);
  P1.zeros();
  arma::mat at(m, n + 1);
  arma::mat att(m, n);
  arma::cube Pt(m, m, n + 1);
  arma::cube Ptt(m, m, n);
  filter(at, att, Pt, Ptt);
  
  unsigned int n_samples = theta.n_cols;
  
  if (predict_type < 3) {
    
    arma::mat mean_pred(n, n_samples);
    arma::mat var_pred(n, n_samples);
    
    for(unsigned int t = 0; t < n; t++) {
      mean_pred(t, 0) = arma::as_scalar(xbeta(t) + 
        Z.col(Ztv * t).t() * at.col(t));
      var_pred(t, 0) = arma::as_scalar(Z.col(Ztv * t).t() * Pt.slice(t) * Z.col(Ztv * t));
    }
    
    if (predict_type == 1) {
      for(unsigned int t = 0; t < n; t++) {
        var_pred(t, 0) += HH(Htv * t);
      }
    }
    
    for (unsigned int i = 1; i < n_samples; i++) {
      arma::vec theta_i = theta.col(i);
      set_theta(theta_i);
      a1 = alpha.col(i);
      P1.zeros();
      filter(at, att, Pt, Ptt);
      
      for(unsigned int t = 0; t < n; t++) {
        mean_pred(t, i) = arma::as_scalar(xbeta(t) + 
          Z.col(Ztv * t).t() * at.col(t));
        var_pred(t, i) = arma::as_scalar(Z.col(Ztv * t).t() * Pt.slice(t) * Z.col(Ztv * t));
      }
      if (predict_type == 1) {
        for(unsigned int t = 0; t < n; t++) {
          var_pred(t, i) += HH(Htv * t);
        }
      }
      
    }
    
    arma::mat expanded_sd = rep_mat(sqrt(var_pred), counts);
    arma::inplace_trans(expanded_sd);
    arma::mat expanded_mean = rep_mat(mean_pred, counts);
    arma::inplace_trans(expanded_mean);
    arma::mat intv = intervals(expanded_mean, expanded_sd, probs, n);
    
    return Rcpp::List::create(Rcpp::Named("intervals") = intv, 
      Rcpp::Named("mean_pred") = expanded_mean,
      Rcpp::Named("sd_pred") = expanded_sd);
  } else {
    arma::cube mean_pred(n, n_samples, m);
    arma::cube var_pred(n, n_samples, m);
    
    for(unsigned int t = 0; t < n; t++) {
      mean_pred.tube(t, 0) = at.col(t);
      var_pred.tube(t, 0) = Pt.slice(t).diag();
    }
    
    for (unsigned int i = 1; i < n_samples; i++) {
      arma::vec theta_i = theta.col(i);
      set_theta(theta_i);
      a1 = alpha.col(i);
      P1.zeros();
      filter(at, att, Pt, Ptt);
      
      for(unsigned int t = 0; t < n; t++) {
        mean_pred.tube(t, i) = at.col(t);
        var_pred.tube(t, i) = Pt.slice(t).diag();
      }
    }
    
    arma::cube intv(n, probs.n_elem, m);
    arma::cube expanded_sd(sum(counts), n, m);
    arma::cube expanded_mean(sum(counts), n, m);
    for (unsigned int i = 0; i < m; i++) {
      arma::mat tmp1 = rep_mat(sqrt(var_pred.slice(i)), counts);
      expanded_sd.slice(i) = tmp1.t();
      arma::mat tmp2 = rep_mat(mean_pred.slice(i), counts);
      expanded_mean.slice(i) = tmp2.t();
      intv.slice(i) = intervals(expanded_mean.slice(i), expanded_sd.slice(i), 
        probs, n);
    }
   
    
    return Rcpp::List::create(Rcpp::Named("intervals") = intv, 
      Rcpp::Named("mean_pred") = expanded_mean,
      Rcpp::Named("sd_pred") = expanded_sd);
  }
}
arma::cube ugg_ssm::predict_sample(const arma::mat& theta,
  const arma::mat& alpha, const arma::uvec& counts, const bool predict_obs) {
  
  unsigned int d = 1;
  if (!predict_obs) d = m;
  
  unsigned int n_samples = theta.n_cols;
  arma::cube sample(d, n, n_samples);
  
  arma::vec theta_i = theta.col(0);
  set_theta(theta_i);
  a1 = alpha.col(0);
  sample.slice(0) = sample_model(predict_obs);
  
  for (unsigned int i = 1; i < n_samples; i++) {
    arma::vec theta_i = theta.col(i);
    set_theta(theta_i);
    a1 = alpha.col(i);
    sample.slice(i) = sample_model(predict_obs);
  }
  
  return rep_cube(sample, counts);
}


arma::mat ugg_ssm::sample_model(const bool simulate_obs) {
  
  arma::mat alpha(m, n);
  alpha.col(0) = a1;
  std::normal_distribution<> normal(0.0, 1.0);
  
  for (unsigned int t = 0; t < (n - 1); t++) {
    arma::vec uk(k);
    for(unsigned int j = 0; j < k; j++) {
      uk(j) = normal(engine);
    }
    alpha.col(t + 1) = C.col(t * Ctv) + T.slice(t * Ttv) * alpha.col(t) + R.slice(t * Rtv) * uk;
  }
  
  if (simulate_obs) {
    arma::mat y(1, n);
    for (unsigned int t = 0; t < n; t++) {
      y(0, t) = xbeta(t) + D(t * Dtv) + 
        arma::as_scalar(Z.col(t * Ztv).t() * alpha.col(t)) + 
        H(t * Htv) * normal(engine);
    }
    return y;
  } else {
    return alpha;
  }
}

// // 
// // Rcpp::List gssm::predict(const arma::uvec& prior_types, const arma::mat& prior_pars,
// //   unsigned int n_iter, unsigned int n_burnin, unsigned int n_thin, double gamma,
// //   double target_acceptance, arma::mat S, unsigned int n_ahead,
// //   unsigned int interval, arma::vec probs) {
// //   
// //   unsigned int n_samples = floor((n_iter - n_burnin) / n_thin);
// //   
// //   unsigned int n_par = prior_types.n_elem;
// //   arma::vec theta = get_theta();
// //   double prior = prior_pdf(theta, prior_types, prior_pars);
// //   
// //   arma::mat mean_pred(n_ahead, n_samples);
// //   arma::mat var_pred(n_ahead, n_samples);
// //   
// //   arma::mat at(m, n + 1);
// //   arma::cube Pt(m, m, n + 1);
// //   arma::mat att(m, n);
// //   arma::cube Ptt(m, m, n);
// //   filter(at, att, Pt, Ptt, true);
// //   
// //   unsigned int j = 0;
// //   
// //   if (n_burnin == 0){
// //     for (unsigned int t = n - n_ahead; t < n; t++) {
// //       mean_pred(t - n + n_ahead, j) = arma::as_scalar(Z.col(Ztv * t).t() * at.col(t));
// //       if(xreg.n_cols > 0) {
// //         mean_pred(t - n + n_ahead, j) += xbeta(t);
// //       }
// //       var_pred(t - n + n_ahead, j) = arma::as_scalar(Z.col(Ztv * t).t() * Pt.slice(t) * Z.col(Ztv * t));
// //       if (interval == 2) {
// //         var_pred(t - n + n_ahead, j) += HH(Htv * t);
// //       }
// //     }
// //     j++;
// //   }
// //   
// //   double ll = log_likelihood(true);
// //   double accept_prob = 0.0;
// //   std::normal_distribution<> normal(0.0, 1.0);
// //   std::uniform_real_distribution<> unif(0.0, 1.0);
// //   for (unsigned int i = 1; i < n_iter; i++) {
// //     
// //     if (i % 16 == 0) {
// //       Rcpp::checkUserInterrupt();
// //     }
// //     
// //     // sample from standard normal distribution
// //     arma::vec u(n_par);
// //     for(unsigned int ii = 0; ii < n_par; ii++) {
// //       u(ii) = normal(engine);
// //     }
// //     // propose new theta
// //     arma::vec theta_prop = theta + S * u;
// //     // check prior
// //     double prior_prop = prior_pdf(theta_prop, prior_types, prior_pars);
// //     
// //     if (prior_prop > -arma::datum::inf) {
// //       // update parameters
// //       set_theta(theta_prop);
// //       // compute log-likelihood with proposed theta
// //       double ll_prop = log_likelihood(true);
// //       //compute the acceptance probability
// //       // use explicit min(...) as we need this value later
// //       double q = proposal(theta, theta_prop);
// //       accept_prob = std::min(1.0, exp(ll_prop - ll + prior_prop - prior + q));
// //       
// //       
// //       //accept
// //       if (unif(engine) < accept_prob) {
// //         ll = ll_prop;
// //         prior = prior_prop;
// //         theta = theta_prop;
// //         filter(at, att, Pt, Ptt, true);
// //       }
// //     } else accept_prob = 0.0;
// //     
// //     if ((i >= n_burnin) && (i % n_thin == 0)) {
// //       set_theta(theta);
// //       for (unsigned int t = n - n_ahead; t < n; t++) {
// //         mean_pred(t - n + n_ahead, j) = arma::as_scalar(Z.col(Ztv * t).t() * at.col(t));
// //         if(xreg.n_cols > 0) {
// //           mean_pred(t - n + n_ahead, j) += xbeta(t);
// //         }
// //         var_pred(t - n + n_ahead, j) = arma::as_scalar(Z.col(Ztv * t).t() * Pt.slice(t) * Z.col(Ztv * t));
// //         if (interval == 2) {
// //           var_pred(t - n + n_ahead, j) += HH(Htv * t);
// //         }
// //       }
// //       j++;
// //     }
// //     
// //     ramcmc::adapt_S(S, u, accept_prob, target_acceptance, i, gamma);
// //     
// //   }
// //   
// //   
// //   arma::inplace_trans(mean_pred);
// //   arma::inplace_trans(var_pred);
// //   var_pred = sqrt(var_pred);
// //   arma::mat intv = intervals(mean_pred, var_pred, probs, n_ahead);
// //   return Rcpp::List::create(Rcpp::Named("intervals") = intv, Rcpp::Named("mean_pred") = mean_pred,
// //     Rcpp::Named("y_sd") = var_pred);
// // }
// // 
// // 
// // arma::mat gssm::predict2(const arma::uvec& prior_types,
// //   const arma::mat& prior_pars, unsigned int n_iter,
// //   unsigned int n_burnin, unsigned int n_thin, double gamma,
// //   double target_acceptance, arma::mat S, unsigned int n_ahead,
// //   unsigned int interval) {
// //   
// //   unsigned int n_samples = floor((n_iter - n_burnin) / n_thin);
// //   
// //   arma::mat pred_store(n_ahead, n_samples);
// //   
// //   unsigned int n_par = prior_types.n_elem;
// //   arma::vec theta = get_theta();
// //   double prior = prior_pdf(theta, prior_types, prior_pars);
// //   arma::cube alpha = sim_smoother(1, true).tube(0, n - n_ahead, m - 1,  n - 1);
// //   
// //   unsigned int j = 0;
// //   std::normal_distribution<> normal(0.0, 1.0);
// //   
// //   if (n_burnin == 0){
// //     for (unsigned int t = 0; t < n_ahead; t++) {
// //       pred_store(t, 0) = arma::as_scalar(Z.col(Ztv * (n - n_ahead + t)).t() * alpha.slice(0).col(t));
// //     }
// //     
// //     if(xreg.n_cols > 0) {
// //       pred_store.col(0) +=  xbeta.subvec(n - n_ahead, n - 1);
// //     }
// //     if (interval == 2) {
// //       for (unsigned int t = 0; t < n_ahead; t++) {
// //         pred_store.row(t).col(0) += H(Htv * (n - n_ahead + t)) * normal(engine);
// //       }
// //     }
// //     j++;
// //   }
// //   
// //   double ll = log_likelihood(true);
// //   double accept_prob = 0.0;
// //   
// //   std::uniform_real_distribution<> unif(0.0, 1.0);
// //   for (unsigned int i = 1; i < n_iter; i++) {
// //     // sample from standard normal distribution
// //     arma::vec u(n_par);
// //     for(unsigned int ii = 0; ii < n_par; ii++) {
// //       u(ii) = normal(engine);
// //     }
// //     // propose new theta
// //     arma::vec theta_prop = theta + S * u;
// //     // check prior
// //     // check prior
// //     double prior_prop = prior_pdf(theta_prop, prior_types, prior_pars);
// //     
// //     if (prior_prop > -arma::datum::inf) {
// //       // update parameters
// //       set_theta(theta_prop);
// //       // compute log-likelihood with proposed theta
// //       double ll_prop = log_likelihood(true);
// //       //compute the acceptance probability
// //       // use explicit min(...) as we need this value later
// //       double q = proposal(theta, theta_prop);
// //       accept_prob = std::min(1.0, exp(ll_prop - ll + prior_prop - prior + q));
// //       
// //       //accept
// //       if (unif(engine) < accept_prob) {
// //         ll = ll_prop;
// //         prior = prior_prop;
// //         theta = theta_prop;
// //         alpha = sim_smoother(1, true).tube(0, n - n_ahead, m - 1,  n - 1);
// //       }
// //     } else accept_prob = 0.0;
// //     
// //     if ((i >= n_burnin) && (i % n_thin == 0)) {
// //       set_theta(theta);
// //       
// //       for (unsigned int t = 0; t < n_ahead; t++) {
// //         pred_store(t, j) = arma::as_scalar(Z.col(Ztv * (n - n_ahead + t)).t() *
// //           alpha.slice(0).col(t));
// //       }
// //       
// //       if(xreg.n_cols > 0) {
// //         pred_store.col(j) +=  xbeta.subvec(n - n_ahead, n - 1);
// //       }
// //       if (interval == 2) {
// //         for (unsigned int t = 0; t < n_ahead; t++) {
// //           pred_store.row(t).col(j) += H(Htv * (n - n_ahead + t)) * normal(engine);
// //         }
// //       }
// //       j++;
// //     }
// //     
// //     ramcmc::adapt_S(S, u, accept_prob, target_acceptance, i, gamma);
// //     
// //   }
// //   
// //   return pred_store;
// //   
// // }
// // 
// //particle filter
// double gssm::particle_filter(unsigned int nsim, arma::cube& alphasim, arma::mat& w, arma::umat& ind) {
//   
//   std::normal_distribution<> normal(0.0, 1.0);
//   std::uniform_real_distribution<> unif(0.0, 1.0);
//   
//   arma::uvec nonzero = arma::find(P1.diag() > 0);
//   arma::mat L_P1(m, m, arma::fill::zeros);
//   if (nonzero.n_elem > 0) {
//     L_P1.submat(nonzero, nonzero) =
//       arma::chol(P1.submat(nonzero, nonzero), "lower");
//   }
//   for (unsigned int i = 0; i < nsim; i++) {
//     arma::vec um(m);
//     for(unsigned int j = 0; j < m; j++) {
//       um(j) = normal(engine);
//     }
//     alphasim.slice(i).col(0) = a1 + L_P1 * um;
//   }
//   
//   arma::vec wnorm(nsim);
//   double ll = 0.0;
//   if(arma::is_finite(y(0))) {
//     for (unsigned int i = 0; i < nsim; i++) {
//       w(i, 0) = R::dnorm(y(0),
//         arma::as_scalar(Z.col(0).t() * alphasim.slice(i).col(0) + xbeta(0)),
//         H(0), 1);
//     }
//     double maxv = w.col(0).max();
//     w.col(0) = exp(w.col(0) - maxv);
//     double sumw = arma::sum(w.col(0));
//     if(sumw > 0.0){
//       wnorm = w.col(0) / sumw;
//     } else {
//       return -arma::datum::inf;
//     }
//     ll = maxv + log(sumw / nsim);
//   } else {
//     w.col(0).ones();
//     wnorm.fill(1.0/nsim);
//   }
//   
//   for (unsigned int t = 0; t < (n - 1); t++) {
//     
//     arma::vec r(nsim);
//     for (unsigned int i = 0; i < nsim; i++) {
//       r(i) = unif(engine);
//     }
//     
//     ind.col(t) = stratified_sample(wnorm, r, nsim);
//     
//     arma::mat alphatmp(m, nsim);
//     
//     for (unsigned int i = 0; i < nsim; i++) {
//       alphatmp.col(i) = alphasim.slice(ind(i, t)).col(t);
//     }
//     for (unsigned int i = 0; i < nsim; i++) {
//       arma::vec uk(k);
//       for(unsigned int j = 0; j < k; j++) {
//         uk(j) = normal(engine);
//       }
//       alphasim.slice(i).col(t + 1) = C.col(t * Ctv) + T.slice(t * Ttv) * alphatmp.col(i) +
//         R.slice(t * Rtv) * uk;
//     }
//     
//     if(arma::is_finite(y(t + 1))) {
//       for (unsigned int i = 0; i < nsim; i++) {
//         w(i, t + 1) = R::dnorm(y(t + 1),
//           arma::as_scalar(Z.col((t + 1) * Ztv).t() * alphasim.slice(i).col(t + 1) + xbeta(t + 1)),
//           H((t + 1) * Htv), 1);
//       }
//       double maxv = w.col(t + 1).max();
//       w.col(t + 1) = exp(w.col(t + 1) - maxv);
//       double sumw = arma::sum(w.col(t + 1));
//       if(sumw > 0.0){
//         wnorm = w.col(t + 1) / sumw;
//       } else {
//         return -arma::datum::inf;
//       }
//       ll += maxv + log(sumw / nsim);
//     } else {
//       w.col(t + 1).ones();
//       wnorm.fill(1.0/nsim);
//     }
//     
//     
//   }
//   
//   return ll;
// }
// 
// void gssm::backtrack_pf2(const arma::cube& alpha, arma::mat& w, const arma::umat& ind) {
//   
//   unsigned int nsim = alpha.n_slices;
//   w.col(n-1) = w.col(n-1) / arma::sum(w.col(n-1));
//   
//   for (int t = n - 1; t > 0; t--) {
//     arma::mat B(nsim, nsim);
//     arma::vec wnorm = w.col(t-1) / arma::sum(w.col(t-1));
//     for (unsigned int i = 0; i < nsim; i++) {
//       for (unsigned int j = 0; j < nsim; j++) {
//         B(j, i) = wnorm(j) * dmvnorm(alpha.slice(i).col(t),
//           C.col((t-1) * Ctv) + T.slice((t-1) * Ttv) * alpha.slice(j).col(t - 1),
//           R.slice((t-1) * Rtv), true, false);
//       }
//     }
//     B.each_row() /= arma::sum(B, 0);
//     w.col(t-1) = B * w.col(t);
//   }
// }
// 
// arma::mat gssm::backward_simulate(arma::cube& alpha, arma::mat& w, arma::umat& ind) {
//   
//   unsigned int nsim = alpha.n_slices;
//   arma::vec I(n);
//   arma::vec wnorm = w.col(n-1) / arma::sum(w.col(n-1));
//   std::discrete_distribution<> sample(wnorm.begin(), wnorm.end());
//   arma::mat alphasim(m, n);
//   I(n-1) = sample(engine);
//   alphasim.col(n - 1) = alpha.slice(I(n - 1)).col(n - 1);
//   for (int t = n - 1; t > 0; t--) {
//     arma::vec b(nsim);
//     arma::vec wnorm = w.col(t-1) / arma::sum(w.col(t-1));
//     for (unsigned int j = 0; j < nsim; j++) {
//       b(j) = wnorm(j) * dmvnorm(alpha.slice(I(t)).col(t),
//         C.col((t-1) * Ctv) + T.slice((t-1) * Ttv) * alpha.slice(j).col(t - 1),
//         R.slice((t-1) * Rtv), true, false);
//     }
//     b /= arma::sum(b);
//     std::discrete_distribution<> sample(b.begin(), b.end());
//     I(t-1) = sample(engine);
//     alphasim.col(t - 1) = alpha.slice(I(t - 1)).col(t - 1);
//   }
//   return alphasim;
// }
// 
// 
// //psi-auxiliary particle filter
// double gssm::psi_filter(unsigned int nsim, arma::cube& alphasim, arma::mat& w, 
//   arma::umat& ind) {
//   
//   arma::mat alphahat(m, n);
//   arma::cube Vt(m, m, n, arma::fill::ones);
//   arma::cube Ct(m, m, n);
//   double ll = log_likelihood(true);
//   smoother_ccov(alphahat, Vt, Ct, true);
//   conditional_dist_helper(Vt, Ct);
//   std::normal_distribution<> normal(0.0, 1.0);
//   std::uniform_real_distribution<> unif(0.0, 1.0);
//   for (unsigned int i = 0; i < nsim; i++) {
//     arma::vec um(m);
//     for(unsigned int j = 0; j < m; j++) {
//       um(j) = normal(engine);
//     }
//     alphasim.slice(i).col(0) = alphahat.col(0) + Vt.slice(0) * um;
//   }
//   
//   for (unsigned int t = 0; t < (n - 1); t++) {
//     
//     for (unsigned int i = 0; i < nsim; i++) {
//       arma::vec um(m);
//       for(unsigned int j = 0; j < m; j++) {
//         um(j) = normal(engine);
//       }
//       alphasim.slice(i).col(t + 1) = alphahat.col(t + 1) + 
//         Ct.slice(t + 1) * (alphasim.slice(i).col(t) - alphahat.col(t)) + Vt.slice(t + 1) * um;
//     }
//   }
//   
//   return ll;
// }
