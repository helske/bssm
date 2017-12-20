#include "ugg_ssm.h"
#include "interval.h"
#include "rep_mat.h"
#include "sample.h"
#include "distr_consts.h"
#include "dmvnorm.h"
#include "psd_chol.h"

// General constructor of ugg_ssm object from Rcpp::List
// with parameter indices
ugg_ssm::ugg_ssm(const Rcpp::List& model,
  const unsigned int seed,
  const arma::uvec& Z_ind_, const arma::uvec& H_ind_,
  const arma::uvec& T_ind_, const arma::uvec& R_ind_) :
  y(Rcpp::as<arma::vec>(model["y"])), Z(Rcpp::as<arma::mat>(model["Z"])),
  H(Rcpp::as<arma::vec>(model["H"])), T(Rcpp::as<arma::cube>(model["T"])),
  R(Rcpp::as<arma::cube>(model["R"])), a1(Rcpp::as<arma::vec>(model["a1"])),
  P1(Rcpp::as<arma::mat>(model["P1"])), xreg(Rcpp::as<arma::mat>(model["xreg"])),
  beta(Rcpp::as<arma::vec>(model["coefs"])),
  D(Rcpp::as<arma::vec>(model["obs_intercept"])),
  C(Rcpp::as<arma::mat>(model["state_intercept"])),
  theta(Rcpp::as<arma::vec>(model["theta"])), 
  prior_distributions(Rcpp::as<arma::uvec>(model["prior_distributions"])), 
  prior_parameters(Rcpp::as<arma::mat>(model["prior_parameters"])),
  Ztv(Z.n_cols > 1), Htv(H.n_elem > 1), Ttv(T.n_slices > 1), Rtv(R.n_slices > 1),
  Dtv(D.n_elem > 1), Ctv(C.n_cols > 1), n(y.n_elem), m(a1.n_elem), k(R.n_cols),
  HH(arma::vec(Htv * (n - 1) + 1)), RR(arma::cube(m, m, Rtv * (n - 1) + 1)),
  xbeta(arma::vec(n, arma::fill::zeros)), engine(seed), zero_tol(1e-8),
  Z_ind(Z_ind_), H_ind(H_ind_), T_ind(T_ind_), R_ind(R_ind_) {
  
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
  const arma::vec& D, const arma::mat& C, 
  const unsigned int seed,
  const arma::vec& theta,
  const arma::uvec& prior_distributions,
  const arma::mat& prior_parameters,  const arma::uvec& Z_ind_,
  const arma::uvec& H_ind_, const arma::uvec& T_ind_, const arma::uvec& R_ind_) :
  y(y), Z(Z), H(H), T(T), R(R), a1(a1), P1(P1), xreg(xreg), beta(beta), D(D), C(C),
  Ztv(Z.n_cols > 1), Htv(H.n_elem > 1), Ttv(T.n_slices > 1), Rtv(R.n_slices > 1),
  Dtv(D.n_elem > 1), Ctv(C.n_cols > 1), n(y.n_elem), m(a1.n_elem), k(R.n_cols),
  HH(arma::vec(Htv * (n - 1) + 1)), RR(arma::cube(m, m, Rtv * (n - 1) + 1)),
  xbeta(arma::vec(n, arma::fill::zeros)), 
  theta(theta), prior_distributions(prior_distributions), 
  prior_parameters(prior_parameters),
  engine(seed), zero_tol(1e-8),
  Z_ind(Z_ind_), H_ind(H_ind_), T_ind(T_ind_), R_ind(R_ind_) {
  
  if(xreg.n_cols > 0) {
    compute_xbeta();
  }
  compute_HH();
  compute_RR();
}

void ugg_ssm::update_model(const arma::vec& new_theta) {
  
  if (Z_ind.n_elem > 0) {
    Z.elem(Z_ind) = new_theta.subvec(0, Z_ind.n_elem - 1);
  }
  if (H_ind.n_elem > 0) {
    H.elem(H_ind) = new_theta.subvec(Z_ind.n_elem, Z_ind.n_elem + H_ind.n_elem - 1);
  }
  if (T_ind.n_elem > 0) {
    T.elem(T_ind) = new_theta.subvec(Z_ind.n_elem + H_ind.n_elem,
      Z_ind.n_elem + H_ind.n_elem + T_ind.n_elem - 1);
  }
  if (R_ind.n_elem > 0) {
    R.elem(R_ind) = new_theta.subvec(Z_ind.n_elem + H_ind.n_elem + T_ind.n_elem,
      Z_ind.n_elem + H_ind.n_elem + T_ind.n_elem + R_ind.n_elem - 1);
  }
  
  if (H_ind.n_elem  > 0) {
    compute_HH();
  }
  if (R_ind.n_elem  > 0) {
    compute_RR();
  }
  if(xreg.n_cols > 0) {
    beta = new_theta.subvec(new_theta.n_elem - xreg.n_cols, new_theta.n_elem - 1);
    compute_xbeta();
  }
  theta = new_theta;
}

double ugg_ssm::log_prior_pdf(const arma::vec& x) const {
  
  double log_prior = 0.0;
  
  for(unsigned int i = 0; i < x.n_elem; i++) {
    switch(prior_distributions(i)) {
    case 0  :
      if (x(i) < prior_parameters(0, i) || x(i) > prior_parameters(1, i)) {
        return -std::numeric_limits<double>::infinity(); 
      }
      break;
    case 1  :
      if (x(i) < 0) {
        return -std::numeric_limits<double>::infinity();
      } else {
        log_prior -= 0.5 * std::pow(x(i) / prior_parameters(0, i), 2);
      }
      break;
    case 2  :
      log_prior -= 0.5 * std::pow((x(i) - prior_parameters(0, i)) / prior_parameters(1, i), 2);
      break;
    }
  }
  return log_prior;
}

double ugg_ssm::log_proposal_ratio(const arma::vec& new_theta, const arma::vec& old_theta) const {
  return 0.0;
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
      logLik -= 0.5 * (LOG2PI + std::log(F) + v * v/F);
    } else {
      at = C.col(t * Ctv) + T.slice(t * Ttv) * at;
      Pt = arma::symmatu(T.slice(t * Ttv) * Pt * T.slice(t * Ttv).t() + RR.slice(t * Rtv));
    }
  }
  
  return logLik;
}


arma::cube ugg_ssm::simulate_states(const unsigned int nsim, const bool use_antithetic) {
  
  arma::vec y_tmp = y;
  
  arma::mat L_P1 = psd_chol(P1);
  
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
      logLik -= 0.5 * (LOG2PI + std::log(F) + v * v/F);
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

Rcpp::List ugg_ssm::predict_interval(const arma::vec& probs, const arma::mat& theta,
  const arma::mat& alpha, const arma::uvec& counts, const unsigned int predict_type) {
  
  arma::vec theta_i = theta.col(0);
  update_model(theta_i);
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
      update_model(theta_i);
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
    
    arma::mat expanded_sd = rep_mat(arma::sqrt(var_pred), counts);
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
      update_model(theta_i);
      a1 = alpha.col(i);
      P1.zeros();
      filter(at, att, Pt, Ptt);
      
      for(unsigned int t = 0; t < n; t++) {
        mean_pred.tube(t, i) = at.col(t);
        var_pred.tube(t, i) = Pt.slice(t).diag();
      }
    }
    
    arma::cube intv(n, probs.n_elem, m);
    arma::cube expanded_sd(arma::accu(counts), n, m);
    arma::cube expanded_mean(arma::accu(counts), n, m);
    for (unsigned int i = 0; i < m; i++) {
      arma::mat tmp1 = rep_mat(arma::sqrt(var_pred.slice(i)), counts);
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
  const arma::mat& alpha, const arma::uvec& counts, const unsigned int predict_type, 
  const unsigned int nsim) {
  
  unsigned int d = 1;
  if (predict_type == 3) d = m;
  
  arma::mat expanded_theta = rep_mat(theta, counts);
  arma::mat expanded_alpha = rep_mat(alpha, counts);
  unsigned int n_samples = expanded_theta.n_cols;
  arma::cube sample(d, n, nsim * n_samples);
  
  for (unsigned int i = 0; i < n_samples; i++) {
    arma::vec theta_i = theta.col(i);
    update_model(theta_i);
    a1 = alpha.col(i);
    sample.slices(i * nsim, (i + 1) * nsim - 1) = sample_model(predict_type, nsim);
  }
  
  return sample;
}


arma::mat ugg_ssm::sample_model(const unsigned int predict_type, const unsigned int nsim) {
  
  arma::cube alpha(m, n, nsim);
  
  std::normal_distribution<> normal(0.0, 1.0);
  for (unsigned int i = 0; i < nsim; i++) {
    
    alpha.slice(i).col(0) = a1;
    
    for (unsigned int t = 0; t < (n - 1); t++) {
      arma::vec uk(k);
      for(unsigned int j = 0; j < k; j++) {
        uk(j) = normal(engine);
      }
      alpha.slice(i).col(t + 1) = C.col(t * Ctv) + 
        T.slice(t * Ttv) * alpha.slice(i).col(t) + R.slice(t * Rtv) * uk;
    }
  }
  
  if (predict_type < 3) {
    arma::cube y(1, n, nsim);
    
    for (unsigned int i = 0; i < nsim; i++) {
      for (unsigned int t = 0; t < n; t++) {
        y(0, t, i) = xbeta(t) + D(t * Dtv) +
          arma::as_scalar(Z.col(t * Ztv).t() * alpha.slice(i).col(t));
        if(predict_type == 1)
          y(0, t, i) += H(t * Htv) * normal(engine);
      }
    }
    return y;
  } else {
    return alpha;
  }
}


double ugg_ssm::bsf_filter(const unsigned int nsim, arma::cube& alpha,
  arma::mat& weights, arma::umat& indices) {
  
  arma::mat L_P1 = psd_chol(P1);
  
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
      double mu = arma::as_scalar(D(0) + Z.col(0).t() *
        alpha.slice(i).col(0));
      weights(i, 0) = -0.5 * std::pow(y(0) - mu, 2.0) / HH(0);
    }
    double max_weight = weights.col(0).max();
    weights.col(0) = arma::exp(weights.col(0) - max_weight);
    double sum_weights = arma::accu(weights.col(0));
    if(sum_weights > 0.0){
      normalized_weights = weights.col(0) / sum_weights;
    } else {
      return -std::numeric_limits<double>::infinity();
    }
    loglik = max_weight + std::log(sum_weights / nsim) + norm_log_const(H(0));
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
        double mu = arma::as_scalar(D((t + 1) * Dtv) + Z.col(Ztv * (t + 1)).t() *
          alpha.slice(i).col(t + 1));
        weights(i, t + 1) = -0.5 * std::pow(y(t + 1) - mu, 2.0) / HH(Htv * (t + 1));
      }
      
      double max_weight = weights.col(t + 1).max();
      weights.col(t + 1) = arma::exp(weights.col(t + 1) - max_weight);
      double sum_weights = arma::accu(weights.col(t + 1));
      if(sum_weights > 0.0){
        normalized_weights = weights.col(t + 1) / sum_weights;
      } else {
        return -std::numeric_limits<double>::infinity();
      }
      loglik += max_weight + std::log(sum_weights / nsim) +
        norm_log_const(H(Htv * (t + 1)));
    } else {
      weights.col(t + 1).ones();
      normalized_weights.fill(1.0/nsim);
    }
  }
  
  return loglik;
}

double ugg_ssm::aux_filter(const unsigned int nsim, arma::cube& alpha,
  arma::mat& weights, arma::umat& indices) {
  
  arma::mat L_P1 = psd_chol(P1);
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
      double mu = arma::as_scalar(D(0) + Z.col(0).t() * alpha.slice(i).col(0));
      weights(i, 0) = -0.5 * std::pow(y(0) - mu, 2.0) / HH(0);
    }
    double max_weight = weights.col(0).max();
    weights.col(0) = arma::exp(weights.col(0) - max_weight);
    double sum_weights = arma::accu(weights.col(0));
    
    if(sum_weights > 0.0){
      normalized_weights = weights.col(0) / sum_weights;
    } else {
      return -std::numeric_limits<double>::infinity();
    }
    loglik = max_weight + std::log(sum_weights / nsim) + norm_log_const(H(0));
  } else {
    weights.col(0).ones();
    normalized_weights.fill(1.0 / nsim);
  }
  
  for (unsigned int t = 0; t < (n - 1); t++) {
    
    arma::vec r(nsim);
    for (unsigned int i = 0; i < nsim; i++) {
      r(i) = unif(engine);
    }
    
    arma::uvec indices_init = stratified_sample(normalized_weights, r, nsim);
    
    arma::mat alphatmp_init(m, nsim);
    arma::vec aux_weights(nsim);
    for (unsigned int i = 0; i < nsim; i++) {
      alphatmp_init.col(i) = alpha.slice(indices_init(i)).col(t);
      double mu = arma::as_scalar(D((t + 1) * Dtv) + Z.col(Ztv * (t + 1)).t() *
        (C.col(t * Ctv) + T.slice(Ttv * t) * alphatmp_init.col(i)));
      aux_weights(i) = -0.5 * std::pow(y(t + 1) - mu, 2.0) / HH(Htv * (t + 1));
    }
    
    double max_aux_weight = aux_weights.max();
    arma::vec normalized_aux_weights = arma::exp(aux_weights-max_aux_weight);
    double sum_aux_weights = arma::accu(normalized_aux_weights);
    normalized_aux_weights /= sum_aux_weights;
    for (unsigned int i = 0; i < nsim; i++) {
      r(i) = unif(engine);
    }
    indices.col(t) = stratified_sample(normalized_aux_weights, r, nsim);
    
    arma::mat alphatmp(m, nsim);
    arma::vec sampled_weights(nsim);
    for (unsigned int i = 0; i < nsim; i++) {
      alphatmp.col(i) = alphatmp_init.col(indices(i, t));
      sampled_weights(i) = aux_weights(indices(i, t));
      indices(i, t) = indices_init(indices(i, t));
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
        double mu = arma::as_scalar(D((t + 1) * Dtv) + Z.col(Ztv * (t + 1)).t() *
          alpha.slice(i).col(t + 1));
        weights(i, t + 1) = -0.5 * std::pow(y(t + 1) - mu, 2.0) / HH(Htv * (t + 1)) -
          sampled_weights(i);
      }
      double max_weight = weights.col(t + 1).max();
      weights.col(t + 1) = arma::exp(weights.col(t + 1) - max_weight);
      double sum_weights = arma::accu(weights.col(t + 1));
      if(sum_weights > 0.0){
        normalized_weights = weights.col(t + 1) / sum_weights;
      } else {
        return -std::numeric_limits<double>::infinity();
      }
      
      loglik += max_weight + std::log(sum_weights / nsim) + std::log(sum_aux_weights / nsim) +
        max_aux_weight + norm_log_const(H(Htv * (t + 1)));
    } else {
      weights.col(t + 1).ones();
      normalized_weights.fill(1.0 / nsim);
    }
  }
  return loglik;
}
double ugg_ssm::oaux_filter(const unsigned int nsim, arma::cube& alpha,
  arma::mat& weights, arma::umat& indices) {
  
  arma::vec att1(m);
  arma::mat Ptt1(m, m);
  aux_update_step(0, y(0), a1, P1, att1, Ptt1);
  
  arma::mat L_P1 = psd_chol(P1);
  std::normal_distribution<> normal(0.0, 1.0);
  for (unsigned int i = 0; i < nsim; i++) {
    
    arma::vec um(m);
    for(unsigned int j = 0; j < m; j++) {
      um(j) = normal(engine);
    }
    
    alpha.slice(i).col(0) = att1 + L_P1 * um;
    
  }
  
  std::uniform_real_distribution<> unif(0.0, 1.0);
  arma::vec normalized_weights(nsim);
  double loglik = 0.0;
  
  if(arma::is_finite(y(0))) {
    for (unsigned int i = 0; i < nsim; i++) {
      double mu = arma::as_scalar(D(0) + Z.col(0).t() * alpha.slice(i).col(0));
      weights(i, 0) = -0.5 * std::pow(y(0) - mu, 2.0) / HH(0) +
        dmvnorm(alpha.slice(i).col(0), a1, P1, false, true) -
        dmvnorm(alpha.slice(i).col(0), att1, L_P1, true, true);
    }
    double max_weight = weights.col(0).max();
    weights.col(0) = arma::exp(weights.col(0) - max_weight);
    double sum_weights = arma::accu(weights.col(0));
    
    if(sum_weights > 0.0){
      normalized_weights = weights.col(0) / sum_weights;
    } else {
      return -std::numeric_limits<double>::infinity();
    }
    loglik = max_weight + std::log(sum_weights / nsim) + norm_log_const(H(0));
  } else {
    weights.col(0).ones();
    normalized_weights.fill(1.0 / nsim);
  }
  
  for (unsigned int t = 0; t < (n - 1); t++) {
    
    arma::vec r(nsim);
    for (unsigned int i = 0; i < nsim; i++) {
      r(i) = unif(engine);
    }
    
    arma::uvec indices_init = stratified_sample(normalized_weights, r, nsim);
    
    arma::mat alphatmp_init(m, nsim);
    arma::vec aux_weights(nsim);
    arma::mat att(m, nsim);
    arma::cube Ptt(m, m, nsim);
    for (unsigned int i = 0; i < nsim; i++) {
      alphatmp_init.col(i) = alpha.slice(indices_init(i)).col(t);
      arma::mat Pt = RR.slice(Rtv * t);
      arma::vec at = C.col(t * Ctv) + T.slice(Ttv * t) * alphatmp_init.col(i);
      arma::vec tmp(m);
      aux_update_step(t + 1, y(t + 1), at, Pt, tmp, Ptt.slice(i));
      att.col(i) = tmp;
      Ptt.slice(i) = psd_chol(Ptt.slice(i));
      double mu = arma::as_scalar(D((t + 1) * Dtv) + Z.col(Ztv * (t + 1)).t() * tmp);
      aux_weights(i) = -0.5 * std::pow(y(t + 1) - mu, 2.0) / HH(Htv * (t + 1));
      
    }
    
    double max_aux_weight = aux_weights.max();
    arma::vec normalized_aux_weights = arma::exp(aux_weights-max_aux_weight);
    double sum_aux_weights = arma::accu(normalized_aux_weights);
    normalized_aux_weights /= sum_aux_weights;
    for (unsigned int i = 0; i < nsim; i++) {
      r(i) = unif(engine);
    }
    arma::uvec indices_second = stratified_sample(normalized_aux_weights, r, nsim);
    
    arma::mat alphatmp(m, nsim);
    arma::vec sampled_weights(nsim);
    for (unsigned int i = 0; i < nsim; i++) {
      alphatmp.col(i) = alphatmp_init.col(indices_second(i));
      sampled_weights(i) = aux_weights(indices_second(i));
      indices(i, t) = indices_init(indices_second(i));
    }
    
    for (unsigned int i = 0; i < nsim; i++) {
      arma::vec um(m);
      for(unsigned int j = 0; j < m; j++) {
        um(j) = normal(engine);
      }
      alpha.slice(i).col(t + 1) = att.col(indices_second(i)) +
        Ptt.slice(indices_second(i)) * um;
    }
    if(arma::is_finite(y(t + 1))) {
      for (unsigned int i = 0; i < nsim; i++) {
        double mu = arma::as_scalar(D((t + 1) * Dtv) + Z.col(Ztv * (t + 1)).t() *
          alpha.slice(i).col(t + 1));
        weights(i, t + 1) = -0.5 * std::pow(y(t + 1) - mu, 2.0) / HH(Htv * (t + 1)) -
          sampled_weights(i) +
          dmvnorm(alpha.slice(i).col(t + 1),
            C.col(t * Ctv) + T.slice(t * Ttv) * alphatmp.col(i),
            RR.slice(Rtv * t), false, true) -
              dmvnorm(alpha.slice(i).col(t + 1),
                att.col(indices_second(i)), Ptt.slice(indices_second(i)), true, true);
      }
      double max_weight = weights.col(t + 1).max();
      weights.col(t + 1) = arma::exp(weights.col(t + 1) - max_weight);
      double sum_weights = arma::accu(weights.col(t + 1));
      if(sum_weights > 0.0){
        normalized_weights = weights.col(t + 1) / sum_weights;
      } else {
        return -std::numeric_limits<double>::infinity();
      }
      
      loglik += max_weight + std::log(sum_weights / nsim) + std::log(sum_aux_weights / nsim) +
        max_aux_weight + norm_log_const(H(Htv * (t + 1)));
    } else {
      weights.col(t + 1).ones();
      normalized_weights.fill(1.0 / nsim);
    }
  }
  return loglik;
}

void ugg_ssm::aux_update_step(const unsigned int t, const double y,
  const arma::vec& at, const arma::mat& Pt, arma::vec& att, arma::mat& Ptt) const {
  
  double F = arma::as_scalar(Z.col(t * Ztv).t() * Pt * Z.col(t * Ztv) + HH(t * Htv));
  if (arma::is_finite(y) && F > zero_tol) {
    double v = arma::as_scalar(y - D(t * Dtv) - Z.col(t * Ztv).t() * at);
    arma::vec K = Pt * Z.col(t * Ztv) / F;
    att = at + K * v;
    Ptt = Pt - K * K.t() * F;
  } else {
    att = at;
    Ptt = Pt;
  }
}
