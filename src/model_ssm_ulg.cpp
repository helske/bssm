#include "model_ssm_ulg.h"
#include "rep_mat.h"
#include "sample.h"
#include "distr_consts.h"
#include "conditional_dist.h"
#include "psd_chol.h"

// General constructor of ssm_ulg object from Rcpp::List
ssm_ulg::ssm_ulg(const Rcpp::List model,
  const unsigned int seed,
  const double zero_tol) 
  :
    y(Rcpp::as<arma::vec>(model["y"])), 
    Z(Rcpp::as<arma::mat>(model["Z"])),
    H(Rcpp::as<arma::vec>(model["H"])), 
    T(Rcpp::as<arma::cube>(model["T"])),
    R(Rcpp::as<arma::cube>(model["R"])), 
    a1(Rcpp::as<arma::vec>(model["a1"])),
    P1(Rcpp::as<arma::mat>(model["P1"])), 
    D(Rcpp::as<arma::vec>(model["D"])),
    C(Rcpp::as<arma::mat>(model["C"])),
    xreg(Rcpp::as<arma::mat>(model["xreg"])),
    beta(Rcpp::as<arma::vec>(model["beta"])),
    n(y.n_elem), m(a1.n_elem), k(R.n_cols),
    Ztv(Z.n_cols > 1), Htv(H.n_elem > 1), Ttv(T.n_slices > 1), Rtv(R.n_slices > 1),
    Dtv(D.n_elem > 1), Ctv(C.n_cols > 1),
    theta(Rcpp::as<arma::vec>(model["theta"])), 
    engine(seed), zero_tol(zero_tol),
    HH(arma::vec(Htv * (n - 1) + 1)), RR(arma::cube(m, m, Rtv * (n - 1) + 1)),
    xbeta(arma::vec(n, arma::fill::zeros)) {
  
  if(xreg.n_cols > 0) {
    compute_xbeta();
  }
  compute_HH();
  compute_RR();
}

// General constructor of ssm_ulg object using arma objects for ng-models
ssm_ulg::ssm_ulg(
  const arma::vec& y, const arma::mat& Z, const arma::vec& H,
  const arma::cube& T, const arma::cube& R, 
  const arma::vec& a1, const arma::mat& P1,
  const arma::vec& D, const arma::mat& C, 
  const arma::mat& xreg, const arma::vec& beta,
  const arma::vec& theta, const unsigned int seed,
  const double zero_tol) :
  y(y), Z(Z), H(H), T(T), R(R), a1(a1), P1(P1), D(D), C(C), 
  xreg(xreg), beta(beta), n(y.n_elem), m(a1.n_elem), k(R.n_cols),
  Ztv(Z.n_cols > 1), Htv(H.n_elem > 1), Ttv(T.n_slices > 1), Rtv(R.n_slices > 1),
  Dtv(D.n_elem > 1), Ctv(C.n_cols > 1),
  theta(theta), engine(seed), zero_tol(zero_tol), 
  HH(arma::vec(Htv * (n - 1) + 1)), RR(arma::cube(m, m, Rtv * (n - 1) + 1)),
  xbeta(arma::vec(n, arma::fill::zeros)) {
  
  if(xreg.n_cols > 0) {
    compute_xbeta();
  }
  compute_HH();
  compute_RR();
}


void ssm_ulg::update_model(const arma::vec& new_theta, const Rcpp::Function update_fn) {
  
  Rcpp::List model_list =
    update_fn(Rcpp::NumericVector(new_theta.begin(), new_theta.end()));
  if (model_list.containsElementNamed("Z")) {
    Z = Rcpp::as<arma::mat>(model_list["Z"]);
  }
  if (model_list.containsElementNamed("H")) {
    H = Rcpp::as<arma::vec>(model_list["H"]);
    compute_HH();
  }
  if (model_list.containsElementNamed("T")) {
    T = Rcpp::as<arma::cube>(model_list["T"]);
  }
  if (model_list.containsElementNamed("R")) {
    R = Rcpp::as<arma::cube>(model_list["R"]);
    compute_RR();
  }
  if (model_list.containsElementNamed("a1")) {
    a1 = Rcpp::as<arma::vec>(model_list["a1"]);
  }
  if (model_list.containsElementNamed("P1")) {
    P1 = Rcpp::as<arma::mat>(model_list["P1"]);
  }
  if (model_list.containsElementNamed("D")) {
    D = Rcpp::as<arma::vec>(model_list["D"]);
  }
  if (model_list.containsElementNamed("C")) {
    C = Rcpp::as<arma::mat>(model_list["C"]);
  }
  
  if (model_list.containsElementNamed("beta")) {
    beta = Rcpp::as<arma::vec>(model_list["beta"]);
    compute_xbeta();
  }
  theta = new_theta;
}

double ssm_ulg::log_prior_pdf(const arma::vec& x, const Rcpp::Function prior_fn) const {
  return Rcpp::as<double>(prior_fn(Rcpp::NumericVector(x.begin(), x.end())));
}

double ssm_ulg::log_likelihood() const {
  
  double logLik = 0;
  if(arma::accu(H) + arma::accu(R) < zero_tol) {
    logLik = -std::numeric_limits<double>::infinity();
  } else {
    
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
  }
  return logLik;
}


arma::cube ssm_ulg::simulate_states(const unsigned int nsim, const bool use_antithetic) {
  
  arma::vec y_tmp = y;
  
  arma::mat L_P1 = psd_chol(P1);
  
  arma::cube asim(m, n + 1, nsim);
  
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
      arma::mat aplus(m, n + 1);
      
      arma::vec um(m);
      for(unsigned int j = 0; j < m; j++) {
        um(j) = normal(engine);
      }
      aplus.col(0) = a1 + L_P1 * um;
      for (unsigned int t = 0; t < n; t++) {
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
      
      asim.slice(i) = -fast_smoother(Ft, Kt, Lt) + aplus;
      if (use_antithetic){
        asim.slice(i + nsim2) = alphahat - asim.slice(i);
      }
      asim.slice(i) += alphahat;
    }
    if ((2 * nsim2) < nsim) {
      
      arma::mat aplus(m, n + 1);
      
      arma::vec um(m);
      for(unsigned int j = 0; j < m; j++) {
        um(j) = normal(engine);
      }
      aplus.col(0) = a1 + L_P1 * um;
      for (unsigned int t = 0; t < n; t++) {
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
    for (unsigned int t = 0; t < n; t++) {
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
    asim.slice(0) += fast_smoother();
  }
 
  y = y_tmp;
  
  return asim;
}

/* Fast state smoothing, only returns smoothed estimates of states
 * which are needed in simulation smoother and Laplace approximation
 */
arma::mat ssm_ulg::fast_smoother() const {
  
  arma::mat at(m, n + 1);
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
  
  for (unsigned int t = 0; t < n; t++) {
    Ft(t) = arma::as_scalar(Z.col(t * Ztv).t() * Pt * Z.col(t * Ztv) + HH(t * Htv));
    if (arma::is_finite(y_tmp(t)) && Ft(t) > zero_tol) {
      Kt.col(t) = Pt * Z.col(t * Ztv) / Ft(t);
      vt(t) = arma::as_scalar(y_tmp(t) - D(t * Dtv) - Z.col(t * Ztv).t() * at.col(t));
      at.col(t + 1) = C.col(t * Ctv) + T.slice(t * Ttv) * (at.col(t) + Kt.col(t) * vt(t));
      //Pt = arma::symmatu(T.slice(t * Ttv) * (Pt - Kt.col(t) * Kt.col(t).t() * Ft(t)) * T.slice(t * Ttv).t() + RR.slice(t * Rtv));
      // Switched to numerically better form
      arma::mat tmp = arma::eye(m, m) - Kt.col(t) * Z.col(t * Ztv).t();
      Pt = arma::symmatu(T.slice(t * Ttv) * (tmp * Pt * tmp.t() + Kt.col(t) * HH(t * Htv) * Kt.col(t).t()) * T.slice(t * Ttv).t() + RR.slice(t * Rtv));
    } else {
      at.col(t + 1) = C.col(t * Ctv) + T.slice(t * Ttv) * at.col(t);
      Pt = arma::symmatu(T.slice(t * Ttv) * Pt * T.slice(t * Ttv).t() + RR.slice(t * Rtv));
    }
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
arma::mat ssm_ulg::fast_smoother(const arma::vec& Ft, const arma::mat& Kt,
  const arma::cube& Lt) const {
  
  arma::mat at(m, n + 1);
  arma::mat Pt(m, m);
  
  arma::vec vt(n);
  
  at.col(0) = a1;
  Pt = P1;
  
  arma::vec y_tmp = y;
  if (xreg.n_cols > 0) {
    y_tmp -= xbeta;
  }
  
  for (unsigned int t = 0; t < n; t++) {
    if (arma::is_finite(y_tmp(t)) && Ft(t) > zero_tol) {
      vt(t) = arma::as_scalar(y_tmp(t) - D(t * Dtv) - Z.col(t * Ztv).t() * at.col(t));
      at.col(t + 1) = C.col(t * Ctv) + T.slice(t * Ttv) * (at.col(t) + Kt.col(t) * vt(t));
    } else {
      at.col(t + 1) = C.col(t * Ctv) + T.slice(t * Ttv) * at.col(t);
    }
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

arma::mat ssm_ulg::fast_precomputing_smoother(arma::vec& Ft, arma::mat& Kt,
  arma::cube& Lt) const {
  
  arma::mat at(m, n + 1);
  arma::mat Pt(m, m);
  arma::vec vt(n);
  
  at.col(0) = a1;
  Pt = P1;
  
  arma::vec y_tmp = y;
  if (xreg.n_cols > 0) {
    y_tmp -= xbeta;
  }
  for (unsigned int t = 0; t < n; t++) {
    Ft(t) = arma::as_scalar(Z.col(t * Ztv).t() * Pt * Z.col(t * Ztv) + HH(t * Htv));
    if (arma::is_finite(y_tmp(t)) && Ft(t) > zero_tol) {
      Kt.col(t) = Pt * Z.col(t * Ztv) / Ft(t);
      vt(t) = arma::as_scalar(y_tmp(t) - D(t * Dtv) - Z.col(t * Ztv).t() * at.col(t));
      at.col(t + 1) = C.col(t * Ctv) + T.slice(t * Ttv) * (at.col(t) + Kt.col(t) * vt(t));
      //Pt = arma::symmatu(T.slice(t * Ttv) * (Pt - Kt.col(t) * Kt.col(t).t() * Ft(t)) * T.slice(t * Ttv).t() + RR.slice(t * Rtv));
      // Switched to numerically better form
      arma::mat tmp = arma::eye(m, m) - Kt.col(t) * Z.col(t * Ztv).t();
      Pt = arma::symmatu(T.slice(t * Ttv) * (tmp * Pt * tmp.t() + Kt.col(t) * HH(t * Htv) * Kt.col(t).t()) * T.slice(t * Ttv).t() + RR.slice(t * Rtv));
    } else {
      at.col(t + 1) = C.col(t * Ctv) + T.slice(t * Ttv) * at.col(t);
      Pt = arma::symmatu(T.slice(t * Ttv) * Pt * T.slice(t * Ttv).t() + RR.slice(t * Rtv));
    }
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
void ssm_ulg::smoother_ccov(arma::mat& at, arma::cube& Pt, arma::cube& ccov) const {
  
  at.col(0) = a1;
  Pt.slice(0) = P1;
  arma::vec vt(n);
  arma::vec Ft(n);
  arma::mat Kt(m, n);
  
  arma::vec y_tmp = y;
  if(xreg.n_cols > 0) {
    y_tmp -= xbeta;
  }
  
  for (unsigned int t = 0; t < n; t++) {
    Ft(t) = arma::as_scalar(Z.col(t * Ztv).t() * Pt.slice(t) * Z.col(t * Ztv) +
      HH(t * Htv));
    if (arma::is_finite(y_tmp(t)) && Ft(t) > zero_tol) {
      Kt.col(t) = Pt.slice(t) * Z.col(t * Ztv) / Ft(t);
      vt(t) = arma::as_scalar(y_tmp(t) - D(t * Dtv) - Z.col(t * Ztv).t() * at.col(t));
      at.col(t + 1) = C.col(t * Ctv) + T.slice(t * Ttv) * (at.col(t) + Kt.col(t) * vt(t));
      //Pt.slice(t + 1) = arma::symmatu(T.slice(t * Ttv) * (Pt.slice(t) -
      //  Kt.col(t) * Kt.col(t).t() * Ft(t)) * T.slice(t * Ttv).t() + RR.slice(t * Rtv));
      // Switched to numerically better form
      arma::mat tmp = arma::eye(m, m) - Kt.col(t) * Z.col(t * Ztv).t();
      Pt.slice(t + 1) = arma::symmatu(T.slice(t * Ttv) * (tmp * Pt.slice(t) * tmp.t() + Kt.col(t) * HH(t * Htv) * Kt.col(t).t()) * T.slice(t * Ttv).t() + RR.slice(t * Rtv));
    } else {
      at.col(t + 1) = C.col(t * Ctv) + T.slice(t * Ttv) * at.col(t);
      Pt.slice(t + 1) = arma::symmatu(T.slice(t * Ttv) * Pt.slice(t) * T.slice(t * Ttv).t() +
        RR.slice(t * Rtv));
    }
    ccov.slice(t) = Pt.slice(t+1); //store for smoothing;
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
  ccov.slice(n).zeros();
}

double ssm_ulg::filter(arma::mat& at, arma::mat& att, arma::cube& Pt,
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
      arma::mat tmp = arma::eye(m, m) - K * Z.col(t * Ztv).t();
      Ptt.slice(t) = tmp * Pt.slice(t) * tmp.t() + K * HH(t * Htv) * K.t();
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

void ssm_ulg::smoother(arma::mat& at, arma::cube& Pt) const {
  
  at.col(0) = a1;
  Pt.slice(0) = P1;
  arma::vec vt(n);
  arma::vec Ft(n);
  arma::mat Kt(m, n);
  
  arma::vec y_tmp = y;
  if (xreg.n_cols > 0) {
    y_tmp -= xbeta;
  }
  
  for (unsigned int t = 0; t < n; t++) {
    Ft(t) = arma::as_scalar(Z.col(t * Ztv).t() * Pt.slice(t) * Z.col(t * Ztv) +
      HH(t * Htv));
    if (arma::is_finite(y_tmp(t)) && Ft(t) > zero_tol) {
      Kt.col(t) = Pt.slice(t) * Z.col(t * Ztv) / Ft(t);
      vt(t) = arma::as_scalar(y_tmp(t) - D(t * Dtv) - Z.col(t * Ztv).t() * at.col(t));
      at.col(t + 1) = C.col(t * Ctv) + T.slice(t * Ttv) * (at.col(t) + Kt.col(t) * vt(t));
      arma::mat tmp = arma::eye(m, m) - Kt.col(t) * Z.col(t * Ztv).t();
      Pt.slice(t + 1) = arma::symmatu(T.slice(t * Ttv) * (tmp * Pt.slice(t) * tmp.t() + Kt.col(t) * HH(t * Htv) * Kt.col(t).t()) * T.slice(t * Ttv).t() + RR.slice(t * Rtv));
    } else {
      at.col(t + 1) = C.col(t * Ctv) + T.slice(t * Ttv) * at.col(t);
      Pt.slice(t + 1) = arma::symmatu(T.slice(t * Ttv) * Pt.slice(t) * T.slice(t * Ttv).t() +
        RR.slice(t * Rtv));
    }
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

double ssm_ulg::bsf_filter(const unsigned int nsim, arma::cube& alpha,
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
  for (unsigned int t = 0; t < n; t++) {
    
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
    
    if ((t < (n - 1)) && arma::is_finite(y(t + 1))) {
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


void ssm_ulg::psi_filter(const unsigned int nsim, arma::cube& alpha) {
  
  arma::mat alphahat(m, n + 1);
  arma::cube Vt(m, m, n + 1);
  arma::cube Ct(m, m, n + 1);
  smoother_ccov(alphahat, Vt, Ct);
  conditional_cov(Vt, Ct);
  
  std::normal_distribution<> normal(0.0, 1.0);
  for (unsigned int i = 0; i < nsim; i++) {
    arma::vec um(m);
    for(unsigned int j = 0; j < m; j++) {
      um(j) = normal(engine);
    }
    alpha.slice(i).col(0) = alphahat.col(0) + Vt.slice(0) * um;
  }
  
  for (unsigned int t = 0; t < n; t++) {
    for (unsigned int i = 0; i < nsim; i++) {
      arma::vec um(m);
      for(unsigned int j = 0; j < m; j++) {
        um(j) = normal(engine);
      }
      alpha.slice(i).col(t + 1) = alphahat.col(t + 1) + Ct.slice(t + 1) * (alpha.slice(i).col(t) - alphahat.col(t)) + Vt.slice(t + 1) * um;
    }
  }
}


arma::cube ssm_ulg::predict_sample(const arma::mat& theta_posterior,
  const arma::mat& alpha, const unsigned int predict_type, const Rcpp::Function update_fn) {
  
  unsigned int d = 1;
  if (predict_type == 3) d = m;
  
  unsigned int n_samples = theta_posterior.n_cols;
  arma::cube sample(d, n, n_samples);
  
  for (unsigned int i = 0; i < n_samples; i++) {
    update_model(theta_posterior.col(i), update_fn);
    a1 = alpha.col(i);
    sample.slice(i) = sample_model(predict_type);
  }
  
  return sample;
}


arma::mat ssm_ulg::sample_model(const unsigned int predict_type) {
  
  arma::mat alpha(m, n);
  
  std::normal_distribution<> normal(0.0, 1.0);
  alpha.col(0) = a1;
  
  for (unsigned int t = 0; t < (n - 1); t++) {
    arma::vec uk(k);
    for(unsigned int j = 0; j < k; j++) {
      uk(j) = normal(engine);
    }
    alpha.col(t + 1) = C.col(t * Ctv) + 
      T.slice(t * Ttv) * alpha.col(t) + R.slice(t * Rtv) * uk;
  }
  
  if (predict_type < 3) {
    arma::mat y(1, n);
    
    for (unsigned int t = 0; t < n; t++) {
      y(0, t) = xbeta(t) + D(t * Dtv) +
        arma::as_scalar(Z.col(t * Ztv).t() * alpha.col(t));
      if(predict_type == 1)
        y(0, t) += H(t * Htv) * normal(engine);
    }
    
    return y;
  } else {
    return alpha;
  }
}


arma::cube ssm_ulg::predict_past(const arma::mat& theta_posterior,
  const arma::cube& alpha, const unsigned int predict_type, const Rcpp::Function update_fn) {
  
  unsigned int n_samples = theta_posterior.n_cols;
  arma::cube samples(p, n, n_samples);
  
  std::normal_distribution<> normal(0.0, 1.0);
  for (unsigned int i = 0; i < n_samples; i++) {
    update_model(theta_posterior.col(i), update_fn);
    for (unsigned int t = 0; t < n; t++) {
      samples.slice(i).col(t) =  xbeta(t) + D(t * Dtv) +
        arma::as_scalar(Z.col(t * Ztv).t() * alpha.slice(i).col(t));
      if(predict_type == 1) {
        arma::vec up(p);
        for(unsigned int j = 0; j < p; j++) {
          up(j) = normal(engine);
        }
        samples.slice(i).col(t) += H(t * Htv) * up;
      }
    }
  }
  return samples;
}
