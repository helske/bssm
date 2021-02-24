#include "model_ssm_mlg.h"
#include "psd_chol.h"
#include "conditional_dist.h"

// General constructor of ssm_mlg object from Rcpp::List
ssm_mlg::ssm_mlg(
  const Rcpp::List model, 
  const unsigned int seed,
  const double zero_tol) 
  :
    y((Rcpp::as<arma::mat>(model["y"])).t()), Z(Rcpp::as<arma::cube>(model["Z"])),
    H(Rcpp::as<arma::cube>(model["H"])), T(Rcpp::as<arma::cube>(model["T"])),
    R(Rcpp::as<arma::cube>(model["R"])), a1(Rcpp::as<arma::vec>(model["a1"])),
    P1(Rcpp::as<arma::mat>(model["P1"])), D(Rcpp::as<arma::mat>(model["D"])),
    C(Rcpp::as<arma::mat>(model["C"])), 
    n(y.n_cols), m(a1.n_elem), k(R.n_cols), p(y.n_rows), 
    Ztv(Z.n_slices > 1), Htv(H.n_slices > 1), Ttv(T.n_slices > 1), 
    Rtv(R.n_slices > 1), Dtv(D.n_cols > 1), Ctv(C.n_cols > 1), 
    theta(Rcpp::as<arma::vec>(model["theta"])), 
    engine(seed), zero_tol(zero_tol),
    HH(arma::cube(p, p, Htv * (n - 1) + 1)), RR(arma::cube(m, m, Rtv * (n - 1) + 1)) {
  
  compute_HH();
  compute_RR();
  
}

// General constructor of ssm_mlg
ssm_mlg::ssm_mlg(const arma::mat& y, const arma::cube& Z, 
  const arma::cube& H, const arma::cube& T, 
  const arma::cube& R, const arma::vec& a1,
  const arma::mat& P1, const arma::mat& D, 
  const arma::mat& C,
  const arma::vec& theta, const unsigned int seed, 
  const double zero_tol) 
  :
    y(y), Z(Z), H(H), T(T), R(R), a1(a1), P1(P1), D(D), C(C),
    n(y.n_cols), m(a1.n_elem), k(R.n_cols), p(y.n_rows),
    Ztv(Z.n_slices > 1), Htv(H.n_slices > 1), 
    Ttv(T.n_slices > 1), Rtv(R.n_slices > 1),
    Dtv(D.n_cols > 1), Ctv(C.n_cols > 1), 
    theta(theta), engine(seed), zero_tol(zero_tol), 
    HH(arma::cube(p, p, Htv * (n - 1) + 1)), 
    RR(arma::cube(m, m, Rtv * (n - 1) + 1)) {
  
  compute_HH();
  compute_RR();
}

 
void ssm_mlg::update_model(const arma::vec& new_theta, const Rcpp::Function update_fn) {
  
  Rcpp::List model_list = 
    update_fn(Rcpp::NumericVector(new_theta.begin(), new_theta.end()));
  if (model_list.containsElementNamed("Z")) {
    Z = Rcpp::as<arma::cube>(model_list["Z"]);
  }
  if (model_list.containsElementNamed("H")) {
    H = Rcpp::as<arma::cube>(model_list["H"]);
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
    D = Rcpp::as<arma::mat>(model_list["D"]);
  }
  if (model_list.containsElementNamed("C")) {
    C = Rcpp::as<arma::mat>(model_list["C"]);
  }
  
  theta = new_theta;
}

double ssm_mlg::log_prior_pdf(const arma::vec& x, const Rcpp::Function prior_fn) const {
  
  return Rcpp::as<double>(prior_fn(Rcpp::NumericVector(x.begin(), x.end())));
}

double ssm_mlg::log_likelihood() const {
  
  double logLik = 0;
  if(arma::accu(H) + arma::accu(R) < zero_tol) {
    logLik = -std::numeric_limits<double>::infinity();
  } else {
    arma::vec at = a1;
    arma::mat Pt = P1;
    
    const double LOG2PI = std::log(2.0 * M_PI);
    
    for (unsigned int t = 0; t < n; t++) {
      arma::uvec obs_y = arma::find_finite(y.col(t));
      
      if (obs_y.n_elem > 0) {
        
        arma::mat Zt = Z.slice(t * Ztv).rows(obs_y);
        
        arma::mat F = Zt * Pt * Zt.t() + HH.slice(t * Htv).submat(obs_y, obs_y);
        // first check to avoid armadillo warnings
        bool chol_ok = F.is_finite();
        if (!chol_ok) return -std::numeric_limits<double>::infinity();
        arma::mat cholF(p, p);
        chol_ok = arma::chol(cholF, F);
        if (!chol_ok) return -std::numeric_limits<double>::infinity();
        
        arma::vec tmp = y.col(t) - D.col(t * Dtv);
        arma::vec v = tmp.rows(obs_y) - Zt * at;
        arma::mat inv_cholF = arma::inv(arma::trimatu(cholF));
        arma::mat K = Pt * Zt.t() * inv_cholF * inv_cholF.t();
        at = C.col(t * Ctv) + T.slice(t * Ttv) * (at + K * v);
        
        arma::mat IKZ = arma::eye(m, m) - K * Zt;
        Pt = arma::symmatu(T.slice(t * Ttv) * (IKZ * Pt * IKZ.t() + K * HH.slice(t * Htv).submat(obs_y, obs_y) * K.t()) * T.slice(t * Ttv).t() + RR.slice(t * Rtv));
        
        arma::vec Fv = inv_cholF.t() * v;
        logLik -= 0.5 * arma::as_scalar(obs_y.n_elem * LOG2PI +
          2.0 * arma::accu(arma::log(arma::diagvec(cholF))) + Fv.t() * Fv);
        
      } else {
        at = C.col(t * Ctv) + T.slice(t * Ttv) * at;
        Pt = arma::symmatu(T.slice(t * Ttv) * Pt * T.slice(t * Ttv).t() + RR.slice(t * Rtv));
      }
      
    }
  }
  return logLik;
}

// Kalman smoother
void ssm_mlg::smoother(arma::mat& at, arma::cube& Pt) const {
  
  
  at.col(0) = a1;
  Pt.slice(0) = P1;
  
  arma::mat vt(p, n, arma::fill::zeros);
  arma::cube ZFinv(m, p, n, arma::fill::zeros);
  arma::cube Kt(m, p, n, arma::fill::zeros);
  
  for (unsigned int t = 0; t < n; t++) {
    arma::uvec na_y = arma::find_nonfinite(y.col(t));
    if (na_y.n_elem < p) {
      arma::mat Zt = Z.slice(t * Ztv);
      arma::mat HHt = HH.slice(t * Htv);
      if (na_y.n_elem > 0) {
        Zt.rows(na_y).zeros();
        HHt.rows(na_y).zeros();
        HHt.cols(na_y).zeros();
        HHt.submat(na_y, na_y) = arma::eye(na_y.n_elem, na_y.n_elem);
      }
      arma::mat Ft = Zt * Pt.slice(t) * Zt.t() + HHt;
      // first check to avoid armadillo warnings
      bool chol_ok = Ft.is_finite() && arma::all(Ft.diag() > 0);
      if (!chol_ok) {
        at.fill(std::numeric_limits<double>::infinity()); 
        Pt.fill(std::numeric_limits<double>::infinity());
        return;
      }
      arma::mat cholF(p, p);
      chol_ok = arma::chol(cholF, Ft);
      if (!chol_ok) {
        at.fill(std::numeric_limits<double>::infinity()); 
        Pt.fill(std::numeric_limits<double>::infinity());
        return;
      }
      
      arma::vec tmpv = y.col(t) - D.col(t * Dtv) - Zt * at.col(t);
      tmpv(na_y).zeros();
      vt.col(t) = tmpv;
      arma::mat inv_cholF = arma::inv(arma::trimatu(cholF));
      ZFinv.slice(t) = Zt.t() * inv_cholF* inv_cholF.t();
      Kt.slice(t) = Pt.slice(t) * ZFinv.slice(t);
      at.col(t + 1) = C.col(t * Ctv) +
        T.slice(t * Ttv) * (at.col(t) + Kt.slice(t) * vt.col(t));
      arma::mat tmp = arma::eye(m, m) - Kt.slice(t) * Zt;
      Pt.slice(t + 1) = arma::symmatu(T.slice(t * Ttv) * (tmp * Pt.slice(t) * tmp.t() + Kt.slice(t) * HHt * Kt.slice(t).t()) * T.slice(t * Ttv).t() + RR.slice(t * Rtv));
    } else {
      at.col(t + 1) = C.col(t * Ctv) + T.slice(t * Ttv) * at.col(t);
      Pt.slice(t + 1) = arma::symmatu(T.slice(t * Ttv) *
        Pt.slice(t) * T.slice(t * Ttv).t() + RR.slice(t * Rtv));
    }
  }
  
  arma::vec rt(m, arma::fill::zeros);
  arma::mat Nt(m, m, arma::fill::zeros);
  
  for (int t = (n - 1); t >= 0; t--) {
    arma::uvec na_y = arma::find_nonfinite(y.col(t));
    
    if (na_y.n_elem < p) {
      arma::mat Zt = Z.slice(t * Ztv);
      Zt.rows(na_y).zeros();
      arma::mat L = T.slice(t * Ttv) * (arma::eye(m, m) -
        Kt.slice(t) * Zt);
      rt = ZFinv.slice(t) * vt.col(t) + L.t() * rt;
      Nt = arma::symmatu(ZFinv.slice(t) * Zt + L.t() * Nt * L);
    } else {
      rt = T.slice(t * Ttv).t() * rt;
      Nt = arma::symmatu(T.slice(t * Ttv).t() * Nt * T.slice(t * Ttv));
    }
    at.col(t) += Pt.slice(t) * rt;
    Pt.slice(t) -= arma::symmatu(Pt.slice(t) * Nt * Pt.slice(t));
  }
}


/* Fast state smoothing, only returns smoothed estimates of states
 * which are needed in simulation smoother and Laplace approximation
 */
arma::mat ssm_mlg::fast_smoother() const {
  
  arma::mat at(m, n + 1);
  arma::mat Pt(m, m);
  
  at.col(0) = a1;
  Pt = P1;
  
  arma::mat vt(p, n, arma::fill::zeros);
  arma::cube ZFinv(m, p, n, arma::fill::zeros);
  arma::cube Kt(m, p, n, arma::fill::zeros);
  
  for (unsigned int t = 0; t < n; t++) {
    
    arma::uvec na_y = arma::find_nonfinite(y.col(t));
    
    if (na_y.n_elem < p) {
      
      arma::mat Zt = Z.slice(t * Ztv);
      arma::mat HHt = HH.slice(t * Htv);
      if (na_y.n_elem > 0) {
        Zt.rows(na_y).zeros();
        HHt.rows(na_y).zeros();
        HHt.cols(na_y).zeros();
        HHt.submat(na_y, na_y) = arma::eye(na_y.n_elem, na_y.n_elem);
      }
      arma::mat Ft = Zt * Pt * Zt.t() + HHt;
      bool chol_ok = Ft.is_finite() && arma::all(Ft.diag() > 0);
      if (!chol_ok) {
        at.fill(-std::numeric_limits<double>::infinity());
        return at;
      }
      arma::mat cholF(p, p);
      chol_ok = arma::chol(cholF, Ft);
      if (!chol_ok) {
        at.fill(-std::numeric_limits<double>::infinity());
        return at;
      }
      
      arma::vec tmpv = y.col(t) - D.col(t * Dtv) - Zt * at.col(t);
      tmpv(na_y).zeros();
      vt.col(t) = tmpv;
      arma::mat inv_cholF = arma::inv(arma::trimatu(cholF));
      ZFinv.slice(t) = Zt.t() * inv_cholF * inv_cholF.t();
      Kt.slice(t) = Pt * ZFinv.slice(t);
      
      at.col(t + 1) = C.col(t * Ctv) +
        T.slice(t * Ttv) * (at.col(t) + Kt.slice(t) * vt.col(t));
      arma::mat tmp = arma::eye(m, m) - Kt.slice(t) * Zt;
      Pt = arma::symmatu(T.slice(t * Ttv) * (tmp * Pt * tmp.t() + Kt.slice(t) * HHt * Kt.slice(t).t()) * T.slice(t * Ttv).t() + RR.slice(t * Rtv));
      
    } else {
      at.col(t + 1) = C.col(t * Ctv) + T.slice(t * Ttv) *  at.col(t);
      Pt = arma::symmatu(T.slice(t * Ttv) * Pt * T.slice(t * Ttv).t() + RR.slice(t * Rtv));
    }
  }
  
  arma::mat rt(m, n);
  rt.col(n - 1).zeros();
  for (int t = (n - 1); t > 0; t--) {
    arma::uvec na_y = arma::find_nonfinite(y.col(t));
    
    if (na_y.n_elem < p) {
      arma::mat Zt = Z.slice(t * Ztv);
      Zt.rows(na_y).zeros();
      arma::mat L = T.slice(t * Ttv) *
        (arma::eye(m, m) - Kt.slice(t) * Zt);
      rt.col(t - 1) = ZFinv.slice(t) * vt.col(t) + L.t() * rt.col(t);
    } else {
      rt.col(t - 1) = T.slice(t * Ttv).t() * rt.col(t);
    }
  }
  arma::uvec na_y = arma::find_nonfinite(y.col(0));
  
  if (na_y.n_elem < p) {
    arma::mat Zt = Z.slice(0);
    Zt.rows(na_y).zeros();
    arma::mat L = T.slice(0) * (arma::eye(m, m) - Kt.slice(0) * Zt);
    at.col(0) = a1 + P1 * (ZFinv.slice(0) * vt.col(0) + L.t() * rt.col(0));
  } else {
    at.col(0) = a1 + P1 * T.slice(0).t() * rt.col(0);
  }
  for (unsigned int t = 0; t < (n - 1); t++) {
    at.col(t + 1) = C.col(t * Ctv)+ T.slice(t * Ttv) * at.col(t) + RR.slice(t * Rtv) * rt.col(t);
  }
  return at;
}

// smoother which returns also cov(alpha_t, alpha_t-1)
// used in psi particle filter
void ssm_mlg::smoother_ccov(arma::mat& at, arma::cube& Pt, arma::cube& ccov) const {
  
  
  at.col(0) = a1;
  Pt.slice(0) = P1;
  
  arma::mat vt(p, n, arma::fill::zeros);
  arma::cube ZFinv(m, p, n, arma::fill::zeros);
  arma::cube Kt(m, p, n, arma::fill::zeros);
  
  for (unsigned int t = 0; t < n; t++) {
    arma::uvec na_y = arma::find_nonfinite(y.col(t));
    if (na_y.n_elem < p) {
      arma::mat Zt = Z.slice(t * Ztv);
      arma::mat HHt = HH.slice(t * Htv);
      if (na_y.n_elem > 0) {
        Zt.rows(na_y).zeros();
        HHt.rows(na_y).zeros();
        HHt.cols(na_y).zeros();
        HHt.submat(na_y, na_y) = arma::eye(na_y.n_elem, na_y.n_elem);
      }
      arma::mat Ft = Zt * Pt.slice(t) * Zt.t() + HHt;
      // first check to avoid armadillo warnings
      bool chol_ok = Ft.is_finite() && arma::all(Ft.diag() > 0);
      if (!chol_ok) {
        at.fill(std::numeric_limits<double>::infinity()); 
        Pt.fill(std::numeric_limits<double>::infinity());
        ccov.fill(std::numeric_limits<double>::infinity());
        return;
      }
      arma::mat cholF(p, p);
      chol_ok = arma::chol(cholF ,Ft);
      if (!chol_ok) {
        at.fill(std::numeric_limits<double>::infinity()); 
        Pt.fill(std::numeric_limits<double>::infinity());
        ccov.fill(std::numeric_limits<double>::infinity());
        return;
      }
      
      
      arma::vec tmpv = y.col(t) - D.col(t * Dtv) - Zt * at.col(t);
      tmpv(na_y).zeros();
      vt.col(t) = tmpv;
      arma::mat inv_cholF = arma::inv(arma::trimatu(cholF));
      ZFinv.slice(t) = Zt.t() * inv_cholF * inv_cholF.t();
      Kt.slice(t) = Pt.slice(t) * ZFinv.slice(t);
      at.col(t + 1) = C.col(t * Ctv) +
        T.slice(t * Ttv) * (at.col(t) + Kt.slice(t) * vt.col(t));
      
      arma::mat tmp = arma::eye(m, m) - Kt.slice(t) * Zt;
      Pt.slice(t + 1) = arma::symmatu(T.slice(t * Ttv) * (tmp * Pt.slice(t) * tmp.t() + Kt.slice(t) * HHt * Kt.slice(t).t()) * T.slice(t * Ttv).t() + RR.slice(t * Rtv));
      
      
    } else {
      at.col(t + 1) = C.col(t * Ctv) + T.slice(t * Ttv) * at.col(t);
      Pt.slice(t + 1) = arma::symmatu(T.slice(t * Ttv) *
        Pt.slice(t) * T.slice(t * Ttv).t() + RR.slice(t * Rtv));
    }
    ccov.slice(t) = Pt.slice(t + 1); //store for smoothing;
  }
  
  arma::vec rt(m, arma::fill::zeros);
  arma::mat Nt(m, m, arma::fill::zeros);
  
  for (int t = (n - 1); t >= 0; t--) {    
    arma::uvec na_y = arma::find_nonfinite(y.col(t));
    if (na_y.n_elem < p) {
      arma::mat Zt = Z.slice(t * Ztv);
      Zt.rows(na_y).zeros();
      arma::mat L = T.slice(t * Ttv) * (arma::eye(m, m) -
        Kt.slice(t) * Zt);
      //P[t+1] stored to ccov_t
      ccov.slice(t) = Pt.slice(t) * L.t() * (arma::eye(m, m) - Nt * ccov.slice(t));
      rt = ZFinv.slice(t) * vt.col(t) + L.t() * rt;
      Nt = arma::symmatu(ZFinv.slice(t) * Zt + L.t() * Nt * L);
    } else {
      ccov.slice(t) = Pt.slice(t) * T.slice(t * Ttv).t() *
        (arma::eye(m, m) - Nt * ccov.slice(t));
      rt = T.slice(t * Ttv).t() * rt;
      Nt = arma::symmatu(T.slice(t * Ttv).t() * Nt * T.slice(t * Ttv));
      //P[t+1] stored to ccov_t
    }
    at.col(t) += Pt.slice(t) * rt;
    Pt.slice(t) -= arma::symmatu(Pt.slice(t) * Nt * Pt.slice(t));
  }
  ccov.slice(n).zeros();
}


double ssm_mlg::filter(arma::mat& at, arma::mat& att,
  arma::cube& Pt, arma::cube& Ptt) const {
  
  at.col(0) = a1;
  Pt.slice(0) = P1;
  
  const double LOG2PI = std::log(2.0 * M_PI);
  double logLik = 0.0;
  for (unsigned int t = 0; t < n; t++) {
    arma::uvec na_y = arma::find_nonfinite(y.col(t));
    
    if (na_y.n_elem < p) {
      arma::mat Zt = Z.slice(t * Ztv);
      arma::mat HHt = HH.slice(t * Htv);
      if (na_y.n_elem > 0) {
        Zt.rows(na_y).zeros();
        HHt.rows(na_y).zeros();
        HHt.cols(na_y).zeros();
        HHt.submat(na_y, na_y) = arma::eye(na_y.n_elem, na_y.n_elem);
      }
      
      arma::mat Ft = Zt * Pt.slice(t) * Zt.t() + HHt;
      
      bool chol_ok = Ft.is_finite() && arma::all(Ft.diag() > 0);
      if (!chol_ok) {
        at.fill(std::numeric_limits<double>::infinity()); 
        Pt.fill(std::numeric_limits<double>::infinity());
        att.fill(std::numeric_limits<double>::infinity());
        Ptt.fill(std::numeric_limits<double>::infinity());
        return -std::numeric_limits<double>::infinity();
      }
      arma::mat cholF(p, p);
      chol_ok = arma::chol(cholF ,Ft);
      
      if (!chol_ok) {
        at.fill(std::numeric_limits<double>::infinity()); 
        at.fill(std::numeric_limits<double>::infinity()); 
        Pt.fill(std::numeric_limits<double>::infinity());
        att.fill(std::numeric_limits<double>::infinity());
        Ptt.fill(std::numeric_limits<double>::infinity());
        return -std::numeric_limits<double>::infinity();
      }
      arma::vec v = y.col(t) - D.col(t * Dtv) - Zt * at.col(t);
      v(na_y).zeros();
      arma::mat inv_cholF = arma::inv(arma::trimatu(cholF));
      arma::mat ZFinv = Zt.t() * inv_cholF * inv_cholF.t();
      arma::mat K = Pt.slice(t) * ZFinv;
      att.col(t) = at.col(t) + K * v;
      at.col(t + 1) = C.col(t * Ctv) +
        T.slice(t * Ttv) * att.col(t);
      
      arma::mat tmp = arma::eye(m, m) - K * Zt;
      Ptt.slice(t) = tmp * Pt.slice(t) * tmp.t() + K * HHt * K.t();
      
      Pt.slice(t + 1) = arma::symmatu(T.slice(t * Ttv) *
        Ptt.slice(t) * T.slice(t * Ttv).t() +
        RR.slice(t * Rtv));
      arma::vec Fv = inv_cholF.t() * v;
      logLik -= 0.5 * arma::as_scalar((p - na_y.n_elem) * LOG2PI +
        2.0 * arma::accu(arma::log(arma::diagvec(cholF))) + Fv.t() * Fv);
    } else {
      att.col(t) = at.col(t);
      at.col(t + 1) = C.col(t * Ctv) + T.slice(t * Ttv) *  att.col(t);
      Ptt.slice(t) = Pt.slice(t);
      Pt.slice(t + 1) = arma::symmatu(T.slice(t * Ttv) *
        Ptt.slice(t) * T.slice(t * Ttv).t() + RR.slice(t * Rtv));
    }
  }
  
  return logLik;
}



void ssm_mlg::psi_filter(const unsigned int nsim, arma::cube& alpha) {
  
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

// simulate states from smoothing distribution
// Note: not optimized at all for multiple replications (compare with ssm_ulg implementation)
arma::cube ssm_mlg::simulate_states(const unsigned int nsim) {
  
  arma::mat L_P1 = psd_chol(P1);
  std::normal_distribution<> normal(0.0, 1.0);
  
  arma::cube asim(m, n + 1, nsim);
  arma::mat y_tmp = y;
  for(unsigned int i = 0; i < nsim; i++) {
    
    arma::vec um(m);
    for(unsigned int j = 0; j < m; j++) {
      um(j) = normal(engine);
    }
    asim.slice(i).col(0) = L_P1 * um;
    
    for (unsigned int t = 0; t < n; t++) {
      arma::uvec na_y = arma::find_nonfinite(y.col(t));
      if (na_y.n_elem < p) {
        arma::vec up(p);
        for(unsigned int j = 0; j < p; j++) {
          up(j) = normal(engine);
        }
        y.col(t) -= Z.slice(t * Ztv) * asim.slice(i).col(t) +
          H.slice(t * Htv) * up;
      }
      arma::vec uk(k);
      for(unsigned int j = 0; j < k; j++) {
        uk(j) = normal(engine);
      }
      asim.slice(i).col(t + 1) = T.slice(t * Ttv) * asim.slice(i).col(t) +
        R.slice(t * Rtv) * uk;
    }
    
    asim.slice(i) += fast_smoother();
    y = y_tmp;
  }
  return asim;
}


arma::cube ssm_mlg::predict_sample(const arma::mat& theta_posterior,
  const arma::mat& alpha, const unsigned int predict_type, const Rcpp::Function update_fn) {
  
  unsigned int d = p;
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


arma::mat ssm_mlg::sample_model(const unsigned int predict_type) {
  
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
    arma::mat y(p, n);
    
    for (unsigned int t = 0; t < n; t++) {
      y.col(t) = D.col(t * Dtv) + Z.slice(t * Ztv) * alpha.col(t);
      if(predict_type == 1) {
        arma::vec up(p);
        for(unsigned int j = 0; j < p; j++) {
          up(j) = normal(engine);
        }
        y.col(t) += H(t * Htv) * up;
      }
    }
    return y;
  } else {
    return alpha;
  }
}


arma::cube ssm_mlg::predict_past(const arma::mat& theta_posterior,
  const arma::cube& alpha, const unsigned int predict_type, const Rcpp::Function update_fn) {
  
  unsigned int n_samples = theta_posterior.n_cols;
  arma::cube samples(p, n, n_samples);
  
  std::normal_distribution<> normal(0.0, 1.0);
  for (unsigned int i = 0; i < n_samples; i++) {
    update_model(theta_posterior.col(i), update_fn);
    for (unsigned int t = 0; t < n; t++) {
      samples.slice(i).col(t) = D.col(t * Dtv) + Z.slice(t * Ztv) * alpha.slice(i).col(t);
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
