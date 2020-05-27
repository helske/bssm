#include "model_mgg_ssm.h"
#include "psd_chol.h"

// General constructor of mgg_ssm object from Rcpp::List
mgg_ssm::mgg_ssm(
  const Rcpp::List& model, 
  const unsigned int seed,
  const double zero_tol) 
  :
    y((Rcpp::as<arma::mat>(model["y"])).t()), 
    Z(Rcpp::as<arma::cube>(model["Z"])),
    H(Rcpp::as<arma::cube>(model["H"])), 
    T(Rcpp::as<arma::cube>(model["T"])),
    R(Rcpp::as<arma::cube>(model["R"])), 
    a1(Rcpp::as<arma::vec>(model["a1"])),
    P1(Rcpp::as<arma::mat>(model["P1"])), 
    D(Rcpp::as<arma::mat>(model["obs_intercept"])),
    C(Rcpp::as<arma::mat>(model["state_intercept"])), 
    n(y.n_cols), m(a1.n_elem), k(R.n_cols), p(y.n_rows), 
    Ztv(Z.n_slices > 1), 
    Htv(H.n_slices > 1), 
    Ttv(T.n_slices > 1), 
    Rtv(R.n_slices > 1),
    Dtv(D.n_cols > 1), 
    Ctv(C.n_cols > 1), 
    engine(seed), zero_tol(zero_tol),
    HH(arma::cube(p, p, Htv * (n - 1) + 1)),
    RR(arma::cube(m, m, Rtv * (n - 1) + 1)) {
  
  compute_HH();
  compute_RR();
  
}

// General constructor of mgg_ssm object for snippet models
mgg_ssm::mgg_ssm(
  const arma::mat& y, 
  const arma::cube& Z, 
  const arma::cube& H,
  const arma::cube& T, 
  const arma::cube& R, 
  const arma::vec& a1,
  const arma::mat& P1, 
  const arma::mat& D, 
  const arma::mat& C, 
  const unsigned int seed,
  const double zero_tol) 
  :
    y(y), Z(Z), H(H), T(T), R(R), a1(a1), P1(P1), D(D), C(C),
    n(y.n_cols), m(a1.n_elem), k(R.n_cols), p(y.n_rows),
    Ztv(Z.n_slices > 1), Htv(H.n_slices > 1), 
    Ttv(T.n_slices > 1), Rtv(R.n_slices > 1),
    Dtv(D.n_cols > 1), Ctv(C.n_cols > 1), 
    engine(seed), zero_tol(zero_tol),
    HH(arma::cube(p, p, Htv * (n - 1) + 1)),
    RR(arma::cube(m, m, Rtv * (n - 1) + 1)) {
  
  compute_HH();
  compute_RR();
}

void mgg_ssm::compute_RR(){
  for (unsigned int t = 0; t < R.n_slices; t++) {
    RR.slice(t) = R.slice(t * Rtv) * R.slice(t * Rtv).t();
  }
}

void mgg_ssm::compute_HH(){
  for (unsigned int t = 0; t < H.n_slices; t++) {
    HH.slice(t) = H.slice(t * Htv) * H.slice(t * Htv).t();
  }
}

double mgg_ssm::log_likelihood() const {
  
  double logLik = 0;
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
  return logLik;
}

// Kalman smoother
void mgg_ssm::smoother(arma::mat& at, arma::cube& Pt) const {
  
  
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
arma::mat mgg_ssm::fast_smoother() const {
  
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
void mgg_ssm::smoother_ccov(arma::mat& at, arma::cube& Pt, arma::cube& ccov) const {
  
  
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


double mgg_ssm::filter(arma::mat& at, arma::mat& att,
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


// simulate states from smoothing distribution
// Note: not optimized at all for multiple replications (compare with ugg_ssm implementation)
arma::cube mgg_ssm::simulate_states(const unsigned int nsim_states) {
  
  arma::mat L_P1 = psd_chol(P1);
  std::normal_distribution<> normal(0.0, 1.0);
  
  arma::cube asim(m, n + 1, nsim_states);
  arma::mat y_tmp = y;
  for(unsigned int i = 0; i < nsim_states; i++) {
    
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
        y.col(t) -= Z.slice(t * Ztv) * asim.slice(0).col(t) +
          H(t * Htv) * up;
      }
      arma::vec uk(k);
      for(unsigned int j = 0; j < k; j++) {
        uk(j) = normal(engine);
      }
      asim.slice(i).col(t + 1) = T.slice(t * Ttv) * asim.slice(0).col(t) +
        R.slice(t * Rtv) * uk;
    }
    
    asim.slice(i) += fast_smoother();
    y = y_tmp;
  }
  return asim;
}
