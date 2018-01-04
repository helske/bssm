#include "mgg_ssm.h"
#include "psd_chol.h"

// General constructor of mgg_ssm object from Rcpp::List
// with parameter indices
mgg_ssm::mgg_ssm(const Rcpp::List& model, const unsigned int seed,
  const arma::uvec& Z_ind_, const arma::uvec& H_ind_,
  const arma::uvec& T_ind_, const arma::uvec& R_ind_) :
  y((Rcpp::as<arma::mat>(model["y"])).t()), Z(Rcpp::as<arma::cube>(model["Z"])),
  H(Rcpp::as<arma::cube>(model["H"])), T(Rcpp::as<arma::cube>(model["T"])),
  R(Rcpp::as<arma::cube>(model["R"])), a1(Rcpp::as<arma::vec>(model["a1"])),
  P1(Rcpp::as<arma::mat>(model["P1"])), xreg(Rcpp::as<arma::cube>(model["xreg"])),
  beta(Rcpp::as<arma::mat>(model["coefs"])),
  D(Rcpp::as<arma::mat>(model["obs_intercept"])),
  C(Rcpp::as<arma::mat>(model["state_intercept"])),  
  theta(Rcpp::as<arma::vec>(model["theta"])), 
  Ztv(Z.n_slices > 1), Htv(H.n_slices > 1), Ttv(T.n_slices > 1), Rtv(R.n_slices > 1),
  Dtv(D.n_cols > 1), Ctv(C.n_cols > 1), n(y.n_cols), m(a1.n_elem), k(R.n_cols),
  p(y.n_rows), HH(arma::cube(p, p, Htv * (n - 1) + 1)),
  RR(arma::cube(m, m, Rtv * (n - 1) + 1)),
  xbeta(arma::mat(n, p, arma::fill::zeros)), engine(seed), zero_tol(1e-8),
  prior_distributions(Rcpp::as<arma::uvec>(model["prior_distributions"])), 
  prior_parameters(Rcpp::as<arma::mat>(model["prior_parameters"])),
  Z_ind(Z_ind_), H_ind(H_ind_), T_ind(T_ind_), R_ind(R_ind_) {
  
  if(xreg.n_elem > 0) {
    compute_xbeta();
  }
  compute_HH();
  compute_RR();
}

// General constructor of mgg_ssm object for approximating models
mgg_ssm::mgg_ssm(const arma::mat& y, const arma::cube& Z, const arma::cube& H,
  const arma::cube& T, const arma::cube& R, const arma::vec& a1,
  const arma::mat& P1, const arma::cube& xreg, const arma::mat& beta,
  const arma::mat& D, const arma::mat& C, 
  const unsigned int seed,
  const arma::vec& theta,
  const arma::uvec& prior_distributions,
  const arma::mat& prior_parameters, 
  const arma::uvec& Z_ind, const arma::uvec& H_ind, const arma::uvec& T_ind,
  const arma::uvec& R_ind) :
  y(y), Z(Z), H(H), T(T), R(R), a1(a1), P1(P1), xreg(xreg), beta(beta), D(D), C(C),
  Ztv(Z.n_slices > 1), Htv(H.n_slices > 1), Ttv(T.n_slices > 1), Rtv(R.n_slices > 1),
  Dtv(D.n_cols > 1), Ctv(C.n_cols > 1), n(y.n_cols), m(a1.n_elem), k(R.n_cols),
  p(y.n_rows), HH(arma::cube(p, p, Htv * (n - 1) + 1)),
  RR(arma::cube(m, m, Rtv * (n - 1) + 1)),
  xbeta(arma::mat(n, p, arma::fill::zeros)),
  theta(theta), prior_distributions(prior_distributions), 
  prior_parameters(prior_parameters),
  engine(seed), zero_tol(1e-8),
  Z_ind(Z_ind), H_ind(H_ind), T_ind(T_ind), R_ind(R_ind) {
  
  if(xreg.n_elem > 0) {
    compute_xbeta();
  }
  compute_HH();
  compute_RR();
}

void mgg_ssm::update_model(const arma::vec& new_theta) {
  
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
  if(xreg.n_elem > 0) {
    for (unsigned int i = 0; i < p; i++){
      beta.col(i) = new_theta.subvec(new_theta.n_elem - xreg.n_cols * (xreg.n_slices - i),
        new_theta.n_elem - xreg.n_cols * (xreg.n_slices - i + 1) - 1);
    }
    compute_xbeta();
  }
  theta = new_theta;
}

double mgg_ssm::log_prior_pdf(const arma::vec& x) const {
  
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

double mgg_ssm::log_proposal_ratio(const arma::vec& new_theta, const arma::vec& old_theta) const {
  return 0.0;
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
void mgg_ssm::compute_xbeta(){
  for (unsigned int i = 0; i < p; i++) {
    xbeta.col(i) = xreg.slice(i) * beta.col(i);
  }
}

double mgg_ssm::log_likelihood() const {
  
  double logLik = 0;
  arma::vec at = a1;
  arma::mat Pt = P1;
  
  arma::mat y_tmp = y;
  if(xreg.n_cols > 0) {
    y_tmp -= xbeta.t();
  }
  
  const double LOG2PI = std::log(2.0 * M_PI);
  
  for (unsigned int t = 0; t < n; t++) {
    arma::uvec obs_y = arma::find_finite(y_tmp.col(t));
    
    if (obs_y.n_elem > 0) {
      arma::mat Zt = Z.slice(t * Ztv).rows(obs_y);
      arma::mat F = Zt * Pt * Zt.t() + HH.slice(t * Htv).submat(obs_y, obs_y);
      // first check to avoid armadillo warnings
      bool chol_ok = F.is_finite();
      if (!chol_ok) return -std::numeric_limits<double>::infinity();
      arma::mat cholF(p, p);
      chol_ok = arma::chol(cholF, F);
      if (!chol_ok) return -std::numeric_limits<double>::infinity();
      
      arma::vec tmp = y_tmp.col(t) - D.col(t * Dtv);
      arma::vec v = tmp.rows(obs_y) - Zt * at;
      arma::mat inv_cholF = arma::inv(arma::trimatu(cholF));
      arma::mat K = Pt * Zt.t() * inv_cholF.t() * inv_cholF;
      at = C.col(t * Ctv) + T.slice(t * Ttv) * (at + K * v);
      Pt = arma::symmatu(T.slice(t * Ttv) *
        (Pt - K * F * K.t()) * T.slice(t * Ttv).t() + RR.slice(t * Rtv));
      arma::vec Fv = inv_cholF * v;
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
  
  arma::mat y_tmp = y;
  if(xreg.n_cols > 0) {
    y_tmp -= xbeta.t();
  }
  
  at.col(0) = a1;
  Pt.slice(0) = P1;
  
  arma::mat vt(p, n, arma::fill::zeros);
  arma::cube ZFinv(m, p, n, arma::fill::zeros);
  arma::cube Kt(m, p, n, arma::fill::zeros);
  
  for (unsigned int t = 0; t < (n - 1); t++) {
    arma::uvec na_y = arma::find_nonfinite(y_tmp.col(t));
    if (na_y.n_elem < p) {
      arma::mat Zt = Z.slice(t * Ztv);
      Zt.rows(na_y).zeros();
      arma::mat HHt = HH.slice(t * Htv);
      HHt.submat(na_y, na_y) = arma::eye(na_y.n_elem, na_y.n_elem);
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
      
      vt.col(t) = y_tmp.col(t) - D.col(t * Dtv) - Zt * at.col(t);
      arma::mat inv_cholF = arma::inv(arma::trimatu(cholF));
      ZFinv.slice(t) = Zt.t() * inv_cholF.t() * inv_cholF;
      Kt.slice(t) = Pt.slice(t) * ZFinv.slice(t);
      at.col(t + 1) = C.col(t * Ctv) +
        T.slice(t * Ttv) * (at.col(t) + Kt.slice(t) * vt.col(t));
      Pt.slice(t + 1) = arma::symmatu(T.slice(t * Ttv) *
        (Pt.slice(t) - Kt.slice(t) * Ft * Kt.slice(t).t()) * T.slice(t * Ttv).t() +
        RR.slice(t * Rtv));
    } else {
      at.col(t + 1) = C.col(t * Ctv) + T.slice(t * Ttv) * at.col(t);
      Pt.slice(t + 1) = arma::symmatu(T.slice(t * Ttv) *
        Pt.slice(t) * T.slice(t * Ttv).t() + RR.slice(t * Rtv));
    }
  }
  
  unsigned int t = n - 1;
  arma::uvec na_y = arma::find_nonfinite(y_tmp.col(t));
  if (na_y.n_elem < p) {
    arma::mat Zt = Z.slice(t * Ztv);
    Zt.rows(na_y).zeros();
    arma::mat HHt = HH.slice(t * Htv);
    HHt.submat(na_y, na_y) = arma::eye(na_y.n_elem, na_y.n_elem);
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
    vt.col(t) = y_tmp.col(t) - D.col(t * Dtv) - Zt * at.col(t);
    arma::mat inv_cholF = arma::inv(arma::trimatu(cholF));
    ZFinv.slice(t) = Zt.t() * inv_cholF.t() * inv_cholF;
    Kt.slice(t) = Pt.slice(t) * ZFinv.slice(t);
  }
  
  arma::vec rt(m, arma::fill::zeros);
  arma::mat Nt(m, m, arma::fill::zeros);
  
  for (int t = (n - 1); t >= 0; t--) {
    na_y = arma::find_nonfinite(y_tmp.col(t));
    
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
  arma::mat y_tmp = y;
  if(xreg.n_cols > 0) {
    y_tmp -= xbeta.t();
  }
  
  arma::mat at(m, n);
  arma::mat Pt(m, m);
  
  at.col(0) = a1;
  Pt = P1;
  
  arma::mat vt(p, n, arma::fill::zeros);
  arma::cube ZFinv(m, p, n, arma::fill::zeros);
  arma::cube Kt(m, p, n, arma::fill::zeros);
  
  for (unsigned int t = 0; t < (n - 1); t++) {
    
    arma::uvec na_y = arma::find_nonfinite(y_tmp.col(t));
    
    if (na_y.n_elem < p) {
      
      arma::mat Zt = Z.slice(t * Ztv);
      Zt.rows(na_y).zeros();
      arma::mat HHt = HH.slice(t * Htv);
      HHt.submat(na_y, na_y) = arma::eye(na_y.n_elem, na_y.n_elem);
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
      
      vt.col(t) = y_tmp.col(t) - D.col(t * Dtv) - Zt * at.col(t);
      arma::mat inv_cholF = arma::inv(arma::trimatu(cholF));
      ZFinv.slice(t) = Zt.t() * inv_cholF.t() * inv_cholF;
      Kt.slice(t) = Pt * ZFinv.slice(t);
      
      at.col(t + 1) = C.col(t * Ctv) +
        T.slice(t * Ttv) * (at.col(t) + Kt.slice(t) * vt.col(t));
      Pt = arma::symmatu(T.slice(t * Ttv) *
        (Pt - Kt.slice(t) * Ft * Kt.slice(t).t()) * T.slice(t * Ttv).t() +
        RR.slice(t * Rtv));
      
    } else {
      at.col(t + 1) = C.col(t * Ctv) + T.slice(t * Ttv) *  at.col(t);
      Pt = arma::symmatu(T.slice(t * Ttv) * Pt * T.slice(t * Ttv).t() + RR.slice(t * Rtv));
    }
  }
  
  unsigned int t = n - 1;
  arma::uvec na_y = arma::find_nonfinite(y_tmp.col(t));
  if (na_y.n_elem < p) {
    arma::mat Zt = Z.slice(t * Ztv);
    Zt.rows(na_y).zeros();
    arma::mat HHt = HH.slice(t * Htv);
    HHt.submat(na_y, na_y) = arma::eye(na_y.n_elem, na_y.n_elem);
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
    
    vt.col(t) = y_tmp.col(t) - D.col(t * Dtv) - Zt * at.col(t);
    arma::mat inv_cholF = arma::inv(arma::trimatu(cholF));
    ZFinv.slice(t) = Zt.t() * inv_cholF.t() * inv_cholF;
    Kt.slice(t) = Pt * ZFinv.slice(t);
    
  }
  arma::mat rt(m, n);
  rt.col(n - 1).zeros();
  for (int t = (n - 1); t > 0; t--) {
    na_y = arma::find_nonfinite(y_tmp.col(t));
    
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
  na_y = arma::find_nonfinite(y_tmp.col(0));
  
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
  
  arma::mat y_tmp = y;
  if(xreg.n_cols > 0) {
    y_tmp -= xbeta.t();
  }
  
  at.col(0) = a1;
  Pt.slice(0) = P1;
  
  arma::mat vt(p, n, arma::fill::zeros);
  arma::cube ZFinv(m, p, n, arma::fill::zeros);
  arma::cube Kt(m, p, n, arma::fill::zeros);
  
  for (unsigned int t = 0; t < (n - 1); t++) {
    arma::uvec na_y = arma::find_nonfinite(y_tmp.col(t));
    if (na_y.n_elem < p) {
      arma::mat Zt = Z.slice(t * Ztv);
      Zt.rows(na_y).zeros();
      arma::mat HHt = HH.slice(t * Htv);
      HHt.submat(na_y, na_y) = arma::eye(na_y.n_elem, na_y.n_elem);
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
      
      
      vt.col(t) = y_tmp.col(t) - D.col(t * Dtv) - Zt * at.col(t);
      arma::mat inv_cholF = arma::inv(arma::trimatu(cholF));
      ZFinv.slice(t) = Zt.t() * inv_cholF.t() * inv_cholF;
      Kt.slice(t) = Pt.slice(t) * ZFinv.slice(t);
      at.col(t + 1) = C.col(t * Ctv) +
        T.slice(t * Ttv) * (at.col(t) + Kt.slice(t) * vt.col(t));
      Pt.slice(t + 1) = arma::symmatu(T.slice(t * Ttv) *
        (Pt.slice(t) - Kt.slice(t) * Ft * Kt.slice(t).t()) * T.slice(t * Ttv).t() +
        RR.slice(t * Rtv));
      
    } else {
      at.col(t + 1) = C.col(t * Ctv) + T.slice(t * Ttv) * at.col(t);
      Pt.slice(t + 1) = arma::symmatu(T.slice(t * Ttv) *
        Pt.slice(t) * T.slice(t * Ttv).t() + RR.slice(t * Rtv));
    }
    ccov.slice(t) = Pt.slice(t + 1); //store for smoothing;
  }
  
  unsigned int t = n - 1;
  arma::uvec na_y = arma::find_nonfinite(y_tmp.col(t));
  if (na_y.n_elem < p) {
    arma::mat Zt = Z.slice(t * Ztv);
    Zt.rows(na_y).zeros();
    arma::mat HHt = HH.slice(t * Htv);
    HHt.submat(na_y, na_y) = arma::eye(na_y.n_elem, na_y.n_elem);
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
    chol_ok = arma::chol(cholF, Ft);
    if (!chol_ok) {
      at.fill(std::numeric_limits<double>::infinity()); 
      Pt.fill(std::numeric_limits<double>::infinity());
      ccov.fill(std::numeric_limits<double>::infinity());
      return;
    }
    vt.col(t) = y_tmp.col(t) - D.col(t * Dtv) - Zt * at.col(t);
    arma::mat inv_cholF = arma::inv(arma::trimatu(cholF));
    ZFinv.slice(t) = Zt.t() * inv_cholF.t() * inv_cholF;
    Kt.slice(t) = Pt.slice(t) * ZFinv.slice(t);
    ccov.slice(t) = arma::symmatu(T.slice(t * Ttv) * (Pt.slice(t) -
      Kt.slice(t) * Ft * Kt.slice(t).t()) * T.slice(t * Ttv).t() + RR.slice(t * Rtv));
    
    
  } else {
    ccov.slice(t) = arma::symmatu(T.slice(t * Ttv) * Pt.slice(t) * T.slice(t * Ttv).t() +
      RR.slice(t * Rtv));
  }
  
  arma::vec rt(m, arma::fill::zeros);
  arma::mat Nt(m, m, arma::fill::zeros);
  
  for (int t = (n - 1); t >= 0; t--) {    
    na_y = arma::find_nonfinite(y_tmp.col(t));
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
      //P[t+1] stored to ccov_t //CHECK THIS
    }
    at.col(t) += Pt.slice(t) * rt;
    Pt.slice(t) -= arma::symmatu(Pt.slice(t) * Nt * Pt.slice(t));
  }
}


double mgg_ssm::filter(arma::mat& at, arma::mat& att,
  arma::cube& Pt, arma::cube& Ptt) const {
  
  arma::mat y_tmp = y;
  if(xreg.n_cols > 0) {
    y_tmp -= xbeta.t();
  }
  
  at.col(0) = a1;
  Pt.slice(0) = P1;
  
  const double LOG2PI = std::log(2.0 * M_PI);
  double logLik = 0.0;
  
  for (unsigned int t = 0; t < (n - 1); t++) {
    arma::uvec na_y = arma::find_nonfinite(y_tmp.col(t));
    
    if (na_y.n_elem < p) {
      arma::mat Zt = Z.slice(t * Ztv);
      Zt.rows(na_y).zeros();
      arma::mat HHt = HH.slice(t * Htv);
      HHt.submat(na_y, na_y) = arma::eye(na_y.n_elem, na_y.n_elem);
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
      arma::vec v = y_tmp.col(t) - D.col(t * Dtv) - Zt * at.col(t);
      arma::mat inv_cholF = arma::inv(arma::trimatu(cholF));
      arma::mat ZFinv = Zt.t() * inv_cholF.t() * inv_cholF;
      arma::mat K = Pt.slice(t) * ZFinv;
      att.col(t) = at.col(t) + K * v;
      at.col(t + 1) = C.col(t * Ctv) +
        T.slice(t * Ttv) * att.col(t);
      Ptt.slice(t) = Pt.slice(t) - K * Ft * K.t();
      Pt.slice(t + 1) = arma::symmatu(T.slice(t * Ttv) *
        Ptt.slice(t) * T.slice(t * Ttv).t() +
        RR.slice(t * Rtv));
      arma::vec Fv = inv_cholF * v;
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
  
  unsigned int t = n - 1;
  arma::uvec na_y = arma::find_nonfinite(y_tmp.col(t));
  if (na_y.n_elem < p) {
    arma::mat Zt = Z.slice(t * Ztv);
    Zt.rows(na_y).zeros();
    arma::mat HHt = HH.slice(t * Htv);
    HHt.submat(na_y, na_y) = arma::eye(na_y.n_elem, na_y.n_elem);
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
    arma::vec v = y_tmp.col(t) - D.col(t * Dtv) - Zt * at.col(t);
    arma::mat inv_cholF = arma::inv(arma::trimatu(cholF));
    arma::mat ZFinv = Zt.t() * inv_cholF.t() * inv_cholF;
    arma::mat K = Pt.slice(t) * ZFinv;
    att.col(t) = at.col(t) + K * v;
    Ptt.slice(t) = Pt.slice(t) - K * Ft * K.t();
    arma::vec Fv = inv_cholF * v;
    logLik -= 0.5 * arma::as_scalar((p - na_y.n_elem) * LOG2PI +
      2.0 * arma::accu(arma::log(arma::diagvec(cholF))) + Fv.t() * Fv);
    
  } else {
    att.col(t) = at.col(t);
    Ptt.slice(t) = Pt.slice(t);
  }
  return logLik;
}


arma::cube mgg_ssm::simulate_states() {
  
  arma::mat L_P1 = psd_chol(P1);
  arma::cube asim(m, n, 1);
  
  std::normal_distribution<> normal(0.0, 1.0);
  
  arma::vec um(m);
  for(unsigned int j = 0; j < m; j++) {
    um(j) = normal(engine);
  }
  asim.slice(0).col(0) = L_P1 * um;
  arma::mat y_tmp = y;
  for (unsigned int t = 0; t < (n - 1); t++) {
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
    asim.slice(0).col(t + 1) = T.slice(t * Ttv) * asim.slice(0).col(t) +
      R.slice(t * Rtv) * uk;
  }
  arma::uvec na_y = arma::find_nonfinite(y.col(n - 1));
  if (na_y.n_elem < p) {
    arma::vec up(p);
    for(unsigned int j = 0; j < p; j++) {
      up(j) = normal(engine);
    }
    y.col(n - 1) -= Z.slice((n - 1) * Ztv) * asim.slice(0).col(n - 1) +
      H((n - 1) * Htv) * up;
  }
  
  asim.slice(0) += fast_smoother();
  y = y_tmp;
  return asim;
}
