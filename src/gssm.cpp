#include "gssm.h"

// from List
gssm::gssm(const List& model, unsigned int seed) :
  y(as<arma::vec>(model["y"])), Z(as<arma::mat>(model["Z"])),
  H(as<arma::vec>(model["H"])), T(as<arma::cube>(model["T"])), R(as<arma::cube>(model["R"])),
  a1(as<arma::vec>(model["a1"])), P1(as<arma::mat>(model["P1"])),
  xreg(as<arma::mat>(model["xreg"])), beta(as<arma::vec>(model["coefs"])),
  Ztv(Z.n_cols > 1), Htv(H.n_elem > 1), Ttv(T.n_slices > 1), Rtv(R.n_slices > 1),
  n(y.n_elem), m(a1.n_elem), k(R.n_cols), HH(arma::vec(Htv * (n - 1) + 1)),
  RR(arma::cube(m, m, Rtv * (n - 1) + 1)), xbeta(arma::vec(n, arma::fill::zeros)),
  engine(seed), zero_tol(1e-8) {
  
  if(xreg.n_cols > 0) {
    compute_xbeta();
  }
  compute_HH();
  compute_RR();
}

// from List
// with parameter indices
gssm::gssm(const List& model, arma::uvec Z_ind, arma::uvec H_ind,
  arma::uvec T_ind, arma::uvec R_ind, unsigned int seed) :
  y(as<arma::vec>(model["y"])), Z(as<arma::mat>(model["Z"])),
  H(as<arma::vec>(model["H"])), T(as<arma::cube>(model["T"])), R(as<arma::cube>(model["R"])),
  a1(as<arma::vec>(model["a1"])), P1(as<arma::mat>(model["P1"])),
  xreg(as<arma::mat>(model["xreg"])), beta(as<arma::vec>(model["coefs"])),
  Ztv(Z.n_cols > 1), Htv(H.n_elem > 1), Ttv(T.n_slices > 1), Rtv(R.n_slices > 1),
  n(y.n_elem), m(a1.n_elem), k(R.n_cols), HH(arma::vec(Htv * (n - 1) + 1)),
  RR(arma::cube(m, m, Rtv * (n - 1) + 1)), xbeta(arma::vec(n, arma::fill::zeros)),
  Z_ind(Z_ind), H_ind(H_ind), T_ind(T_ind), R_ind(R_ind), engine(seed), zero_tol(1e-8) {
  
  if(xreg.n_cols > 0) {
    compute_xbeta();
  }
  compute_HH();
  compute_RR();
}
// from List for non-gaussian models, ng value is not actually used, cheap trick...
gssm::gssm(const List& model, unsigned int seed, bool ng) :
  y(as<arma::vec>(model["y"])), Z(as<arma::mat>(model["Z"])),
  H(arma::vec(y.n_elem)), T(as<arma::cube>(model["T"])), R(as<arma::cube>(model["R"])),
  a1(as<arma::vec>(model["a1"])), P1(as<arma::mat>(model["P1"])),
  xreg(as<arma::mat>(model["xreg"])), beta(as<arma::vec>(model["coefs"])),
  Ztv(Z.n_cols > 1), Htv(H.n_elem > 1), Ttv(T.n_slices > 1), Rtv(R.n_slices > 1),
  n(y.n_elem), m(a1.n_elem), k(R.n_cols), HH(arma::vec(Htv * (n - 1) + 1)),
  RR(arma::cube(m, m, Rtv * (n - 1) + 1)), xbeta(arma::vec(n, arma::fill::zeros)),
  engine(seed), zero_tol(1e-8) {
  
  if(xreg.n_cols > 0) {
    compute_xbeta();
  }
  compute_HH();
  compute_RR();
}
// from List for non-gaussian models, ng value is not actually used, cheap trick...
// with parameter indices
gssm::gssm(const List& model, arma::uvec Z_ind,
  arma::uvec T_ind, arma::uvec R_ind, unsigned int seed, bool ng) :
  y(as<arma::vec>(model["y"])), Z(as<arma::mat>(model["Z"])),
  H(arma::vec(y.n_elem)), T(as<arma::cube>(model["T"])), R(as<arma::cube>(model["R"])),
  a1(as<arma::vec>(model["a1"])), P1(as<arma::mat>(model["P1"])),
  xreg(as<arma::mat>(model["xreg"])), beta(as<arma::vec>(model["coefs"])),
  Ztv(Z.n_cols > 1), Htv(H.n_elem > 1), Ttv(T.n_slices > 1), Rtv(R.n_slices > 1),
  n(y.n_elem), m(a1.n_elem), k(R.n_cols), HH(arma::vec(Htv * (n - 1) + 1)),
  RR(arma::cube(m, m, Rtv * (n - 1) + 1)), xbeta(arma::vec(n, arma::fill::zeros)),
  Z_ind(Z_ind), H_ind(arma::uvec(1)), T_ind(T_ind), R_ind(R_ind), engine(seed), zero_tol(1e-8) {
  
  if(xreg.n_cols > 0) {
    compute_xbeta();
  }
  compute_HH();
  compute_RR();
}

//general constructor
gssm::gssm(arma::vec y, arma::mat Z, arma::vec H, arma::cube T,
  arma::cube R, arma::vec a1, arma::mat P1, arma::mat xreg,
  arma::vec beta, unsigned int seed) :
  y(y), Z(Z), H(H), T(T), R(R), a1(a1), P1(P1),  xreg(xreg), beta(beta), Ztv(Z.n_cols > 1),
  Htv(H.n_elem > 1), Ttv(T.n_slices > 1), Rtv(R.n_slices > 1),
  n(y.n_elem), m(a1.n_elem), k(R.n_cols), HH(arma::vec(Htv * (n - 1) + 1)),
  RR(arma::cube(m, m, Rtv * (n - 1) + 1)),
  xbeta(arma::vec(n, arma::fill::zeros)), engine(seed), zero_tol(1e-8) {
  
  if(xreg.n_cols > 0) {
    compute_xbeta();
  }
  compute_HH();
  compute_RR();
}


//general constructor with parameter indices
gssm::gssm(arma::vec y, arma::mat Z, arma::vec H, arma::cube T,
  arma::cube R, arma::vec a1, arma::mat P1, arma::mat xreg,
  arma::vec beta, arma::uvec Z_ind, arma::uvec H_ind,
  arma::uvec T_ind, arma::uvec R_ind, unsigned int seed) :
  y(y), Z(Z), H(H), T(T), R(R), a1(a1), P1(P1),  xreg(xreg), beta(beta), Ztv(Z.n_cols > 1),
  Htv(H.n_elem > 1), Ttv(T.n_slices > 1), Rtv(R.n_slices > 1),
  n(y.n_elem), m(a1.n_elem), k(R.n_cols), HH(arma::vec(Htv * (n - 1) + 1)),
  RR(arma::cube(m, m, Rtv * (n - 1) + 1)),
  xbeta(arma::vec(n, arma::fill::zeros)),
  Z_ind(Z_ind), H_ind(H_ind), T_ind(T_ind), R_ind(R_ind), engine(seed),
  zero_tol(1e-8) {
  
  if(xreg.n_cols > 0) {
    compute_xbeta();
  }
  compute_HH();
  compute_RR();
}


double gssm::prior_pdf(const arma::vec& theta, const arma::uvec& prior_types,
  const arma::mat& params) {
  
  double q = 0.0;
  
  for(unsigned int i = 0; i < theta.n_elem; i++) {
    switch(prior_types(i)) {
    case 0  :
      q += R::dunif(theta(i), params(0, i), params(1, i), 1);
      break;
    case 1  :
      if (theta(i) < 0) {
        return -arma::datum::inf;
      } else {
        q += log(2.0) + R::dnorm(theta(i), 0, params(0, i), 1);
      }
      break;
    case 2  :
      q += R::dnorm(theta(i), params(0, i), params(1, i), 1);
      break;
    }
  }
  return q;
}


double gssm::proposal(const arma::vec& theta, const arma::vec& theta_prop) {
  return 0.0;
}

void gssm::compute_RR(void){
  for (unsigned int t = 0; t < R.n_slices; t++) {
    RR.slice(t) = R.slice(t * Rtv) * R.slice(t * Rtv).t();
  }
}
void gssm::compute_HH(void){
  HH = square(H);
}

void gssm::compute_xbeta(void){
  xbeta = xreg * beta;
}



void gssm::update_model(arma::vec theta) {
  
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

arma::vec gssm::get_theta(void) {
  
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

double gssm::log_likelihood(bool demean) {
  
  double logLik = 0;
  arma::vec at = a1;
  arma::mat Pt = P1;
  
  if (demean && xreg.n_cols > 0) {
    for (unsigned int t = 0; t < n; t++) {
      logLik += uv_filter(y(t) - xbeta(t), Z.unsafe_col(t * Ztv), HH(t * Htv),
        T.slice(t * Ttv), RR.slice(t * Rtv), at, Pt, zero_tol);
    }
  } else {
    for (unsigned int t = 0; t < n; t++) {
      logLik += uv_filter(y(t), Z.unsafe_col(t * Ztv), HH(t * Htv),
        T.slice(t * Ttv), RR.slice(t * Rtv), at, Pt, zero_tol);
    }
  }
  
  return logLik;
}


double gssm::filter(arma::mat& at, arma::mat& att, arma::cube& Pt,
  arma::cube& Ptt, bool demean) {
  
  double logLik = 0;
  
  at.col(0) = a1;
  Pt.slice(0) = P1;
  
  if (demean && xreg.n_cols > 0) {
    for (unsigned int t = 0; t < n; t++) {
      // update
      logLik += uv_filter_update(y(t) - xbeta(t), Z.col(t * Ztv), HH(t * Htv),
        at.col(t), Pt.slice(t), att.col(t), Ptt.slice(t), zero_tol);
      // prediction
      uv_filter_predict(T.slice(t * Ttv), RR.slice(t * Rtv), att.col(t),
        Ptt.slice(t), at.col(t + 1),  Pt.slice(t + 1));
    }
  } else {
    for (unsigned int t = 0; t < n; t++) {
      // update
      logLik += uv_filter_update(y(t), Z.col(t * Ztv), HH(t * Htv),
        at.col(t), Pt.slice(t), att.col(t), Ptt.slice(t), zero_tol);
      // prediction
      uv_filter_predict(T.slice(t * Ttv), RR.slice(t * Rtv), att.col(t),
        Ptt.slice(t), at.col(t + 1),  Pt.slice(t + 1));
    }
  }
  return logLik;
  
}

/* Fast state smoothing, only returns smoothed estimates of states
* which are needed in simulation smoother
*/
arma::mat gssm::fast_smoother(bool demean) {
  
  arma::mat at(m, n);
  arma::mat Pt(m, m);
  
  arma::vec vt(n);
  arma::vec Ft(n);
  arma::mat Kt(m, n);
  
  at.col(0) = a1;
  Pt = P1;
  arma::vec y_tmp = y;
  if (demean && xreg.n_cols > 0) {
    y -= xbeta;
  }
  for (unsigned int t = 0; t < (n - 1); t++) {
    Ft(t) = arma::as_scalar(Z.col(t * Ztv).t() * Pt * Z.col(t * Ztv) + HH(t * Htv));
    if (arma::is_finite(y(t)) && Ft(t) > zero_tol) {
      Kt.col(t) = Pt * Z.col(t * Ztv) / Ft(t);
      vt(t) = arma::as_scalar(y(t) - Z.col(t * Ztv).t() * at.col(t));
      at.col(t + 1) = T.slice(t * Ttv) * (at.col(t) + Kt.col(t) * vt(t));
      Pt = arma::symmatu(T.slice(t * Ttv) * (Pt - Kt.col(t) * Kt.col(t).t() * Ft(t)) * T.slice(t * Ttv).t() + RR.slice(t * Rtv));
    } else {
      at.col(t + 1) = T.slice(t * Ttv) * at.col(t);
      Pt = arma::symmatu(T.slice(t * Ttv) * Pt * T.slice(t * Ttv).t() + RR.slice(t * Rtv));
    }
  }
  unsigned int t = n - 1;
  Ft(t) = arma::as_scalar(Z.col(t * Ztv).t() * Pt * Z.col(t * Ztv) + HH(t * Htv));
  if (arma::is_finite(y(t)) && Ft(t) > zero_tol) {
    vt(t) = arma::as_scalar(y(t) - Z.col(t * Ztv).t() * at.col(t));
    Kt.col(t) = Pt * Z.col(t * Ztv) / Ft(t);
  }
  arma::mat rt(m, n);
  rt.col(n - 1).zeros();
  
  for (int t = (n - 1); t > 0; t--) {
    if (arma::is_finite(y(t)) && Ft(t) > zero_tol){
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
    at.col(t + 1) = T.slice(t * Ttv) * at.col(t) + RR.slice(t * Rtv) * rt.col(t);
  }
  if (demean && xreg.n_cols > 0) {
    y = y_tmp;
  }
  return at;
}


/* Fast state smoothing which returns also Ft, Kt and Lt which can be used
* in subsequent calls of smoother in simulation smoother.
*/

arma::mat gssm::fast_smoother2(arma::vec& Ft, arma::mat& Kt, arma::cube& Lt, bool demean) {
  
  arma::mat at(m, n);
  arma::mat Pt(m, m);
  
  arma::vec vt(n);
  
  at.col(0) = a1;
  Pt = P1;
  
  arma::vec y_tmp = y;
  if (demean && xreg.n_cols > 0) {
    y -= xbeta;
  }
  for (unsigned int t = 0; t < (n - 1); t++) {
    Ft(t) = arma::as_scalar(Z.col(t * Ztv).t() * Pt * Z.col(t * Ztv) + HH(t * Htv));
    if (arma::is_finite(y(t)) && Ft(t) > zero_tol) {
      Kt.col(t) = Pt * Z.col(t * Ztv) / Ft(t);
      vt(t) = arma::as_scalar(y(t) - Z.col(t * Ztv).t() * at.col(t));
      at.col(t + 1) = T.slice(t * Ttv) * (at.col(t) + Kt.col(t) * vt(t));
      Pt = arma::symmatu(T.slice(t * Ttv) * (Pt - Kt.col(t) * Kt.col(t).t() * Ft(t)) * T.slice(t * Ttv).t() + RR.slice(t * Rtv));
    } else {
      at.col(t + 1) = T.slice(t * Ttv) * at.col(t);
      Pt = arma::symmatu(T.slice(t * Ttv) * Pt * T.slice(t * Ttv).t() + RR.slice(t * Rtv));
    }
  }
  unsigned int t = n - 1;
  Ft(t) = arma::as_scalar(Z.col(t * Ztv).t() * Pt * Z.col(t * Ztv) + HH(t * Htv));
  if (arma::is_finite(y(t)) && Ft(t) > zero_tol) {
    vt(t) = arma::as_scalar(y(t) - Z.col(t * Ztv).t() * at.col(t));
    Kt.col(t) = Pt * Z.col(t * Ztv) / Ft(t);
  }
  arma::mat rt(m, n);
  rt.col(n - 1).zeros();
  
  for (int t = (n - 1); t > 0; t--) {
    if (arma::is_finite(y(t)) && Ft(t) > zero_tol){
      Lt.slice(t) = T.slice(t * Ttv) * (arma::eye(m, m) - Kt.col(t) * Z.col(t * Ztv).t());
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
    at.col(t + 1) = T.slice(t * Ttv) * at.col(t) + RR.slice(t * Rtv) * rt.col(t);
  }
  
  if (demean && xreg.n_cols > 0) {
    y = y_tmp;
  }
  return at;
}

/* Fast state smoothing which uses precomputed Ft, Kt and Lt.
*/

arma::mat gssm::precomp_fast_smoother(const arma::vec& Ft, const arma::mat& Kt,
  const arma::cube& Lt, bool demean) {
  
  arma::mat at(m, n);
  arma::mat Pt(m, m);
  
  arma::vec vt(n);
  
  at.col(0) = a1;
  Pt = P1;
  
  arma::vec y_tmp = y;
  if (demean && xreg.n_cols > 0) {
    y -= xbeta;
  }
  
  for (unsigned int t = 0; t < (n - 1); t++) {
    if (arma::is_finite(y(t)) && Ft(t) > zero_tol) {
      vt(t) = arma::as_scalar(y(t) - Z.col(t * Ztv).t() * at.col(t));
      at.col(t + 1) = T.slice(t * Ttv) * (at.col(t) + Kt.col(t) * vt(t));
    } else {
      at.col(t + 1) = T.slice(t * Ttv) * at.col(t);
    }
  }
  unsigned int t = n - 1;
  if (arma::is_finite(y(t)) && Ft(t) > zero_tol) {
    vt(t) = arma::as_scalar(y(t) - Z.col(t * Ztv).t() * at.col(t));
  }
  arma::mat rt(m, n);
  rt.col(n - 1).zeros();
  
  for (int t = (n - 1); t > 0; t--) {
    if (arma::is_finite(y(t)) && Ft(t) > zero_tol){
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
    at.col(t + 1) = T.slice(t * Ttv) * at.col(t) + RR.slice(t * Rtv) * rt.col(t);
  }
  
  if (demean && xreg.n_cols > 0) {
    y = y_tmp;
  }
  
  return at;
}

arma::cube gssm::sim_smoother(unsigned int nsim, bool demean) {
  
  // keep importance sampling more comparative with particle filtering
  bool antithetic = false;
  ///////
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
    
    arma::mat alphahat = fast_smoother2(Ft, Kt, Lt, demean);
    
    
    unsigned int nsim2;
    if(antithetic) {
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
          y(t) = arma::as_scalar(Z.col(t * Ztv).t() * aplus.col(t)) +
            H(t * Htv) * normal(engine);
        }
        arma::vec uk(k);
        for(unsigned int j = 0; j < k; j++) {
          uk(j) = normal(engine);
        }
        aplus.col(t + 1) = T.slice(t * Ttv) * aplus.col(t) + R.slice(t * Rtv) * uk;
      }
      if (arma::is_finite(y(n - 1))) {
        y(n - 1) = arma::as_scalar(Z.col((n - 1) * Ztv).t() * aplus.col(n - 1)) +
          H((n - 1) * Htv) * normal(engine);
      }
      
      asim.slice(i) = -precomp_fast_smoother(Ft, Kt, Lt, false) + aplus;
      if (antithetic){
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
          y(t) = arma::as_scalar(Z.col(t * Ztv).t() * aplus.col(t)) +
            H(t * Htv) * normal(engine);
        }
        arma::vec uk(k);
        for(unsigned int j = 0; j < k; j++) {
          uk(j) = normal(engine);
        }
        aplus.col(t + 1) = T.slice(t * Ttv) * aplus.col(t) + R.slice(t * Rtv) * uk;
      }
      if (arma::is_finite(y(n - 1))) {
        y(n - 1) = arma::as_scalar(Z.col((n - 1) * Ztv).t() * aplus.col(n - 1)) +
          H((n - 1) * Htv) * normal(engine);
      }
      
      asim.slice(nsim - 1) = alphahat - precomp_fast_smoother(Ft, Kt, Lt, false) + aplus;
    }
    
  } else {
    // for _single simulation_ this version is faster:
    //no xbeta as it is not part of the states
    // same thing also for a1 (although it is not that clear, see,
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
    
    asim.slice(0) += fast_smoother(demean);
    
  }
  
  y = y_tmp;
  
  return asim;
}

void gssm::smoother(arma::mat& at, arma::cube& Pt, bool demean) {
  
  
  at.col(0) = a1;
  Pt.slice(0) = P1;
  arma::vec vt(n);
  arma::vec Ft(n);
  arma::mat Kt(m, n);
  
  arma::vec y_tmp = y;
  if (demean && xreg.n_cols > 0) {
    y -= xbeta;
  }
  
  for (unsigned int t = 0; t < (n - 1); t++) {
    Ft(t) = arma::as_scalar(Z.col(t * Ztv).t() * Pt.slice(t) * Z.col(t * Ztv) +
      HH(t * Htv));
    if (arma::is_finite(y(t)) && Ft(t) > zero_tol) {
      Kt.col(t) = Pt.slice(t) * Z.col(t * Ztv) / Ft(t);
      vt(t) = arma::as_scalar(y(t) - Z.col(t * Ztv).t() * at.col(t));
      at.col(t + 1) = T.slice(t * Ttv) * (at.col(t) + Kt.col(t) * vt(t));
      Pt.slice(t + 1) = arma::symmatu(T.slice(t * Ttv) * (Pt.slice(t) -
        Kt.col(t) * Kt.col(t).t() * Ft(t)) * T.slice(t * Ttv).t() + RR.slice(t * Rtv));
    } else {
      at.col(t + 1) = T.slice(t * Ttv) * at.col(t);
      Pt.slice(t + 1) = arma::symmatu(T.slice(t * Ttv) * Pt.slice(t) * T.slice(t * Ttv).t() +
        RR.slice(t * Rtv));
    }
  }
  unsigned int t = n - 1;
  Ft(t) = arma::as_scalar(Z.col(t * Ztv).t() * Pt.slice(t) * Z.col(t * Ztv) +
    HH(t * Htv));
  if (arma::is_finite(y(t)) && Ft(t) > zero_tol) {
    vt(t) = arma::as_scalar(y(t) - Z.col(t * Ztv).t() * at.col(t));
    Kt.col(t) = Pt.slice(t) * Z.col(t * Ztv) / Ft(t);
  }
  
  
  arma::vec rt(m, arma::fill::zeros);
  arma::mat Nt(m, m, arma::fill::zeros);
  
  for (int t = (n - 1); t >= 0; t--) {
    if (arma::is_finite(y(t)) && Ft(t) > zero_tol){
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
  
  if (demean && xreg.n_cols > 0) {
    y = y_tmp;
  }
}



//smoother which returns also cov(alpha_t, alpha_t-1)
void gssm::smoother_ccov(arma::mat& at, arma::cube& Pt, arma::cube& ccov, bool demean) {

  at.col(0) = a1;
  Pt.slice(0) = P1;
  arma::vec vt(n);
  arma::vec Ft(n);
  arma::mat Kt(m, n);

  arma::vec y_tmp = y;
  if (demean && xreg.n_cols > 0) {
    y -= xbeta;
  }

  for (unsigned int t = 0; t < (n - 1); t++) {
    Ft(t) = arma::as_scalar(Z.col(t * Ztv).t() * Pt.slice(t) * Z.col(t * Ztv) +
      HH(t * Htv));
    if (arma::is_finite(y(t)) && Ft(t) > zero_tol) {
      Kt.col(t) = Pt.slice(t) * Z.col(t * Ztv) / Ft(t);
      vt(t) = arma::as_scalar(y(t) - Z.col(t * Ztv).t() * at.col(t));
      at.col(t + 1) = T.slice(t * Ttv) * (at.col(t) + Kt.col(t) * vt(t));
      Pt.slice(t + 1) = arma::symmatu(T.slice(t * Ttv) * (Pt.slice(t) -
        Kt.col(t) * Kt.col(t).t() * Ft(t)) * T.slice(t * Ttv).t() + RR.slice(t * Rtv));
    } else {
      at.col(t + 1) = T.slice(t * Ttv) * at.col(t);
      Pt.slice(t + 1) = arma::symmatu(T.slice(t * Ttv) * Pt.slice(t) * T.slice(t * Ttv).t() +
        RR.slice(t * Rtv));
    }
    ccov.slice(t) = Pt.slice(t+1); //store for smoothing;
  }
  unsigned int t = n - 1;
  Ft(t) = arma::as_scalar(Z.col(t * Ztv).t() * Pt.slice(t) * Z.col(t * Ztv) +
    HH(t * Htv));
  if (arma::is_finite(y(t)) && Ft(t) > zero_tol) {
    vt(t) = arma::as_scalar(y(t) - Z.col(t * Ztv).t() * at.col(t));
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
    if (arma::is_finite(y(t)) && Ft(t) > zero_tol){
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

  if (demean && xreg.n_cols > 0) {
    y = y_tmp;
  }
}

double gssm::run_mcmc(const arma::uvec& prior_types, const arma::mat& prior_pars,
  unsigned int n_iter, bool sim_states, unsigned int n_burnin,
  unsigned int n_thin, double gamma, double target_acceptance, arma::mat& S,
  bool end_ram, arma::mat& theta_store, arma::vec& posterior_store,
  arma::cube& alpha_store) {
  
  unsigned int n_par = prior_types.n_elem;
  unsigned int n_samples = floor((n_iter - n_burnin) / n_thin);
  double acceptance_rate = 0.0;
  
  arma::vec theta = get_theta();
  double prior = prior_pdf(theta, prior_types, prior_pars);
  double ll = log_likelihood(true);
  
  arma::mat alpha(m, n * sim_states);
  if (sim_states) {
    alpha = sim_smoother(1, true);
  }
  
  unsigned int j = 0;
  
  if (n_burnin == 0) {
    if (sim_states) {
      alpha_store.slice(0) = alpha;
    }
    theta_store.col(0) = theta;
    posterior_store(0) = ll + prior;
    acceptance_rate++;
    j++;
  }
  
  double accept_prob = 0.0;
  
  std::normal_distribution<> normal(0.0, 1.0);
  std::uniform_real_distribution<> unif(0.0, 1.0);
  
  for (unsigned int i = 1; i < n_iter; i++) {
    
    // sample from standard normal distribution
    arma::vec u(n_par);
    for(unsigned int ii = 0; ii < n_par; ii++) {
      u(ii) = normal(engine);
    }
    
    // propose new theta
    arma::vec theta_prop = theta + S * u;
    // compute prior
    double prior_prop = prior_pdf(theta_prop, prior_types, prior_pars);
    
    if (prior_prop > -arma::datum::inf) {
      // update parameters
      update_model(theta_prop);
      // compute log-likelihood with proposed theta
      double ll_prop = log_likelihood(true);
      //compute the acceptance probability
      // use explicit min(...) as we need this value later
      double q = proposal(theta, theta_prop);
      accept_prob = std::min(1.0, exp(ll_prop - ll + prior_prop - prior + q));
      //accept
      if (unif(engine) < accept_prob) {
        if (i >= n_burnin) {
          acceptance_rate++;
        }
        ll = ll_prop;
        prior = prior_prop;
        theta = theta_prop;
        if (sim_states) {
          alpha = sim_smoother(1, true);
        }
      }
    } else accept_prob = 0.0;
    
    //store
    if ((i >= n_burnin) && (i % n_thin == 0) && j < n_samples) {
      posterior_store(j) = ll + prior;
      theta_store.col(j) = theta;
      if (sim_states) {
        alpha_store.slice(j) = alpha;
      }
      
      j++;
    }
    
    if (!end_ram || i < n_burnin) {
      adjust_S(S, u, accept_prob, target_acceptance, i, gamma);
    }
    
  }
  
  return acceptance_rate / (n_iter - n_burnin);
  
}

double gssm::mcmc_summary(const arma::uvec& prior_types, const arma::mat& prior_pars,
  unsigned int n_iter, unsigned int n_burnin,
  unsigned int n_thin, double gamma, double target_acceptance, arma::mat& S,
  bool end_ram, arma::mat& theta_store, arma::vec& posterior_store,
  arma::mat& alphahat, arma::cube& Vt) {
  
  unsigned int n_samples = posterior_store.n_elem;
  arma::cube alpha_store(m, n, 0);
  
  double acceptance_rate = run_mcmc(prior_types, prior_pars, n_iter, false,
    n_burnin, n_thin, gamma, target_acceptance, S, end_ram,
    theta_store, posterior_store, alpha_store);
  
  arma::cube Valpha(m, m, n, arma::fill::zeros);
  
  arma::vec theta = theta_store.col(0);
  update_model(theta);
  smoother(alphahat, Vt, true);
  arma::mat alphahat_i = alphahat;
  arma::cube Vt_i = Vt;
  for (unsigned int i = 1; i < n_samples; i++) {
    if(arma::any(theta_store.col(i) != theta_store.col(i-1))) {
      
      arma::vec theta = theta_store.col(i);
      update_model(theta);
      smoother(alphahat_i, Vt_i, true);
    }
    arma::mat diff = (alphahat_i - alphahat);
    alphahat += diff / (i + 1);
    for (unsigned int t = 0; t < n; t++) {
      Valpha.slice(t) += diff.col(t) * (alphahat_i.col(t) - alphahat.col(t)).t();
    }
    Vt += (Vt_i - Vt) / (i + 1);
  }
  Vt += Valpha / n_samples; // Var[E(alpha)] + E[Var(alpha)]
  
  return acceptance_rate;
}

List gssm::predict(const arma::uvec& prior_types, const arma::mat& prior_pars,
  unsigned int n_iter, unsigned int n_burnin, unsigned int n_thin, double gamma,
  double target_acceptance, arma::mat S, unsigned int n_ahead,
  unsigned int interval, arma::vec probs) {
  
  unsigned int n_samples = floor((n_iter - n_burnin) / n_thin);
  
  unsigned int n_par = prior_types.n_elem;
  arma::vec theta = get_theta();
  double prior = prior_pdf(theta, prior_types, prior_pars);
  
  arma::mat y_mean(n_ahead, n_samples);
  arma::mat y_var(n_ahead, n_samples);
  
  arma::mat at(m, n + 1);
  arma::cube Pt(m, m, n + 1);
  arma::mat att(m, n);
  arma::cube Ptt(m, m, n);
  filter(at, att, Pt, Ptt, true);
  
  unsigned int j = 0;
  
  if (n_burnin == 0){
    for (unsigned int t = n - n_ahead; t < n; t++) {
      y_mean(t - n + n_ahead, j) = arma::as_scalar(Z.col(Ztv * t).t() * at.col(t));
      if(xreg.n_cols > 0) {
        y_mean(t - n + n_ahead, j) += xbeta(t);
      }
      y_var(t - n + n_ahead, j) = arma::as_scalar(Z.col(Ztv * t).t() * Pt.slice(t) * Z.col(Ztv * t));
      if (interval == 2) {
        y_var(t - n + n_ahead, j) += HH(Htv * t);
      }
    }
    j++;
  }
  
  double ll = log_likelihood(true);
  double accept_prob = 0.0;
  std::normal_distribution<> normal(0.0, 1.0);
  std::uniform_real_distribution<> unif(0.0, 1.0);
  for (unsigned int i = 1; i < n_iter; i++) {
    // sample from standard normal distribution
    arma::vec u(n_par);
    for(unsigned int ii = 0; ii < n_par; ii++) {
      u(ii) = normal(engine);
    }
    // propose new theta
    arma::vec theta_prop = theta + S * u;
    // check prior
    double prior_prop = prior_pdf(theta_prop, prior_types, prior_pars);
    
    if (prior_prop > -arma::datum::inf) {
      // update parameters
      update_model(theta_prop);
      // compute log-likelihood with proposed theta
      double ll_prop = log_likelihood(true);
      //compute the acceptance probability
      // use explicit min(...) as we need this value later
      double q = proposal(theta, theta_prop);
      accept_prob = std::min(1.0, exp(ll_prop - ll + prior_prop - prior + q));
      
      
      //accept
      if (unif(engine) < accept_prob) {
        ll = ll_prop;
        prior = prior_prop;
        theta = theta_prop;
        filter(at, att, Pt, Ptt, true);
      }
    } else accept_prob = 0.0;
    
    if ((i >= n_burnin) && (i % n_thin == 0)) {
      update_model(theta);
      for (unsigned int t = n - n_ahead; t < n; t++) {
        y_mean(t - n + n_ahead, j) = arma::as_scalar(Z.col(Ztv * t).t() * at.col(t));
        if(xreg.n_cols > 0) {
          y_mean(t - n + n_ahead, j) += xbeta(t);
        }
        y_var(t - n + n_ahead, j) = arma::as_scalar(Z.col(Ztv * t).t() * Pt.slice(t) * Z.col(Ztv * t));
        if (interval == 2) {
          y_var(t - n + n_ahead, j) += HH(Htv * t);
        }
      }
      j++;
    }
    
    adjust_S(S, u, accept_prob, target_acceptance, i, gamma);
    
  }
  
  
  arma::inplace_trans(y_mean);
  arma::inplace_trans(y_var);
  y_var = sqrt(y_var);
  arma::mat intv = intervals(y_mean, y_var, probs, n_ahead);
  return List::create(Named("intervals") = intv, Named("y_mean") = y_mean,
    Named("y_sd") = y_var);
}


arma::mat gssm::predict2(const arma::uvec& prior_types,
  const arma::mat& prior_pars, unsigned int n_iter,
  unsigned int n_burnin, unsigned int n_thin, double gamma,
  double target_acceptance, arma::mat S, unsigned int n_ahead,
  unsigned int interval) {
  
  unsigned int n_samples = floor((n_iter - n_burnin) / n_thin);
  
  arma::mat pred_store(n_ahead, n_samples);
  
  unsigned int n_par = prior_types.n_elem;
  arma::vec theta = get_theta();
  double prior = prior_pdf(theta, prior_types, prior_pars);
  arma::cube alpha = sim_smoother(1, true).tube(0, n - n_ahead, m - 1,  n - 1);
  
  unsigned int j = 0;
  std::normal_distribution<> normal(0.0, 1.0);
  
  if (n_burnin == 0){
    for (unsigned int t = 0; t < n_ahead; t++) {
      pred_store(t, 0) = arma::as_scalar(Z.col(Ztv * (n - n_ahead + t)).t() * alpha.slice(0).col(t));
    }
    
    if(xreg.n_cols > 0) {
      pred_store.col(0) +=  xbeta.subvec(n - n_ahead, n - 1);
    }
    if (interval == 2) {
      for (unsigned int t = 0; t < n_ahead; t++) {
        pred_store.row(t).col(0) += H(Htv * (n - n_ahead + t)) * normal(engine);
      }
    }
    j++;
  }
  
  double ll = log_likelihood(true);
  double accept_prob = 0.0;
  
  std::uniform_real_distribution<> unif(0.0, 1.0);
  for (unsigned int i = 1; i < n_iter; i++) {
    // sample from standard normal distribution
    arma::vec u(n_par);
    for(unsigned int ii = 0; ii < n_par; ii++) {
      u(ii) = normal(engine);
    }
    // propose new theta
    arma::vec theta_prop = theta + S * u;
    // check prior
    // check prior
    double prior_prop = prior_pdf(theta_prop, prior_types, prior_pars);
    
    if (prior_prop > -arma::datum::inf) {
      // update parameters
      update_model(theta_prop);
      // compute log-likelihood with proposed theta
      double ll_prop = log_likelihood(true);
      //compute the acceptance probability
      // use explicit min(...) as we need this value later
      double q = proposal(theta, theta_prop);
      accept_prob = std::min(1.0, exp(ll_prop - ll + prior_prop - prior + q));
      
      //accept
      if (unif(engine) < accept_prob) {
        ll = ll_prop;
        prior = prior_prop;
        theta = theta_prop;
        alpha = sim_smoother(1, true).tube(0, n - n_ahead, m - 1,  n - 1);
      }
    } else accept_prob = 0.0;
    
    if ((i >= n_burnin) && (i % n_thin == 0)) {
      update_model(theta);
      
      for (unsigned int t = 0; t < n_ahead; t++) {
        pred_store(t, j) = arma::as_scalar(Z.col(Ztv * (n - n_ahead + t)).t() *
          alpha.slice(0).col(t));
      }
      
      if(xreg.n_cols > 0) {
        pred_store.col(j) +=  xbeta.subvec(n - n_ahead, n - 1);
      }
      if (interval == 2) {
        for (unsigned int t = 0; t < n_ahead; t++) {
          pred_store.row(t).col(j) += H(Htv * (n - n_ahead + t)) * normal(engine);
        }
      }
      j++;
    }
    
    adjust_S(S, u, accept_prob, target_acceptance, i, gamma);
    
  }
  
  return pred_store;
  
}

//particle filter
double gssm::particle_filter(unsigned int nsim, arma::cube& alphasim, arma::mat& V, arma::umat& ind) {
  
  std::normal_distribution<> normal(0.0, 1.0);
  std::uniform_real_distribution<> unif(0.0, 1.0);
  
  arma::uvec nonzero = arma::find(P1.diag() > 0);
  arma::mat L_P1(m, m, arma::fill::zeros);
  if (nonzero.n_elem > 0) {
    L_P1.submat(nonzero, nonzero) =
      arma::chol(P1.submat(nonzero, nonzero), "lower");
  }
  for (unsigned int i = 0; i < nsim; i++) {
    arma::vec um(m);
    for(unsigned int j = 0; j < m; j++) {
      um(j) = normal(engine);
    }
    alphasim.slice(i).col(0) = a1 + L_P1 * um;
  }
  
  arma::vec Vnorm(nsim);
  double logU = 0.0;
  if(arma::is_finite(y(0))) {
    for (unsigned int i = 0; i < nsim; i++) {
      V(i, 0) = R::dnorm(y(0),
        arma::as_scalar(Z.col(0).t() * alphasim.slice(i).col(0) + xbeta(0)),
        H(0), 0);
    }
    double sumw = arma::sum(V.col(0));
    if(sumw > 0.0){
      Vnorm = V.col(0) / sumw;
    } else {
      return -arma::datum::inf;
    }
    logU = log(arma::mean(V.col(0)));
  } else {
    V.col(0).ones();
    Vnorm.fill(1.0/nsim);
  }
  
  for (unsigned int t = 0; t < (n - 1); t++) {
    
    arma::vec r(nsim);
    for (unsigned int i = 0; i < nsim; i++) {
      r(i) = unif(engine);
    }
    
    ind.col(t) = stratified_sample(Vnorm, r, nsim);
    
    arma::mat alphatmp(m, nsim);
    
    for (unsigned int i = 0; i < nsim; i++) {
      alphatmp.col(i) = alphasim.slice(ind(i, t)).col(t);
    }
    for (unsigned int i = 0; i < nsim; i++) {
      arma::vec uk(k);
      for(unsigned int j = 0; j < k; j++) {
        uk(j) = normal(engine);
      }
      alphasim.slice(i).col(t + 1) = T.slice(t * Ttv) * alphatmp.col(i) +
        R.slice(t * Rtv) * uk;
    }
    
    if(arma::is_finite(y(t + 1))) {
      for (unsigned int i = 0; i < nsim; i++) {
        V(i, t + 1) = R::dnorm(y(t + 1),
          arma::as_scalar(Z.col((t + 1) * Ztv).t() * alphasim.slice(i).col(t + 1) + xbeta(t + 1)),
          H((t + 1) * Htv), 0);
      }
      double sumw = arma::sum(V.col(t + 1));
      if(sumw > 0.0){
        Vnorm = V.col(t + 1) / sumw;
      } else {
        return -arma::datum::inf;
      }
      logU += log(arma::mean(V.col(t + 1)));
    } else {
      V.col(t + 1).ones();
      Vnorm.fill(1.0/nsim);
    }
    
    
  }
  
  return logU;
}
// 
// //psi-auxiliary particle filter
// double gssm::psi_filter(unsigned int nsim, arma::cube& alphasim, arma::mat& V, arma::umat& ind) {
// 
//   arma::mat alphahat(m, n);
//   arma::cube Vt(m, m, n);
//   arma::cube Ct(m, m, n);
//   smoother_ccov(alphahat, Vt, Ct, true);
//   conditional_dist_helper(Vt, Ct);
//   
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
//   arma::vec Vnorm(nsim);
//   double logU = 0.0;
//   if(arma::is_finite(y(0))) {
//     V.col(0).fill(log_likelihood(true));
// //    for (unsigned int i = 0; i < nsim; i++) {
//       // V(i, 0) = R::dnorm(y(0),
//       //   arma::as_scalar(Z.col(0).t() * alphasim.slice(i).col(0) + xbeta(0)),
//       //   H(0), 0);
//     }
//     double sumw = arma::sum(V.col(0));
//     if(sumw > 0.0){
//       Vnorm = V.col(0) / sumw;
//     } else {
//       return -arma::datum::inf;
//     }
//     logU = log(arma::mean(V.col(0)));
//   } else {
//     V.col(0).ones();
//     Vnorm.fill(1.0/nsim);
//   }
// 
//   for (unsigned int t = 0; t < (n - 1); t++) {
// 
//     arma::vec r(nsim);
//     for (unsigned int i = 0; i < nsim; i++) {
//       r(i) = unif(engine);
//     }
// 
//     ind.col(t) = stratified_sample(Vnorm, r, nsim);
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
//       alphasim.slice(i).col(t + 1) = T.slice(t * Ttv) * alphatmp.col(i) +
//         R.slice(t * Rtv) * uk;
//     }
// 
//     if(arma::is_finite(y(t + 1))) {
//       for (unsigned int i = 0; i < nsim; i++) {
//         V(i, t + 1) = R::dnorm(y(t + 1),
//           arma::as_scalar(Z.col((t + 1) * Ztv).t() * alphasim.slice(i).col(t + 1) + xbeta(t + 1)),
//           H((t + 1) * Htv), 0);
//       }
//       double sumw = arma::sum(V.col(t + 1));
//       if(sumw > 0.0){
//         Vnorm = V.col(t + 1) / sumw;
//       } else {
//         return -arma::datum::inf;
//       }
//       logU += log(arma::mean(V.col(t + 1)));
//     } else {
//       V.col(t + 1).ones();
//       Vnorm.fill(1.0/nsim);
//     }
// 
// 
//   }
// 
//   return logU;
// }

void gssm::backtrack_pf2(const arma::cube& alpha, arma::mat& V, const arma::umat& ind) {
  
  unsigned int nsim = alpha.n_slices;
  V.col(n-1) = V.col(n-1) / arma::sum(V.col(n-1));
  
  for (int t = n - 1; t > 0; t--) {
    arma::mat B(nsim, nsim);
    arma::vec Vnorm = V.col(t-1) / arma::sum(V.col(t-1));
    for (unsigned int i = 0; i < nsim; i++) {
      for (unsigned int j = 0; j < nsim; j++) {
        B(j, i) = Vnorm(j) * dmvnorm1(alpha.slice(i).col(t),
          T.slice((t-1) * Ttv) * alpha.slice(j).col(t - 1),
          R.slice((t-1) * Rtv), true, false);
      }
    }
    B.each_row() /= arma::sum(B, 0);
    V.col(t-1) = B * V.col(t);
  }
}

arma::mat gssm::backward_simulate(arma::cube& alpha, arma::mat& V, arma::umat& ind) {
  
  unsigned int nsim = alpha.n_slices;
  arma::vec I(n);
  arma::vec Vnorm = V.col(n-1) / arma::sum(V.col(n-1));
  std::discrete_distribution<> sample(Vnorm.begin(), Vnorm.end());
  arma::mat alphasim(m, n);
  I(n-1) = sample(engine);
  alphasim.col(n - 1) = alpha.slice(I(n - 1)).col(n - 1);
  for (int t = n - 1; t > 0; t--) {
    arma::vec b(nsim);
    arma::vec Vnorm = V.col(t-1) / arma::sum(V.col(t-1));
    for (unsigned int j = 0; j < nsim; j++) {
      b(j) = Vnorm(j) * dmvnorm1(alpha.slice(I(t)).col(t),
        T.slice((t-1) * Ttv) * alpha.slice(j).col(t - 1),
        R.slice((t-1) * Rtv), true, false);
    }
    b /= arma::sum(b);
    std::discrete_distribution<> sample(b.begin(), b.end());
    I(t-1) = sample(engine);
    alphasim.col(t - 1) = alpha.slice(I(t - 1)).col(t - 1);
  }
  return alphasim;
}


//psi-auxiliary particle filter
double gssm::psi_filter(unsigned int nsim, arma::cube& alphasim, arma::mat& V, 
  arma::umat& ind) {
  
  arma::mat alphahat(m, n);
  arma::cube Vt(m, m, n, arma::fill::ones);
  arma::cube Ct(m, m, n);
  double ll = log_likelihood(true);
  smoother_ccov(alphahat, Vt, Ct, true);
  conditional_dist_helper(Vt, Ct);
  std::normal_distribution<> normal(0.0, 1.0);
  std::uniform_real_distribution<> unif(0.0, 1.0);
  for (unsigned int i = 0; i < nsim; i++) {
    arma::vec um(m);
    for(unsigned int j = 0; j < m; j++) {
      um(j) = normal(engine);
    }
    alphasim.slice(i).col(0) = alphahat.col(0) + Vt.slice(0) * um;
  }
  
  for (unsigned int t = 0; t < (n - 1); t++) {
    
    for (unsigned int i = 0; i < nsim; i++) {
      arma::vec um(m);
      for(unsigned int j = 0; j < m; j++) {
        um(j) = normal(engine);
      }
      alphasim.slice(i).col(t + 1) = alphahat.col(t + 1) + 
        Ct.slice(t + 1) * (alphasim.slice(i).col(t) - alphahat.col(t)) + Vt.slice(t + 1) * um;
    }
  }
  
  return ll;
}


