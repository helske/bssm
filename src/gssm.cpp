#include "gssm.h"
//general constructor
gssm::gssm(arma::vec y, arma::mat Z, arma::vec H, arma::cube T,
  arma::cube R, arma::vec a1, arma::mat P1, arma::mat xreg,
  arma::vec beta, unsigned int seed) :
  y(y), Z(Z), H(H), T(T), R(R), a1(a1), P1(P1), Ztv(Z.n_cols > 1),
  Htv(H.n_elem > 1), Ttv(T.n_slices > 1), Rtv(R.n_slices > 1),
  n(y.n_elem), m(a1.n_elem), k(R.n_cols), HH(arma::vec(Htv * (n - 1) + 1)),
  RR(arma::cube(m, m, Rtv * (n - 1) + 1)), xreg(xreg), beta(beta),
  xbeta(arma::vec(n, arma::fill::zeros)), engine(seed) {

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
  y(y), Z(Z), H(H), T(T), R(R), a1(a1), P1(P1), Ztv(Z.n_cols > 1),
  Htv(H.n_elem > 1), Ttv(T.n_slices > 1), Rtv(R.n_slices > 1),
  n(y.n_elem), m(a1.n_elem), k(R.n_cols), HH(arma::vec(Htv * (n - 1) + 1)),
  RR(arma::cube(m, m, Rtv * (n - 1) + 1)), xreg(xreg), beta(beta),
  xbeta(arma::vec(n, arma::fill::zeros)),
  Z_ind(Z_ind), H_ind(H_ind), T_ind(T_ind), R_ind(R_ind), engine(seed) {

  if(xreg.n_cols > 0) {
    compute_xbeta();
  }
  compute_HH();
  compute_RR();
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

double gssm::log_likelihood(void) {

  double logLik = 0;

  arma::vec at = a1;
  arma::mat Pt = P1;

  for (unsigned int t = 0; t < n; t++) {
    // update
    logLik += uv_filter(y(t), Z.col(t * Ztv), HH(t * Htv),
      xbeta(t), T.slice(t * Ttv), RR.slice(t * Rtv), at, Pt);
  }
  return logLik;
}


double gssm::filter(arma::mat& at, arma::mat& att, arma::cube& Pt,
  arma::cube& Ptt) {

  double logLik = 0;

  at.col(0) = a1;
  Pt.slice(0) = P1;

  for (unsigned int t = 0; t < n; t++) {
    // update
    logLik += uv_filter_update(y(t), Z.col(t * Ztv), HH(t * Htv),
      xbeta(t), at.col(t), Pt.slice(t), att.col(t), Ptt.slice(t));
    // prediction
    uv_filter_predict(T.slice(t * Ttv), RR.slice(t * Rtv), att.col(t),
      Ptt.slice(t), at.col(t + 1),  Pt.slice(t + 1));
  }
  return logLik;

}

/* Fast state smoothing, only returns smoothed estimates of states
 * which are needed in simulation smoother
 */
arma::mat gssm::fast_smoother(void) {

  arma::mat at(m, n);
  arma::mat Pt(m, m);

  arma::vec vt(n);
  arma::vec Ft(n);
  arma::mat Kt(m, n);

  at.col(0) = a1;
  Pt = P1;

  for (unsigned int t = 0; t < (n - 1); t++) {
    if (arma::is_finite(y(t))) {
      Ft(t) = arma::as_scalar(Z.col(t * Ztv).t() * Pt * Z.col(t * Ztv) + HH(t * Htv));
      Kt.col(t) = Pt * Z.col(t * Ztv) / Ft(t);
      vt(t) = arma::as_scalar(y(t) - Z.col(t * Ztv).t() * at.col(t) - xbeta(t));
      at.col(t + 1) = T.slice(t * Ttv) * (at.col(t) + Kt.col(t) * vt(t));
      Pt = arma::symmatu(T.slice(t * Ttv) * (Pt - Kt.col(t) * Kt.col(t).t() * Ft(t)) * T.slice(t * Ttv).t() + RR.slice(t * Rtv));
    } else {
      at.col(t + 1) = T.slice(t * Ttv) * at.col(t);
      Pt = arma::symmatu(T.slice(t * Ttv) * Pt * T.slice(t * Ttv).t() + RR.slice(t * Rtv));
    }
  }
  unsigned int t = n - 1;
  if (arma::is_finite(y(t))) {
    vt(t) = arma::as_scalar(y(t) - Z.col(t * Ztv).t() * at.col(t) - xbeta(t));
    Ft(t) = arma::as_scalar(Z.col(t * Ztv).t() * Pt * Z.col(t * Ztv) + HH(t * Htv));
    Kt.col(t) = Pt * Z.col(t * Ztv) / Ft(t);
  }
  arma::mat rt(m, n);
  rt.col(n - 1).zeros();

  for (int t = (n - 1); t > 0; t--) {
    if (arma::is_finite(y(t))){
      arma::mat L = T.slice(t * Ttv) * (arma::eye(m, m) - Kt.col(t) * Z.col(t * Ztv).t());
      rt.col(t - 1) = Z.col(t * Ztv) / Ft(t) * vt(t) + L.t() * rt.col(t);
    } else {
      rt.col(t - 1) = T.slice(t * Ttv).t() * rt.col(t);
    }
  }
  if (arma::is_finite(y(0))){
    arma::mat L = T.slice(0) * (arma::eye(m, m) - Kt.col(0) * Z.col(0).t());
    at.col(0) = a1 + P1 * (Z.col(0) / Ft(0) * vt(0) + L.t() * rt.col(0));
  } else {
    at.col(0) = a1 + P1 * T.slice(0).t() * rt.col(0);
  }
  for (unsigned int t = 0; t < (n - 1); t++) {
    at.col(t + 1) = T.slice(t * Ttv) * at.col(t) + RR.slice(t * Rtv) * rt.col(t);
  }

  return at;
}


/* Fast state smoothing which returns also Ft, Kt and Lt which can be used
 * in subsequent calls of smoother in simulation smoother.
 */

arma::mat gssm::fast_smoother2(arma::vec& Ft, arma::mat& Kt, arma::cube& Lt) {

  arma::mat at(m, n);
  arma::mat Pt(m, m);

  arma::vec vt(n);

  at.col(0) = a1;
  Pt = P1;

  for (unsigned int t = 0; t < (n - 1); t++) {
    if (arma::is_finite(y(t))) {
      Ft(t) = arma::as_scalar(Z.col(t * Ztv).t() * Pt * Z.col(t * Ztv) + HH(t * Htv));
      Kt.col(t) = Pt * Z.col(t * Ztv) / Ft(t);
      vt(t) = arma::as_scalar(y(t) - Z.col(t * Ztv).t() * at.col(t) - xbeta(t));
      at.col(t + 1) = T.slice(t * Ttv) * (at.col(t) + Kt.col(t) * vt(t));
      Pt = arma::symmatu(T.slice(t * Ttv) * (Pt - Kt.col(t) * Kt.col(t).t() * Ft(t)) * T.slice(t * Ttv).t() + RR.slice(t * Rtv));
    } else {
      at.col(t + 1) = T.slice(t * Ttv) * at.col(t);
      Pt = arma::symmatu(T.slice(t * Ttv) * Pt * T.slice(t * Ttv).t() + RR.slice(t * Rtv));
    }
  }
  unsigned int t = n - 1;
  if (arma::is_finite(y(t))) {
    vt(t) = arma::as_scalar(y(t) - Z.col(t * Ztv).t() * at.col(t) - xbeta(t));
    Ft(t) = arma::as_scalar(Z.col(t * Ztv).t() * Pt * Z.col(t * Ztv) + HH(t * Htv));
    Kt.col(t) = Pt * Z.col(t * Ztv) / Ft(t);
  }
  arma::mat rt(m, n);
  rt.col(n - 1).zeros();

  for (int t = (n - 1); t > 0; t--) {
    if (arma::is_finite(y(t))){
      Lt.slice(t) = T.slice(t * Ttv) * (arma::eye(m, m) - Kt.col(t) * Z.col(t * Ztv).t());
      rt.col(t - 1) = Z.col(t * Ztv) / Ft(t) * vt(t) + Lt.slice(t).t() * rt.col(t);
    } else {
      rt.col(t - 1) = T.slice(t * Ttv).t() * rt.col(t);
    }
  }
  if (arma::is_finite(y(0))){
    arma::mat L = T.slice(0) * (arma::eye(m, m) - Kt.col(0) * Z.col(0).t());
    at.col(0) = a1 + P1 * (Z.col(0) / Ft(0) * vt(0) + L.t() * rt.col(0));
  } else {
    at.col(0) = a1 + P1 * T.slice(0).t() * rt.col(0);
  }
  for (unsigned int t = 0; t < (n - 1); t++) {
    at.col(t + 1) = T.slice(t * Ttv) * at.col(t) + RR.slice(t * Rtv) * rt.col(t);
  }

  return at;
}

/* Fast state smoothing which uses precomputed Ft, Kt and Lt.
 */

arma::mat gssm::precomp_fast_smoother(const arma::vec& Ft, const arma::mat& Kt,
  const arma::cube& Lt) {

  arma::mat at(m, n);
  arma::mat Pt(m, m);

  arma::vec vt(n);

  at.col(0) = a1;
  Pt = P1;

  for (unsigned int t = 0; t < (n - 1); t++) {
    if (arma::is_finite(y(t))) {
      vt(t) = arma::as_scalar(y(t) - Z.col(t * Ztv).t() * at.col(t) - xbeta(t));
      at.col(t + 1) = T.slice(t * Ttv) * (at.col(t) + Kt.col(t) * vt(t));
    } else {
      at.col(t + 1) = T.slice(t * Ttv) * at.col(t);
    }
  }
  unsigned int t = n - 1;
  if (arma::is_finite(y(t))) {
    vt(t) = arma::as_scalar(y(t) - Z.col(t * Ztv).t() * at.col(t) - xbeta(t));
  }
  arma::mat rt(m, n);
  rt.col(n - 1).zeros();

  for (int t = (n - 1); t > 0; t--) {
    if (arma::is_finite(y(t))){
      rt.col(t - 1) = Z.col(t * Ztv) / Ft(t) * vt(t) + Lt.slice(t).t() * rt.col(t);
    } else {
      rt.col(t - 1) = T.slice(t * Ttv).t() * rt.col(t);
    }
  }
  if (arma::is_finite(y(0))){
    arma::mat L = T.slice(0) * (arma::eye(m, m) - Kt.col(0) * Z.col(0).t());
    at.col(0) = a1 + P1 * (Z.col(0) / Ft(0) * vt(0) + L.t() * rt.col(0));
  } else {
    at.col(0) = a1 + P1 * T.slice(0).t() * rt.col(0);
  }

  for (unsigned int t = 0; t < (n - 1); t++) {
    at.col(t + 1) = T.slice(t * Ttv) * at.col(t) + RR.slice(t * Rtv) * rt.col(t);
  }

  return at;
}

arma::cube gssm::sim_smoother(unsigned int nsim) {

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

    arma::mat alphahat = fast_smoother2(Ft, Kt, Lt);

    unsigned int nsim2 = std::floor(nsim / 2);

    for(unsigned int i = 0; i < nsim2; i++) {
      arma::mat aplus(m, n);

      arma::vec um(m);
      for(unsigned int j = 0; j < m; j++) {
        um(j) = normal(engine);
      }
      aplus.col(0) = a1 + L_P1 * um;
      for (unsigned int t = 0; t < (n - 1); t++) {
        if (arma::is_finite(y(t))) {
          y(t) = arma::as_scalar(Z.col(t * Ztv).t() * aplus.col(t) + xbeta(t)) +
            H(t * Htv) * normal(engine);
        }
        arma::vec uk(k);
        for(unsigned int j = 0; j < k; j++) {
          uk(j) = normal(engine);
        }
        aplus.col(t + 1) = T.slice(t * Ttv) * aplus.col(t) + R.slice(t * Rtv) * uk;
      }
      if (arma::is_finite(y(n - 1))) {
        y(n - 1) = arma::as_scalar(Z.col((n - 1) * Ztv).t() * aplus.col(n - 1) + xbeta(n - 1)) +
          H((n - 1) * Htv) * normal(engine);
      }

      asim.slice(i) = -precomp_fast_smoother(Ft, Kt, Lt) + aplus;
      asim.slice(i + nsim2) = alphahat - asim.slice(i);
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
          y(t) = arma::as_scalar(Z.col(t * Ztv).t() * aplus.col(t) + xbeta(t)) +
            H(t * Htv) * normal(engine);
        }
        arma::vec uk(k);
        for(unsigned int j = 0; j < k; j++) {
          uk(j) = normal(engine);
        }
        aplus.col(t + 1) = T.slice(t * Ttv) * aplus.col(t) + R.slice(t * Rtv) * uk;
      }
      if (arma::is_finite(y(n - 1))) {
        y(n - 1) = arma::as_scalar(Z.col((n - 1) * Ztv).t() * aplus.col(n - 1) + xbeta(n - 1)) +
          H((n - 1) * Htv) * normal(engine);
      }

      asim.slice(nsim - 1) = alphahat - precomp_fast_smoother(Ft, Kt, Lt) + aplus;
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

    asim.slice(0) += fast_smoother();

  }

  y = y_tmp;

  return asim;
}

void gssm::smoother(arma::mat& at, arma::cube& Pt) {


  at.col(0) = a1;
  Pt.slice(0) = P1;
  arma::vec vt(n);
  arma::vec Ft(n);
  arma::mat Kt(m, n);

  for (unsigned int t = 0; t < (n - 1); t++) {
    if (arma::is_finite(y(t))) {
      Ft(t) = arma::as_scalar(Z.col(t * Ztv).t() * Pt.slice(t) * Z.col(t * Ztv) +
        HH(t * Htv));
      Kt.col(t) = Pt.slice(t) * Z.col(t * Ztv) / Ft(t);
      vt(t) = arma::as_scalar(y(t) - Z.col(t * Ztv).t() * at.col(t) - xbeta(t));
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
  if (arma::is_finite(y(t))) {
    vt(t) = arma::as_scalar(y(t) - Z.col(t * Ztv).t() * at.col(t) - xbeta(t));
    Ft(t) = arma::as_scalar(Z.col(t * Ztv).t() * Pt.slice(t) * Z.col(t * Ztv) +
      HH(t * Htv));
    Kt.col(t) = Pt.slice(t) * Z.col(t * Ztv) / Ft(t);
  }


  arma::vec rt(m, arma::fill::zeros);
  arma::mat Nt(m, m, arma::fill::zeros);

  for (int t = (n - 1); t > 0; t--) {
    if (arma::is_finite(y(t))){
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
  if (arma::is_finite(y(0))){
    arma::mat L = T.slice(0) * (arma::eye(m, m) - Kt.col(0) * Z.col(0).t());
    rt = Z.col(0) / Ft(0) * vt(0) + L.t() * rt;
    Nt = arma::symmatu(Z.col(0) * Z.col(0).t() / Ft(0) + L.t() * Nt * L);
  } else {
    rt = T.slice(0).t() * rt;
    Nt = arma::symmatu(T.slice(0).t() * Nt * T.slice(0));
  }
  at.col(0) += Pt.slice(0) * rt;
  Pt.slice(0) -= arma::symmatu(Pt.slice(0) * Nt * Pt.slice(0));

}


List gssm::mcmc_full(arma::vec theta_lwr, arma::vec theta_upr,
  unsigned int n_iter, unsigned int nsim_states, unsigned int n_burnin,
  unsigned int n_thin, double gamma, double target_acceptance, arma::mat S) {

  unsigned int npar = theta_lwr.n_elem;

  unsigned int n_samples = floor((n_iter - n_burnin) / n_thin);
  arma::mat theta_store(npar, n_samples);
  arma::cube alpha_store(m, n, nsim_states * n_samples);
  arma::vec ll_store(n_samples);
  double acceptance_rate = 0.0;

  arma::vec theta = get_theta();

  // everything is conditional on beta
  double ll = log_likelihood();

  arma::cube alpha = sim_smoother(nsim_states);

  unsigned int j = 0;

  if (n_burnin == 0) {
    theta_store.col(0) = theta;
    alpha_store.slices(0, nsim_states - 1) = alpha;
    ll_store(0) = ll;
    acceptance_rate++;
    j++;
  }

  double accept_prob = 0;
  double ll_prop = 0;
  std::normal_distribution<> normal(0.0, 1.0);
  std::uniform_real_distribution<> unif(0.0, 1.0);

  for (unsigned int i = 1; i < n_iter; i++) {
    // sample from standard normal distribution
    arma::vec u(npar);
    for(unsigned int ii = 0; ii < npar; ii++) {
      u(ii) = normal(engine);
    }
    // propose new theta
    arma::vec theta_prop = theta + S * u;
    // check prior
    bool inrange = sum(theta_prop >= theta_lwr && theta_prop <= theta_upr) == npar;

    if (inrange) {
      // update parameters
      update_model(theta_prop);
      // compute log-likelihood with proposed theta
      ll_prop = log_likelihood();
      //compute the acceptance probability
      // use explicit min(...) as we need this value later
      double q = proposal(theta, theta_prop);
      accept_prob = std::min(1.0, exp(ll_prop - ll + q));
    } else accept_prob = 0;


    //accept
    if (inrange && unif(engine) < accept_prob) {
      if (i >= n_burnin) {
        acceptance_rate++;
      }
      ll = ll_prop;
      theta = theta_prop;
      alpha = sim_smoother(nsim_states);
    }
    //store
    if ((i >= n_burnin) && (i % n_thin == 0)) {
      ll_store(j) = ll;
      theta_store.col(j) = theta;
      alpha_store.slices(j * nsim_states, (j + 1) * nsim_states - 1) = alpha;
      j++;
    }

    double change = accept_prob - target_acceptance;
    u = S * u / arma::norm(u) * sqrt(std::min(1.0, npar * pow(i, -gamma)) *
      std::abs(change));


    if(change > 0) {
      S = cholupdate(S, u);

    } else {
      if(change < 0){
        //update S unless numerical problems occur
        arma::mat Stmp = choldowndate(S, u);
        arma::uvec cond = arma::find(arma::diagvec(Stmp) < 0);
        if (cond.n_elem == 0) {
          S = Stmp;
        }
      }
    }

  }
  arma::inplace_trans(theta_store);

  return List::create(Named("alpha") = alpha_store,
    Named("theta") = theta_store,
    Named("acceptance_rate") = acceptance_rate / (n_iter - n_burnin),
    Named("S") = S,  Named("logLik") = ll_store);

}


List gssm::mcmc_param(arma::vec theta_lwr, arma::vec theta_upr,
  unsigned int n_iter, unsigned int n_burnin,
  unsigned int n_thin, double gamma, double target_acceptance, arma::mat S) {

  unsigned int npar = theta_lwr.n_elem;

  unsigned int n_samples = floor((n_iter - n_burnin) / n_thin);
  arma::mat theta_store(npar, n_samples);
  arma::vec ll_store(n_samples);
  double acceptance_rate = 0.0;

  arma::vec theta = get_theta();
  double ll = log_likelihood();

  unsigned int j = 0;

  if (n_burnin == 0) {
    theta_store.col(0) = theta;
    ll_store(0) = ll;
    acceptance_rate++;
    j++;
  }

  double accept_prob = 0;
  double ll_prop = 0;
  std::normal_distribution<> normal(0.0, 1.0);
  std::uniform_real_distribution<> unif(0.0, 1.0);

  for (unsigned int i = 1; i < n_iter; i++) {
    // sample from standard normal distribution
    arma::vec u(npar);
    for(unsigned int ii = 0; ii < npar; ii++) {
      u(ii) = normal(engine);
    }
    // propose new theta
    arma::vec theta_prop = theta + S * u;
    // check prior
    bool inrange = sum(theta_prop >= theta_lwr && theta_prop <= theta_upr) == npar;

    if (inrange) {
      // update parameters
      update_model(theta_prop);
      // compute log-likelihood with proposed theta
      ll_prop = log_likelihood();
      //compute the acceptance probability
      // use explicit min(...) as we need this value later
      double q = proposal(theta, theta_prop);
      accept_prob = std::min(1.0, exp(ll_prop - ll + q));
    } else accept_prob = 0;


    //accept
    if (inrange && unif(engine) < accept_prob) {
      if (i >= n_burnin) {
        acceptance_rate++;
      }
      ll = ll_prop;
      theta = theta_prop;
    }
    //store
    if ((i >= n_burnin) && (i % n_thin == 0)) {
      ll_store(j) = ll;
      theta_store.col(j) = theta;
      j++;
    }

    double change = accept_prob - target_acceptance;
    u = S * u / arma::norm(u) * sqrt(std::min(1.0, npar * pow(i, -gamma)) *
      std::abs(change));


    if(change > 0) {
      S = cholupdate(S, u);

    } else {
      if(change < 0){
        //update S unless numerical problems occur
        arma::mat Stmp = choldowndate(S, u);
        arma::uvec cond = arma::find(arma::diagvec(Stmp) < 0);
        if (cond.n_elem == 0) {
          S = Stmp;
        }
      }
    }

  }
  arma::inplace_trans(theta_store);
  return List::create(Named("theta") = theta_store,
    Named("acceptance_rate") = acceptance_rate / (n_iter - n_burnin),
    Named("S") = S,  Named("logLik") = ll_store);

}

List gssm::mcmc_summary(arma::vec theta_lwr, arma::vec theta_upr,
  unsigned int n_iter, unsigned int n_burnin,
  unsigned int n_thin, double gamma, double target_acceptance, arma::mat S) {

  unsigned int npar = theta_lwr.n_elem;
  arma::vec theta = get_theta();

  arma::mat alphahat(m, n, arma::fill::zeros);
  arma::cube Vt(m, m, n, arma::fill::zeros);
  arma::cube Valpha(m, m, n, arma::fill::zeros);

  arma::mat alphahat_i(m, n, arma::fill::zeros);
  arma::cube Vt_i(m, m, n, arma::fill::zeros);

  smoother(alphahat_i, Vt_i);

  double ll = log_likelihood();

  double acceptance_rate = 0.0;
  double accept_prob = 0;
  double ll_prop = 0;

  unsigned int n_samples = floor((n_iter - n_burnin) / n_thin);
  arma::mat theta_store(npar, n_samples);
  arma::vec ll_store(n_samples);
  unsigned int j = 0;

  if (n_burnin == 0){
    alphahat = alphahat_i;
    for (unsigned int t = 0; t < n; t++) {
      Valpha.slice(t) = alphahat_i.col(t) * alphahat_i.col(t).t();
    }
    Vt = Vt_i;
    theta_store.col(0) = theta;
    ll_store(0) = ll;
    acceptance_rate++;
    j++;
  }
  std::normal_distribution<> normal(0.0, 1.0);
  std::uniform_real_distribution<> unif(0.0, 1.0);
  for (unsigned int i = 1; i < n_iter; i++) {
    // sample from standard normal distribution
    arma::vec u(npar);
    for(unsigned int ii = 0; ii < npar; ii++) {
      u(ii) = normal(engine);
    }
    // propose new theta
    arma::vec theta_prop = theta + S * u;
    // check prior
    bool inrange = sum(theta_prop >= theta_lwr && theta_prop <= theta_upr) == npar;


    if (inrange) {
      // update parameters
      update_model(theta_prop);
      // compute log-likelihood with proposed theta
      ll_prop = log_likelihood();
      //compute the acceptance probability
      // use explicit min(...) as we need this value later
      double q = proposal(theta, theta_prop);
      accept_prob = std::min(1.0, exp(ll_prop - ll + q));
    } else accept_prob = 0;

    //accept
    if (inrange && unif(engine) < accept_prob) {
      if (i >= n_burnin) {
        acceptance_rate++;
      }
      ll = ll_prop;
      theta = theta_prop;
      smoother(alphahat_i, Vt_i);
    }

    if ((i >= n_burnin) && (i % n_thin == 0)) {
      theta_store.col(j) = theta;
      ll_store(j) = ll;
      arma::mat diff = (alphahat_i - alphahat);
      alphahat += diff / (j + 1);
      for (unsigned int t = 0; t < n; t++) {
        Valpha.slice(t) += diff.col(t) * (alphahat_i.col(t) - alphahat.col(t)).t();
      }
      Vt += (Vt_i - Vt) / (j + 1);
      j++;
    }

    double change = accept_prob - target_acceptance;
    u = S * u / arma::norm(u) * sqrt(std::min(1.0, npar * pow(i, -gamma)) * std::abs(change));

    if(change > 0) {
      S = cholupdate(S, u);

    } else {
      if(change < 0){
        //update S unless numerical problems occur
        arma::mat Stmp = choldowndate(S, u);
        arma::uvec cond = arma::find(arma::diagvec(Stmp) < 0);
        if (cond.n_elem == 0) {
          S = Stmp;
        }
      }
    }

  }

  Vt += Valpha / (j + 1);
  arma::inplace_trans(alphahat);
  arma::inplace_trans(theta_store);
  return List::create(Named("alphahat") = alphahat,
    Named("Vt") = Vt, Named("theta") = theta_store,
    Named("acceptance_rate") = acceptance_rate / (n_iter - n_burnin),
    Named("S") = S,  Named("logLik") = ll_store);
}

List gssm::predict(arma::vec theta_lwr,
  arma::vec theta_upr, unsigned int n_iter,
  unsigned int n_burnin, unsigned int n_thin, double gamma,
  double target_acceptance, arma::mat S, unsigned int n_ahead,
  unsigned int interval, arma::vec probs) {

  unsigned int n_samples = floor((n_iter - n_burnin) / n_thin);

  unsigned int npar = theta_lwr.n_elem;
  arma::vec theta = get_theta();



  arma::mat y_mean(n_ahead, n_samples);
  arma::mat y_var(n_ahead, n_samples);

  arma::mat at(m, n + 1);
  arma::cube Pt(m, m, n + 1);
  arma::mat att(m, n);
  arma::cube Ptt(m, m, n);
  filter(at, att, Pt, Ptt);

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

  double ll = log_likelihood();
  double accept_prob = 0;
  double ll_prop = 0;
  std::normal_distribution<> normal(0.0, 1.0);
  std::uniform_real_distribution<> unif(0.0, 1.0);
  for (unsigned int i = 1; i < n_iter; i++) {
    // sample from standard normal distribution
    arma::vec u(npar);
    for(unsigned int ii = 0; ii < npar; ii++) {
      u(ii) = normal(engine);
    }
    // propose new theta
    arma::vec theta_prop = theta + S * u;
    // check prior
    bool inrange = sum(theta_prop >= theta_lwr && theta_prop <= theta_upr) == npar;

    if (inrange) {
      // update parameters
      update_model(theta_prop);
      // compute log-likelihood with proposed theta
      ll_prop = log_likelihood();
      //compute the acceptance probability
      // use explicit min(...) as we need this value later
      double q = proposal(theta, theta_prop);
      accept_prob = std::min(1.0, exp(ll_prop - ll + q));
    } else accept_prob = 0;

    //accept
    if (inrange && unif(engine) < accept_prob) {
      ll = ll_prop;
      theta = theta_prop;
      filter(at, att, Pt, Ptt);
    }

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
    double change = accept_prob - target_acceptance;
    u = S * u * sqrt(std::min(1.0, npar * pow(i, -gamma)) * std::abs(change)) /
    arma::norm(u);

    if(change > 0) {
      S = cholupdate(S, u);

    } else {
      if(change < 0){
        //update S unless numerical problems occur
        arma::mat Stmp = choldowndate(S, u);
        arma::uvec cond = arma::find(arma::diagvec(Stmp) < 0);
        if (cond.n_elem == 0) {
          S = Stmp;
        }
      }
    }

  }


  arma::inplace_trans(y_mean);
  arma::inplace_trans(y_var);
  y_var = sqrt(y_var);
  arma::mat intv = intervals(y_mean, y_var, probs, n_ahead);
  return List::create(Named("intervals") = intv, Named("y_mean") = y_mean,
    Named("y_sd") = y_var);
}


arma::mat gssm::predict2(arma::vec theta_lwr,
  arma::vec theta_upr, unsigned int n_iter, unsigned int nsim_states,
  unsigned int n_burnin, unsigned int n_thin, double gamma,
  double target_acceptance, arma::mat S, unsigned int n_ahead,
  unsigned int interval) {

  unsigned int n_samples = floor((n_iter - n_burnin) / n_thin);

  arma::mat pred_store(n_ahead, nsim_states * n_samples);

  unsigned int npar = theta_lwr.n_elem;
  arma::vec theta = get_theta();
  arma::cube alpha = sim_smoother(nsim_states).tube(0, n - n_ahead, m - 1,  n - 1);

  unsigned int j = 0;
  std::normal_distribution<> normal(0.0, 1.0);

  if (n_burnin == 0){
    for (unsigned int ii = 0; ii < nsim_states; ii++) {
      for (unsigned int t = 0; t < n_ahead; t++) {
        pred_store(t, ii) = arma::as_scalar(Z.col(Ztv * (n - n_ahead + t)).t() * alpha.slice(ii).col(t));
      }
    }
    if(xreg.n_cols > 0) {
      for (unsigned int ii = 0; ii < nsim_states; ii++) {
        pred_store.col(ii) +=  xbeta.subvec(n - n_ahead, n - 1);
      }
    }
    if (interval == 2) {
      for (unsigned int t = 0; t < n_ahead; t++) {
        arma::vec u2(nsim_states);
        for(unsigned int ii = 0; ii < nsim_states; ii++) {
          u2(ii) = normal(engine);
        }
        pred_store.row(t).cols(0, nsim_states - 1) += H(Htv * (n - n_ahead + t)) * u2;
      }
    }
    j++;
  }

  double ll = log_likelihood();
  double accept_prob = 0;
  double ll_prop = 0;

  std::uniform_real_distribution<> unif(0.0, 1.0);
  for (unsigned int i = 1; i < n_iter; i++) {
    // sample from standard normal distribution
    arma::vec u(npar);
    for(unsigned int ii = 0; ii < npar; ii++) {
      u(ii) = normal(engine);
    }
    // propose new theta
    arma::vec theta_prop = theta + S * u;
    // check prior
    bool inrange = sum(theta_prop >= theta_lwr && theta_prop <= theta_upr) == npar;

    if (inrange) {
      // update parameters
      update_model(theta_prop);
      // compute log-likelihood with proposed theta
      ll_prop = log_likelihood();
      //compute the acceptance probability
      // use explicit min(...) as we need this value later
      double q = proposal(theta, theta_prop);
      accept_prob = std::min(1.0, exp(ll_prop - ll + q));
    } else accept_prob = 0;

    //accept
    if (inrange && unif(engine) < accept_prob) {
      ll = ll_prop;
      theta = theta_prop;
      alpha = sim_smoother(nsim_states).tube(0, n - n_ahead, m - 1,  n - 1);
    }

    if ((i >= n_burnin) && (i % n_thin == 0)) {

      for (unsigned int ii = j * nsim_states; ii < (j + 1) * nsim_states; ii++) {
        for (unsigned int t = 0; t < n_ahead; t++) {
          pred_store(t, ii) = arma::as_scalar(Z.col(Ztv * (n - n_ahead + t)).t() *
            alpha.slice(ii - j * nsim_states).col(t));
        }
      }

      if(xreg.n_cols > 0) {
        update_model(theta);
        for (unsigned int ii = j * nsim_states; ii < (j + 1) * nsim_states; ii++) {
          pred_store.col(ii) +=  xbeta.subvec(n - n_ahead, n - 1);
        }
      }
      if (interval == 2) {
        for (unsigned int t = 0; t < n_ahead; t++) {
          arma::vec u2(nsim_states);
          for(unsigned int ii = 0; ii < nsim_states; ii++) {
            u2(ii) = normal(engine);
          }
          pred_store.row(t).cols(j * nsim_states, (j + 1) * nsim_states - 1) +=
            H(Htv * (n - n_ahead + t)) * u2;
        }
      }
      j++;
    }

    double change = accept_prob - target_acceptance;
    u = S * u * sqrt(std::min(1.0, npar * pow(i, -gamma)) * std::abs(change)) /
    arma::norm(u);

    if(change > 0) {
      S = cholupdate(S, u);

    } else {
      if(change < 0){
        //update S unless numerical problems occur
        arma::mat Stmp = choldowndate(S, u);
        arma::uvec cond = arma::find(arma::diagvec(Stmp) < 0);
        if (cond.n_elem == 0) {
          S = Stmp;
        }
      }
    }

  }

  return pred_store;

}
// [[Rcpp::plugins(openmp)]]
arma::cube sample_states(gssm mod, const arma::mat& theta,
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
